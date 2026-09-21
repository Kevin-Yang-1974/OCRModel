#!/usr/bin/env bash
# One four-GPU BSCC task: official GLM-OCR prompt-only baseline on a fixed
# 64-page MTHv2 validation subset, with decoder visual-attention probe samples.
#
# The four workers are disjoint eval-only shards.  They do not train, read the
# test manifest, or use layout routing/mask feedback.  All pages keep scalar
# attention reductions; four pre-registered pages additionally retain a few
# patch-level maps for offline heatmap rendering.
#SBATCH --job-name=glmattn64
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${BSCC_GLM_EXPERIMENT_ROOT:-${workspace}/glm_ocr_layout_mask_routing}"
code_root="${BSCC_GLM_CODE_ROOT:-${experiment_root}/code/ocrmodel}"
# The branch-specific code root is isolated, while the established environment,
# model, and dataset remain shared read-only assets outside the source tree.
env_dir="${BSCC_GLM_ENV:-${workspace}/glm_ocr_layout_ot/envs/glm-ocr-py311-cu128}"
model_dir="${BSCC_GLM_MODEL:-${workspace}/models/GLM-OCR-ca5d8b3}"
dataset_root="${BSCC_MTHV2_PAGE_ROOT:-${workspace}/datasets/MTHv2/converted/mthv2_layout_page_v1_real}"
pytorch_package="/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311"
nvidia_env="${BSCC_GLM_NVIDIA_ENV:-${workspace}/envs/anandasky}"

run_id="${GLMOCR_BSCC_RUN_ID:-glmocr_mthv2_baseline_attention_64val_bscc_260921_v1}"
source_branch="${GLMOCR_SOURCE_BRANCH:-glm-ocr-layout-mask-routing}"
seed=42
page_count=64
shard_count=4
num_queries=512
max_pixels="${GLMOCR_ATTENTION_MAX_PIXELS:-1003520}"
max_eval_new_tokens="${GLMOCR_ATTENTION_MAX_NEW_TOKENS:-1536}"
probe_layers="${GLMOCR_ATTENTION_PROBE_LAYERS:-0,4,8,12}"
probe_heads="${GLMOCR_ATTENTION_PROBE_HEADS:-2,3,8,10,11,12,14,15}"
heatmap_max_steps="${GLMOCR_ATTENTION_HEATMAP_MAX_STEPS:-8}"
render_layer="${GLMOCR_ATTENTION_RENDER_LAYER:-8}"

run_root="${experiment_root}/attention_runs/${run_id}"
subset_dir="${run_root}/subset"
subset_manifest="${subset_dir}/validation_subset_64.jsonl"
protocol_file="${run_root}/protocol.train_validation_64_no_test.json"
workers_root="${run_root}/workers"
merged_root="${run_root}/merged"
heatmap_root="${run_root}/heatmaps"
analysis_root="${run_root}/analysis"
python="${env_dir}/bin/python"

gpu_ids="${GLMOCR_BSCC_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0,1,2,3}}}"
[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    printf '{"event":"glmocr_bscc_attention_failed","error":"invalid_run_id"}\n' >&2
    exit 64
}
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]] || {
    printf '{"event":"glmocr_bscc_attention_failed","error":"exactly_four_numeric_gpus_required","gpu_ids":"%s"}\n' "${gpu_ids}" >&2
    exit 64
}
[[ -x "${python}" && -f "${pytorch_package}/env.sh" ]] || {
    printf '{"event":"glmocr_bscc_attention_failed","error":"missing_python_environment"}\n' >&2
    exit 66
}
[[ -f "${model_dir}/model.safetensors" && -f "${model_dir}/config.json" ]] || {
    printf '{"event":"glmocr_bscc_attention_failed","error":"missing_staged_glmocr_model"}\n' >&2
    exit 66
}
[[ -f "${dataset_root}/train/manifest.jsonl" && -f "${dataset_root}/validation/manifest.jsonl" ]] || {
    printf '{"event":"glmocr_bscc_attention_failed","error":"missing_mthv2_train_or_validation_manifest"}\n' >&2
    exit 66
}
for required in \
    "${code_root}/src/layout_ocr/train_screen.py" \
    "${code_root}/src/layout_ocr/attention_probe.py" \
    "${code_root}/tools/subset_mthv2_manifest.py" \
    "${code_root}/tools/audit_mthv2_manifest.py" \
    "${code_root}/tools/merge_decoder_attention_shards.py" \
    "${code_root}/tools/analyze_decoder_attention.py"; do
    [[ -f "${required}" ]] || {
        printf '{"event":"glmocr_bscc_attention_failed","error":"missing_code_file","path":"%s"}\n' "${required}" >&2
        exit 66
    }
done
[[ ! -e "${run_root}" ]] || {
    printf '{"event":"glmocr_bscc_attention_failed","error":"run_already_exists","run_root":"%s"}\n' "${run_root}" >&2
    exit 74
}

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
declare -A seen_gpu=()
for gpu in "${gpu_array[@]}"; do
    [[ -z "${seen_gpu[${gpu}]+present}" ]] || {
        printf '{"event":"glmocr_bscc_attention_failed","error":"duplicate_gpu_id","gpu":"%s"}\n' "${gpu}" >&2
        exit 64
    }
    seen_gpu["${gpu}"]=1
done

query_gpus() {
    command -v nvidia-smi >/dev/null 2>&1 || {
        printf '{"event":"glmocr_bscc_attention_failed","error":"nvidia_smi_missing"}\n' >&2
        exit 69
    }
    declare -A observed=()
    while IFS=',' read -r observed_id utilization; do
        observed_id="${observed_id//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [[ "${observed_id}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || {
            printf '{"event":"glmocr_bscc_attention_failed","error":"cannot_parse_gpu_utilization"}\n' >&2
            exit 69
        }
        observed["${observed_id}"]="${utilization}"
    done < <(nvidia-smi -i "${gpu_ids}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    for gpu in "${gpu_array[@]}"; do
        [[ -n "${observed[${gpu}]+present}" ]] || {
            printf '{"event":"glmocr_bscc_attention_failed","error":"requested_gpu_not_reported","gpu":"%s"}\n' "${gpu}" >&2
            exit 69
        }
        (( observed[${gpu}] < 50 )) || {
            printf '{"event":"glmocr_bscc_attention_failed","error":"gpu_admission_failed","gpu":"%s","utilization":%s}\n' "${gpu}" "${observed[${gpu}]}" >&2
            exit 75
        }
    done
}

source "${pytorch_package}/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"

torch_lib="${env_dir}/lib/python3.11/site-packages/torch/lib"
machine_arch="$(uname -m)"
case "${machine_arch}" in
    x86_64) cuda_arch="x86_64" ;;
    aarch64|arm64) cuda_arch="aarch64" ;;
    *) cuda_arch="${machine_arch}" ;;
esac
cuda_libraries="${torch_lib}"
system_cuda="/usr/local/cuda/targets/${cuda_arch}-linux/lib"
if [[ -d "${system_cuda}" ]]; then
    cuda_libraries="${system_cuda}:${cuda_libraries}"
fi
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    component_dir="${env_dir}/lib/python3.11/site-packages/nvidia/${component}/lib"
    if [[ -d "${component_dir}" ]]; then
        cuda_libraries="${cuda_libraries}:${component_dir}"
    fi
    component_dir="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    if [[ -d "${component_dir}" ]]; then
        cuda_libraries="${cuda_libraries}:${component_dir}"
    fi
done
export LD_LIBRARY_PATH="${cuda_libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

query_gpus
mkdir -p "${run_root}" "${subset_dir}" "${workers_root}" "${heatmap_root}" "${experiment_root}/tmp"
printf '{"event":"glmocr_bscc_attention_started","run_id":"%s","source_branch":"%s","pages":%s,"gpu_ids":"%s","layers":"%s","heads":"%s","test_manifest_read":false}\n' \
    "${run_id}" "${source_branch}" "${page_count}" "${gpu_ids}" "${probe_layers}" "${probe_heads}"

"${python}" "${code_root}/tools/subset_mthv2_manifest.py" \
    --input "${dataset_root}/validation/manifest.jsonl" \
    --output "${subset_manifest}" \
    --count "${page_count}" --seed "${seed}" --stratify-regions \
    > "${run_root}/subset_selection.json"

"${python}" "${code_root}/tools/audit_mthv2_manifest.py" \
    --train-manifest "${dataset_root}/train/manifest.jsonl" \
    --validation-manifest "${subset_manifest}" \
    --without-test --allow-count-mismatch --num-queries "${num_queries}" \
    --dataset-label "MTHv2_validation64_attention" \
    --protocol-label "glm_ocr_mthv2_baseline_attention_64val_v1" \
    --output "${protocol_file}" \
    > "${run_root}/protocol_audit.json"

heatmap_pages="${run_root}/heatmap_pages.txt"
"${python}" - "${subset_manifest}" "${workers_root}" "${heatmap_pages}" "${shard_count}" <<'PY'
import json
import sys
from pathlib import Path

manifest, workers_root, heatmap_pages, shard_count = map(Path, sys.argv[1:4])
shard_count = int(sys.argv[4])
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != 64:
    raise SystemExit(f"expected 64 subset rows, got {len(rows)}")
for index in range(shard_count):
    shard = rows[index::shard_count]
    if len(shard) != 16:
        raise SystemExit(f"shard {index} has {len(shard)} rows, expected 16")
    path = workers_root / "manifests" / f"shard-{index}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in shard),
        encoding="utf-8",
    )

ordered = sorted(rows, key=lambda row: (len(row.get("regions") or []), str(row["page_id"])))
indices = sorted({0, len(ordered) // 3, (2 * len(ordered)) // 3, len(ordered) - 1})
chosen = [ordered[index] for index in indices]
heatmap_pages.write_text("".join(f"{row['page_id']}\n" for row in chosen), encoding="utf-8")
(workers_root.parent / "heatmap_selection.json").write_text(
    json.dumps(
        {
            "rule": "min/one-third/two-thirds/max by region count then page_id",
            "pages": [row["page_id"] for row in chosen],
            "region_counts": [len(row.get("regions") or []) for row in chosen],
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

run_worker() {
    local index="$1"
    local gpu="${gpu_array[${index}]}"
    local worker_root="${workers_root}/shard-${index}"
    local worker_manifest="${workers_root}/manifests/shard-${index}.jsonl"
    local worker_log="${workers_root}/shard-${index}.log"
    local worker_tmp="${experiment_root}/tmp/${run_id}/shard-${index}"
    mkdir -p "${worker_tmp}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export TMPDIR="${worker_tmp}"
        export HF_HOME="${worker_tmp}/huggingface"
        export TRANSFORMERS_CACHE="${HF_HOME}"
        export GLMOCR_ATTENTION_PROBE="${worker_root}/attention.jsonl"
        export GLMOCR_ATTENTION_HEATMAP_DIR="${heatmap_root}/shard-${index}"
        export GLMOCR_ATTENTION_HEATMAP_PAGES="${heatmap_pages}"
        export GLMOCR_ATTENTION_HEATMAP_MAX_STEPS="${heatmap_max_steps}"
        export GLMOCR_ATTENTION_PROBE_LAYERS="${probe_layers}"
        export GLMOCR_ATTENTION_PROBE_HEADS="${probe_heads}"
        mkdir -p "${HF_HOME}"
        cd "${code_root}"
        exec "${python}" -m layout_ocr.train_screen \
            --mode content_only \
            --model-path "${model_dir}" \
            --train-manifest "${dataset_root}/train/manifest.jsonl" \
            --validation-manifest "${worker_manifest}" \
            --protocol-file "${protocol_file}" \
            --output-dir "${worker_root}" \
            --max-steps 1 --lr-schedule-steps 1 --warmup-steps 0 \
            --num-queries "${num_queries}" --seed "${seed}" \
            --experiment-label "glmocr_baseline_attention_64val_shard${index}" \
            --decoder-adaptation frozen \
            --initial-residual-scale 0.0 \
            --auxiliary-weight 0.0 --auxiliary-weight-start 0.0 \
            --max-pixels "${max_pixels}" \
            --max-eval-new-tokens "${max_eval_new_tokens}" \
            --validation-interval 1 --log-steps 16 \
            --adapter-precision fp32 \
            --layout-loss-profile ocr_only --query-assignment hungarian \
            --processor-mode fast --generation-mode plain \
            --layout-attention-probe \
            --layout-attention-probe-layers "${probe_layers}" \
            --layout-attention-probe-heads "${probe_heads}" \
            --eval-only
    ) > "${worker_log}" 2>&1
}

pids=()
for index in 0 1 2 3; do
    run_worker "${index}" &
    pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if (( failed != 0 )); then
    printf '{"event":"glmocr_bscc_attention_failed","error":"worker_failed","run_root":"%s"}\n' "${run_root}" >&2
    exit 1
fi

"${python}" "${code_root}/tools/merge_decoder_attention_shards.py" \
    --workers-root "${workers_root}" --output-dir "${merged_root}" \
    --expected-shards "${shard_count}" --expected-pages "${page_count}"

"${python}" "${code_root}/tools/analyze_decoder_attention.py" \
    --probe "${merged_root}/attention.jsonl" \
    --manifest "${subset_manifest}" \
    --predictions "${merged_root}/validation_predictions.jsonl" \
    --heatmap-dir "${heatmap_root}" \
    --output-dir "${analysis_root}" \
    --render-layer "${render_layer}" --render-pages 4 --render-steps-per-page 3 \
    --max-edge 2400

"${python}" - "${run_root}" "${analysis_root}/attention_summary.json" "${merged_root}/workers_summary.json" <<'PY'
import json
import sys
from pathlib import Path

run_root, attention_path, workers_path = map(Path, sys.argv[1:])
attention = json.loads(attention_path.read_text(encoding="utf-8"))
workers = json.loads(workers_path.read_text(encoding="utf-8"))
attention.update(
    {
        "run_id": run_root.name,
        "baseline": {
            "mode": "content_only",
            "decoder_adaptation": "frozen",
            "training_updates": 0,
            "model": "GLM-OCR-ca5d8b3",
            "max_pixels": 1003520,
            "max_eval_new_tokens": 1536,
            "generation_mode": "plain",
        },
        "workers": workers,
        "test_manifest_read": False,
        "test_used_for_selection": False,
    }
)
(run_root / "summary.json").write_text(
    json.dumps(attention, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(run_root / "DECODER_ATTENTION_COMPLETE").touch()
print(json.dumps({
    "event": "glmocr_bscc_attention_complete",
    "run_id": run_root.name,
    "pages": attention["pages"],
    "summary": str(run_root / "summary.json"),
}, ensure_ascii=False, separators=(",", ":")))
PY
