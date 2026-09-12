#!/usr/bin/env bash
#SBATCH --job-name=glmocr_val4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/glm_ocr_layout_ot"
code_root="${experiment_root}/code/ocrmodel"
env_dir="${experiment_root}/envs/glm-ocr-py311-cu128"
model_dir="${workspace}/models/GLM-OCR-ca5d8b3"
dataset_root="${BSCC_MTHV2_PAGE_ROOT:-${workspace}/datasets/MTHv2/converted/mthv2_layout_page_v1_real}"
pytorch_package="/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311"
run_id="${GLMOCR_BSCC_RUN_ID:-glmocr_mthv2_decoder_lora_lr1e5_20k_4gpu_official_layout_260912_v1}"
seed=42
gpu_utilization_limit=50
group_root="${experiment_root}/training_runs/${run_id}"
run_dir="${group_root}/seed${seed}"
protocol_file="${experiment_root}/protocols/${run_id}.train_validation_no_test.json"
validation_root="${run_dir}/parallel-validation"
validation_log="${group_root}/logs/seed${seed}.parallel-validation.log"

gpu_ids="${GLMOCR_BSCC_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0,1,2,3}}}"
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]] || {
    printf '{"event":"glmocr_bscc_validation_failed","error":"exactly_four_numeric_gpus_required","gpu_ids":"%s"}\n' "${gpu_ids}" >&2
    exit 64
}
IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
declare -A seen_gpu=()
for gpu in "${gpu_array[@]}"; do
    [[ -z "${seen_gpu[${gpu}]+present}" ]] || {
        printf '{"event":"glmocr_bscc_validation_failed","error":"duplicate_gpu_id","gpu":"%s"}\n' "${gpu}" >&2
        exit 64
    }
    seen_gpu["${gpu}"]=1
done

[[ -f "${pytorch_package}/env.sh" && -x "${env_dir}/bin/python" ]] || {
    printf '{"event":"glmocr_bscc_validation_failed","error":"missing_bscc_python_environment"}\n' >&2
    exit 66
}
source "${pytorch_package}/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "${group_root}/logs" "${group_root}/tmp"

"${env_dir}/bin/python" - "${run_dir}" "${protocol_file}" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
protocol_file = Path(sys.argv[2])
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
protocol = json.loads(protocol_file.read_text(encoding="utf-8"))
if summary.get("status") != "complete" or metadata.get("status") != "complete":
    raise SystemExit("training run is not complete")
if summary.get("test_manifest_read") is not False or metadata.get("test_manifest_read") is not False:
    raise SystemExit("training run is not test-free")
if metadata.get("world_size") != 4 or metadata.get("distributed_strategy") != "ddp":
    raise SystemExit("training run is not four-card DDP")
if metadata.get("decoder_adaptation") != "lora":
    raise SystemExit("training run does not use decoder LoRA")
if not math.isclose(float((metadata.get("decoder_lora_config") or {}).get("learning_rate")), 1e-5):
    raise SystemExit("training decoder LoRA learning rate is not 1e-5")
training = summary.get("training") or {}
natural_loop = training.get("natural_loop") or {}
if natural_loop.get("enabled") is not False:
    raise SystemExit("training run unexpectedly enabled natural-loop objective")
for name in ("scheduled_sampling", "loop_escape", "continuation_head"):
    if (training.get(name) or {}).get("enabled") is not False:
        raise SystemExit(f"training run enabled extra objective: {name}")
objective = training.get("loss_objective") or {}
if objective.get("formula") != "L_official + auxiliary_weight * L_layout":
    raise SystemExit(f"unexpected training loss objective: {objective}")
if not math.isclose(float(objective.get("layout_weight", -1.0)), 0.2):
    raise SystemExit("training layout loss weight is not 0.2")
if objective.get("extra_terms") != []:
    raise SystemExit(f"unexpected extra loss terms: {objective.get('extra_terms')}")
if protocol.get("test_manifest_read") is not False:
    raise SystemExit("validation protocol reads the test split")
expected_steps = [5000, 10000, 15000, 20000]
recorded_steps = [int(step) for step in (summary.get("training") or {}).get("checkpoint_steps", [])]
if recorded_steps != expected_steps:
    raise SystemExit(f"unexpected checkpoint steps: {recorded_steps}")
for step in expected_steps:
    checkpoint = run_dir / f"checkpoint-{step}"
    for filename in ("adapter.safetensors", "decoder_lora.safetensors", "checkpoint_health.json"):
        if not (checkpoint / filename).is_file():
            raise SystemExit(f"missing checkpoint artifact: {checkpoint / filename}")
PY

query_gpus() {
    command -v nvidia-smi >/dev/null 2>&1 || {
        printf '{"event":"glmocr_bscc_validation_failed","error":"nvidia_smi_missing"}\n' >&2
        exit 69
    }
    declare -A observed=()
    while IFS=',' read -r observed_id utilization; do
        observed_id="${observed_id//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [[ "${observed_id}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || {
            printf '{"event":"glmocr_bscc_validation_failed","error":"cannot_parse_gpu_utilization"}\n' >&2
            exit 69
        }
        observed["${observed_id}"]="${utilization}"
    done < <(nvidia-smi -i "${gpu_ids}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    for gpu in "${gpu_array[@]}"; do
        [[ -n "${observed[${gpu}]+present}" ]] || {
            printf '{"event":"glmocr_bscc_validation_failed","error":"requested_gpu_not_reported","gpu":"%s"}\n' "${gpu}" >&2
            exit 69
        }
        (( observed[${gpu}] < gpu_utilization_limit )) || {
            printf '{"event":"glmocr_bscc_validation_failed","error":"gpu_admission_failed","gpu":"%s","utilization":%s,"limit":%s}\n' "${gpu}" "${observed[${gpu}]}" "${gpu_utilization_limit}" >&2
            exit 75
        }
    done
}

[[ -f "${protocol_file}" && -f "${code_root}/src/layout_ocr/train_screen.py" ]] || {
    printf '{"event":"glmocr_bscc_validation_failed","error":"missing_validation_inputs"}\n' >&2
    exit 66
}
[[ ! -e "${validation_root}" ]] || {
    printf '{"event":"glmocr_bscc_validation_failed","error":"validation_output_already_exists","path":"%s"}\n' "${validation_root}" >&2
    exit 74
}
query_gpus
mkdir -p "${validation_root}"

train_manifest="${dataset_root}/train/manifest.jsonl"
validation_manifest="${dataset_root}/validation/manifest.jsonl"
torch_lib="${env_dir}/lib/python3.11/site-packages/torch/lib"
machine_arch="$(uname -m)"
case "${machine_arch}" in
    x86_64) cuda_target_arch="x86_64" ;;
    aarch64|arm64) cuda_target_arch="aarch64" ;;
    *) cuda_target_arch="${machine_arch}" ;;
esac
cuda_library_path="${torch_lib}"
system_cuda_library="/usr/local/cuda/targets/${cuda_target_arch}-linux/lib"
if [[ -d "${system_cuda_library}" ]]; then
    cuda_library_path="${system_cuda_library}:${cuda_library_path}"
fi

steps=(5000 10000 15000 20000)
pids=()
for index in "${!steps[@]}"; do
    step="${steps[${index}]}"
    gpu="${gpu_array[${index}]}"
    eval_dir="${validation_root}/step-${step}"
    eval_tmp="${group_root}/tmp/validation-seed${seed}-step${step}"
    eval_log="${group_root}/logs/seed${seed}.validation.step${step}.log"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export TMPDIR="${eval_tmp}"
        export HF_HOME="${TMPDIR}/huggingface"
        export TRANSFORMERS_CACHE="${HF_HOME}"
        export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
        export LD_LIBRARY_PATH="${cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
        mkdir -p "${TMPDIR}" "${HF_HOME}"
        cd "${code_root}"
        exec "${env_dir}/bin/python" -m layout_ocr.train_screen \
            --mode geometry \
            --model-path "${model_dir}" \
            --train-manifest "${train_manifest}" \
            --validation-manifest "${validation_manifest}" \
            --protocol-file "${protocol_file}" \
            --output-dir "${eval_dir}" \
            --per-device-batch-size 1 \
            --gradient-accumulation-steps 1 \
            --max-steps 20000 \
            --num-queries 512 \
            --seed "${seed}" \
            --experiment-label "bscc_decoder_lora_lr1e5_20k_4gpu_validation_step${step}" \
            --learning-rate 5e-5 \
            --decoder-adaptation lora \
            --decoder-lora-rank 8 \
            --decoder-lora-alpha 8 \
            --decoder-lora-dropout 0 \
            --decoder-learning-rate 1e-5 \
            --warmup-steps 216 \
            --lr-schedule-steps 20000 \
            --min-lr-ratio 0.1 \
            --residual-scale-cap 0.03 \
            --initial-residual-scale 0 \
            --auxiliary-weight 0.2 \
            --auxiliary-weight-start 0.2 \
            --auxiliary-ramp-steps 0 \
            --gate-freeze-steps 0 \
            --max-grad-norm 1.0 \
            --max-pixels 1003520 \
            --max-eval-new-tokens 1536 \
            --validation-interval 5000 \
            --log-steps 16 \
            --adapter-precision fp32 \
            --layout-loss-profile full \
            --query-assignment hungarian \
            --processor-mode fast \
            --generation-mode loop_recovery \
            --eval-checkpoint-dir "${run_dir}/checkpoint-${step}" \
            --eval-only
    ) > "${eval_log}" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failed=1
    fi
done
if (( failed != 0 )); then
    printf '{"event":"glmocr_bscc_validation_failed","error":"validation_worker_failed","run_id":"%s","log":"%s"}\n' "${run_id}" "${validation_log}" >&2
    exit 1
fi

"${env_dir}/bin/python" "${code_root}/tools/summarize_glmocr_parallel_validation.py" \
    --run-dir "${run_dir}" \
    --validation-root "${validation_root}" \
    --group-root "${group_root}" \
    --steps 5000,10000,15000,20000 \
    --expected-world-size 4 \
    > "${validation_log}" 2>&1

"${env_dir}/bin/python" - "${run_id}" "${run_dir}/selection.json" <<'PY'
import json
import sys

run_id, selection_path = sys.argv[1:]
selection = json.loads(open(selection_path, encoding="utf-8").read())
print(json.dumps({
    "event": "glmocr_bscc_parallel_validation_complete",
    "run_id": run_id,
    "world_size": 4,
    "checkpoint_steps": selection["checkpoint_steps"],
    "selected_step": selection["selected_step"],
    "selection_metric": selection["selection_metric"],
    "test_manifest_read": selection["test_manifest_read"],
    "test_used_for_selection": selection["test_used_for_selection"],
}, ensure_ascii=False, separators=(",", ":")))
PY
