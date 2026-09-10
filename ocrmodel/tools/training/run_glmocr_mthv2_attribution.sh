#!/usr/bin/env bash
# Run the three seed42 capacity-attribution groups serially on a fixed
# validation-only MTHv2 subset.  The script never opens the test manifest.
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code_root="${GLMOCR_A100_CODE_ROOT:-${remote_root}/code/ocrmodel}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
dataset_root="${GLMOCR_A100_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"
bundle_id="${GLMOCR_ATTRIBUTION_BUNDLE_ID:-glmocr_mthv2_attribution_128_v1}"
validation_pages="${GLMOCR_VALIDATION_SUBSET_PAGES:-32}"
validation_seed="${GLMOCR_VALIDATION_SUBSET_SEED:-42}"
gpu_ids="${GLMOCR_ATTRIBUTION_GPU_IDS:-0,1,2,3,4}"
noop_run_id_override=""
adapter_run_id_override=""
lora_run_id_override=""
validation_manifest_override=""
protocol_file_override=""
ddp_timeout_seconds="${GLMOCR_ATTRIBUTION_DDP_TIMEOUT_SECONDS:-3600}"
gpu_utilization_limit=50
foreground=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --foreground) foreground=1; shift ;;
        --bundle-id) bundle_id="$2"; shift 2 ;;
        --validation-pages) validation_pages="$2"; shift 2 ;;
        --validation-seed) validation_seed="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --noop-run-id) noop_run_id_override="$2"; shift 2 ;;
        --adapter-run-id) adapter_run_id_override="$2"; shift 2 ;;
        --lora-run-id) lora_run_id_override="$2"; shift 2 ;;
        --validation-manifest) validation_manifest_override="$2"; shift 2 ;;
        --protocol-file) protocol_file_override="$2"; shift 2 ;;
        --ddp-timeout-seconds) ddp_timeout_seconds="$2"; shift 2 ;;
        --remote-root) remote_root="$2"; shift 2 ;;
        --code-root) code_root="$2"; shift 2 ;;
        --env-dir) env_dir="$2"; shift 2 ;;
        --model-dir) model_dir="$2"; shift 2 ;;
        --dataset-root) dataset_root="$2"; shift 2 ;;
        *) printf '{"event":"glmocr_attribution_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

[[ "${bundle_id}" =~ ^[A-Za-z0-9_.-]+$ && "${validation_pages}" =~ ^[1-9][0-9]*$ && "${validation_seed}" =~ ^[0-9]+$ ]] || {
    printf '{"event":"glmocr_attribution_failed","error":"invalid_bundle_or_subset"}\n' >&2
    exit 64
}
[[ "${ddp_timeout_seconds}" =~ ^[1-9][0-9]*$ ]] || {
    printf '{"event":"glmocr_attribution_failed","error":"invalid_ddp_timeout_seconds"}\n' >&2
    exit 64
}
[[ "${gpu_ids}" == "0,1,2,3,4" ]] || {
    printf '{"event":"glmocr_attribution_failed","error":"attribution_requires_five_a100_ids","gpu_ids":"%s"}\n' "${gpu_ids}" >&2
    exit 64
}

python="${env_dir}/bin/python"
train_manifest="${dataset_root}/train/manifest.jsonl"
validation_source="${dataset_root}/validation/manifest.jsonl"
validation_manifest="${validation_manifest_override:-${remote_root}/protocols/${bundle_id}.validation${validation_pages}.seed${validation_seed}.jsonl}"
protocol_file="${protocol_file_override:-${remote_root}/protocols/${bundle_id}.train_validation${validation_pages}.without_test.json}"
bundle_root="${remote_root}/training_runs/${bundle_id}"
bundle_log="${remote_root}/runs/${bundle_id}.bundle.log"
status_file="${bundle_root}/bundle_status.json"
noop_run_id="${bundle_id}_A_noop"
adapter_run_id="${bundle_id}_B_adapter_only"
lora_run_id="${bundle_id}_C_decoder_lora"
[[ -n "${noop_run_id_override}" ]] && noop_run_id="${noop_run_id_override}"
[[ -n "${adapter_run_id_override}" ]] && adapter_run_id="${adapter_run_id_override}"
[[ -n "${lora_run_id_override}" ]] && lora_run_id="${lora_run_id_override}"
summary_file="${bundle_root}/attribution_summary.json"

torch_lib="${env_dir}/lib/python3.11/site-packages/torch/lib"
cuda_library_path="/usr/local/cuda/targets/x86_64-linux/lib:${torch_lib}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    component_lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    if [[ -d "${component_lib}" ]]; then
        cuda_library_path="${cuda_library_path}:${component_lib}"
    fi
done

write_bundle_status() {
    local status="$1"
    local phase="$2"
    local detail="${3:-}"
    mkdir -p "${bundle_root}"
    printf '{"status":"%s","phase":"%s","detail":"%s","bundle_id":"%s","validation_pages":%s,"validation_seed":%s,"test_manifest_read":false,"test_used_for_selection":false}\n' \
        "${status}" "${phase}" "${detail}" "${bundle_id}" "${validation_pages}" "${validation_seed}" > "${status_file}.tmp"
    mv -f -- "${status_file}.tmp" "${status_file}"
}

on_error() {
    local rc=$?
    write_bundle_status failed "error" "rc=${rc}"
    exit "${rc}"
}
trap on_error ERR

preflight() {
    [[ -x "${python}" && -f "${code_root}/tools/subset_mthv2_manifest.py" && -f "${code_root}/tools/audit_mthv2_manifest.py" ]] || {
        printf '{"event":"glmocr_attribution_failed","error":"missing_runtime_or_tools"}\n' >&2
        exit 66
    }
    [[ -f "${train_manifest}" && -f "${validation_source}" ]] || {
        printf '{"event":"glmocr_attribution_failed","error":"missing_train_or_validation_manifest"}\n' >&2
        exit 66
    }
    for run_id in "${noop_run_id}" "${adapter_run_id}" "${lora_run_id}"; do
        run_dir="${remote_root}/training_runs/${run_id}/seed42"
        if [[ -e "${run_dir}" && ( ! -f "${run_dir}/summary.json" || ! -f "${run_dir}/COMPLETED" ) ]]; then
            printf '{"event":"glmocr_attribution_failed","error":"existing_incomplete_run","run_dir":"%s"}\n' "${run_dir}" >&2
            exit 74
        fi
    done
    mkdir -p "${remote_root}/protocols" "${remote_root}/runs" "${bundle_root}"
    if [[ ! -f "${validation_manifest}" ]]; then
        "${python}" "${code_root}/tools/subset_mthv2_manifest.py" \
            --input "${validation_source}" \
            --output "${validation_manifest}" \
            --count "${validation_pages}" \
            --seed "${validation_seed}"
    fi
    audit_args=(
        --train-manifest "${train_manifest}"
        --validation-manifest "${validation_manifest}"
        --num-queries 512
        --allow-count-mismatch
        --without-test
    )
    if [[ ! -f "${protocol_file}" ]]; then
        "${python}" "${code_root}/tools/audit_mthv2_manifest.py" "${audit_args[@]}" \
            --output "${protocol_file}" >/dev/null
    else
        "${python}" "${code_root}/tools/audit_mthv2_manifest.py" "${audit_args[@]}" >/dev/null
    fi
    "${python}" - "${protocol_file}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if payload.get("test_manifest_read") is not False:
    raise SystemExit("attribution protocol reads test manifest")
PY
}

query_single_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 || {
        printf '{"event":"glmocr_attribution_failed","error":"nvidia_smi_missing"}\n' >&2
        exit 69
    }
    observed="$(nvidia-smi -i 0 --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)"
    observed_id="${observed%%,*}"
    utilization="${observed##*,}"
    observed_id="${observed_id//[[:space:]]/}"
    utilization="${utilization//[[:space:]]/}"
    [[ "${observed_id}" == "0" && "${utilization}" =~ ^[0-9]+$ ]] || {
        printf '{"event":"glmocr_attribution_failed","error":"cannot_parse_gpu_utilization"}\n' >&2
        exit 69
    }
    (( utilization < gpu_utilization_limit )) || {
        printf '{"event":"glmocr_attribution_failed","error":"gpu_admission_failed","gpu":0,"utilization":%s,"limit":%s}\n' "${utilization}" "${gpu_utilization_limit}" >&2
        exit 75
    }
    printf '{"event":"glmocr_attribution_gpu_admission_ok","gpu":0,"utilization":%s}\n' "${utilization}"
}

common_env() {
    export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="${cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export NCCL_P2P_DISABLE=1
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
}

run_noop() {
    local run_dir="${remote_root}/training_runs/${noop_run_id}/seed42"
    if [[ -f "${run_dir}/summary.json" && -f "${run_dir}/COMPLETED" ]]; then
        printf '{"event":"glmocr_attribution_group_skipped","group":"A_noop","run_id":"%s"}\n' "${noop_run_id}"
        return 0
    fi
    query_single_gpu
    write_bundle_status running A_noop
    mkdir -p "${remote_root}/training_runs/${noop_run_id}/logs" "${bundle_root}/tmp/noop"
    common_env
    export CUDA_VISIBLE_DEVICES=0
    export TMPDIR="${bundle_root}/tmp/noop"
    export HF_HOME="${TMPDIR}/huggingface" TRANSFORMERS_CACHE="${HF_HOME}"
    mkdir -p "${TMPDIR}" "${HF_HOME}"
    "${python}" -m layout_ocr.train_screen \
        --distributed-strategy none \
        --mode content_only \
        --model-path "${model_dir}" \
        --train-manifest "${train_manifest}" \
        --validation-manifest "${validation_manifest}" \
        --protocol-file "${protocol_file}" \
        --output-dir "${run_dir}" \
        --per-device-batch-size 1 \
        --max-steps 1 --lr-schedule-steps 1 --warmup-steps 0 \
        --num-queries 512 --seed 42 \
        --experiment-label A_noop \
        --learning-rate 1e-6 --decoder-adaptation frozen \
        --auxiliary-weight 0 --auxiliary-weight-start 0 \
        --layout-loss-profile ocr_only --query-assignment hungarian \
        --adapter-precision fp32 --max-pixels 1003520 \
        --max-eval-new-tokens 1536 --validation-interval 1 \
        --diagnostic-steps 0 --log-steps 1 --eval-only \
        > "${remote_root}/training_runs/${noop_run_id}/logs/seed42.eval.log" 2>&1
}

run_train_group() {
    local label="$1"
    local run_id="$2"
    local decoder_adaptation="$3"
    local run_dir="${remote_root}/training_runs/${run_id}/seed42"
    if [[ -f "${run_dir}/summary.json" && -f "${run_dir}/COMPLETED" ]]; then
        printf '{"event":"glmocr_attribution_group_skipped","group":"%s","run_id":"%s"}\n' "${label}" "${run_id}"
        return 0
    fi
    write_bundle_status running "${label}"
    export GLMOCR_DDP_TIMEOUT_SECONDS="${ddp_timeout_seconds}"
    bash "${script_dir}/run_glmocr_mthv2_ddp.sh" \
        --foreground \
        --run-id "${run_id}" \
        --experiment-label "${label}" \
        --seed 42 \
        --gpu-ids "${gpu_ids}" \
        --max-steps 128 --lr-schedule-steps 128 \
        --learning-rate 2.5e-5 --decoder-adaptation "${decoder_adaptation}" \
        --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
        --decoder-learning-rate 1e-6 \
        --warmup-steps 32 --min-lr-ratio 0.5 \
        --gradient-accumulation-steps 4 \
        --initial-residual-scale 0.01 --gate-freeze-steps 128 \
        --auxiliary-weight-start 0.05 --auxiliary-weight 0.2 --auxiliary-ramp-steps 128 \
        --layout-loss-profile validity_assignment \
        --initial-valid-probability 0.066 --validity-gating-mode raw_mass \
        --validity-use-transport-evidence \
        --validation-manifest "${validation_manifest}" \
        --protocol-file "${protocol_file}" \
        --remote-root "${remote_root}" --code-root "${code_root}" \
        --env-dir "${env_dir}" --model-dir "${model_dir}" \
        --dataset-root "${dataset_root}" \
        --diagnostic-steps 0,128 --validation-interval 128 \
        --log-steps 16 --max-eval-new-tokens 1536 \
        --skip-selection --without-test
}

run_bundle() {
    preflight
    write_bundle_status running preflight
    run_noop
    run_train_group B_adapter_only "${adapter_run_id}" frozen
    run_train_group C_decoder_lora "${lora_run_id}" lora
    write_bundle_status running aggregation
    common_env
    "${python}" "${code_root}/tools/summarize_attribution.py" \
        --bundle-root "${remote_root}/training_runs" \
        --protocol-file "${protocol_file}" \
        --output "${summary_file}" \
        --group "A_noop=${noop_run_id}" \
        --group "B_adapter_only=${adapter_run_id}" \
        --group "C_decoder_lora=${lora_run_id}"
    write_bundle_status complete complete
    printf '{"event":"glmocr_attribution_complete","bundle_id":"%s","summary":"%s","test_manifest_read":false,"test_used_for_selection":false}\n' "${bundle_id}" "${summary_file}"
}

preflight
if (( foreground == 0 )); then
    command -v tmux >/dev/null 2>&1 || {
        printf '{"event":"glmocr_attribution_failed","error":"tmux_missing"}\n' >&2
        exit 69
    }
    session="${bundle_id}_seed42"
    tmux has-session -t "${session}" 2>/dev/null && {
        printf '{"event":"glmocr_attribution_failed","error":"session_already_exists","session":"%s"}\n' "${session}" >&2
        exit 73
    }
    mkdir -p "${remote_root}/runs"
    script_path="$(realpath -- "${BASH_SOURCE[0]}")"
    tmux new-session -d -s "${session}" \
        "cd $(printf '%q' "${code_root}") && exec bash $(printf '%q' "${script_path}") --foreground --bundle-id $(printf '%q' "${bundle_id}") --validation-pages $(printf '%q' "${validation_pages}") --validation-seed $(printf '%q' "${validation_seed}") --gpu-ids $(printf '%q' "${gpu_ids}") --noop-run-id $(printf '%q' "${noop_run_id_override}") --adapter-run-id $(printf '%q' "${adapter_run_id_override}") --lora-run-id $(printf '%q' "${lora_run_id_override}") --validation-manifest $(printf '%q' "${validation_manifest_override}") --protocol-file $(printf '%q' "${protocol_file_override}") --ddp-timeout-seconds $(printf '%q' "${ddp_timeout_seconds}") --remote-root $(printf '%q' "${remote_root}") --code-root $(printf '%q' "${code_root}") --env-dir $(printf '%q' "${env_dir}") --model-dir $(printf '%q' "${model_dir}") --dataset-root $(printf '%q' "${dataset_root}") >$(printf '%q' "${bundle_log}") 2>&1"
    printf '{"event":"glmocr_attribution_armed","session":"%s","bundle_id":"%s","groups":["A_noop","B_adapter_only","C_decoder_lora"],"validation_pages":%s,"gpu_ids":"%s","test_manifest_read":false,"log":"%s"}\n' \
        "${session}" "${bundle_id}" "${validation_pages}" "${gpu_ids}" "${bundle_log}"
else
    run_bundle
fi
