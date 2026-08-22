#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="$(cd -- "${script_dir}/../.." && pwd -P)"
source "${ocrmodel_root}/config/paths.env"

session="mthv2_pvld_p1_validation_diagnostics_20260822"
run_prefix="mthv2_page_pvld_ablation_20260821_r1"
gpu_ids="0,1"
utilization_limit=50
session_inner=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --gpu-utilization-limit) utilization_limit="$2"; shift 2 ;;
        --session-inner) session_inner=1; shift ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

IFS=',' read -r -a gpus <<< "${gpu_ids}"
(( ${#gpus[@]} == 2 )) || { printf 'ERROR: exactly two GPU IDs are required.\n' >&2; exit 64; }
for gpu in "${gpus[@]}"; do
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
    (( utilization < utilization_limit )) || { printf 'ERROR: GPU%s utilization=%s.\n' "${gpu}" "${utilization}" >&2; exit 75; }
done

dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1"
model_root="${GOT_TRAINING_RUNS}/${run_prefix}_C5_pvld_p1_p2_seed42/p1/model"
output_root="${GOT_EVALUATION_RUNS}/${run_prefix}_C5_p1_validation_diagnostics_20260822"
log_root="${GOT_TRAINING_RUNS}/${run_prefix}_${session}_logs"
launcher_log="${log_root}/launcher.log"

run_candidate() {
    local step="$1" gpu="$2" output="$3" log="$4"
    CUDA_VISIBLE_DEVICES="${gpu}" bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/src/GOT-OCR-2.0/scripts/evaluate_GOT_layout.py" \
        --model-name-or-path "${model_root}/checkpoint-${step}" \
        --model-kind pvld \
        --tokenizer-name-or-path "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --layout-manifest "${dataset_root}/validation/manifest.jsonl" \
        --layout-image-root "${dataset_root}/validation" \
        --layout-split validation \
        --output-dir "${output}" \
        --max-regions 512 --max-records 512 \
        --max-new-tokens 2048 --no-repeat-ngram-size 20 \
        >"${log}" 2>&1
}

run_pipeline() {
    mkdir -p "${log_root}"
    [[ ! -e "${output_root}" ]] || { printf 'ERROR: output exists: %s\n' "${output_root}" >&2; return 73; }
    mkdir -p "${output_root}"
    run_candidate 9000 "${gpus[0]}" "${output_root}/step-00009000" "${log_root}/p1_step_9000.log" &
    pid_9000=$!
    run_candidate 12000 "${gpus[1]}" "${output_root}/step-00012000" "${log_root}/p1_step_12000.log" &
    pid_12000=$!
    failed=0
    wait "${pid_9000}" || failed=1
    wait "${pid_12000}" || failed=1
    (( failed == 0 )) || { printf '%s\n' '{"event":"pvld_p1_validation_diagnostics_failed"}'; return 1; }

    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/evaluation/analyze_pvld_validation_diagnostics.py" \
        --prediction "P1-9000=${output_root}/step-00009000/layout_validation_predictions.jsonl" \
        --prediction "P1-12000=${output_root}/step-00012000/layout_validation_predictions.jsonl" \
        --output-dir "${output_root}/offline_analysis"
    printf '{"event":"pvld_p1_validation_diagnostics_completed","output_root":"%s"}\n' "${output_root}"
}

if (( session_inner == 1 )); then
    run_pipeline
    exit
fi

tmux has-session -t "${session}" 2>/dev/null && { printf 'ERROR: tmux session exists.\n' >&2; exit 73; }
mkdir -p "${log_root}"
script_path="$(realpath "${BASH_SOURCE[0]}")"
tmux new-session -d -s "${session}" \
    "cd '${ocrmodel_root}' && exec bash '${script_path}' --session-inner --session '${session}' --gpu-ids '${gpu_ids}' --gpu-utilization-limit '${utilization_limit}' >'${launcher_log}' 2>&1"
sleep 5
tmux has-session -t "${session}" 2>/dev/null || { tail -n 20 "${launcher_log}" >&2 || true; exit 1; }
printf '{"event":"pvld_p1_validation_diagnostics_started","session":"%s","gpu_ids":"%s","output_root":"%s"}\n' \
    "${session}" "${gpu_ids}" "${output_root}"
