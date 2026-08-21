#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
paths_env="${ocrmodel_root}/config/paths.env"
[[ ! -f "${paths_env}" ]] || source "${paths_env}"

session="mthv2_page_visual_routing_pipeline_20260821"
run_prefix="mthv2_page_visual_routing_20260821"
gpu_ids="0,1,2,3,4"
dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1"
utilization_limit="50"
session_inner=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="${2:-}"; shift 2 ;;
        --run-prefix) run_prefix="${2:-}"; shift 2 ;;
        --gpu-ids) gpu_ids="${2:-}"; shift 2 ;;
        --gpu-utilization-limit) utilization_limit="${2:-}"; shift 2 ;;
        --session-inner) session_inner=1; shift ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

log_root="${GOT_TRAINING_RUNS}/${run_prefix}_pipeline_tmux_logs"
log_path="${log_root}/launcher.log"
mkdir -p "${log_root}"

run_pipeline() {
    printf '{"event":"page_visual_routing_pipeline_started","run_prefix":"%s","gpu_ids":"%s"}\n' "${run_prefix}" "${gpu_ids}"
    bash "${script_dir}/run_mthv2_chunk_ablation_tmux.sh" \
        --session-inner --dataset-root "${dataset_root}" --run-prefix "${run_prefix}" \
        --parallel-gpu-ids "${gpu_ids}" --ablations C1,C2,C3,C4,C5 \
        --p1-steps 12000 --p2-steps 30000 --checkpoint-steps 3000 \
        --gpu-utilization-limit "${utilization_limit}" --seed 42 \
        --max-regions 512 --vlqa-writeback-mode visual_value_layout_routing
    printf '%s\n' '{"event":"page_visual_routing_training_completed"}'
    bash "${ocrmodel_root}/tools/evaluation/run_mthv2_chunk_validation_test_tmux.sh" \
        --session-inner --gpu-ids "${gpu_ids}" --run-prefix "${run_prefix}" \
        --dataset-root "${dataset_root}" --gpu-utilization-limit "${utilization_limit}" \
        --max-regions 512
    printf '%s\n' '{"event":"page_visual_routing_pipeline_completed","selection":"validation_only","test":"selection_locked"}'
}

if (( session_inner == 1 )); then
    run_pipeline
    exit
fi

tmux has-session -t "${session}" 2>/dev/null && { printf 'ERROR: tmux session exists.\n' >&2; exit 73; }
IFS=',' read -r -a selected_gpus <<< "${gpu_ids}"
for gpu in "${selected_gpus[@]}"; do
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
    (( utilization < utilization_limit )) || { printf 'ERROR: GPU%s utilization=%s.\n' "${gpu}" "${utilization}" >&2; exit 75; }
done

script_path="$(realpath "${BASH_SOURCE[0]}")"
tmux new-session -d -s "${session}" \
    "cd '${ocrmodel_root}' && exec bash '${script_path}' --session-inner --session '${session}' --run-prefix '${run_prefix}' --gpu-ids '${gpu_ids}' --gpu-utilization-limit '${utilization_limit}' >'${log_path}' 2>&1"
sleep 5
tmux has-session -t "${session}" 2>/dev/null || { tail -n 20 "${log_path}" >&2 || true; exit 1; }
printf '{"event":"page_visual_routing_pipeline_tmux_started","session":"%s","run_prefix":"%s","gpu_ids":"%s","log":"%s"}\n' "${session}" "${run_prefix}" "${gpu_ids}" "${log_path}"
