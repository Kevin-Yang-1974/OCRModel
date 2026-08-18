#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

session=""
dataset_id=""
run_prefix=""
gpu_ids=""
p1_steps="12000"
p2_steps="24000"
checkpoint_steps="2000"
gpu_utilization_limit="50"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="${2:-}"; shift 2 ;;
        --dataset-id) dataset_id="${2:-}"; shift 2 ;;
        --run-prefix) run_prefix="${2:-}"; shift 2 ;;
        --gpu-id) gpu_ids="${2:-}"; shift 2 ;;
        --gpu-ids) gpu_ids="${2:-}"; shift 2 ;;
        --p1-steps) p1_steps="${2:-}"; shift 2 ;;
        --p2-steps) p2_steps="${2:-}"; shift 2 ;;
        --checkpoint-steps) checkpoint_steps="${2:-}"; shift 2 ;;
        --gpu-utilization-limit) gpu_utilization_limit="${2:-}"; shift 2 ;;
        *) printf '{"event":"diverse_training_launch_failed","error":"unknown argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

safe_name='^[A-Za-z0-9_.-]+$'
[[ "${session}" =~ ${safe_name} && "${dataset_id}" =~ ${safe_name} && "${run_prefix}" =~ ${safe_name} ]] || {
    printf '%s\n' '{"event":"diverse_training_launch_failed","error":"invalid session, dataset id, or run prefix"}' >&2
    exit 64
}
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || {
    printf '%s\n' '{"event":"diverse_training_launch_failed","error":"GPU ids must be comma-separated numeric ids"}' >&2
    exit 64
}
[[ "${p1_steps}" =~ ^[1-9][0-9]*$ && "${p2_steps}" =~ ^[1-9][0-9]*$ && "${checkpoint_steps}" =~ ^[1-9][0-9]*$ ]] || {
    printf '%s\n' '{"event":"diverse_training_launch_failed","error":"step values must be positive integers"}' >&2
    exit 64
}
[[ "${gpu_utilization_limit}" =~ ^[1-9][0-9]*$ ]] && (( gpu_utilization_limit <= 100 )) || {
    printf '%s\n' '{"event":"diverse_training_launch_failed","error":"GPU utilization limit must be an integer in 1..100"}' >&2
    exit 64
}

dataset_root="${GOT_LAYOUT_DATA}/${dataset_id}"
run_root="${GOT_TRAINING_RUNS}/${run_prefix}_vlqa_layout_p1_p2_seed42"
log_root="${GOT_TRAINING_RUNS}/${run_prefix}_tmux_logs"
stamp="$(date +%Y%m%d_%H%M%S)"
log_path="${log_root}/${stamp}.log"

command -v tmux >/dev/null 2>&1 || {
    printf '%s\n' '{"event":"diverse_training_launch_failed","error":"tmux is unavailable"}' >&2
    exit 69
}
[[ -d "${dataset_root}" ]] || {
    printf '{"event":"diverse_training_launch_failed","error":"dataset is missing","dataset":"%s"}\n' "${dataset_root}" >&2
    exit 66
}
[[ ! -e "${run_root}" ]] || {
    printf '{"event":"diverse_training_launch_failed","error":"run already exists","run":"%s"}\n' "${run_root}" >&2
    exit 73
}
if tmux has-session -t "${session}" 2>/dev/null; then
    printf '{"event":"diverse_training_launch_failed","error":"tmux session already exists","session":"%s"}\n' "${session}" >&2
    exit 73
fi

IFS=',' read -r -a selected_gpus <<< "${gpu_ids}"
declare -A seen=()
for gpu in "${selected_gpus[@]}"; do
    [[ -z "${seen[$gpu]:-}" ]] || {
        printf '{"event":"diverse_training_launch_failed","error":"duplicate GPU id","gpu":"%s"}\n' "${gpu}" >&2
        exit 64
    }
    seen[$gpu]=1
    output="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>&1)" || {
        printf '{"event":"diverse_training_launch_failed","error":"GPU query failed","gpu":"%s"}\n' "${gpu}" >&2
        exit 66
    }
    utilization="${output//[[:space:]]/}"
    [[ "${utilization}" =~ ^[0-9]+$ ]] || {
        printf '{"event":"diverse_training_launch_failed","error":"GPU utilization query was not numeric","gpu":"%s","value":"%s"}\n' "${gpu}" "${output}" >&2
        exit 66
    }
    (( utilization < gpu_utilization_limit )) || {
        printf '{"event":"diverse_training_launch_failed","error":"GPU utilization at or above sharing limit","gpu":"%s","utilization":%s,"limit":%s}\n' "${gpu}" "${utilization}" "${gpu_utilization_limit}" >&2
        exit 75
    }
done

mkdir -p "${log_root}"
args=(
    bash "${script_dir}/run_diverse_synthetic_ancientdoc.sh" train-synthetic
    --dataset-id "${dataset_id}"
    --run-prefix "${run_prefix}"
    --p1-steps "${p1_steps}"
    --p2-steps "${p2_steps}"
    --checkpoint-steps "${checkpoint_steps}"
    --gpu-utilization-limit "${gpu_utilization_limit}"
)
if [[ "${gpu_ids}" == *,* ]]; then
    args+=(--gpu-ids "${gpu_ids}")
else
    args+=(--gpu-id "${gpu_ids}")
fi
printf -v launch_command '%q ' "${args[@]}"
printf -v quoted_root '%q' "${ocrmodel_root}"
printf -v quoted_log '%q' "${log_path}"
tmux new-session -d -s "${session}" \
    "cd ${quoted_root} && exec ${launch_command}>${quoted_log} 2>&1"

sleep 8
if ! tmux has-session -t "${session}" 2>/dev/null; then
    tail -n 20 "${log_path}" >&2 || true
    printf '{"event":"diverse_training_launch_failed","error":"tmux process exited during startup","session":"%s","log":"%s"}\n' \
        "${session}" "${log_path}" >&2
    exit 1
fi
pane_pid="$(tmux display-message -p -t "${session}:0.0" '#{pane_pid}')"
total_steps=$((p1_steps + p2_steps))
printf '{"event":"diverse_training_started","session":"%s","pane_pid":%s,"dataset_id":"%s","run_prefix":"%s","run_root":"%s","physical_gpu_ids":"%s","p1_steps":%s,"p2_steps":%s,"total_steps":%s,"checkpoint_steps":%s,"gpu_utilization_limit":%s,"log":"%s"}\n' \
    "${session}" "${pane_pid}" "${dataset_id}" "${run_prefix}" "${run_root}" "${gpu_ids}" "${p1_steps}" "${p2_steps}" "${total_steps}" "${checkpoint_steps}" "${gpu_utilization_limit}" "${log_path}"
