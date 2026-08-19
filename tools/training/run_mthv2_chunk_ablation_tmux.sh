#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_mthv2_chunk_ablation_tmux.sh --session NAME \
    [--gpu-id ID | --parallel-gpu-ids ID[,ID...]] \
    [--dataset-root PATH] [--run-prefix NAME] [--ablations C1,C2,C3,C4,C5]

Train the MTHv2 oracle-chunk ablation controls in one tmux session. C0-page
and C0-chunk are zero-shot references and are recorded, not optimized here.
The default budget is C1-C4 direct P2=42000 steps and C5 P1=12000 -> P2=30000;
all checkpoints are saved every 3000 steps. This launcher intentionally does
not run the synthetic-layout audit because MTHv2 chunk manifests use a real-
data schema. Results are interim chunk-level runs until grouped evaluation is run.
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
paths_env="${ocrmodel_root}/config/paths.env"
if [[ -f "${paths_env}" ]]; then
    # shellcheck source=/dev/null
    source "${paths_env}"
fi

session=""
dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_column_chunks16_v1"
run_prefix="mthv2_chunk_ablation_20260819"
gpu_id=""
parallel_gpu_ids=""
ablations="C1,C2,C3,C4,C5"
p1_steps="12000"
p2_steps="30000"
checkpoint_steps="3000"
gpu_utilization_limit="50"
seed="42"
session_inner=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="${2:-}"; shift 2 ;;
        --dataset-root) dataset_root="${2:-}"; shift 2 ;;
        --run-prefix) run_prefix="${2:-}"; shift 2 ;;
        --gpu-id) gpu_id="${2:-}"; shift 2 ;;
        --parallel-gpu-ids) parallel_gpu_ids="${2:-}"; shift 2 ;;
        --ablations) ablations="${2:-}"; shift 2 ;;
        --p1-steps) p1_steps="${2:-}"; shift 2 ;;
        --p2-steps) p2_steps="${2:-}"; shift 2 ;;
        --checkpoint-steps) checkpoint_steps="${2:-}"; shift 2 ;;
        --gpu-utilization-limit) gpu_utilization_limit="${2:-}"; shift 2 ;;
        --seed) seed="${2:-}"; shift 2 ;;
        --session-inner) session_inner=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage; exit 64 ;;
    esac
done

safe_name='^[A-Za-z0-9_.-]+$'
[[ "${run_prefix}" =~ ${safe_name} ]] || {
    printf 'ERROR: session and run-prefix must be simple names.\n' >&2; exit 64;
}
if [[ "${session_inner}" -eq 0 && ! "${session}" =~ ${safe_name} ]]; then
    printf 'ERROR: session and run-prefix must be simple names.\n' >&2; exit 64;
fi
gpu_mode_count=0
[[ -n "${gpu_id}" ]] && ((gpu_mode_count+=1))
[[ -n "${parallel_gpu_ids}" ]] && ((gpu_mode_count+=1))
(( gpu_mode_count == 1 )) || { printf 'ERROR: provide exactly one of --gpu-id or --parallel-gpu-ids.\n' >&2; exit 64; }
[[ -z "${gpu_id}" || "${gpu_id}" =~ ^[0-9]+$ ]] || { printf 'ERROR: --gpu-id must be numeric.\n' >&2; exit 64; }
[[ -z "${parallel_gpu_ids}" || "${parallel_gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { printf 'ERROR: --parallel-gpu-ids must be comma-separated numeric ids.\n' >&2; exit 64; }
[[ "${p1_steps}" =~ ^[1-9][0-9]*$ && "${p2_steps}" =~ ^[1-9][0-9]*$ && "${checkpoint_steps}" =~ ^[1-9][0-9]*$ ]] || {
    printf 'ERROR: step values must be positive integers.\n' >&2; exit 64;
}
[[ "${gpu_utilization_limit}" =~ ^[1-9][0-9]*$ ]] && (( gpu_utilization_limit <= 100 )) || {
    printf 'ERROR: GPU utilization limit must be 1..100.\n' >&2; exit 64;
}
[[ -d "${dataset_root}/train" && -f "${dataset_root}/train/manifest.jsonl" ]] || {
    printf 'ERROR: MTHv2 chunk dataset is missing: %s\n' "${dataset_root}" >&2; exit 66;
}
if [[ "${session_inner}" -eq 0 ]]; then
    command -v tmux >/dev/null 2>&1 || { printf 'ERROR: tmux is unavailable.\n' >&2; exit 69; }
fi
if [[ "${session_inner}" -eq 0 ]] && tmux has-session -t "${session}" 2>/dev/null; then
    printf 'ERROR: tmux session already exists: %s\n' "${session}" >&2; exit 73
fi

IFS=',' read -r -a requested <<< "${ablations}"
for ablation in "${requested[@]}"; do
    case "${ablation}" in C1|C2|C3|C4|C5) ;; *) printf 'ERROR: unsupported control: %s\n' "${ablation}" >&2; exit 64 ;; esac
done

run_root="${GOT_TRAINING_RUNS}/${run_prefix}"
log_root="${GOT_TRAINING_RUNS}/${run_prefix}_tmux_logs"
mkdir -p "${log_root}"
log_path="${log_root}/launcher.log"

require_gpu_below_limit() {
    local target_gpu="$1" output utilization
    output="$(nvidia-smi -i "${target_gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>&1)" || {
        printf 'ERROR: GPU%s query failed: %s\n' "${target_gpu}" "${output}" >&2; exit 66;
    }
    utilization="${output//[[:space:]]/}"
    [[ "${utilization}" =~ ^[0-9]+$ ]] || { printf 'ERROR: GPU%s utilization is not numeric: %s\n' "${target_gpu}" "${output}" >&2; exit 66; }
    (( utilization < gpu_utilization_limit )) || {
        printf 'ERROR: GPU%s utilization=%s is not below limit=%s.\n' "${target_gpu}" "${utilization}" "${gpu_utilization_limit}" >&2; exit 75;
    }
}

IFS=',' read -r -a selected_gpus <<< "${parallel_gpu_ids:-${gpu_id}}"
declare -A seen_gpus=()
for selected_gpu in "${selected_gpus[@]}"; do
    [[ -z "${seen_gpus[${selected_gpu}]:-}" ]] || { printf 'ERROR: duplicate GPU id: %s\n' "${selected_gpu}" >&2; exit 64; }
    seen_gpus[${selected_gpu}]=1
    require_gpu_below_limit "${selected_gpu}"
done
if [[ -n "${parallel_gpu_ids}" && ${#requested[@]} -ne ${#selected_gpus[@]} ]]; then
    printf 'ERROR: --parallel-gpu-ids count must equal --ablations count.\n' >&2
    exit 64
fi

train_one() {
    local control="$1" assigned_gpu="$2" ablation_id layout_preset mode run_id
    case "${control}" in
        C1) ablation_id="projector_only"; layout_preset="layout_none"; mode="ablation"; run_id="${run_prefix}_C1_projector_only_seed${seed}" ;;
        C2) ablation_id="generic_adapter_projector"; layout_preset="layout_none"; mode="ablation"; run_id="${run_prefix}_C2_generic_adapter_seed${seed}" ;;
        C3) ablation_id="vlqa_ocr_only"; layout_preset="layout_none"; mode="ablation"; run_id="${run_prefix}_C3_vlqa_ocr_only_seed${seed}" ;;
        C4) ablation_id="vlqa_layout_direct"; layout_preset="layout_full"; mode="ablation"; run_id="${run_prefix}_C4_vlqa_layout_direct_seed${seed}" ;;
        C5) ablation_id="vlqa_layout_p1_p2"; layout_preset="layout_full"; mode="ablation"; run_id="${run_prefix}_C5_vlqa_p1_p2_seed${seed}" ;;
    esac
    require_gpu_below_limit "${assigned_gpu}"
    local args=(
        --mode "${mode}" --ablation "${ablation_id}"
        --dataset-root "${dataset_root}/train"
        --manifest "${dataset_root}/train/manifest.jsonl"
        --validation-manifest "${dataset_root}/validation/manifest.jsonl"
        --validation-image-root "${dataset_root}/validation"
        --test-manifest "${dataset_root}/test/manifest.jsonl"
        --test-image-root "${dataset_root}/test"
        --source-model "${GOT_SOURCE_MODEL}"
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}"
        --p2-max-steps "$([[ "${control}" == "C5" ]] && printf '%s' "${p2_steps}" || printf '%s' "$((p1_steps + p2_steps))")"
        --checkpoint-steps "${checkpoint_steps}"
        --gpu-utilization-limit "${gpu_utilization_limit}"
        --seed "${seed}" --run-id "${run_id}" --gpu-ids "${assigned_gpu}"
        --max-regions 16 --layout-loss-preset "${layout_preset}"
        --skip-post-training-validation
    )
    if [[ "${control}" == "C5" ]]; then
        args+=(--p1-max-steps "${p1_steps}")
    fi
    printf '{"event":"mthv2_chunk_control_started","control":"%s","run_id":"%s"}\n' "${control}" "${run_id}"
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/training/run_layout_a100.py" "${args[@]}"
    printf '{"event":"mthv2_chunk_control_completed","control":"%s","run_id":"%s"}\n' "${control}" "${run_id}"
}

run_all() {
    printf '{"event":"mthv2_chunk_ablation_started","dataset_root":"%s","controls":"%s","gpu_ids":"%s","p1_steps":%s,"p2_steps":%s,"direct_p2_steps":%s,"checkpoint_steps":%s,"gpu_condition":"utilization<limit"}\n' \
        "${dataset_root}" "${ablations}" "${parallel_gpu_ids:-${gpu_id}}" "${p1_steps}" "${p2_steps}" "$((p1_steps + p2_steps))" "${checkpoint_steps}"
    printf '%s\n' '{"event":"zero_shot_references","controls":["C0-page","C0-chunk"],"status":"recorded_only"}'
    if [[ -n "${parallel_gpu_ids}" ]]; then
        pids=()
        for index in "${!requested[@]}"; do
            train_one "${requested[${index}]}" "${selected_gpus[${index}]}" &
            pids+=("$!")
        done
        failed=0
        for pid in "${pids[@]}"; do
            wait "${pid}" || failed=1
        done
        (( failed == 0 )) || return 1
    else
        for control in "${requested[@]}"; do
            train_one "${control}" "${gpu_id}"
        done
    fi
    printf '%s\n' '{"event":"mthv2_chunk_ablation_finished","status":"training_completed","evaluation":"grouped_source_page_evaluation_pending"}'
}

printf -v quoted_root '%q' "${ocrmodel_root}"
printf -v quoted_log '%q' "${log_path}"
printf -v quoted_script '%q' "${BASH_SOURCE[0]}"
if [[ "${session_inner}" -eq 1 ]]; then
    run_all
else
    if [[ -n "${parallel_gpu_ids}" ]]; then
        inner_gpu_args="--parallel-gpu-ids '${parallel_gpu_ids}'"
    else
        inner_gpu_args="--gpu-id '${gpu_id}'"
    fi
    tmux new-session -d -s "${session}" \
        "cd ${quoted_root} && exec bash ${quoted_script} --session-inner --dataset-root '${dataset_root}' --run-prefix '${run_prefix}' ${inner_gpu_args} --ablations '${ablations}' --p1-steps '${p1_steps}' --p2-steps '${p2_steps}' --checkpoint-steps '${checkpoint_steps}' --gpu-utilization-limit '${gpu_utilization_limit}' --seed '${seed}' >${quoted_log} 2>&1"

    sleep 5
    if ! tmux has-session -t "${session}" 2>/dev/null; then
        tail -n 20 "${log_path}" >&2 || true
        printf 'ERROR: tmux process exited during startup.\n' >&2
        exit 1
    fi
    printf '{"event":"mthv2_chunk_ablation_tmux_started","session":"%s","gpu_ids":"%s","log":"%s","dataset_root":"%s"}\n' "${session}" "${parallel_gpu_ids:-${gpu_id}}" "${log_path}" "${dataset_root}"
fi
