#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

session="mthv2_pvld_causal_c3_c5_20260822"
run_prefix="mthv2_pvld_causal_20260822_v1"
gpu_ids="0,1,2,3,4"
dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1"
utilization_limit=50
session_inner=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="$2"; shift 2 ;;
        --run-prefix) run_prefix="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --gpu-utilization-limit) utilization_limit="$2"; shift 2 ;;
        --session-inner) session_inner=1; shift ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

IFS=',' read -r -a gpus <<< "${gpu_ids}"
[[ ${#gpus[@]} -ge 1 ]] || { printf 'ERROR: at least one target GPU is required.\n' >&2; exit 64; }
for gpu in "${gpus[@]}"; do
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "${utilization}" =~ ^[0-9]+$ ]] || { printf 'ERROR: GPU%s utilization query failed.\n' "${gpu}" >&2; exit 75; }
    (( utilization < utilization_limit )) || { printf 'ERROR: GPU%s utilization=%s.\n' "${gpu}" "${utilization}" >&2; exit 75; }
done

log_root="${GOT_TRAINING_RUNS}/${run_prefix}_${session}_logs"
log_path="${log_root}/launcher.log"
mkdir -p "${log_root}"

train_control() {
    local gpu_group="$1" control="$2" ablation="$3" preset="$4" stages="$5" p2_steps="$6"
    local run_id="${run_prefix}_${control}_seed42"
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/training/run_variable_layout_a100.py" \
        --dataset-root "${dataset_root}/train" \
        --manifest "${dataset_root}/train/manifest.jsonl" \
        --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
        --test-manifest "${dataset_root}/test/manifest.jsonl" \
        --source-model "${GOT_SOURCE_MODEL}" \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --stages "${stages}" --ablation "${ablation}" --layout-loss-preset "${preset}" \
        --num-layout-prompt-queries 32 --max-layout-records 512 \
        --max-layout-tokens 2048 --layout-decoder-layers 2 \
        --layout-decoder-hidden-size 256 --layout-decoder-num-heads 8 \
        --p1-max-steps 12000 --p2-max-steps "${p2_steps}" \
        --checkpoint-steps 2000 --checkpoint-retention 2 \
        --gpu-utilization-limit "${utilization_limit}" --gpu-ids "${gpu_group}" \
        --seed 42 --run-id "${run_id}"
}

evaluate_control() {
    local gpu="$1" control="$2" ablation="$3"
    local run_id="${run_prefix}_${control}_seed42"
    local model_root="${GOT_TRAINING_RUNS}/${run_id}/p2/model"
    local selection_root="${GOT_EVALUATION_RUNS}/${run_prefix}_${control}_validation_selection"
    local test_root="${GOT_EVALUATION_RUNS}/${run_prefix}_${control}_test"
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/evaluation/select_layout_ablation_checkpoint.py" \
        --ablation "${ablation}" --model-root "${model_root}" --model-kind pvld \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
        --validation-image-root "${dataset_root}/validation" --output-dir "${selection_root}" \
        --project-root "${ocrmodel_root}/src/GOT-OCR-2.0" --max-regions 512 \
        --gpu-id "${gpu}" --gpu-utilization-limit "${utilization_limit}"
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/evaluation/evaluate_layout_ablation_test.py" \
        --selection "${selection_root}/selection.json" --test-category Real-OOD \
        --test-manifest "${dataset_root}/test/manifest.jsonl" \
        --test-image-root "${dataset_root}/test" --model-kind pvld \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --project-root "${ocrmodel_root}/src/GOT-OCR-2.0" --output-dir "${test_root}" \
        --max-regions 512 --gpu-id "${gpu}" --gpu-utilization-limit "${utilization_limit}"
}

run_pipeline() {
    local controls=(C3 C4 C5)
    local ablations=(vlqa_ocr_only vlqa_layout_direct vlqa_layout_p1_p2)
    local presets=(layout_none layout_full layout_full)
    local stages=(p2 p2 p1,p2)
    local p2_steps=(42000 42000 30000)
    local assignments=()
    if (( ${#gpus[@]} >= ${#controls[@]} )); then
        assignments=("${gpus[0]}" "${gpus[1]}" "${gpus[2]}")
    else
        assignments=("${gpu_ids}" "${gpu_ids}" "${gpu_ids}")
    fi

    local failed=0 index pid
    if (( ${#gpus[@]} >= ${#controls[@]} )); then
        local pids=()
        for index in "${!controls[@]}"; do
            train_control "${assignments[index]}" "${controls[index]}" "${ablations[index]}" \
                "${presets[index]}" "${stages[index]}" "${p2_steps[index]}" \
                >"${log_root}/${controls[index]}_train.log" 2>&1 &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
    else
        for index in "${!controls[@]}"; do
            train_control "${assignments[index]}" "${controls[index]}" "${ablations[index]}" \
                "${presets[index]}" "${stages[index]}" "${p2_steps[index]}" \
                >"${log_root}/${controls[index]}_train.log" 2>&1 || failed=1
            (( failed == 0 )) || break
        done
    fi
    (( failed == 0 )) || { printf '%s\n' '{"event":"mthv2_pvld_c3_c5_training_failed"}'; return 1; }

    local eval_pids=()
    if (( ${#gpus[@]} >= ${#controls[@]} )); then
        for index in "${!controls[@]}"; do
            evaluate_control "${assignments[index]}" "${controls[index]}" "${ablations[index]}" \
                >"${log_root}/${controls[index]}_validation_test.log" 2>&1 &
            eval_pids+=("$!")
        done
        for pid in "${eval_pids[@]}"; do wait "${pid}" || failed=1; done
    else
        for index in "${!controls[@]}"; do
            evaluate_control "${gpus[0]}" "${controls[index]}" "${ablations[index]}" \
                >"${log_root}/${controls[index]}_validation_test.log" 2>&1 || failed=1
            (( failed == 0 )) || break
        done
    fi
    (( failed == 0 )) || { printf '%s\n' '{"event":"mthv2_pvld_c3_c5_validation_test_failed"}'; return 1; }
    printf '%s\n' '{"event":"mthv2_pvld_c3_c5_completed","selection":"validation_only","test":"selection_locked","test_used_for_tuning":false}'
}

if (( session_inner == 1 )); then
    run_pipeline
    exit
fi

tmux has-session -t "${session}" 2>/dev/null && { printf 'ERROR: tmux session exists.\n' >&2; exit 73; }
script_path="$(realpath "${BASH_SOURCE[0]}")"
tmux new-session -d -s "${session}" \
    "cd '${ocrmodel_root}' && exec bash '${script_path}' --session-inner --session '${session}' --run-prefix '${run_prefix}' --gpu-ids '${gpu_ids}' --gpu-utilization-limit '${utilization_limit}' >'${log_path}' 2>&1"
sleep 5
tmux has-session -t "${session}" 2>/dev/null || { tail -n 20 "${log_path}" >&2 || true; exit 1; }
printf '{"event":"mthv2_pvld_c3_c5_started","session":"%s","run_prefix":"%s","gpu_ids":"%s","dataset_root":"%s","log":"%s"}\n' \
    "${session}" "${run_prefix}" "${gpu_ids}" "${dataset_root}" "${log_path}"
