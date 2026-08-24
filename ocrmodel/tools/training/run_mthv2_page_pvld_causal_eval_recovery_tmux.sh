#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

session="mthv2_pvld_causal_eval_recovery_20260824_v1"
run_prefix="mthv2_pvld_causal_20260822_v1"
evaluation_suffix="_causal_recovery_20260824_v1"
gpu_ids="0,1,3,4"
dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1"
utilization_limit=50
session_inner=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="$2"; shift 2 ;;
        --run-prefix) run_prefix="$2"; shift 2 ;;
        --evaluation-suffix) evaluation_suffix="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --gpu-utilization-limit) utilization_limit="$2"; shift 2 ;;
        --session-inner) session_inner=1; shift ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

IFS=',' read -r -a gpus <<< "${gpu_ids}"
(( ${#gpus[@]} >= 3 )) || { printf 'ERROR: at least three target GPUs are required.\n' >&2; exit 64; }
for gpu in "${gpus[@]}"; do
    [[ "${gpu}" != "2" ]] || { printf 'ERROR: GPU2 is excluded from this recovery.\n' >&2; exit 64; }
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "${utilization}" =~ ^[0-9]+$ ]] || { printf 'ERROR: GPU%s utilization query failed.\n' "${gpu}" >&2; exit 75; }
    (( utilization < utilization_limit )) || { printf 'ERROR: GPU%s utilization=%s.\n' "${gpu}" "${utilization}" >&2; exit 75; }
done

log_root="${GOT_TRAINING_RUNS}/${run_prefix}_${session}_logs"
log_path="${log_root}/launcher.log"
mkdir -p "${log_root}"

resume_c5_p2() {
    local gpu="${gpus[3]}"
    local run_id="${run_prefix}_C5_seed42"
    local selection="${GOT_TRAINING_RUNS}/${run_id}/p1/validation_selection/selection.json"
    local selected_model
    selected_model="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["selected"]["model_path"])' "${selection}")"
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/training/run_variable_layout_a100.py" \
        --resume-existing-run --stages p2 \
        --dataset-root "${dataset_root}/train" \
        --manifest "${dataset_root}/train/manifest.jsonl" \
        --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
        --test-manifest "${dataset_root}/test/manifest.jsonl" \
        --source-model "${selected_model}" \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --ablation vlqa_layout_p1_p2 --layout-loss-preset layout_full \
        --num-layout-prompt-queries 32 --max-layout-records 512 \
        --max-layout-tokens 2048 --layout-decoder-layers 2 \
        --layout-decoder-hidden-size 256 --layout-decoder-num-heads 8 \
        --p2-max-steps 30000 --checkpoint-steps 2000 --checkpoint-retention 2 \
        --gpu-utilization-limit "${utilization_limit}" --gpu-ids "${gpu}" \
        --seed 42 --run-id "${run_id}"
}

evaluate_control() {
    local gpu="$1" control="$2" ablation="$3"
    local run_id="${run_prefix}_${control}_seed42"
    local model_root="${GOT_TRAINING_RUNS}/${run_id}/p2/model"
    local selection_root="${GOT_EVALUATION_RUNS}/${run_prefix}_${control}_validation_selection${evaluation_suffix}"
    local test_root="${GOT_EVALUATION_RUNS}/${run_prefix}_${control}_test${evaluation_suffix}"
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
    resume_c5_p2 >"${log_root}/C5_p2_recovery.log" 2>&1 || {
        printf '%s\n' '{"event":"pvld_causal_c5_p2_recovery_failed"}'
        return 1
    }

    local controls=(C3 C4 C5)
    local ablations=(vlqa_ocr_only vlqa_layout_direct vlqa_layout_p1_p2)
    local pids=() failed=0 index pid
    for index in "${!controls[@]}"; do
        evaluate_control "${gpus[index]}" "${controls[index]}" "${ablations[index]}" \
            >"${log_root}/${controls[index]}_validation_test.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
    (( failed == 0 )) || {
        printf '%s\n' '{"event":"pvld_causal_c3_c5_validation_test_failed"}'
        return 1
    }
    printf '%s\n' '{"event":"pvld_causal_c3_c5_validation_test_completed","selection":"validation_only","test":"selection_locked","test_used_for_tuning":false}'
}

if (( session_inner == 1 )); then
    run_pipeline
    exit
fi

tmux has-session -t "${session}" 2>/dev/null && { printf 'ERROR: tmux session exists.\n' >&2; exit 73; }
script_path="$(realpath "${BASH_SOURCE[0]}")"
tmux new-session -d -s "${session}" \
    "cd '${ocrmodel_root}' && exec bash '${script_path}' --session-inner --session '${session}' --run-prefix '${run_prefix}' --evaluation-suffix '${evaluation_suffix}' --gpu-ids '${gpu_ids}' --gpu-utilization-limit '${utilization_limit}' >'${log_path}' 2>&1"
sleep 5
tmux has-session -t "${session}" 2>/dev/null || { tail -n 20 "${log_path}" >&2 || true; exit 1; }
printf '{"event":"pvld_causal_c3_c5_recovery_started","session":"%s","run_prefix":"%s","evaluation_suffix":"%s","gpu_ids":"%s","log":"%s"}\n' \
    "${session}" "${run_prefix}" "${evaluation_suffix}" "${gpu_ids}" "${log_path}"
