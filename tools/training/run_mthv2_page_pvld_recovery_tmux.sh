#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

session="mthv2_page_pvld_recovery_20260821_r1"
run_prefix="mthv2_page_pvld_ablation_20260821_r1"
gpu_ids="0,1,2,3,4"
dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1"
utilization_limit=50
control_indices="0,1,2,3,4"
evaluation_indices="0,1,2,3,4"
evaluation_suffix=""
skip_training=0
session_inner=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="$2"; shift 2 ;;
        --run-prefix) run_prefix="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --gpu-utilization-limit) utilization_limit="$2"; shift 2 ;;
        --control-indices) control_indices="$2"; shift 2 ;;
        --evaluation-indices) evaluation_indices="$2"; shift 2 ;;
        --evaluation-suffix) evaluation_suffix="$2"; shift 2 ;;
        --skip-training) skip_training=1; shift ;;
        --session-inner) session_inner=1; shift ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

IFS=',' read -r -a gpus <<< "${gpu_ids}"
for gpu in "${gpus[@]}"; do
    utilization="$(nvidia-smi -i "${gpu}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
    (( utilization < utilization_limit )) || { printf 'ERROR: GPU%s utilization=%s.\n' "${gpu}" "${utilization}" >&2; exit 75; }
done

log_root="${GOT_TRAINING_RUNS}/${run_prefix}_${session}_logs"
log_path="${log_root}/launcher.log"
mkdir -p "${log_root}"

resume_fixed() {
    local gpu="$1" ablation="$2" run_id="$3"
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/training/resume_layout_stage_a100.py" \
        --mode ablation --ablation "${ablation}" \
        --dataset-root "${dataset_root}/train" \
        --manifest "${dataset_root}/train/manifest.jsonl" \
        --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
        --validation-image-root "${dataset_root}/validation" \
        --test-manifest "${dataset_root}/test/manifest.jsonl" \
        --test-image-root "${dataset_root}/test" \
        --source-model "${GOT_SOURCE_MODEL}" \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --p2-max-steps 42000 --checkpoint-steps 3000 \
        --gpu-utilization-limit "${utilization_limit}" --gpu-id "${gpu}" \
        --seed 42 --run-id "${run_id}" --max-regions 512 \
        --layout-loss-preset layout_none --skip-post-training-validation
}

resume_pvld() {
    local gpu="$1" ablation="$2" preset="$3" source="$4" run_id="$5" max_steps="$6"
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/training/run_variable_layout_a100.py" \
        --resume-existing-run --stages p2 \
        --dataset-root "${dataset_root}/train" \
        --manifest "${dataset_root}/train/manifest.jsonl" \
        --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
        --test-manifest "${dataset_root}/test/manifest.jsonl" \
        --source-model "${source}" \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --ablation "${ablation}" --layout-loss-preset "${preset}" \
        --num-layout-prompt-queries 32 --max-layout-records 512 \
        --max-layout-tokens 2048 --layout-decoder-layers 2 \
        --layout-decoder-hidden-size 256 --layout-decoder-num-heads 8 \
        --p2-max-steps "${max_steps}" --checkpoint-steps 3000 --checkpoint-retention 2 \
        --gpu-utilization-limit "${utilization_limit}" --gpu-ids "${gpu}" \
        --seed 42 --run-id "${run_id}"
}

evaluate_control() {
    local control="$1" gpu="$2" ablation="$3" kind="$4" run_id="$5"
    local model_root="${GOT_TRAINING_RUNS}/${run_id}/p2/model"
    local selection_root="${GOT_EVALUATION_RUNS}/${run_prefix}_${control}_validation_selection${evaluation_suffix}"
    local test_root="${GOT_EVALUATION_RUNS}/${run_prefix}_${control}_test${evaluation_suffix}"
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/evaluation/select_layout_ablation_checkpoint.py" \
        --ablation "${ablation}" --model-root "${model_root}" --model-kind "${kind}" \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
        --validation-image-root "${dataset_root}/validation" --output-dir "${selection_root}" \
        --project-root "${ocrmodel_root}/src/GOT-OCR-2.0" --max-regions 512 \
        --gpu-id "${gpu}" --gpu-utilization-limit "${utilization_limit}"
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/evaluation/evaluate_layout_ablation_test.py" \
        --selection "${selection_root}/selection.json" --test-category Real-OOD \
        --test-manifest "${dataset_root}/test/manifest.jsonl" \
        --test-image-root "${dataset_root}/test" --model-kind "${kind}" \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --project-root "${ocrmodel_root}/src/GOT-OCR-2.0" --output-dir "${test_root}" \
        --max-regions 512 --gpu-id "${gpu}" --gpu-utilization-limit "${utilization_limit}"
}

run_training_control() {
    local index="$1" gpu="$2"
    case "${index}" in
        0) resume_fixed "${gpu}" projector_only "${run_ids[0]}" ;;
        1) resume_fixed "${gpu}" generic_adapter_projector "${run_ids[1]}" ;;
        2) resume_pvld "${gpu}" vlqa_ocr_only layout_none "${GOT_SOURCE_MODEL}" "${run_ids[2]}" 42000 ;;
        3) resume_pvld "${gpu}" vlqa_layout_direct layout_full "${GOT_SOURCE_MODEL}" "${run_ids[3]}" 42000 ;;
        4) resume_pvld "${gpu}" vlqa_layout_p1_p2 layout_full "${GOT_TRAINING_RUNS}/${run_ids[4]}/p1/model" "${run_ids[4]}" 30000 ;;
    esac
}

run_pipeline() {
    run_ids=(
        "${run_prefix}_C1_projector_only_seed42"
        "${run_prefix}_C2_generic_adapter_seed42"
        "${run_prefix}_C3_pvld_ocr_only_seed42"
        "${run_prefix}_C4_pvld_layout_direct_seed42"
        "${run_prefix}_C5_pvld_p1_p2_seed42"
    )
    local failed=0 start index offset gpu pid
    local pids=()
    local train_indices eval_indices
    IFS=',' read -r -a train_indices <<< "${control_indices}"
    IFS=',' read -r -a eval_indices <<< "${evaluation_indices}"
    for ((start=0; skip_training==0 && start<${#train_indices[@]}; start+=${#gpus[@]})); do
        pids=()
        for ((offset=0; offset<${#gpus[@]} && start+offset<${#train_indices[@]}; offset++)); do
            index="${train_indices[start + offset]}"
            gpu="${gpus[offset]}"
            run_training_control "${index}" "${gpu}" >"${log_root}/C$((index + 1))_recovery.log" 2>&1 &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
        (( failed == 0 )) || { printf '%s\n' '{"event":"pvld_recovery_training_failed"}'; return 1; }
    done

    local controls=(C1 C2 C3 C4 C5)
    local ablations=(projector_only generic_adapter_projector vlqa_ocr_only vlqa_layout_direct vlqa_layout_p1_p2)
    local kinds=(baseline generic pvld pvld pvld)
    for ((start=0; start<${#eval_indices[@]}; start+=${#gpus[@]})); do
        pids=()
        for ((offset=0; offset<${#gpus[@]} && start+offset<${#eval_indices[@]}; offset++)); do
            index="${eval_indices[start + offset]}"
            gpu="${gpus[offset]}"
            evaluate_control "${controls[index]}" "${gpu}" "${ablations[index]}" "${kinds[index]}" "${run_ids[index]}" >"${log_root}/${controls[index]}_eval.log" 2>&1 &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do wait "${pid}" || failed=1; done
        (( failed == 0 )) || { printf '%s\n' '{"event":"pvld_recovery_validation_test_failed"}'; return 1; }
    done
    printf '%s\n' '{"event":"mthv2_page_pvld_recovery_completed","selection":"validation_only","test":"selection_locked"}'
}

if (( session_inner == 1 )); then
    run_pipeline
    exit
fi

tmux has-session -t "${session}" 2>/dev/null && { printf 'ERROR: tmux session exists.\n' >&2; exit 73; }
script_path="$(realpath "${BASH_SOURCE[0]}")"
tmux new-session -d -s "${session}" \
    "cd '${ocrmodel_root}' && exec bash '${script_path}' --session-inner --session '${session}' --run-prefix '${run_prefix}' --gpu-ids '${gpu_ids}' --gpu-utilization-limit '${utilization_limit}' --control-indices '${control_indices}' --evaluation-indices '${evaluation_indices}' --evaluation-suffix '${evaluation_suffix}' $([[ ${skip_training} -eq 1 ]] && printf '%s' '--skip-training') >'${log_path}' 2>&1"
sleep 5
tmux has-session -t "${session}" 2>/dev/null || { tail -n 20 "${log_path}" >&2 || true; exit 1; }
printf '{"event":"mthv2_page_pvld_recovery_started","session":"%s","run_prefix":"%s","gpu_ids":"%s","dataset_root":"%s","log":"%s"}\n' \
    "${session}" "${run_prefix}" "${gpu_ids}" "${dataset_root}" "${log_path}"
