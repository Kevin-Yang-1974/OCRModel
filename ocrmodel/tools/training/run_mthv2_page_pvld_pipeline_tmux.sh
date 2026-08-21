#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

session="mthv2_page_pvld_pipeline_20260821"
run_prefix="mthv2_page_pvld_ablation_20260821"
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
    (( utilization < utilization_limit )) || { printf 'ERROR: GPU%s utilization=%s.\n' "${gpu}" "${utilization}" >&2; exit 75; }
done

log_root="${GOT_TRAINING_RUNS}/${run_prefix}_pipeline_tmux_logs"
log_path="${log_root}/launcher.log"
mkdir -p "${log_root}"

train_fixed_control() {
    local control="$1" gpu="$2" ablation="$3" run_id="$4"
    local gpu_args
    if [[ "${gpu}" == *,* ]]; then gpu_args=(--gpu-ids "${gpu}"); else gpu_args=(--gpu-id "${gpu}"); fi
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/training/run_layout_a100.py" \
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
        --gpu-utilization-limit "${utilization_limit}" "${gpu_args[@]}" \
        --seed 42 --run-id "${run_id}" --max-regions 512 \
        --layout-loss-preset layout_none --skip-post-training-validation
}

train_pvld_control() {
    local control="$1" gpu="$2" ablation="$3" preset="$4" stages="$5" run_id="$6"
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
        --p1-max-steps 12000 --p2-max-steps "$([[ "${stages}" == p2 ]] && printf 42000 || printf 30000)" \
        --checkpoint-steps 3000 --gpu-utilization-limit "${utilization_limit}" \
        --gpu-ids "${gpu}" --seed 42 --run-id "${run_id}"
}

evaluate_control() {
    local control="$1" gpu="$2" ablation="$3" kind="$4" run_id="$5"
    local model_root="${GOT_TRAINING_RUNS}/${run_id}/p2/model"
    local selection_root="${GOT_EVALUATION_RUNS}/${run_prefix}_${control}_validation_selection"
    local test_root="${GOT_EVALUATION_RUNS}/${run_prefix}_${control}_test"
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

run_pipeline() {
    local run_ids=(
        "${run_prefix}_C1_projector_only_seed42"
        "${run_prefix}_C2_generic_adapter_seed42"
        "${run_prefix}_C3_pvld_ocr_only_seed42"
        "${run_prefix}_C4_pvld_layout_direct_seed42"
        "${run_prefix}_C5_pvld_p1_p2_seed42"
    )
    failed=0
    if (( ${#gpus[@]} >= 5 )); then
        train_fixed_control C1 "${gpus[0]}" projector_only "${run_ids[0]}" >"${log_root}/C1.log" 2>&1 & p1=$!
        train_fixed_control C2 "${gpus[1]}" generic_adapter_projector "${run_ids[1]}" >"${log_root}/C2.log" 2>&1 & p2=$!
        train_pvld_control C3 "${gpus[2]}" vlqa_ocr_only layout_none p2 "${run_ids[2]}" >"${log_root}/C3.log" 2>&1 & p3=$!
        train_pvld_control C4 "${gpus[3]}" vlqa_layout_direct layout_full p2 "${run_ids[3]}" >"${log_root}/C4.log" 2>&1 & p4=$!
        train_pvld_control C5 "${gpus[4]}" vlqa_layout_p1_p2 layout_full p1,p2 "${run_ids[4]}" >"${log_root}/C5.log" 2>&1 & p5=$!
        for pid in "${p1}" "${p2}" "${p3}" "${p4}" "${p5}"; do wait "${pid}" || failed=1; done
    else
        train_fixed_control C1 "${gpu_ids}" projector_only "${run_ids[0]}" >"${log_root}/C1.log" 2>&1 || failed=1
        (( failed )) || train_fixed_control C2 "${gpu_ids}" generic_adapter_projector "${run_ids[1]}" >"${log_root}/C2.log" 2>&1 || failed=1
        (( failed )) || train_pvld_control C3 "${gpu_ids}" vlqa_ocr_only layout_none p2 "${run_ids[2]}" >"${log_root}/C3.log" 2>&1 || failed=1
        (( failed )) || train_pvld_control C4 "${gpu_ids}" vlqa_layout_direct layout_full p2 "${run_ids[3]}" >"${log_root}/C4.log" 2>&1 || failed=1
        (( failed )) || train_pvld_control C5 "${gpu_ids}" vlqa_layout_p1_p2 layout_full p1,p2 "${run_ids[4]}" >"${log_root}/C5.log" 2>&1 || failed=1
    fi
    (( failed == 0 )) || { printf '%s\n' '{"event":"pvld_training_failed"}'; return 1; }
    local controls=(C1 C2 C3 C4 C5)
    local ablations=(projector_only generic_adapter_projector vlqa_ocr_only vlqa_layout_direct vlqa_layout_p1_p2)
    local kinds=(baseline generic pvld pvld pvld)
    local eval_pids=()
    failed=0
    for index in "${!controls[@]}"; do
        gpu="${gpus[$((index % ${#gpus[@]}))]}"
        evaluate_control "${controls[index]}" "${gpu}" "${ablations[index]}" "${kinds[index]}" "${run_ids[index]}" >"${log_root}/${controls[index]}_eval.log" 2>&1 &
        eval_pids+=("$!")
        if (( ${#eval_pids[@]} == ${#gpus[@]} || index == 4 )); then
            for pid in "${eval_pids[@]}"; do wait "${pid}" || failed=1; done
            eval_pids=()
        fi
    done
    (( failed == 0 )) || { printf '%s\n' '{"event":"pvld_validation_test_failed"}'; return 1; }
    printf '%s\n' '{"event":"mthv2_page_pvld_pipeline_completed","selection":"validation_only","test":"selection_locked"}'
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
printf '{"event":"mthv2_page_pvld_pipeline_started","session":"%s","run_prefix":"%s","gpu_ids":"%s","dataset_root":"%s","log":"%s"}\n' \
    "${session}" "${run_prefix}" "${gpu_ids}" "${dataset_root}" "${log_path}"
