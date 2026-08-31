#!/usr/bin/env bash
# A100 launcher for the BSCC-data 50,000-step Legacy P1 experiment.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

remote_root="${OCR_REMOTE_ROOT:-/data3/yky/yangky_ocr_models}"
dataset_root="${BSCC_MTHV2_ROOT_A100:-${remote_root}/datasets/MTHv2/converted/mthv2_layout_page_v1}"
source_model="${BSCC_GOT_SOURCE_MODEL_A100:-/data4/hyf/backup/GOT-OCR2.0/GOT-OCR-2.0-master/model_weights/original}"
training_root="${BSCC_P1_TRAINING_ROOT_A100:-${remote_root}/training_runs/GOT}"
run_prefix="bscc_p1_legacy_50000_a100_20260829_v1"
session="${run_prefix}"
gpu_ids=""
gpu_utilization_limit=50
seed=42
session_inner=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root) dataset_root="$2"; shift 2 ;;
        --source-model) source_model="$2"; shift 2 ;;
        --training-root) training_root="$2"; shift 2 ;;
        --run-prefix) run_prefix="$2"; shift 2 ;;
        --session) session="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --gpu-utilization-limit) gpu_utilization_limit="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --session-inner) session_inner=1; shift ;;
        *) printf '{"event":"bscc_p1_a100_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done

[[ "${run_prefix}" =~ ^[A-Za-z0-9_.-]+$ && "${session}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
for path in "${dataset_root}/train/manifest.jsonl" "${dataset_root}/validation/manifest.jsonl" "${dataset_root}/test/manifest.jsonl" "${source_model}/model.safetensors" "${source_model}/config.json"; do
    [[ -e "${path}" ]] || { printf '{"event":"bscc_p1_a100_failed","error":"missing_path","path":"%s"}\n' "${path}" >&2; exit 66; }
done
run_id="${run_prefix}"
run_root="${training_root}/${run_id}"
[[ ! -e "${run_root}" ]] || { printf '{"event":"bscc_p1_a100_failed","error":"output_already_exists","path":"%s"}\n' "${run_root}" >&2; exit 74; }

runner="${ocrmodel_root}/tools/training/run_variable_layout_a100.py"
project_root="${ocrmodel_root}/src/GOT-OCR-2.0"
tokenizer_model="${BSCC_GOT_TOKENIZER_MODEL_A100:-${source_model}}"
validation_page_count="$(wc -l < "${dataset_root}/validation/manifest.jsonl")"

run_training() {
    local -a command=(
        bash "${ocrmodel_root}/tools/environment/run_got2.sh" "${runner}"
        --dataset-root "${dataset_root}/train"
        --manifest "${dataset_root}/train/manifest.jsonl"
        --validation-manifest "${dataset_root}/validation/manifest.jsonl"
        --validation-image-root "${dataset_root}/validation"
        --test-manifest "${dataset_root}/test/manifest.jsonl"
        --source-model "${source_model}"
        --tokenizer-model "${tokenizer_model}"
        --stages p1 --ablation vlqa_layout_p1_p2 --layout-loss-preset layout_full
        --layout-memory-resolution 64 --num-layout-prompt-queries 32
        --max-layout-records 512 --max-layout-tokens 2048
        --layout-decoder-layers 2 --layout-decoder-hidden-size 256 --layout-decoder-num-heads 8
        --layout-boundary-loss-weight 0 --layout-count-condition-strength 0
        --p1-max-steps 50000 --p1-checkpoint-steps 10000
        --p1-candidate-steps 20000,30000,40000,50000 --checkpoint-retention 5
        --p1-learning-rate 1e-4 --p1-vision-learning-rate 1e-6
        --p1-projector-learning-rate 1e-5 --p1-layout-learning-rate 1e-4
        --p1-qwen-learning-rate 0 --p1-gate-learning-rate 0 --p1-lm-head-learning-rate 0
        --per-device-batch-size 1 --gradient-accumulation-steps 1
        --gpu-utilization-limit "${gpu_utilization_limit}"
        --distributed-strategy deepspeed_zero2 --nccl-p2p-disable --seed "${seed}"
        --protocol-version bscc_long_p1_freeze_v1 --variant bscc_legacy_pvld_p1
        --validation-page-count "${validation_page_count}"
        --runs-root "${training_root}" --project-root "${project_root}" --run-id "${run_id}"
    )
    [[ -n "${gpu_ids}" ]] && command+=(--gpu-ids "${gpu_ids}")
    "${command[@]}"
}

if (( session_inner == 0 )); then
    command -v tmux >/dev/null 2>&1 || exit 69
    tmux has-session -t "${session}" 2>/dev/null && exit 73
    log_root="${training_root}/${run_prefix}_pipeline_logs"
    mkdir -p "${log_root}"
    log_path="${log_root}/${session}.log"
    inner=(bash "$(realpath "${BASH_SOURCE[0]}")" --session-inner
        --dataset-root "${dataset_root}" --source-model "${source_model}"
        --training-root "${training_root}" --run-prefix "${run_prefix}" --session "${session}"
        --gpu-utilization-limit "${gpu_utilization_limit}" --seed "${seed}")
    [[ -n "${gpu_ids}" ]] && inner+=(--gpu-ids "${gpu_ids}")
    printf -v command '%q ' "${inner[@]}"
    tmux new-session -d -s "${session}" "cd '${ocrmodel_root}' && exec ${command} >'${log_path}' 2>&1"
    sleep 3
    tmux has-session -t "${session}" 2>/dev/null || { tail -n 20 "${log_path}" >&2 || true; exit 1; }
    printf '{"event":"bscc_p1_a100_armed","session":"%s","run_prefix":"%s","p1_steps":50000,"candidate_steps":[20000,30000,40000,50000],"protocol_version":"bscc_long_p1_freeze_v1","variant":"bscc_legacy_pvld_p1","validation_page_count":%s,"test_used_for_selection":false,"log":"%s"}\n' "${session}" "${run_prefix}" "${validation_page_count}" "${log_path}"
    exit 0
fi

run_training
printf '{"event":"bscc_p1_a100_completed","run_prefix":"%s","run_root":"%s","selection":"%s/p1/validation_selection/selection.json","test_used_for_selection":false}\n' "${run_prefix}" "${run_root}" "${run_root}"
