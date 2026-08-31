#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"
remote_root="${OCR_REMOTE_ROOT:-/data3/yky/yangky_ocr_models}"
GOT_TRAINING_RUNS="${GOT_TRAINING_RUNS:-${remote_root}/training_runs/GOT}"
GOT_EVALUATION_RUNS="${GOT_EVALUATION_RUNS:-${remote_root}/evaluation_runs/GOT}"
GOT_SOURCE_MODEL="${GOT_SOURCE_MODEL:-/data4/hyf/backup/GOT-OCR2.0/GOT-OCR-2.0-master/model_weights/original}"
dataset_root="${TIME_CONSTRAINED_DATASET_ROOT:-${remote_root}/training_data/got_layout_pages/ancient_photo_diverse_formal_s3s4_dense_20260827_v4}"
mthv2_root="${TIME_CONSTRAINED_MTHV2_ROOT:-${remote_root}/datasets/MTHv2/converted/mthv2_layout_page_v1}"
p1_run_root="${TIME_CONSTRAINED_P1_RUN_ROOT:-${remote_root}/training_runs/GOT/lavp_p1_p3_formal_20260828_v9_legacy_p1_seed42/p1/model}"
run_prefix="time_constrained_original_pvld_20260829_v1"
session="time_constrained_original_pvld_20260829_v1"
gpu_ids=""
gpu_utilization_limit=50
seed=42
p1_candidate_steps="6000,9000,12000"
previous_test_summary=""
session_inner=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root) dataset_root="$2"; shift 2 ;;
        --mthv2-root) mthv2_root="$2"; shift 2 ;;
        --p1-run-root) p1_run_root="$2"; shift 2 ;;
        --run-prefix) run_prefix="$2"; shift 2 ;;
        --session) session="$2"; shift 2 ;;
        --gpu-ids) gpu_ids="$2"; shift 2 ;;
        --gpu-utilization-limit) gpu_utilization_limit="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --p1-candidate-steps) p1_candidate_steps="$2"; shift 2 ;;
        --previous-test-summary) previous_test_summary="$2"; shift 2 ;;
        --session-inner) session_inner=1; shift ;;
        *) printf '{"event":"time_constrained_baseline_failed","error":"unknown_argument","argument":"%s"}\n' "$1" >&2; exit 64 ;;
    esac
done
[[ "${run_prefix}" =~ ^[A-Za-z0-9_.-]+$ && "${session}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
[[ "${gpu_utilization_limit}" =~ ^[1-9][0-9]*$ && "${gpu_utilization_limit}" -le 100 ]] || exit 64
IFS=',' read -r -a configured_p1_steps <<<"${p1_candidate_steps}"
[[ "${#configured_p1_steps[@]}" -ge 1 ]] || exit 64
for step in "${configured_p1_steps[@]}"; do
    [[ "${step}" =~ ^[1-9][0-9]*$ ]] || exit 64
done

train_manifest="${dataset_root}/train/manifest.jsonl"
source_validation_manifest="${dataset_root}/validation/manifest.jsonl"
validation_root="${dataset_root}/validation"
test_manifest="${dataset_root}/test/manifest.jsonl"
test_root="${dataset_root}/test"
evaluation_root="${GOT_EVALUATION_RUNS}/${run_prefix}"
training_root="${GOT_TRAINING_RUNS}"
fixed_validation_manifest="${evaluation_root}/validation/validation_400.jsonl"
project_root="${ocrmodel_root}/src/GOT-OCR-2.0"
runner="${ocrmodel_root}/tools/training/run_variable_layout_a100.py"
selector="${ocrmodel_root}/tools/evaluation/select_layout_ablation_checkpoint.py"
tester="${ocrmodel_root}/tools/evaluation/evaluate_layout_ablation_test.py"
summarizer="${ocrmodel_root}/tools/evaluation/summarize_time_constrained_pvld.py"
lock_script="${ocrmodel_root}/tools/preprocessing/prepare_time_constrained_validation.py"
tokenizer_model="${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}"

require_path() { [[ -e "$1" ]] || { printf '{"event":"time_constrained_baseline_failed","error":"missing_path","path":"%s"}\n' "$1" >&2; exit 66; }; }
selection_gpus() {
    if [[ -n "${gpu_ids}" ]]; then
        printf '%s\n' "${gpu_ids}"
        return
    fi
    local selected
    selected="$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | awk -F, -v limit="${gpu_utilization_limit}" '{gsub(/[[:space:]]/,"",$1); gsub(/[[:space:]]/,"",$2); if ($2 ~ /^[0-9]+$/ && $2 < limit) printf "%s,",$1}' | sed 's/,$//')"
    [[ -n "${selected}" ]] || { printf '%s\n' '{"event":"time_constrained_baseline_failed","error":"no_eligible_gpu"}' >&2; return 75; }
    printf '%s\n' "${selected}"
}
for path in "${train_manifest}" "${source_validation_manifest}" "${test_manifest}" "${validation_root}" "${test_root}" "${p1_run_root}" "${mthv2_root}/train/manifest.jsonl" "${mthv2_root}/test/manifest.jsonl" "${GOT_SOURCE_MODEL}/model.safetensors" "${tokenizer_model}"; do require_path "${path}"; done
for path in "${training_root}/${run_prefix}_p2_seed${seed}" "${training_root}/${run_prefix}_p3_seed${seed}" "${evaluation_root}"; do
    [[ ! -e "${path}" ]] || { printf '{"event":"time_constrained_baseline_failed","error":"output_already_exists","path":"%s"}\n' "$path" >&2; exit 74; }
done
for step in "${configured_p1_steps[@]}"; do
    require_path "${p1_run_root}/checkpoint-${step}/model.safetensors"
    require_path "${p1_run_root}/checkpoint-${step}/config.json"
done

if (( session_inner == 0 )); then
    command -v tmux >/dev/null 2>&1 || exit 69
    tmux has-session -t "${session}" 2>/dev/null && exit 73
    log_root="${training_root}/${run_prefix}_pipeline_logs"
    mkdir -p "${log_root}"
    log_path="${log_root}/${session}.log"
    inner=(bash "$(realpath "${BASH_SOURCE[0]}")" --session-inner --dataset-root "${dataset_root}" --mthv2-root "${mthv2_root}" --p1-run-root "${p1_run_root}" --run-prefix "${run_prefix}" --session "${session}" --gpu-ids "${gpu_ids}" --gpu-utilization-limit "${gpu_utilization_limit}" --seed "${seed}" --p1-candidate-steps "${p1_candidate_steps}")
    [[ -n "${previous_test_summary}" ]] && inner+=(--previous-test-summary "${previous_test_summary}")
    printf -v command '%q ' "${inner[@]}"
    tmux new-session -d -s "${session}" "cd '${ocrmodel_root}' && exec ${command} >'${log_path}' 2>&1"
    sleep 3
    tmux has-session -t "${session}" 2>/dev/null || { tail -n 20 "${log_path}" >&2 || true; exit 1; }
    printf '{"event":"time_constrained_original_pvld_armed","session":"%s","run_prefix":"%s","protocol_version":"time_constrained_freeze_strategy_v1","variant":"original_pvld_freeze_strategy","validation_page_count":400,"test_used_for_selection":false,"log":"%s"}\n' "${session}" "${run_prefix}" "${log_path}"
    exit 0
fi

mkdir -p "${evaluation_root}/validation"
bash "${ocrmodel_root}/tools/environment/run_got2.sh" "${lock_script}" --source-manifest "${source_validation_manifest}" --output-manifest "${fixed_validation_manifest}" --metadata "${evaluation_root}/validation/validation_400.lock.json" --seed "${seed}"

common=(--dataset-root "${dataset_root}/train" --manifest "${train_manifest}" --validation-manifest "${fixed_validation_manifest}" --validation-image-root "${validation_root}" --test-manifest "${test_manifest}" --tokenizer-model "${tokenizer_model}" --layout-memory-resolution 64 --max-layout-records 512 --max-layout-tokens 2048 --layout-decoder-layers 2 --layout-decoder-hidden-size 256 --layout-decoder-num-heads 8 --layout-loss-preset layout_full --ablation vlqa_layout_p1_p2 --protocol-version time_constrained_freeze_strategy_v1 --variant original_pvld_freeze_strategy --validation-page-count 400 --gpu-utilization-limit "${gpu_utilization_limit}" --distributed-strategy deepspeed_zero2 --nccl-p2p-disable --seed "${seed}" --runs-root "${training_root}" --project-root "${project_root}")
[[ -n "${gpu_ids}" ]] && common+=(--gpu-ids "${gpu_ids}")

p1_selection="${evaluation_root}/p1_validation_selection/selection.json"
bash "${ocrmodel_root}/tools/environment/run_got2.sh" "${selector}" --ablation vlqa_layout_p1_p2 --model-root "${p1_run_root}" --model-kind pvld --selection-purpose p1_layout --tokenizer-model "${tokenizer_model}" --validation-manifest "${fixed_validation_manifest}" --validation-image-root "${validation_root}" --output-dir "${evaluation_root}/p1_validation_selection" --project-root "${project_root}" --max-regions 512 --max-records 0 --max-new-tokens 2048 --no-repeat-ngram-size 20 --candidate-steps "${p1_candidate_steps}" --prefer-periodic-checkpoint --protocol-version time_constrained_freeze_strategy_v1 --variant original_pvld_freeze_strategy --expected-validation-page-count 400 --parallel-gpu-ids "$(selection_gpus)" --gpu-utilization-limit "${gpu_utilization_limit}"
selected_p1="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["selected"]["model_path"])' "${p1_selection}")"

p2_id="${run_prefix}_p2_seed${seed}"
bash "${ocrmodel_root}/tools/environment/run_got2.sh" "${runner}" "${common[@]}" --source-model "${selected_p1}" --stages p2 --p2-max-steps 30000 --p2-checkpoint-steps 10000 --checkpoint-retention 3 --source-validation-selection "${p1_selection}" --run-id "${p2_id}"

p2_selection="${evaluation_root}/p2_validation_selection/selection.json"
bash "${ocrmodel_root}/tools/environment/run_got2.sh" "${selector}" --ablation vlqa_layout_p1_p2 --model-root "${training_root}/${p2_id}/p2/model" --model-kind pvld --selection-purpose ocr --tokenizer-model "${tokenizer_model}" --validation-manifest "${fixed_validation_manifest}" --validation-image-root "${validation_root}" --output-dir "${evaluation_root}/p2_validation_selection" --project-root "${project_root}" --max-regions 512 --max-records 0 --max-new-tokens 2048 --no-repeat-ngram-size 20 --candidate-steps 10000,20000,30000 --prefer-periodic-checkpoint --protocol-version time_constrained_freeze_strategy_v1 --variant original_pvld_freeze_strategy --expected-validation-page-count 400 --parallel-gpu-ids "$(selection_gpus)" --gpu-utilization-limit "${gpu_utilization_limit}"
selected_p2="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["selected"]["model_path"])' "${p2_selection}")"

p3_id="${run_prefix}_p3_seed${seed}"
p3_common=(--dataset-root "${mthv2_root}/train" --manifest "${mthv2_root}/train/manifest.jsonl" --test-manifest "${mthv2_root}/test/manifest.jsonl" --source-model "${selected_p2}" --tokenizer-model "${tokenizer_model}" --layout-memory-resolution 64 --max-layout-records 512 --max-layout-tokens 2048 --layout-decoder-layers 2 --layout-decoder-hidden-size 256 --layout-decoder-num-heads 8 --layout-loss-preset layout_full --ablation vlqa_layout_p1_p2 --protocol-version time_constrained_freeze_strategy_v1 --variant original_pvld_freeze_strategy --validation-page-count 400 --gpu-utilization-limit "${gpu_utilization_limit}" --distributed-strategy deepspeed_zero2 --nccl-p2p-disable --seed "${seed}" --runs-root "${training_root}" --project-root "${project_root}" --validation-manifest "${fixed_validation_manifest}" --validation-image-root "${validation_root}")
[[ -n "${gpu_ids}" ]] && p3_common+=(--gpu-ids "${gpu_ids}")
bash "${ocrmodel_root}/tools/environment/run_got2.sh" "${runner}" "${p3_common[@]}" --stages p3 --p3-max-steps 40000 --checkpoint-steps 8000 --checkpoint-retention 5 --source-validation-selection "${p2_selection}" --run-id "${p3_id}"

p3_selection="${evaluation_root}/p3_validation_selection/selection.json"
bash "${ocrmodel_root}/tools/environment/run_got2.sh" "${selector}" --ablation vlqa_layout_p1_p2 --model-root "${training_root}/${p3_id}/p3/model" --model-kind pvld --selection-purpose ocr --tokenizer-model "${tokenizer_model}" --validation-manifest "${fixed_validation_manifest}" --validation-image-root "${validation_root}" --output-dir "${evaluation_root}/p3_validation_selection" --project-root "${project_root}" --max-regions 512 --max-records 0 --max-new-tokens 2048 --no-repeat-ngram-size 20 --candidate-steps 8000,16000,24000,32000,40000 --prefer-periodic-checkpoint --protocol-version time_constrained_freeze_strategy_v1 --variant original_pvld_freeze_strategy --expected-validation-page-count 400 --parallel-gpu-ids "$(selection_gpus)" --gpu-utilization-limit "${gpu_utilization_limit}"

bash "${ocrmodel_root}/tools/environment/run_got2.sh" "${tester}" --selection "${p3_selection}" --test-category Real-OOD --test-manifest "${mthv2_root}/test/manifest.jsonl" --test-image-root "${mthv2_root}/test" --model-kind pvld --tokenizer-model "${tokenizer_model}" --project-root "${project_root}" --output-dir "${evaluation_root}/p3_selection_locked_test" --max-regions 512 --max-records 0 --max-new-tokens 2048 --no-repeat-ngram-size 20 --parallel-gpu-ids "$(selection_gpus)" --gpu-utilization-limit "${gpu_utilization_limit}"

summary_args=(--validation-lock "${evaluation_root}/validation/validation_400.lock.json" --p1-selection "${p1_selection}" --p2-selection "${p2_selection}" --p3-selection "${p3_selection}" --p2-training-metrics "${training_root}/${p2_id}/p2/model/layout_training_metrics.json" --p3-training-metrics "${training_root}/${p3_id}/p3/model/layout_training_metrics.json" --test-summary "${evaluation_root}/p3_selection_locked_test/summary.json" --output "${evaluation_root}/summary.json")
[[ -n "${previous_test_summary}" ]] && summary_args+=(--previous-test-summary "${previous_test_summary}")
bash "${ocrmodel_root}/tools/environment/run_got2.sh" "${summarizer}" "${summary_args[@]}"
printf '{"event":"time_constrained_original_pvld_baseline_completed","run_prefix":"%s","summary":"%s/summary.json","protocol_version":"time_constrained_freeze_strategy_v1","variant":"original_pvld_freeze_strategy","validation_page_count":400,"test_used_for_selection":false}\n' "${run_prefix}" "${evaluation_root}"
