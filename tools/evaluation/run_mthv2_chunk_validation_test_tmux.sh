#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_mthv2_chunk_validation_test_tmux.sh --session NAME \
    [--gpu-ids 0,1,2,3,4] [--run-prefix NAME]

Select each C1-C5 checkpoint using validation, then evaluate only the selected
checkpoint on test. GPU admission uses utilization.gpu < 50, allowing sharing.
This evaluates chunk records; grouped source-page aggregation remains separate.
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
paths_env="${ocrmodel_root}/config/paths.env"
if [[ -f "${paths_env}" ]]; then source "${paths_env}"; fi

session=""
gpu_ids="0,1,2,3,4"
run_prefix="mthv2_chunk_ablation_20260819_multi"
dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_column_chunks16_v1"
utilization_limit="50"
max_regions="16"
session_inner=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="${2:-}"; shift 2 ;;
        --gpu-ids) gpu_ids="${2:-}"; shift 2 ;;
        --run-prefix) run_prefix="${2:-}"; shift 2 ;;
        --dataset-root) dataset_root="${2:-}"; shift 2 ;;
        --gpu-utilization-limit) utilization_limit="${2:-}"; shift 2 ;;
        --max-regions) max_regions="${2:-}"; shift 2 ;;
        --session-inner) session_inner=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage; exit 64 ;;
    esac
done

safe_name='^[A-Za-z0-9_.-]+$'
[[ "${run_prefix}" =~ ${safe_name} ]] || { printf 'ERROR: invalid run-prefix.\n' >&2; exit 64; }
if [[ "${session_inner}" -eq 0 && ! "${session}" =~ ${safe_name} ]]; then
    printf 'ERROR: invalid session.\n' >&2; exit 64
fi
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+)*$ ]] || { printf 'ERROR: invalid gpu ids.\n' >&2; exit 64; }
[[ "${utilization_limit}" =~ ^[1-9][0-9]*$ ]] && (( utilization_limit <= 100 )) || { printf 'ERROR: invalid utilization limit.\n' >&2; exit 64; }
[[ -f "${dataset_root}/validation/manifest.jsonl" && -f "${dataset_root}/test/manifest.jsonl" ]] || { printf 'ERROR: dataset manifests missing.\n' >&2; exit 66; }
if [[ "${session_inner}" -eq 0 ]]; then
    command -v tmux >/dev/null 2>&1 || { printf 'ERROR: tmux unavailable.\n' >&2; exit 69; }
    tmux has-session -t "${session}" 2>/dev/null && { printf 'ERROR: tmux session exists.\n' >&2; exit 73; }
fi

IFS=',' read -r -a gpus <<< "${gpu_ids}"
controls=(C1 C2 C3 C4 C5)
[[ ${#gpus[@]} -eq ${#controls[@]} ]] || { printf 'ERROR: exactly five GPUs are required for C1-C5.\n' >&2; exit 64; }
declare -A seen=()
for gpu in "${gpus[@]}"; do
    [[ -z "${seen[$gpu]:-}" ]] || { printf 'ERROR: duplicate GPU id.\n' >&2; exit 64; }
    seen[$gpu]=1
done

model_kind_for() { case "$1" in C1) echo baseline;; C2) echo generic;; C3|C4|C5) echo vlqa;; esac; }
ablation_for() { case "$1" in C1) echo projector_only;; C2) echo generic_adapter_projector;; C3) echo vlqa_ocr_only;; C4) echo vlqa_layout_direct;; C5) echo vlqa_layout_p1_p2;; esac; }
model_for() {
    case "$1" in
        C1) echo "${GOT_TRAINING_RUNS}/${run_prefix}_C1_projector_only_seed42/p2/model" ;;
        C2) echo "${GOT_TRAINING_RUNS}/${run_prefix}_C2_generic_adapter_seed42/p2/model" ;;
        C3) echo "${GOT_TRAINING_RUNS}/${run_prefix}_C3_vlqa_ocr_only_seed42/p2/model" ;;
        C4) echo "${GOT_TRAINING_RUNS}/${run_prefix}_C4_vlqa_layout_direct_seed42/p2/model" ;;
        C5) echo "${GOT_TRAINING_RUNS}/${run_prefix}_C5_vlqa_p1_p2_seed42/p2/model" ;;
    esac
}

run_control() {
    local control="$1" gpu="$2" ablation kind model eval_root selection test_root
    ablation="$(ablation_for "${control}")"
    kind="$(model_kind_for "${control}")"
    model="$(model_for "${control}")"
    eval_root="${GOT_EVALUATION_RUNS}/${run_prefix}_${control}_validation_selection"
    selection="${eval_root}/selection.json"
    test_root="${GOT_EVALUATION_RUNS}/${run_prefix}_${control}_test"
    [[ -f "${model}/layout_training_metrics.json" ]] || { echo "ERROR missing model ${model}" >&2; return 66; }
    echo "{\"event\":\"mthv2_chunk_validation_started\",\"control\":\"${control}\",\"gpu\":\"${gpu}\"}"
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
      "${ocrmodel_root}/tools/evaluation/select_layout_ablation_checkpoint.py" \
      --ablation "${ablation}" --model-root "${model}" --model-kind "${kind}" \
      --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
      --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
      --validation-image-root "${dataset_root}/validation" --output-dir "${eval_root}" \
      --project-root "${ocrmodel_root}/src/GOT-OCR-2.0" --max-regions "${max_regions}" \
      --gpu-id "${gpu}" --gpu-utilization-limit "${utilization_limit}"
    echo "{\"event\":\"mthv2_chunk_test_started\",\"control\":\"${control}\",\"gpu\":\"${gpu}\"}"
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
      "${ocrmodel_root}/tools/evaluation/evaluate_layout_ablation_test.py" \
      --selection "${selection}" --test-category Real-OOD --test-manifest "${dataset_root}/test/manifest.jsonl" \
      --test-image-root "${dataset_root}/test" --model-kind "${kind}" \
      --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
      --project-root "${ocrmodel_root}/src/GOT-OCR-2.0" --output-dir "${test_root}" \
      --max-regions "${max_regions}" --gpu-id "${gpu}" --gpu-utilization-limit "${utilization_limit}"
    echo "{\"event\":\"mthv2_chunk_test_completed\",\"control\":\"${control}\"}"
}

run_all() {
    echo "{\"event\":\"mthv2_chunk_validation_test_started\",\"gpu_ids\":\"${gpu_ids}\",\"condition\":\"utilization<${utilization_limit}\"}"
    pids=()
    for i in "${!controls[@]}"; do run_control "${controls[$i]}" "${gpus[$i]}" >"${GOT_EVALUATION_RUNS}/${run_prefix}_${controls[$i]}_validation_test.log" 2>&1 & pids+=("$!"); done
    failed=0
    for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    (( failed == 0 )) || { echo '{"event":"mthv2_chunk_validation_test_failed"}'; return 1; }
    echo '{"event":"mthv2_chunk_validation_test_finished","status":"ok","granularity":"chunk","grouped_source_page_evaluation":"pending"}'
}

log_root="${GOT_EVALUATION_RUNS}/${run_prefix}_validation_test_tmux_logs"
mkdir -p "${log_root}"
log_path="${log_root}/launcher.log"
if [[ "${session_inner}" -eq 1 ]]; then
    run_all >"${log_path}" 2>&1
else
    script_path="$(realpath "${BASH_SOURCE[0]}")"
    tmux new-session -d -s "${session}" "cd '${ocrmodel_root}' && exec bash '${script_path}' --session-inner --gpu-ids '${gpu_ids}' --run-prefix '${run_prefix}' --dataset-root '${dataset_root}' --gpu-utilization-limit '${utilization_limit}' --max-regions '${max_regions}'"
    sleep 5
    tmux has-session -t "${session}" 2>/dev/null || { tail -n 20 "${log_path}" >&2 || true; exit 1; }
    echo "{\"event\":\"mthv2_chunk_validation_test_tmux_started\",\"session\":\"${session}\",\"log\":\"${log_path}\"}"
fi
