#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

run_id="${1:?run id required}"
session="${2:?tmux session required}"
gpu_ids="${3:?comma-separated evaluation GPUs required}"
dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1"
run_root="${GOT_TRAINING_RUNS}/${run_id}"
selection_root="${GOT_EVALUATION_RUNS}/${run_id}_p2_validation_selection_auto"
test_root="${GOT_EVALUATION_RUNS}/${run_id}_test_auto"
status_path="${run_root}/metadata/status.txt"
log_root="${GOT_EVALUATION_RUNS}/${run_id}_auto_eval_logs"
pipeline_log="${log_root}/${session}.log"
mkdir -p "${log_root}"

wait_for_training() {
    while true; do
        if [[ -f "${status_path}" ]]; then
            status="$(python3 - "${status_path}" <<'PY'
import json, sys
from pathlib import Path
payload=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("status",""), payload.get("stage_status",""))
PY
)"
            case "${status}" in
                "training_completed completed") return 0 ;;
                "failed"*|*" failed")
                    printf '{"event":"pvld_auto_eval_blocked","status":"%s"}\n' "${status}"
                    return 1
                    ;;
            esac
        fi
        sleep 30
    done
}

run_pipeline() {
    wait_for_training
    [[ -f "${run_root}/p2/model/layout_training_metrics.json" ]] || {
        printf '%s\n' '{"event":"pvld_auto_eval_missing_final_model"}'
        return 1
    }
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/evaluation/select_layout_ablation_checkpoint.py" \
        --ablation vlqa_layout_p1_p2 --model-root "${run_root}/p2/model" \
        --model-kind pvld --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
        --validation-image-root "${dataset_root}/validation" \
        --output-dir "${selection_root}" --project-root "${ocrmodel_root}/src/GOT-OCR-2.0" \
        --max-regions 512 --max-records 0 --max-new-tokens 2048 \
        --no-repeat-ngram-size 20 --parallel-gpu-ids "${gpu_ids}" \
        --gpu-utilization-limit 50 --resume
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/evaluation/evaluate_layout_ablation_test.py" \
        --selection "${selection_root}/selection.json" --test-category Real-OOD \
        --test-manifest "${dataset_root}/test/manifest.jsonl" \
        --test-image-root "${dataset_root}/test" --model-kind pvld \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --project-root "${ocrmodel_root}/src/GOT-OCR-2.0" --output-dir "${test_root}" \
        --max-regions 512 --parallel-gpu-ids "${gpu_ids}" --gpu-utilization-limit 50
    printf '{"event":"pvld_auto_validation_test_completed","run_id":"%s","selection":"%s","test":"%s","test_used_for_selection":false}\n' "${run_id}" "${selection_root}/selection.json" "${test_root}"
}

if [[ "${AUTO_EVAL_SESSION_INNER:-0}" == 1 ]]; then
    run_pipeline >"${pipeline_log}" 2>&1
    exit
fi

tmux has-session -t "${session}" 2>/dev/null && { printf 'ERROR: tmux session exists: %s\n' "${session}" >&2; exit 73; }
script_path="$(realpath "${BASH_SOURCE[0]}")"
tmux new-session -d -s "${session}" \
    "AUTO_EVAL_SESSION_INNER=1 bash '${script_path}' '${run_id}' '${session}' '${gpu_ids}'"
printf '{"event":"pvld_auto_eval_armed","session":"%s","run_id":"%s","selection":"%s","test":"%s","log":"%s"}\n' "${session}" "${run_id}" "${selection_root}" "${test_root}" "${pipeline_log}"
