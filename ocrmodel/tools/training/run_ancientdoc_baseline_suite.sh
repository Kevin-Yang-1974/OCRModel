#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_ancientdoc_baseline_suite.sh [options]

Runs the implemented AncientDoc whole-page baselines from one entry point.
The source data is read-only. Converted layout-page manifests are written to
$GOT_LAYOUT_DATA/<ancient-dataset-id>.

Implemented training baselines:
  c4_vlqa_ocr_only       P2-8000 VLQA init + AncientDoc OCR-only.
  c5_vlqa_ocr_replay     C4 + synthetic OCR replay.
  c6_vlqa_layout_replay  C4 + synthetic replay with layout supervision.

Zero-shot C0/C1 and structural controls that still need model code are recorded
in the suite summary and handled by the validation entry point.

Options:
  --ancientdoc-root <path>        Default: $ANCIENTDOC_ROOT
  --ancient-dataset-id <id>       Default: ancientdoc_layout_260707
  --synthetic-dataset-id <id>     Default: formal_pdf_short_seed20260812
  --p2-model <path>               Default: $GOT_TRAINING_RUNS/layout_joint-train_8000_20260813/p2/model
  --c4-model <path>               Required with --start-from c5/c6; completed C4 p2/model
  --steps <n>                     Default: 2000
  --max-train-records <n>         Default: 0 (all)
  --max-validation-records <n>    Default: 0 (all)
  --max-test-records <n>          Default: 0 (all)
  --synthetic-replay-records <n>  Default: 0 (all)
  --run-prefix <prefix>           Default: ancientdoc_baseline_YYYYmmdd_HHMMSS
  --start-from <baseline>         c4, c5, or c6; skip earlier baselines
  --prepare-only                  Only convert/audit AncientDoc manifests.
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
paths_env="${ocrmodel_root}/config/paths.env"
if [[ -f "${paths_env}" ]]; then
    # shellcheck source=/dev/null
    source "${paths_env}"
fi

ancientdoc_root="${ANCIENTDOC_ROOT:-}"
ancient_dataset_id="ancientdoc_layout_260707"
synthetic_dataset_id="formal_pdf_short_seed20260812"
p2_model="${GOT_TRAINING_RUNS:-}/layout_joint-train_8000_20260813/p2/model"
c4_model=""
steps="2000"
max_train_records="0"
max_validation_records="0"
max_test_records="0"
synthetic_replay_records="0"
run_prefix="ancientdoc_baseline_$(date +%Y%m%d_%H%M%S)"
start_from="c4"
prepare_only="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ancientdoc-root) ancientdoc_root="${2:-}"; shift 2 ;;
        --ancient-dataset-id) ancient_dataset_id="${2:-}"; shift 2 ;;
        --synthetic-dataset-id) synthetic_dataset_id="${2:-}"; shift 2 ;;
        --p2-model) p2_model="${2:-}"; shift 2 ;;
        --c4-model) c4_model="${2:-}"; shift 2 ;;
        --steps) steps="${2:-}"; shift 2 ;;
        --max-train-records) max_train_records="${2:-}"; shift 2 ;;
        --max-validation-records) max_validation_records="${2:-}"; shift 2 ;;
        --max-test-records) max_test_records="${2:-}"; shift 2 ;;
        --synthetic-replay-records) synthetic_replay_records="${2:-}"; shift 2 ;;
        --run-prefix) run_prefix="${2:-}"; shift 2 ;;
        --start-from) start_from="${2:-}"; shift 2 ;;
        --prepare-only) prepare_only="1"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage; exit 64 ;;
    esac
done

case "${start_from}" in
    c4|c5|c6) ;;
    *)
        printf 'ERROR: --start-from must be c4, c5, or c6: %s\n' "${start_from}" >&2
        exit 64
        ;;
esac

if [[ -z "${GOT_LAYOUT_DATA:-}" || -z "${GOT_TRAINING_RUNS:-}" ]]; then
    printf 'ERROR: GOT_LAYOUT_DATA and GOT_TRAINING_RUNS must be set.\n' >&2
    exit 64
fi
if [[ -z "${ancientdoc_root}" || ! -d "${ancientdoc_root}" ]]; then
    printf 'ERROR: AncientDoc root is missing: %s\n' "${ancientdoc_root}" >&2
    exit 66
fi
if [[ ! -d "${p2_model}" ]]; then
    printf 'ERROR: P2 model is missing: %s\n' "${p2_model}" >&2
    exit 66
fi
if [[ "${start_from}" != "c4" && ( -z "${c4_model}" || ! -d "${c4_model}" ) ]]; then
    printf 'ERROR: --c4-model is required and must be a directory when --start-from=%s.\n' "${start_from}" >&2
    exit 66
fi

ancient_dataset_root="${GOT_LAYOUT_DATA}/${ancient_dataset_id}"
synthetic_dataset_root="${GOT_LAYOUT_DATA}/${synthetic_dataset_id}"

bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
    "${ocrmodel_root}/tools/preprocessing/prepare_ancientdoc_layout_dataset.py" \
    --ancientdoc-root "${ancientdoc_root}" \
    --output-root "${ancient_dataset_root}" \
    --max-train-records "${max_train_records}" \
    --max-validation-records "${max_validation_records}" \
    --max-test-records "${max_test_records}" \
    --symlink-images \
    --overwrite

bash "${script_dir}/check_layout_dataset_mount.sh" --dataset-root "${ancient_dataset_root}" >/dev/null
if [[ "${prepare_only}" == "1" ]]; then
    python3 - "${ancient_dataset_root}" <<'PY'
from __future__ import annotations
import json, sys
print(json.dumps({"event":"ancientdoc_baseline_suite_prepared","dataset_root":sys.argv[1]}, ensure_ascii=False, separators=(",", ":")))
PY
    exit 0
fi

suite_root="${GOT_TRAINING_RUNS}/${run_prefix}"
mkdir -p "${suite_root}"
summary="${suite_root}/suite_summary.jsonl"
: > "${summary}"

run_one() {
    local baseline_id="$1"
    shift
    local run_id="${run_prefix}_${baseline_id}"
    local source_model="${p2_model}"
    if [[ "${baseline_id}" != "c4_vlqa_ocr_only" ]]; then
        source_model="${c4_model}"
    fi
    python3 - "$baseline_id" "$run_id" "$summary" <<'PY'
from __future__ import annotations
import json, sys
baseline, run_id, summary = sys.argv[1:4]
line = json.dumps({"event":"baseline_started","baseline":baseline,"run_id":run_id}, ensure_ascii=False, separators=(",", ":"))
print(line)
with open(summary, "a", encoding="utf-8") as handle:
    handle.write(line + "\n")
PY
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/training/run_layout_a100.py" \
        --dataset-root "${ancient_dataset_root}/train" \
        --manifest "${ancient_dataset_root}/train/manifest.jsonl" \
        --validation-manifest "${ancient_dataset_root}/validation/manifest.jsonl" \
        --validation-image-root "${ancient_dataset_root}/validation" \
        --test-manifest "${ancient_dataset_root}/test/manifest.jsonl" \
        --test-image-root "${ancient_dataset_root}/test" \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --source-model "${source_model}" \
        --p2-max-steps "${steps}" \
        --mode adapt \
        --run-id "${run_id}" \
        "$@"
    python3 - "$baseline_id" "$run_id" "${GOT_TRAINING_RUNS}/${run_id}/p2/model" "$summary" <<'PY'
from __future__ import annotations
import json, sys
baseline, run_id, model, summary = sys.argv[1:5]
line = json.dumps({"event":"baseline_finished","baseline":baseline,"run_id":run_id,"model":model}, ensure_ascii=False, separators=(",", ":"))
print(line)
with open(summary, "a", encoding="utf-8") as handle:
    handle.write(line + "\n")
PY
    if [[ "${baseline_id}" == "c4_vlqa_ocr_only" ]]; then
        c4_model="${GOT_TRAINING_RUNS}/${run_id}/p2/model"
    fi
}

run_if_requested() {
    local baseline_id="$1"
    shift
    local baseline_rank
    case "${baseline_id}" in
        c4_vlqa_ocr_only) baseline_rank=4 ;;
        c5_vlqa_ocr_replay) baseline_rank=5 ;;
        c6_vlqa_layout_replay) baseline_rank=6 ;;
        *) printf 'ERROR: unknown baseline id: %s\n' "${baseline_id}" >&2; exit 64 ;;
    esac
    if (( baseline_rank < ${start_from#c} )); then
        local run_id="${run_prefix}_${baseline_id}"
        printf '{"event":"baseline_skipped","baseline":"%s","reason":"start_from_%s"}\n' \
            "${baseline_id}" "${start_from}"
        printf '{"event":"baseline_skipped","baseline":"%s","reason":"start_from_%s"}\n' \
            "${baseline_id}" "${start_from}" >>"${summary}"
        return 0
    fi
    run_one "${baseline_id}" "$@"
}

run_if_requested c4_vlqa_ocr_only --layout-loss-weight 0

run_if_requested c5_vlqa_ocr_replay \
    --layout-loss-weight 0 \
    --replay-manifest "${synthetic_dataset_root}/train/manifest.jsonl" \
    --replay-image-root "${synthetic_dataset_root}/train" \
    --replay-split train \
    --replay-max-records "${synthetic_replay_records}" \
    --primary-per-replay 3

run_if_requested c6_vlqa_layout_replay \
    --replay-manifest "${synthetic_dataset_root}/train/manifest.jsonl" \
    --replay-image-root "${synthetic_dataset_root}/train" \
    --replay-split train \
    --replay-max-records "${synthetic_replay_records}" \
    --primary-per-replay 3

cat >>"${summary}" <<EOF
{"event":"baseline_not_implemented","baseline":"c2_original_got2_lora","reason":"LoRA-only whole-page baseline is not wired into the VLQA launcher yet."}
{"event":"baseline_not_implemented","baseline":"c7_equal_parameter_visual_adaptor","reason":"Requires separate non-VLQA adaptor model code."}
{"event":"baseline_not_implemented","baseline":"c8_unsupervised_queries","reason":"Requires a structural switch that keeps queries but disables layout targets without reusing C4."}
{"event":"baseline_not_implemented","baseline":"c10_full_finetune_upper_bound","reason":"Requires explicit full/partial decoder unfreeze policy and memory budget."}
EOF

python3 - "$suite_root" "$summary" <<'PY'
from __future__ import annotations
import json, sys
suite_root, summary = sys.argv[1:3]
print(json.dumps({"event":"ancientdoc_baseline_suite_completed","suite_root":suite_root,"summary":summary}, ensure_ascii=False, separators=(",", ":")))
PY
