#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_ancientdoc_baseline_suite.sh --phase core [options]
  run_ancientdoc_baseline_suite.sh --phase replay --c4-selection <selection.json> [options]

Phases:
  core    Train C1 and C4 only, then stop for validation-only C4 selection.
  replay  Resolve one frozen C4-best from selection.json and independently train C5/C6.

Common options:
  --ancient-dataset-id <id>       Default: ancientdoc_layout_260707_group_isolated_seed20260815
  --synthetic-dataset-id <id>     Default: formal_pdf_short_seed20260812
  --steps <n>                     Default: 12000
  --checkpoint-steps <n>          Default: 2000
  --learning-rate <value>         Default: 2e-5
  --gpu-id <id>                   Explicit physical GPU (default: 0)
  --max-train-records <n>         Default: 0 (all)
  --synthetic-replay-records <n>  Default: 0 (all)
  --run-prefix <prefix>           Required for formal runs
  --prepare-only                  Verify data and exit

Core options:
  --p2-model <path>               Synthetic P2 initialization for C4

Replay options:
  --c4-selection <path>           Required formal C4 validation selection
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
if [[ -f "${ocrmodel_root}/config/paths.env" ]]; then
    # shellcheck source=/dev/null
    source "${ocrmodel_root}/config/paths.env"
fi

phase=""
ancient_dataset_id="ancientdoc_layout_260707_group_isolated_seed20260815"
synthetic_dataset_id="formal_pdf_short_seed20260812"
p2_model="${GOT_TRAINING_RUNS:-}/layout_joint-train_8000_20260813/p2/model"
c4_selection=""
steps="12000"
checkpoint_steps="2000"
learning_rate="2e-5"
gpu_id="${GOT_PHYSICAL_GPU:-0}"
max_train_records="0"
synthetic_replay_records="0"
run_prefix=""
prepare_only="0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) phase="${2:-}"; shift 2 ;;
        --ancient-dataset-id) ancient_dataset_id="${2:-}"; shift 2 ;;
        --synthetic-dataset-id) synthetic_dataset_id="${2:-}"; shift 2 ;;
        --p2-model) p2_model="${2:-}"; shift 2 ;;
        --c4-selection) c4_selection="${2:-}"; shift 2 ;;
        --steps) steps="${2:-}"; shift 2 ;;
        --checkpoint-steps) checkpoint_steps="${2:-}"; shift 2 ;;
        --learning-rate) learning_rate="${2:-}"; shift 2 ;;
        --gpu-id) gpu_id="${2:-}"; shift 2 ;;
        --max-train-records) max_train_records="${2:-}"; shift 2 ;;
        --synthetic-replay-records) synthetic_replay_records="${2:-}"; shift 2 ;;
        --run-prefix) run_prefix="${2:-}"; shift 2 ;;
        --prepare-only) prepare_only="1"; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage; exit 64 ;;
    esac
done

if [[ "${phase}" != "core" && "${phase}" != "replay" ]]; then
    printf 'ERROR: --phase must be core or replay.\n' >&2
    exit 64
fi
if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    printf 'ERROR: --gpu-id must be one numeric physical GPU id.\n' >&2
    exit 64
fi
if [[ -z "${GOT_LAYOUT_DATA:-}" || -z "${GOT_TRAINING_RUNS:-}" ]]; then
    printf 'ERROR: GOT_LAYOUT_DATA and GOT_TRAINING_RUNS must be set.\n' >&2
    exit 64
fi

ancient_dataset_root="${GOT_LAYOUT_DATA}/${ancient_dataset_id}"
synthetic_dataset_root="${GOT_LAYOUT_DATA}/${synthetic_dataset_id}"
python3 "${ocrmodel_root}/tools/preprocessing/verify_ancientdoc_group_audit.py" \
    "${ancient_dataset_root}/audit/ancientdoc_split_leakage/split_leakage_audit.json" \
    --split-audit "${ancient_dataset_root}/split_audit.json" \
    --max-ratio-deviation 0.03 >/dev/null
bash "${script_dir}/check_layout_dataset_mount.sh" \
    --dataset-root "${ancient_dataset_root}" >/dev/null

if [[ "${prepare_only}" == "1" ]]; then
    printf '{"event":"ancientdoc_%s_suite_prepared","dataset_root":"%s"}\n' \
        "${phase}" "${ancient_dataset_root}"
    exit 0
fi
if [[ -z "${run_prefix}" ]]; then
    printf 'ERROR: --run-prefix is required for formal training.\n' >&2
    exit 64
fi

suite_root="${GOT_TRAINING_RUNS}/${run_prefix}"
if [[ -e "${suite_root}" ]]; then
    printf 'ERROR: suite output already exists: %s\n' "${suite_root}" >&2
    exit 74
fi
mkdir -p "${suite_root}"
summary="${suite_root}/suite_summary.jsonl"
: > "${summary}"

record_event() {
    local event="$1"
    local baseline="$2"
    local run_id="$3"
    local model="$4"
    local extra_json="{}"
    if [[ $# -ge 5 ]]; then
        extra_json="$5"
    fi
    python3 - "${event}" "${baseline}" "${run_id}" "${model}" "${extra_json}" "${summary}" <<'PY'
from __future__ import annotations
import json, sys
event, baseline, run_id, model, extra, summary = sys.argv[1:7]
payload = {"event": event, "baseline": baseline, "run_id": run_id}
if model:
    payload["model"] = model
payload.update(json.loads(extra))
line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
print(line, flush=True)
with open(summary, "a", encoding="utf-8") as handle:
    handle.write(line + "\n")
PY
}

common_layout_args=(
    --dataset-root "${ancient_dataset_root}/train"
    --manifest "${ancient_dataset_root}/train/manifest.jsonl"
    --validation-manifest "${ancient_dataset_root}/validation/manifest.jsonl"
    --validation-image-root "${ancient_dataset_root}/validation"
    --test-manifest "${ancient_dataset_root}/test/manifest.jsonl"
    --test-image-root "${ancient_dataset_root}/test"
    --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}"
    --p2-max-steps "${steps}"
    --p2-learning-rate "${learning_rate}"
    --p2-train-scope decoder_adapter_projector
    --checkpoint-steps "${checkpoint_steps}"
    --lr-scheduler-type cosine
    --warmup-ratio 0.03
    --weight-decay 0.01
    --p2-max-records "${max_train_records}"
    --gpu-id "${gpu_id}"
    --mode adapt
    --skip-post-training-validation
)

if [[ "${phase}" == "core" ]]; then
    if [[ -z "${GOT_SOURCE_MODEL:-}" || ! -d "${GOT_SOURCE_MODEL}" ]]; then
        printf 'ERROR: GOT_SOURCE_MODEL is missing.\n' >&2
        exit 66
    fi
    if [[ ! -d "${p2_model}" ]]; then
        printf 'ERROR: synthetic P2 model is missing: %s\n' "${p2_model}" >&2
        exit 66
    fi

    c1_id="${run_prefix}_c1_got2_ocr_only"
    record_event baseline_started c1_got2_ocr_only "${c1_id}" ""
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/training/run_got2_page_ocr_a100.py" \
        --dataset-root "${ancient_dataset_root}/train" \
        --manifest "${ancient_dataset_root}/train/manifest.jsonl" \
        --validation-manifest "${ancient_dataset_root}/validation/manifest.jsonl" \
        --test-manifest "${ancient_dataset_root}/test/manifest.jsonl" \
        --source-model "${GOT_SOURCE_MODEL}" \
        --max-steps "${steps}" \
        --max-train-records "${max_train_records}" \
        --learning-rate "${learning_rate}" \
        --checkpoint-steps "${checkpoint_steps}" \
        --lr-scheduler-type cosine \
        --warmup-ratio 0.03 \
        --weight-decay 0.01 \
        --train-scope decoder_projector \
        --gpu-id "${gpu_id}" \
        --run-id "${c1_id}"
    c1_model="${GOT_TRAINING_RUNS}/${c1_id}/model"
    record_event baseline_finished c1_got2_ocr_only "${c1_id}" "${c1_model}"

    c4_id="${run_prefix}_c4_vlqa_ocr_only"
    record_event baseline_started c4_vlqa_ocr_only "${c4_id}" ""
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/training/run_layout_a100.py" \
        "${common_layout_args[@]}" \
        --source-model "${p2_model}" \
        --layout-loss-weight 0 \
        --run-id "${c4_id}"
    c4_model="${GOT_TRAINING_RUNS}/${c4_id}/p2/model"
    record_event baseline_finished c4_vlqa_ocr_only "${c4_id}" "${c4_model}"
    printf '{"event":"ancientdoc_train_core_completed","suite_root":"%s","next":"select-c4"}\n' "${suite_root}"
    exit 0
fi

if [[ -z "${c4_selection}" ]]; then
    printf 'ERROR: replay phase requires --c4-selection.\n' >&2
    exit 64
fi
selection_output="$(python3 "${script_dir}/c4_selection_contract.py" \
    --selection "${c4_selection}" --format lines)"
mapfile -t selection_fields <<<"${selection_output}"
if [[ "${#selection_fields[@]}" -ne 8 ]]; then
    printf 'ERROR: C4 selection resolver returned incomplete output.\n' >&2
    exit 65
fi
c4_model="${selection_fields[0]}"
c4_step="${selection_fields[1]}"
c4_cer="${selection_fields[2]}"
c4_weights_sha="${selection_fields[5]}"
c4_selection_resolved="${selection_fields[7]}"
if [[ ! -d "${c4_model}" ]]; then
    printf 'ERROR: selected C4 model is missing: %s\n' "${c4_model}" >&2
    exit 66
fi

branch_extra="$(python3 - "${c4_selection_resolved}" "${c4_step}" "${c4_cer}" "${c4_weights_sha}" <<'PY'
import json, sys
print(json.dumps({
    "c4_selection": sys.argv[1],
    "selected_c4_step": int(sys.argv[2]),
    "selected_c4_validation_page_cer": float(sys.argv[3]),
    "selected_c4_weights_sha256": sys.argv[4],
}, separators=(",", ":")))
PY
)"

run_replay_branch() {
    local baseline="$1"
    local layout_loss="$2"
    local run_id="${run_prefix}_${baseline}"
    record_event baseline_started "${baseline}" "${run_id}" "" "${branch_extra}"
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/training/run_layout_a100.py" \
        "${common_layout_args[@]}" \
        --source-model "${c4_model}" \
        --c4-selection "${c4_selection_resolved}" \
        --layout-loss-weight "${layout_loss}" \
        --replay-manifest "${synthetic_dataset_root}/train/manifest.jsonl" \
        --replay-image-root "${synthetic_dataset_root}/train" \
        --replay-split train \
        --replay-max-records "${synthetic_replay_records}" \
        --primary-per-replay 3 \
        --run-id "${run_id}"
    local model="${GOT_TRAINING_RUNS}/${run_id}/p2/model"
    record_event baseline_finished "${baseline}" "${run_id}" "${model}" "${branch_extra}"
}

run_replay_branch c5_vlqa_ocr_replay 0
run_replay_branch c6_vlqa_layout_replay 1

python3 - "${GOT_TRAINING_RUNS}/${run_prefix}_c5_vlqa_ocr_replay/p2/model/layout_training_metrics.json" \
    "${GOT_TRAINING_RUNS}/${run_prefix}_c6_vlqa_layout_replay/p2/model/layout_training_metrics.json" <<'PY'
from __future__ import annotations
import json, sys
left, right = (json.load(open(path, encoding="utf-8"))["branch_initialization"] for path in sys.argv[1:3])
keys = ("selection_path", "selected_c4_step", "selected_c4_model_path", "selected_c4_weights_sha256")
if any(left.get(key) != right.get(key) for key in keys):
    raise SystemExit("C5/C6 initial C4 checkpoint provenance differs")
if left.get("optimizer_state_initialization") != "fresh" or right.get("optimizer_state_initialization") != "fresh":
    raise SystemExit("C5/C6 optimizer state was not initialized fresh")
print(json.dumps({"event":"ancientdoc_train_replay_completed","shared_c4":{key:left[key] for key in keys}}, separators=(",", ":")))
PY
