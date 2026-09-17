#!/usr/bin/env bash
# Merge a completed zero-shot shard array and compute unified benchmark OCR metrics.
#
# Usage: finalize_sota_zeroshot.sh <model> <split> <run_id>
set -euo pipefail

model="${1:?model required}"
split="${2:?split required}"
run_id="${3:?run_id required}"

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/glm_ocr_layout_ot"
code_root="${experiment_root}/code/ocrmodel"
dataset_root="${BSCC_DATASET_ROOT:-${workspace}/datasets/dunhuang_local_gazetteer_q32_v1_portable}"
python_bin="${SOTA_FINALIZE_PYTHON:-${experiment_root}/envs/sota-transformers/bin/python}"

manifest="${dataset_root}/manifests/${split}.jsonl"
run_root="${workspace}/evaluation_runs/SOTA/${run_id}/${model}/${split}"
expected=$(wc -l < "${manifest}")

export PYTHONPATH="${code_root}:${PYTHONPATH:-}"
"${python_bin}" "${code_root}/tools/sota/merge_predictions.py" \
    --manifest "${manifest}" \
    --shard-root "${run_root}" \
    --output "${run_root}/predictions.jsonl" \
    --expected-pages "${expected}"

"${python_bin}" "${code_root}/tools/sota/summarize_metrics.py" \
    --manifest "${manifest}" \
    --predictions "${run_root}/predictions.jsonl" \
    --output "${run_root}/unified_metrics.json" \
    --expected-pages "${expected}"
