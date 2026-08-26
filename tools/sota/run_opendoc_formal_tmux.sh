#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"

session="sota_opendoc_formal_20260824_v2"
run_id="sota_opendoc_formal_20260824_v2"
session_inner=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --session) session="$2"; shift 2 ;;
        --run-id) run_id="$2"; shift 2 ;;
        --session-inner) session_inner=1; shift ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

evaluation_root="/data3/yky/yangky_ocr_models/evaluation_runs/SOTA"
run_root="${evaluation_root}/${run_id}"
base="${run_root}/opendoc_0_1b"
dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1"
model_root="/data3/yky/yangky_ocr_models/models/sota/opendoc_0_1b/a377e00d62c01b6544603e2a90f2cffe2a0388e1"
python_bin="/data3/yky/yangky_ocr_models/envs/anandasky/bin/python"
site_packages="/data3/yky/yangky_ocr_models/envs/sota/opendoc_onnx_overlay_20260824:/data3/yky/yangky_ocr_models/envs/sota/opendoc_overlay_20260823:/data3/yky/yangky_ocr_models/envs/sota/opendoc_0_1b/site-packages:/data3/yky/yangky_ocr_models/envs/anandasky/lib/python3.11/site-packages"
onnx_cache="/data3/yky/yangky_ocr_models/models/sota/opendoc_onnx_cache_20260824"

validate_predictions() {
    local path="$1" expected="$2" split="$3"
    "${python_bin}" - "${path}" "${expected}" "${split}" <<'PY'
import json
import sys
from pathlib import Path

path, expected, split = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
records = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
if len(records) != expected:
    raise SystemExit(f"{split} prediction count mismatch: {len(records)} != {expected}")
failed = [record.get("page_id") for record in records if record.get("status") != "ok"]
if failed:
    raise SystemExit(f"{split} contains {len(failed)} non-ok predictions; first={failed[:3]}")
if any(not isinstance(record.get("normalized_text"), str) for record in records):
    raise SystemExit(f"{split} contains invalid normalized_text")
print(json.dumps({"event": "opendoc_predictions_validated", "split": split, "pages": len(records), "failed": 0}, separators=(",", ":")))
PY
}

run_pipeline() {
    mkdir -p "${base}"
    trap 'rc=$?; printf "{\"event\":\"opendoc_formal_failed\",\"exit_code\":%s,\"test_used_for_selection\":false}\n" "$rc" >"${base}/failed.json"; exit "$rc"' ERR
    export PYTHONPATH="${ocrmodel_root}:${site_packages}:${PYTHONPATH:-}"
    export OPENOCR_PYTHONPATH="${site_packages}"
    export OPENOCR_ONNX=1
    export OPENOCR_ONNX_CACHE="${onnx_cache}"

    printf '%s\n' '{"event":"formal_training_blocked","model":"opendoc_0_1b","reason":"official_finetuning_unavailable","test_used":false}' >"${base}/training_blocked.json"
    "${python_bin}" -m tools.sota.run_zero_shot \
        --model opendoc_0_1b --model-root "${model_root}" \
        --manifest "${dataset_root}/validation/manifest.jsonl" \
        --image-root "${dataset_root}/validation" --output-dir "${base}/validation" \
        --split validation --device cpu --dtype fp32 >"${base}/validation.log" 2>&1
    validate_predictions "${base}/validation/predictions.jsonl" 240 validation | tee "${base}/validation_check.json"

    "${python_bin}" -m tools.sota.select_validation \
        --predictions "${base}/validation/predictions.jsonl" \
        --output "${base}/selection.json" --model opendoc_0_1b \
        --checkpoint "${model_root}" >"${base}/selection.log" 2>&1

    "${python_bin}" -m tools.sota.run_selection_locked_test \
        --selection "${base}/selection.json" --model opendoc_0_1b \
        --model-root "${model_root}" --test-manifest "${dataset_root}/test/manifest.jsonl" \
        --image-root "${dataset_root}/test" --output-dir "${base}/test" \
        --device cpu --dtype fp32 --allow-formal-test >"${base}/test.log" 2>&1
    validate_predictions "${base}/test/predictions.jsonl" 800 test | tee "${base}/test_check.json"

    "${python_bin}" -m tools.sota.summarize_metrics \
        --manifest "${dataset_root}/test/manifest.jsonl" \
        --predictions "${base}/test/predictions.jsonl" \
        --output "${base}/test/unified_metrics.json" --expected-pages 800 \
        >"${base}/metrics.log" 2>&1
    printf '%s\n' '{"event":"opendoc_formal_completed","selection":"validation_only","test":"selection_locked","test_used_for_selection":false,"device":"cpu","provider":"official_openocr_onnx"}' | tee "${base}/finished.json"
}

if (( session_inner == 1 )); then
    run_pipeline
    exit
fi

tmux has-session -t "${session}" 2>/dev/null && { printf 'ERROR: tmux session exists.\n' >&2; exit 73; }
[[ ! -e "${run_root}" ]] || { printf 'ERROR: run directory exists: %s\n' "${run_root}" >&2; exit 73; }
mkdir -p "${run_root}"
script_path="$(realpath "${BASH_SOURCE[0]}")"
tmux new-session -d -s "${session}" \
    "cd '${ocrmodel_root}' && exec bash '${script_path}' --session-inner --session '${session}' --run-id '${run_id}' >'${run_root}/launcher.log' 2>&1"
sleep 5
tmux has-session -t "${session}" 2>/dev/null || {
    [[ ! -f "${run_root}/launcher.log" ]] || tail -n 20 "${run_root}/launcher.log" >&2
    exit 1
}
printf '{"event":"opendoc_formal_started","session":"%s","run_id":"%s","device":"cpu","gpu_queried":false,"run_root":"%s"}\n' \
    "${session}" "${run_id}" "${run_root}"
