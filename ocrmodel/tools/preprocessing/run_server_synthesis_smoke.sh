#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PLAYWRIGHT_BROWSERS_PATH="${OCR_WORKSPACE:-$(cd -- "${ocrmodel_root}/.." && pwd -P)}/cache/ms-playwright"

python_bin=""
python_candidates=(
    "${OCR_WORKSPACE:-}/envs/layout-synthesis/bin/python"
    "${OCR_WORKSPACE:-}/envs/got2/bin/python"
)
if command -v python3 >/dev/null 2>&1; then
    python_candidates+=("$(command -v python3)")
fi
for candidate in "${python_candidates[@]}"; do
    if [[ -x "${candidate}" ]] && "${candidate}" -c 'import numpy, PIL, playwright' >/dev/null 2>&1; then
        python_bin="${candidate}"
        break
    fi
done
if [[ -z "${python_bin}" ]]; then
    printf '%s\n' '{"event":"server_synthesis_smoke_failed","stage":"environment","error":"No existing Python environment imports numpy, Pillow, and playwright."}' >&2
    exit 66
fi

browser=""
for name in chromium chromium-browser google-chrome google-chrome-stable; do
    if command -v "${name}" >/dev/null 2>&1; then
        browser="$(command -v "${name}")"
        break
    fi
done

stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${GOT_LAYOUT_DATA}/server_synthesis_smoke_${stamp}"
log_root="${GOT_LAYOUT_DATA}/_server_synthesis_smoke_logs"
log_path="${log_root}/${stamp}.log"
content_manifest="${SYNTHESIS_SMOKE_CONTENT_MANIFEST:-${ocrmodel_root}/config/synthetic_content.example.jsonl}"
content_root="${SYNTHESIS_SMOKE_CONTENT_ROOT:-}"
smoke_num_pages="${SYNTHESIS_SMOKE_NUM_PAGES:-1}"
[[ "${smoke_num_pages}" =~ ^[1-9][0-9]*$ ]] || {
    printf '%s\n' '{"event":"server_synthesis_smoke_failed","stage":"preflight","error":"SYNTHESIS_SMOKE_NUM_PAGES must be positive"}' >&2
    exit 64
}
mkdir -p "${log_root}"
if [[ -e "${output_dir}" ]]; then
    printf '{"event":"server_synthesis_smoke_failed","stage":"preflight","error":"output exists","output":"%s"}\n' "${output_dir}" >&2
    exit 74
fi

generator_args=(
    --content-manifest "${content_manifest}"
    --output-dir "${output_dir}"
    --split train
    --tier s0-html-text
    --num-pages "${smoke_num_pages}"
    --seed 20260817
    --config "${ocrmodel_root}/config/synthetic_layout.ancient_photo_diverse_v1.json"
    --progress-every 1
)
if [[ -n "${content_root}" ]]; then
    generator_args+=(--content-root "${content_root}")
fi
if [[ -n "${browser}" ]]; then
    generator_args+=(--chromium-executable "${browser}")
fi

if ! timeout 600 "${python_bin}" "${script_dir}/generate_synthetic_layout.py" \
    "${generator_args[@]}" >"${log_path}" 2>&1; then
    tail -n 20 "${log_path}" >&2
    printf '{"event":"server_synthesis_smoke_failed","stage":"render","output":"%s","log":"%s"}\n' \
        "${output_dir}" "${log_path}" >&2
    exit 1
fi
if ! "${python_bin}" "${script_dir}/audit_synthetic_layout.py" \
    --manifest "${output_dir}/manifest.jsonl" \
    --summary-json "${output_dir}/audit_summary.json" >>"${log_path}" 2>&1; then
    tail -n 20 "${log_path}" >&2
    printf '{"event":"server_synthesis_smoke_failed","stage":"audit","output":"%s","log":"%s"}\n' \
        "${output_dir}" "${log_path}" >&2
    exit 1
fi

"${python_bin}" - "${output_dir}" "${python_bin}" "${browser}" "${log_path}" "${content_manifest}" <<'PY'
from __future__ import annotations
import json, sys
from pathlib import Path

output, python_bin, browser, log_path, content_manifest = sys.argv[1:6]
root = Path(output)
metadata = json.loads((root / "dataset_meta.json").read_text(encoding="utf-8"))
audit = json.loads((root / "audit_summary.json").read_text(encoding="utf-8"))
print(json.dumps({
    "event": "server_synthesis_smoke_completed",
    "output": str(root),
    "python": python_bin,
    "browser_executable": browser or None,
    "chromium_version": metadata.get("chromium_version"),
    "pages": metadata.get("num_pages"),
    "audit_status": audit.get("status"),
    "input_level": "whole_page_image",
    "content_manifest": content_manifest,
    "cuda_visible_devices": "",
    "log": log_path,
}, ensure_ascii=False, separators=(",", ":")))
PY
