#!/usr/bin/env bash
set -euo pipefail

umask 027

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

workspace="${OCR_WORKSPACE:-$(cd -- "${ocrmodel_root}/.." && pwd -P)}"
env_dir="${workspace}/envs/layout-synthesis"
python_bin="${env_dir}/bin/python"
lock_file="${script_dir}/requirements-layout-synthesis.lock.txt"
micromamba="${workspace}/.tools/micromamba"

export CUDA_VISIBLE_DEVICES=""
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="${workspace}/cache/pip"
export PLAYWRIGHT_BROWSERS_PATH="${workspace}/cache/ms-playwright"
export MAMBA_ROOT_PREFIX="${workspace}/.micromamba"

[[ -n "${workspace}" && "${workspace}" != "/" ]] || {
    printf '%s\n' '{"event":"layout_synthesis_environment_failed","error":"unsafe workspace"}' >&2
    exit 64
}
[[ -f "${lock_file}" ]] || {
    printf '{"event":"layout_synthesis_environment_failed","error":"missing lock file","path":"%s"}\n' "${lock_file}" >&2
    exit 66
}

mkdir -p "${workspace}/envs" "${PIP_CACHE_DIR}" "${PLAYWRIGHT_BROWSERS_PATH}"
if [[ ! -x "${python_bin}" ]]; then
    [[ ! -e "${env_dir}" ]] || {
        printf '{"event":"layout_synthesis_environment_failed","error":"incomplete environment exists","path":"%s"}\n' "${env_dir}" >&2
        exit 73
    }
    if [[ -x "${micromamba}" ]]; then
        "${micromamba}" create -y -p "${env_dir}" -c conda-forge python=3.11 pip
    elif command -v python3 >/dev/null 2>&1 && python3 -m venv --help >/dev/null 2>&1; then
        python3 -m venv "${env_dir}"
    else
        printf '%s\n' '{"event":"layout_synthesis_environment_failed","error":"no micromamba or usable python3 venv"}' >&2
        exit 69
    fi
fi

"${python_bin}" -m pip install --disable-pip-version-check -r "${lock_file}"
"${python_bin}" -m playwright install chromium
"${python_bin}" -m pip check
"${python_bin}" -c 'import numpy, PIL, fitz, playwright'

"${python_bin}" - "${env_dir}" "${PLAYWRIGHT_BROWSERS_PATH}" <<'PY'
from __future__ import annotations

import json
import sys

import PIL
import fitz
import numpy
import playwright

print(json.dumps({
    "event": "layout_synthesis_environment_ready",
    "environment": sys.argv[1],
    "browser_cache": sys.argv[2],
    "python": sys.version.split()[0],
    "numpy": numpy.__version__,
    "pillow": PIL.__version__,
    "pymupdf": fitz.VersionBind,
    "playwright": getattr(playwright, "__version__", "1.62.0"),
    "cuda_visible_devices": "",
}, ensure_ascii=False, separators=(",", ":")))
PY
