#!/usr/bin/env bash
set -euo pipefail

umask 027

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
workspace="${OCR_WORKSPACE:-$(cd -- "${ocrmodel_root}/.." && pwd -P)}"
env_bin="${workspace}/envs/anandasky/bin"
python="${env_bin}/python"
expected_version="2.7.4.post1"
default_wheel="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
wheel_url="${FLASH_ATTN_WHEEL_URL:-${default_wheel}}"
build_from_source="${FLASH_ATTN_BUILD_FROM_SOURCE:-0}"
cuda_home="${CUDA_HOME:-/usr/local/cuda-12.8}"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ -x "${python}" ]] || die "Missing AnandaSky environment: ${python}"
export PATH="${env_bin}:${PATH}"

"${python}" - <<'PY'
import platform
import sys

import torch

if platform.machine() != "x86_64":
    raise SystemExit("The pinned FlashAttention wheel requires x86_64.")
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Expected Python 3.11, found {platform.python_version()}.")
if not torch.__version__.startswith("2.5.") or torch.version.cuda is None:
    raise SystemExit(f"Expected PyTorch 2.5 with CUDA, found {torch.__version__}.")
print(f"flash_attn_target torch={torch.__version__} cuda={torch.version.cuda}")
PY

capability="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1)"
[[ "${capability%%.*}" =~ ^[0-9]+$ ]] || die "Cannot determine GPU compute capability."
(( ${capability%%.*} >= 8 )) || die "FlashAttention requires Ampere (SM80) or newer."

current_version="$(${python} - <<'PY'
try:
    import flash_attn
except ImportError:
    print("missing")
else:
    print(flash_attn.__version__)
PY
)"

if [[ "${current_version}" != "${expected_version}" ]]; then
    if [[ "${build_from_source}" == "1" ]]; then
        [[ -x "${cuda_home}/bin/nvcc" ]] || die "Missing nvcc: ${cuda_home}/bin/nvcc"
        "${python}" -m pip install --no-deps "ninja==1.13.0"
        CUDA_HOME="${cuda_home}" \
        MAX_JOBS="${MAX_JOBS:-8}" \
        NVCC_THREADS="${NVCC_THREADS:-4}" \
        FLASH_ATTENTION_FORCE_BUILD=TRUE \
            "${python}" -m pip install --no-deps --no-build-isolation \
            "flash-attn==${expected_version}"
    else
        "${python}" -m pip install --no-deps "${wheel_url}"
    fi
fi

"${python}" - <<PY
import flash_attn
import torch

assert flash_attn.__version__ == "${expected_version}", flash_attn.__version__
assert callable(flash_attn.flash_attn_varlen_func)
print(f"FLASH_ATTN_OK version={flash_attn.__version__} torch={torch.__version__} cuda={torch.version.cuda}")
PY

"${python}" -m pip check
