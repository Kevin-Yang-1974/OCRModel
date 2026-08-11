#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
workspace="${OCR_WORKSPACE:-$(cd -- "${ocrmodel_root}/.." && pwd -P)}"
env_dir="${workspace}/envs/got2"
python_lib="${env_dir}/lib/python3.10/site-packages"

if [[ ! -x "${env_dir}/bin/python" ]]; then
    printf 'ERROR: GOT environment is missing: %s\n' "${env_dir}" >&2
    exit 66
fi

export OCR_WORKSPACE="${workspace}"
export OCRMODEL_ROOT="${ocrmodel_root}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${workspace}/cache/huggingface}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="${python_lib}/torch/lib:${python_lib}/nvidia/cuda_runtime/lib:${python_lib}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"

exec "${env_dir}/bin/python" "$@"
