#!/usr/bin/env bash
# Create the x86_64 CUDA 12.8 environment used by the standalone GLMOCR A100 run.
set -Eeuo pipefail

remote_root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
env_dir="${GLMOCR_A100_ENV:-${remote_root}/envs/glmocr_a100_py311_cu128}"
micromamba="${GLMOCR_MICROMAMBA:-/data3/yky/yangky_ocr_models/.tools/micromamba}"
mamba_root="${GLMOCR_MAMBA_ROOT:-/data3/yky/yangky_ocr_models/.micromamba}"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"

[[ -x "${micromamba}" ]] || {
    printf '{"event":"glmocr_a100_env_failed","error":"missing_micromamba","path":"%s"}\n' "${micromamba}" >&2
    exit 66
}

if [[ ! -e "${env_dir}" ]]; then
    mkdir -p "$(dirname "${env_dir}")"
    MAMBA_ROOT_PREFIX="${mamba_root}" "${micromamba}" create -y \
        -p "${env_dir}" -c conda-forge python=3.11 pip
elif [[ ! -x "${env_dir}/bin/python" ]]; then
    printf '{"event":"glmocr_a100_env_failed","error":"incomplete_existing_environment","path":"%s"}\n' "${env_dir}" >&2
    exit 74
fi

python="${env_dir}/bin/python"
"${python}" -m pip install --upgrade pip
# The host already provides CUDA 12.8 runtime and cuBLAS.  Installing the
# PyTorch wheel without its pip-level NVIDIA bundles avoids duplicating the
# host toolchain (and avoids pulling a second multi-hundred-MB cuBLAS wheel).
if [[ ! -f "${env_dir}/lib/python3.11/site-packages/torch/version.py" ]]; then
    "${python}" -m pip install --no-cache-dir --no-deps \
        --index-url https://download.pytorch.org/whl/cu128 \
        'torch==2.8.0+cu128' 'torchvision==0.23.0+cu128'
fi
"${python}" -m pip install --no-cache-dir \
    'filelock' 'typing-extensions>=4.10' 'sympy>=1.13.3' 'networkx' \
    'jinja2' 'fsspec' 'numpy' \
    --index-url https://pypi.org/simple
"${python}" -m pip install --no-cache-dir \
    'transformers==5.3.0' 'pillow>=11' 'safetensors>=0.4' \
    'sentencepiece>=0.2'

# The A100 image already carries compatible CUDA 12.x libraries from the
# maintained AnandaSky environment.  Only NCCL is added to the private env,
# because torch 2.8 requires a newer NCCL symbol than that older environment.
if [[ ! -d "${env_dir}/lib/python3.11/site-packages/nvidia/nccl/lib" ]]; then
    "${python}" -m pip install --no-cache-dir --no-deps \
        --index-url https://pypi.org/simple 'nvidia-nccl-cu12==2.27.3'
fi

torch_lib="${env_dir}/lib/python3.11/site-packages/torch/lib"
cusparse_lt="$(find "${nvidia_env}/lib/python3.11/site-packages/torch/lib" \
    -maxdepth 1 -type f -name 'libcusparseLt-*.so.0' -print -quit 2>/dev/null || true)"
if [[ -n "${cusparse_lt}" && ! -e "${torch_lib}/libcusparseLt.so.0" ]]; then
    ln -s "${cusparse_lt}" "${torch_lib}/libcusparseLt.so.0"
fi

cuda_library_path="/usr/local/cuda/targets/x86_64-linux/lib:${torch_lib}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    component_lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    if [[ -d "${component_lib}" ]]; then
        cuda_library_path="${cuda_library_path}:${component_lib}"
    fi
done
export LD_LIBRARY_PATH="${cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

printf '%s\n' \
    'import torch' \
    'import transformers' \
    'import safetensors' \
    'assert torch.__version__.startswith("2.8.0")' \
    'assert transformers.__version__ == "5.3.0"' \
    'assert torch.version.cuda == "12.8"' \
    'assert torch.cuda.is_available()' \
    'print({"python": __import__("sys").version.split()[0], "torch": torch.__version__, "torch_cuda": torch.version.cuda, "transformers": transformers.__version__, "cuda_available": torch.cuda.is_available()})' \
    | "${python}" -

printf '{"event":"glmocr_a100_env_ready","env":"%s","torch":"2.8.0+cu128","transformers":"5.3.0"}\n' "${env_dir}"
