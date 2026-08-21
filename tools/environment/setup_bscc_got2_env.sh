#!/usr/bin/env bash
set -euo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
env_dir="${BSCC_GOT_ENV:-${workspace}/envs/got2-py310-cu118}"
pytorch_package="/home/bingxing2/apps/package/pytorch/2.0.1+cu118_cp310"

mkdir -p \
    "${workspace}/models" \
    "${workspace}/samples" \
    "${workspace}/cache/pip" \
    "${workspace}/cache/huggingface" \
    "${workspace}/cache/torch_extensions" \
    "${workspace}/tmp" \
    "${workspace}/runs"

export TMPDIR="${workspace}/tmp"
export PIP_CACHE_DIR="${workspace}/cache/pip"
export HF_HOME="${workspace}/cache/huggingface"
export TORCH_EXTENSIONS_DIR="${workspace}/cache/torch_extensions"

source "${pytorch_package}/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -x "${env_dir}/bin/python" ]]; then
    conda create -y --prefix "${env_dir}" python=3.10 pip
fi
set +u
conda activate "${env_dir}"
set -u

python -m pip install --no-deps \
    "${pytorch_package}/torch-2.0.1+cu118-cp310-cp310-linux_aarch64.whl" \
    "${pytorch_package}/torchvision-0.15.2+cu118-cp310-cp310-linux_aarch64.whl" \
    /home/bingxing2/apps/package/opencv_python/opencv_python-4.9.0.80-cp37-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

python -m pip install --index-url https://mirrors.aliyun.com/pypi/simple/ \
    'numpy==1.26.4' \
    'pillow<12' \
    'transformers==4.37.2' \
    'accelerate==0.28.0' \
    'tokenizers==0.15.2' \
    'sentencepiece==0.1.99' \
    'safetensors>=0.4,<1' \
    'requests==2.28.1' \
    'httpx==0.24.0' \
    'einops==0.6.1' \
    'einops-exts==0.0.4' \
    'timm==0.6.13' \
    'tiktoken==0.6.0' \
    'jinja2>=3.1,<4' \
    'sympy>=1.11,<2' \
    'networkx>=2.8,<4' \
    markdown2 \
    shortuuid

python -m pip install --no-deps -e "${workspace}/ocrmodel/src/GOT-OCR-2.0"

python -c 'import cv2, torch, torchvision, transformers; print("ENV_OK torch=%s cuda=%s torchvision=%s transformers=%s cv2=%s" % (torch.__version__, torch.version.cuda, torchvision.__version__, transformers.__version__, cv2.__version__))'
