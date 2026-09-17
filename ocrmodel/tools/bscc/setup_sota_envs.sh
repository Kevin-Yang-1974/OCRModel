#!/usr/bin/env bash
# Create the isolated BSCC environments used by the external-SOTA zero-shot runs.
#
# Two environments are created:
#   sota-transformers  torch + transformers (PaddleOCR-VL, MinerU2.5-Pro)
#   sota-opendoc       torch + onnxruntime + OpenCV (OpenDoc-0.1B via OpenOCR ONNX)
#
# Existing environments are left untouched. Run on the login node.
set -euo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/glm_ocr_layout_ot"
env_root="${experiment_root}/envs"
pytorch_package="/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311"
pip_index="https://mirrors.aliyun.com/pypi/simple/"

source "${pytorch_package}/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"

torch_wheels=(
    "${pytorch_package}/torch-2.8.0+cu128-cp311-cp311-linux_aarch64.whl"
    "${pytorch_package}/torchvision-0.23.0+cu128-cp311-cp311-linux_aarch64.whl"
)

make_env() {
    local prefix="$1"
    if [[ ! -x "${prefix}/bin/python" ]]; then
        conda create -y --prefix "${prefix}" python=3.11 pip
    fi
    set +u
    conda activate "${prefix}"
    set -u
    python -m pip install --no-cache-dir --no-deps "${torch_wheels[@]}"
}

# --- transformers env (PaddleOCR-VL + MinerU) -------------------------------
tf_env="${env_root}/sota-transformers"
if [[ ! -f "${tf_env}/.sota_ready" ]]; then
    make_env "${tf_env}"
    python -m pip install --no-cache-dir --index-url "${pip_index}" \
        'transformers>=5.10.1,<6' 'accelerate>=1.5' 'safetensors>=0.4,<1' \
        'sentencepiece>=0.2,<1' 'pillow>=11,<13' 'numpy>=1.26,<3' packaging psutil
    python -m pip install --no-cache-dir --index-url "${pip_index}" 'mineru-vl-utils>=2.0,<3'
    touch "${tf_env}/.sota_ready"
fi

# --- OpenDoc ONNX env --------------------------------------------------------
od_env="${env_root}/sota-opendoc"
if [[ ! -f "${od_env}/.sota_ready" ]]; then
    make_env "${od_env}"
    python -m pip install --no-cache-dir --index-url "${pip_index}" \
        'onnxruntime>=1.20' 'opencv-python-headless>=4.10' 'numpy>=1.26,<3' \
        'pyyaml>=6' 'pillow>=11,<13' 'tqdm>=4.67' 'rapidfuzz>=3' 'pydantic>=2' \
        safetensors packaging psutil
    touch "${od_env}/.sota_ready"
fi

printf '{"event":"sota_envs_ready","transformers_env":"%s","opendoc_env":"%s"}\n' "${tf_env}" "${od_env}"
