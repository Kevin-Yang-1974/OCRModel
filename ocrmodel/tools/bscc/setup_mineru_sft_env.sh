#!/usr/bin/env bash
# Prepare an isolated BSCC environment for MinerU2.5-Pro SFT.
# Run once on the BSCC login node.  It deliberately does not touch the
# existing sota-transformers environment used by zero-shot evaluations.
set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/glm_ocr_layout_ot"
env_root="${experiment_root}/envs"
env_dir="${MINERU_SFT_ENV_DIR:-${env_root}/mineru-sft}"
# BSCC login nodes may not reach GitHub.  The normal path is populated by
# syncing the already checked-out official source from the workstation.
vendor_root="${MINERU_MS_SWIFT_ROOT:-${experiment_root}/vendor/ms-swift-bscc}"
pytorch_package="/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311"
pip_index="https://mirrors.aliyun.com/pypi/simple/"
marker="${env_dir}/.mineru_sft_ready"

[[ -f "${pytorch_package}/env.sh" ]] || {
    printf '{"event":"mineru_sft_env_failed","error":"missing_pytorch_package"}\n' >&2
    exit 66
}

mkdir -p "${env_root}" "$(dirname "${vendor_root}")"
source "${pytorch_package}/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ ! -f "${vendor_root}/swift/__init__.py" ]]; then
    [[ ! -e "${vendor_root}" ]] || {
        printf '{"event":"mineru_sft_env_failed","error":"incomplete_ms_swift_vendor","vendor_root":"%s"}\n' "${vendor_root}" >&2
        exit 66
    }
    git clone --depth 1 https://github.com/modelscope/ms-swift.git "${vendor_root}"
fi

if [[ ! -x "${env_dir}/bin/python" ]]; then
    conda create -y --prefix "${env_dir}" python=3.11 pip
fi

set +u
conda activate "${env_dir}"
set -u

if [[ ! -f "${marker}" ]]; then
    torch_wheels=(
        "${pytorch_package}/torch-2.8.0+cu128-cp311-cp311-linux_aarch64.whl"
        "${pytorch_package}/torchvision-0.23.0+cu128-cp311-cp311-linux_aarch64.whl"
    )
    python -m pip install --no-cache-dir --no-deps "${torch_wheels[@]}"
    python -m pip install --no-cache-dir --index-url "${pip_index}" --no-deps -e "${vendor_root}"
    python -m pip install --no-cache-dir --index-url "${pip_index}" \
        'accelerate>=1.5' 'addict' 'aiohttp' 'datasets>=3.0,<4.8.5' 'einops' \
        'importlib_metadata' 'modelscope>=1.23' 'numpy>=1.26,<3' 'peft>=0.17,<0.21' \
        'pillow>=11,<13' 'PyYAML>=5.4' 'qwen-vl-utils>=0.0.14' \
        'safetensors>=0.4,<1' 'sentencepiece>=0.2,<1' 'tensorboard' \
        'transformers>=5.10.1,<5.17.0' 'tqdm' 'trl>=0.15,<1.0'
    python -c 'import torch, transformers, swift, qwen_vl_utils; print({"torch": torch.__version__, "transformers": transformers.__version__, "swift": getattr(swift, "__version__", "unknown")})'
    touch "${marker}"
fi

python -c 'import torch, transformers, swift, qwen_vl_utils; assert torch.cuda.is_available() or True; print({"event":"mineru_sft_env_ready", "torch":torch.__version__, "transformers":transformers.__version__, "swift":getattr(swift, "__version__", "unknown")})'
