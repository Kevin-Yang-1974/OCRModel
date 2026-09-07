#!/usr/bin/env bash
#SBATCH --job-name=glmocr_setup
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -euo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/glm_ocr_layout_ot"
code_root="${experiment_root}/code/ocrmodel"
env_dir="${experiment_root}/envs/glm-ocr-py311-cu128"
model_dir="${workspace}/models/GLM-OCR-ca5d8b3"
protocol_dir="${experiment_root}/protocols/r1_r2_128_seed42_v1"
dataset_root="${BSCC_MTHV2_PAGE_ROOT:-${workspace}/datasets/MTHv2/converted/mthv2_layout_page_v1_real}"
pytorch_package="/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311"
model_revision="ca5d8b3e287e52589e37c28385d9655ee4372f9d"

[[ -f "${code_root}/pyproject.toml" ]] || { printf 'ERROR: synced GLM-OCR code is missing.\n' >&2; exit 66; }
[[ -f "${dataset_root}/train/manifest.jsonl" ]] || { printf 'ERROR: MTHv2 source manifests are missing.\n' >&2; exit 66; }

mkdir -p "${experiment_root}/envs" "${experiment_root}/protocols" "${workspace}/models" \
    "${workspace}/runs" "${experiment_root}/tmp"
export TMPDIR="${experiment_root}/tmp"
export HF_HOME="${experiment_root}/cache/huggingface"
export PIP_CACHE_DIR="${experiment_root}/cache/pip"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1

source "${pytorch_package}/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
if [[ ! -x "${env_dir}/bin/python" ]]; then
    conda create -y --prefix "${env_dir}" python=3.11 pip
fi
set +u
conda activate "${env_dir}"
set -u

python -m pip install --no-cache-dir --no-deps \
    "${pytorch_package}/torch-2.8.0+cu128-cp311-cp311-linux_aarch64.whl" \
    "${pytorch_package}/torchvision-0.23.0+cu128-cp311-cp311-linux_aarch64.whl"
python -m pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ \
    'transformers>=5.3,<5.4' 'accelerate>=1.10' 'pillow>=11,<13' 'safetensors>=0.4,<1' \
    'sentencepiece>=0.2,<1' 'pytest>=8,<10' 'numpy>=1.26,<3' packaging psutil
python -m pip install --no-cache-dir --no-deps -e "${code_root}"

# Compute nodes are intentionally offline: the pinned model is staged by the
# sync workflow.  Fail early on an incomplete or wrong revision instead of
# attempting an unreachable Hugging Face download.
expected_model_size=2650579464
expected_model_sha256=a16eb0de98d199293371c560f95f83130d2a2c9612449df16839f08ff9498815
[[ -f "${model_dir}/model.safetensors" ]] || { echo "missing staged GLM-OCR weights" >&2; exit 2; }
[[ "$(stat -c %s "${model_dir}/model.safetensors")" == "${expected_model_size}" ]] || {
    echo "GLM-OCR weight size mismatch" >&2; exit 2;
}
actual_model_sha256="$(sha256sum "${model_dir}/model.safetensors" | awk '{print $1}')"
[[ "${actual_model_sha256}" == "${expected_model_sha256}" ]] || {
    echo "GLM-OCR weight SHA256 mismatch" >&2; exit 2;
}
for required_file in config.json generation_config.json preprocessor_config.json tokenizer.json tokenizer_config.json chat_template.jinja; do
    [[ -f "${model_dir}/${required_file}" ]] || { echo "missing staged model file: ${required_file}" >&2; exit 2; }
done

if [[ ! -f "${protocol_dir}/protocol.json" ]]; then
    python "${code_root}/tools/prepare_bscc_protocol.py" \
        --source-root "${dataset_root}" \
        --output-root "${protocol_dir}" \
        --train-pages 128 --validation-pages 64 --test-pages 64 \
        --max-regions 32 --seed 42 --dhash-threshold 4
fi

python - <<'PY'
import json, torch, transformers
print(json.dumps({
    "event":"glm_ocr_setup_complete",
    "torch":torch.__version__,
    "cuda":torch.version.cuda,
    "transformers":transformers.__version__,
}, separators=(",",":")))
PY
