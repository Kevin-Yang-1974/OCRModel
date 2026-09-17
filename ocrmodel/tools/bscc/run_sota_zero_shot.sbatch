#!/usr/bin/env bash
#SBATCH --job-name=sota_zeroshot
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_sota_%A_%a.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_sota_%A_%a.log
#
# Zero-shot inference for one external SOTA model on one merged-benchmark split/shard.
#
# Required env:
#   SOTA_RUN_ID     evaluation run id (creates evaluation_runs/SOTA/<id>)
#   SOTA_MODEL      paddleocr_vl_1_6 | mineru2_5_pro | opendoc_0_1b
#   SOTA_SPLIT      validation | test
# Optional env:
#   SOTA_SHARD_COUNT (default 1), SOTA_LIMIT, SOTA_DEVICE, SOTA_DTYPE,
#   SOTA_MODEL_ROOT, SOTA_ENV_DIR, SOTA_ALLOW_TEST (test only),
#   SOTA_MAX_OUTPUT_TOKENS (default 1536)
set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/glm_ocr_layout_ot"
code_root="${experiment_root}/code/ocrmodel"
models_root="${SOTA_MODELS_ROOT:-${workspace}/models/sota}"
dataset_root="${BSCC_DATASET_ROOT:-${workspace}/datasets/dunhuang_local_gazetteer_q32_v1_portable}"
pytorch_package="/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311"

run_id="${SOTA_RUN_ID:?SOTA_RUN_ID is required}"
model="${SOTA_MODEL:?SOTA_MODEL is required}"
split="${SOTA_SPLIT:?SOTA_SPLIT is required}"
shard_count="${SOTA_SHARD_COUNT:-1}"
shard_index="${SLURM_ARRAY_TASK_ID:-0}"
limit="${SOTA_LIMIT:-}"
device="${SOTA_DEVICE:-cuda}"
dtype="${SOTA_DTYPE:-bf16}"
allow_test="${SOTA_ALLOW_TEST:-0}"
max_output_tokens="${SOTA_MAX_OUTPUT_TOKENS:-1536}"

[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid SOTA_RUN_ID" >&2; exit 64; }
[[ "${split}" == "validation" || "${split}" == "test" ]] || { echo "invalid SOTA_SPLIT" >&2; exit 64; }
if [[ "${split}" == "test" && "${allow_test}" != "1" ]]; then
    echo "benchmark test is locked; set SOTA_ALLOW_TEST=1 for an explicit zero-shot test." >&2
    exit 77
fi
if (( shard_index >= shard_count )); then
    echo "shard index out of range" >&2
    exit 64
fi
[[ "${max_output_tokens}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid SOTA_MAX_OUTPUT_TOKENS" >&2; exit 64; }

declare -A model_roots=(
    [paddleocr_vl_1_6]="${models_root}/paddleocr_vl_1_6/c5630abae1d940eafe0697512a0325494b02ab42"
    [mineru2_5_pro]="${models_root}/mineru2_5_pro/bff20d4ae2bf202df9f45284b4d43681555a97ed"
    [opendoc_0_1b]="${models_root}/opendoc_onnx_20260912"
)
declare -A env_dirs=(
    [paddleocr_vl_1_6]="${experiment_root}/envs/sota-transformers"
    [mineru2_5_pro]="${experiment_root}/envs/sota-transformers"
    [opendoc_0_1b]="${experiment_root}/envs/sota-opendoc"
)
model_root="${SOTA_MODEL_ROOT:-${model_roots[${model}]:-}}"
env_dir="${SOTA_ENV_DIR:-${env_dirs[${model}]:-}}"
[[ -n "${model_root}" && -d "${model_root}" ]] || { echo "missing model root for ${model}" >&2; exit 66; }
[[ -x "${env_dir}/bin/python" ]] || { echo "missing env for ${model}: ${env_dir}" >&2; exit 66; }

run_root="${workspace}/evaluation_runs/SOTA/${run_id}/${model}/${split}"
shard_dir="${run_root}/shard-$(printf '%03d' "${shard_index}")"
[[ ! -e "${shard_dir}" ]] || { echo "shard dir already exists: ${shard_dir}" >&2; exit 74; }

mkdir -p "${run_root}" "${workspace}/runs" "${experiment_root}/tmp"
export TMPDIR="${experiment_root}/tmp/${run_id}_${model}_${split}_${shard_index}"
export HF_HOME="${experiment_root}/cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "${TMPDIR}" "${HF_HOME}"

source "${pytorch_package}/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

export PYTHONPATH="${code_root}:${PYTHONPATH:-}"

# OpenDoc needs the OpenOCR package tree and the pre-staged ONNX assets.
if [[ "${model}" == "opendoc_0_1b" ]]; then
    openocr_parent="${SOTA_OPENOCR_PARENT:-${HOME}}"
    onnx_cache="${SOTA_OPENOCR_ONNX_CACHE:-${models_root}}"
    export OPENOCR_ONNX=1
    export OPENOCR_ONNX_CACHE="${onnx_cache}"
    export OPENOCR_PYTHONPATH="${openocr_parent}"
    export PYTHONPATH="${code_root}:${openocr_parent}:${PYTHONPATH:-}"
fi

manifest="${dataset_root}/manifests/${split}.jsonl"
image_root="${dataset_root}"
[[ -d "${dataset_root}" && -f "${manifest}" ]] || { echo "missing benchmark dataset or manifest: ${manifest}" >&2; exit 66; }
limit_args=()
[[ -n "${limit}" ]] && limit_args=(--limit "${limit}")
allow_args=()
[[ "${allow_test}" == "1" ]] && allow_args=(--allow-test)

python -m tools.sota.run_zero_shot \
    --model "${model}" \
    --model-root "${model_root}" \
    --manifest "${manifest}" \
    --image-root "${image_root}" \
    --output-dir "${shard_dir}" \
    --split "${split}" \
    --shard-index "${shard_index}" \
    --shard-count "${shard_count}" \
    --device "${device}" \
    --dtype "${dtype}" \
    --max-output-tokens "${max_output_tokens}" \
    --progress-every 5 \
    "${limit_args[@]}" "${allow_args[@]}"
