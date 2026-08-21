#!/usr/bin/env bash
#SBATCH --job-name=got2_page_proto
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/got2_page_proto_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/got2_page_proto_%j.log

set -euo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
env_dir="${BSCC_GOT_ENV:-${workspace}/envs/got2-py310-cu118}"
model_dir="${BSCC_GOT_MODEL:-${workspace}/models/GOT-OCR2-original}"
image_path="${BSCC_SAMPLE_IMAGE:-${workspace}/samples/MTH1000__009.jpg}"
run_dir="${workspace}/runs/got2_page_proto_${SLURM_JOB_ID}"

mkdir -p "${run_dir}" "${workspace}/tmp" "${workspace}/cache/huggingface"
export TMPDIR="${workspace}/tmp"
export HF_HOME="${workspace}/cache/huggingface"
export TORCH_EXTENSIONS_DIR="${workspace}/cache/torch_extensions"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

source /home/bingxing2/apps/package/pytorch/2.0.1+cu118_cp310/env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

python -c 'import json, torch; print(json.dumps({"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, sort_keys=True))' > "${run_dir}/cuda.json"

cat > "${run_dir}/metadata.txt" <<EOF
job_id=${SLURM_JOB_ID}
host=$(hostname)
model=${model_dir}
image=${image_path}
input_granularity=whole_page_image
prompt=OCR:
EOF

cd "${workspace}/ocrmodel/src/GOT-OCR-2.0"
python GOT/demo/run_ocr_2.0.py \
    --model-name "${model_dir}" \
    --image-file "${image_path}" \
    --type ocr \
    > "${run_dir}/inference.log" 2>&1

touch "${run_dir}/GOT2_PROTOTYPE_FINISHED"
printf '{"event":"got2_prototype_completed","job_id":"%s","run_dir":"%s"}\n' \
    "${SLURM_JOB_ID}" "${run_dir}"
