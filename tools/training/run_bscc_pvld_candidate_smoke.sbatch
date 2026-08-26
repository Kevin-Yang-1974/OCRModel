#!/usr/bin/env bash
# Bounded BSCC candidate smoke; never a formal training or evaluation job.
#SBATCH --job-name=pvld_candidate_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -euo pipefail
workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
env_dir="${BSCC_GOT_ENV:-${workspace}/envs/got2-py310-cu118}"
run_id="${BSCC_RUN_ID:-pvld_m2_m3_m4_smoke_${SLURM_JOB_ID}}"
run_dir="${workspace}/runs/${run_id}"
mkdir -p "${run_dir}" "${workspace}/tmp" "${workspace}/cache/huggingface" "${workspace}/cache/torch_extensions"
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
project_root="${workspace}/ocrmodel/src/GOT-OCR-2.0"
script="${workspace}/ocrmodel/tools/training/smoke_pvld_causal_decoder_cuda.py"
summary="${run_dir}/summary.json"
printf 'job_id=%s\nhost=%s\ncuda_visible_devices=%s\n' "${SLURM_JOB_ID}" "$(hostname)" "${CUDA_VISIBLE_DEVICES:-}" > "${run_dir}/metadata.txt"
python "${script}" --project-root "${project_root}" --output "${summary}" --m2 --m3-scale 0.25 --m4 > "${run_dir}/smoke.log" 2>&1
touch "${run_dir}/BSCC_SMOKE_FINISHED"
printf '{"status":"ok","job_id":"%s","run_dir":"%s","test_read":false}\n' "${SLURM_JOB_ID}" "${run_dir}"
