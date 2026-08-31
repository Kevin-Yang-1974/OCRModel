#!/usr/bin/env bash
# BSCC/Slurm launcher for the 100,000-step synthetic-data Legacy P1 run.
#SBATCH --job-name=bscc_synth_p1_100k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=240:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -euo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
env_dir="${BSCC_GOT_ENV:-${workspace}/envs/got2-py310-cu118}"
dataset_root="${BSCC_SYNTHETIC_ROOT:-${workspace}/datasets/got_layout_pages/ancient_photo_diverse_formal_s3s4_dense_20260827_v4}"
mthv2_root="${BSCC_MTHV2_PAGE_ROOT:-${workspace}/datasets/MTHv2/converted/mthv2_layout_page_v1}"
source_model="${BSCC_GOT_MODEL:-${workspace}/models/GOT-OCR2-original}"
run_prefix="${BSCC_SYNTHETIC_P1_RUN_PREFIX:-bscc_synth_p1_legacy_100000_20260829_v1}"
run_root="${workspace}/training_runs/GOT/${run_prefix}"
runner="${workspace}/ocrmodel/tools/training/run_variable_layout_a100.py"
project_root="${workspace}/ocrmodel/src/GOT-OCR-2.0"

for p in \
    "${dataset_root}/train/manifest.jsonl" \
    "${dataset_root}/validation/manifest.jsonl" \
    "${dataset_root}/test/manifest.jsonl" \
    "${mthv2_root}/train/manifest.jsonl" \
    "${source_model}/model.safetensors" \
    "${source_model}/config.json"; do
    [[ -f "${p}" ]] || { printf 'ERROR: missing required path: %s\n' "${p}" >&2; exit 66; }
done
[[ ! -e "${run_root}" ]] || { printf 'ERROR: output run already exists: %s\n' "${run_root}" >&2; exit 74; }

gpu_ids="${CUDA_VISIBLE_DEVICES:?Slurm did not assign CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a assigned_gpus <<< "${gpu_ids}"
[[ "${#assigned_gpus[@]}" -eq 4 ]] || { printf 'ERROR: BSCC P1 requires four assigned GPUs.\n' >&2; exit 75; }

mkdir -p "${workspace}/training_runs/GOT" "${workspace}/runs" "${workspace}/tmp" \
    "${workspace}/cache/huggingface" "${workspace}/cache/torch_extensions"
export TMPDIR="${workspace}/tmp"
export HF_HOME="${workspace}/cache/huggingface"
export TORCH_EXTENSIONS_DIR="${workspace}/cache/torch_extensions"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source /home/bingxing2/apps/package/pytorch/2.0.1+cu118_cp310/env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

validation_page_count="$(wc -l < "${dataset_root}/validation/manifest.jsonl")"
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export NCCL_P2P_DISABLE=1

python "${runner}" \
    --dataset-root "${dataset_root}/train" \
    --manifest "${dataset_root}/train/manifest.jsonl" \
    --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
    --validation-image-root "${dataset_root}/validation" \
    --test-manifest "${dataset_root}/test/manifest.jsonl" \
    --source-model "${source_model}" --tokenizer-model "${source_model}" \
    --stages p1 --ablation vlqa_layout_p1_p2 --layout-loss-preset layout_full \
    --layout-memory-resolution 64 --num-layout-prompt-queries 32 \
    --max-layout-records 512 --max-layout-tokens 2048 \
    --layout-decoder-layers 2 --layout-decoder-hidden-size 256 --layout-decoder-num-heads 8 \
    --layout-boundary-loss-weight 0 --layout-count-condition-strength 0 \
    --p1-max-steps 100000 --p1-checkpoint-steps 25000 \
    --p1-candidate-steps 25000,50000,75000,100000 --checkpoint-retention 5 \
    --replay-manifest "${mthv2_root}/train/manifest.jsonl" \
    --replay-image-root "${mthv2_root}/train" \
    --p1-learning-rate 1e-4 --p1-vision-learning-rate 1e-6 \
    --p1-projector-learning-rate 1e-5 --p1-layout-learning-rate 1e-4 \
    --p1-qwen-learning-rate 0 --p1-gate-learning-rate 0 --p1-lm-head-learning-rate 0 \
    --per-device-batch-size 1 --gradient-accumulation-steps 1 \
    --gpu-ids "${gpu_ids}" --gpu-utilization-limit 50 \
    --distributed-strategy deepspeed_zero2 --nccl-p2p-disable --seed 42 \
    --protocol-version bscc_synthetic_p1_100k_v1 --variant bscc_synthetic_legacy_pvld_p1 \
    --validation-page-count "${validation_page_count}" \
    --runs-root "${workspace}/training_runs/GOT" --project-root "${project_root}" \
    --run-id "${run_prefix}" \
    > "${workspace}/runs/${run_prefix}.train.log" 2>&1

printf '{"event":"bscc_synthetic_p1_completed","run_prefix":"%s","run_root":"%s","selection":"%s/p1/validation_selection/selection.json","p1_steps":100000,"candidate_steps":[25000,50000,75000,100000],"protocol_version":"bscc_synthetic_p1_100k_v1","variant":"bscc_synthetic_legacy_pvld_p1","validation_page_count":%s,"test_used_for_selection":false}\n' \
    "${run_prefix}" "${run_root}" "${run_root}" "${validation_page_count}"
