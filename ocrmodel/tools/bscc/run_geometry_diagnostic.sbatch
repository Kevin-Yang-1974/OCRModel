#!/usr/bin/env bash
#SBATCH --job-name=glmocr_geom_diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -euo pipefail

# First-pass diagnosis only: geometry, seed 42, and validation points 0/64/128.
# This launcher deliberately does not load the held-out split and is not part
# of the existing multi-seed mechanism screen.
workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/glm_ocr_layout_ot"
code_root="${experiment_root}/code/ocrmodel"
env_dir="${experiment_root}/envs/glm-ocr-py311-cu128"
model_dir="${workspace}/models/GLM-OCR-ca5d8b3"
protocol_dir="${experiment_root}/protocols/r1_r2_128_seed42_v1"
screen_id="${GLM_OCR_DIAGNOSTIC_ID:-geometry_seed42_steps128_v1}"
adapter_precision="${GLM_OCR_ADAPTER_PRECISION:-mixed_bf16}"
layout_loss_profile="${GLM_OCR_LAYOUT_LOSS_PROFILE:-full}"
query_assignment="${GLM_OCR_QUERY_ASSIGNMENT:-fixed_order}"
processor_mode="${GLM_OCR_PROCESSOR_MODE:-fast}"
max_steps="${GLM_OCR_MAX_STEPS:-128}"
lr_schedule_steps="${GLM_OCR_LR_SCHEDULE_STEPS:-}"
lr_schedule_steps_json="${GLM_OCR_LR_SCHEDULE_STEPS:-null}"
diagnostic_steps="${GLM_OCR_DIAGNOSTIC_STEPS:-0,64,128}"
diagnostic_steps_json="${GLM_OCR_DIAGNOSTIC_STEPS_JSON:-[0,64,128]}"
validation_interval="${GLM_OCR_VALIDATION_INTERVAL:-64}"
run_root="${experiment_root}/training_runs/${screen_id}"
output_dir="${run_root}/seed42/geometry_aux0.2"
pytorch_package="/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311"

[[ "${processor_mode}" == "fast" || "${processor_mode}" == "slow" ]] || {
    printf 'ERROR: GLM_OCR_PROCESSOR_MODE must be fast or slow.\n' >&2; exit 64;
}

if [[ "${GLM_OCR_PRINT_TASK_MAP:-0}" == "1" ]]; then
    printf '{"mode":"geometry","seed":42,"auxiliary_weight":0.2,"max_steps":%s,"lr_schedule_steps":%s,"diagnostic_steps":%s,"adapter_precision":"%s","layout_loss_profile":"%s","query_assignment":"%s","processor_mode":"%s","output_dir":"%s","reads_test":false}\n' \
        "${max_steps}" "${lr_schedule_steps_json}" "${diagnostic_steps_json}" "${adapter_precision}" "${layout_loss_profile}" \
        "${query_assignment}" "${processor_mode}" "${output_dir}"
    exit 0
fi

[[ -x "${env_dir}/bin/python" && -f "${model_dir}/model.safetensors" ]] || {
    printf 'ERROR: GLM-OCR setup is incomplete.\n' >&2; exit 66;
}
[[ -f "${protocol_dir}/protocol.json" && -f "${protocol_dir}/train.jsonl" \
    && -f "${protocol_dir}/validation.jsonl" && -f "${protocol_dir}/test.jsonl" ]] || {
    printf 'ERROR: locked train/validation protocol is missing.\n' >&2; exit 66;
}
[[ ! -e "${output_dir}" ]] || {
    printf 'ERROR: run already exists: %s\n' "${output_dir}" >&2; exit 74;
}

mkdir -p "${experiment_root}/training_runs" "${experiment_root}/tmp" "${workspace}/runs"
export TMPDIR="${experiment_root}/tmp"
export HF_HOME="${experiment_root}/cache/huggingface"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
source "${pytorch_package}/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

cd "${code_root}"
python -m layout_ocr.train_screen \
    --mode geometry \
    --model-path "${model_dir}" \
    --train-manifest "${protocol_dir}/train.jsonl" \
    --validation-manifest "${protocol_dir}/validation.jsonl" \
    --protocol-file "${protocol_dir}/protocol.json" \
    --output-dir "${output_dir}" \
    --max-steps "${max_steps}" --num-queries 32 --seed 42 \
    --learning-rate 5e-5 --warmup-steps 64 --min-lr-ratio 0.1 \
    --lr-schedule-steps "${lr_schedule_steps:-none}" \
    --residual-scale-cap 0.03 --auxiliary-weight 0.2 --max-grad-norm 1.0 \
    --max-pixels 1003520 --max-eval-new-tokens 768 \
    --validation-interval "${validation_interval}" --log-steps 16 --diagnostic-steps "${diagnostic_steps}" \
    --adapter-precision "${adapter_precision}" \
    --layout-loss-profile "${layout_loss_profile}" \
    --query-assignment "${query_assignment}" \
    --processor-mode "${processor_mode}"
