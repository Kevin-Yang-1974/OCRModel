#!/usr/bin/env bash
#SBATCH --job-name=glmocr_screen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
# Stabilization confirmation: six trained mode/seed runs plus one eval-only baseline.
# Each task runs once; no task is restarted under the same output directory.
#SBATCH --array=0-6
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%A_%a.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%A_%a.log

set -euo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/glm_ocr_layout_ot"
code_root="${experiment_root}/code/ocrmodel"
env_dir="${experiment_root}/envs/glm-ocr-py311-cu128"
model_dir="${workspace}/models/GLM-OCR-ca5d8b3"
protocol_dir="${experiment_root}/protocols/r1_r2_128_seed42_v1"
screen_id="${GLM_OCR_SCREEN_ID:-mechanism_stable_128_3seed_v1}"
run_root="${experiment_root}/training_runs/${screen_id}"
pytorch_package="/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311"
modes=("attention" "geometry")
seeds=(42 43 44)
task_id="${SLURM_ARRAY_TASK_ID:?Slurm array task ID is missing}"
if (( task_id < 6 )); then
    mode_index=$((task_id % ${#modes[@]}))
    seed_index=$((task_id / ${#modes[@]}))
    mode="${modes[mode_index]}"
    seed="${seeds[seed_index]}"
    auxiliary_weight="0.2"
    output_dir="${run_root}/seed${seed}/${mode}_aux${auxiliary_weight}"
    eval_only=()
else
    mode="content_only"
    seed="42"
    auxiliary_weight="0.0"
    output_dir="${run_root}/content_only_eval"
    eval_only=(--eval-only)
fi

if [[ "${GLM_OCR_PRINT_TASK_MAP:-0}" == "1" ]]; then
    printf '{"task_id":%s,"mode":"%s","seed":%s,"auxiliary_weight":%s,"output_dir":"%s","eval_only":%s}\n' \
        "${task_id}" "${mode}" "${seed}" "${auxiliary_weight}" "${output_dir}" \
        "$([[ ${#eval_only[@]} -eq 1 ]] && printf true || printf false)"
    exit 0
fi

[[ -x "${env_dir}/bin/python" && -f "${model_dir}/model.safetensors" ]] || {
    printf 'ERROR: GLM-OCR setup is incomplete.\n' >&2; exit 66;
}
[[ -f "${protocol_dir}/protocol.json" && -f "${protocol_dir}/train.jsonl" \
    && -f "${protocol_dir}/validation.jsonl" && -f "${protocol_dir}/test.jsonl" ]] || {
    printf 'ERROR: locked R1/R2 protocol is missing.\n' >&2; exit 66;
}
[[ ! -e "${output_dir}" ]] || { printf 'ERROR: run already exists: %s\n' "${output_dir}" >&2; exit 74; }

mkdir -p "${experiment_root}/training_runs" "${experiment_root}/tmp" "${workspace}/runs"
export TMPDIR="${experiment_root}/tmp"
export HF_HOME="${experiment_root}/cache/huggingface"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "${pytorch_package}/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

python -m layout_ocr.train_screen \
    --mode "${mode}" \
    --model-path "${model_dir}" \
    --train-manifest "${protocol_dir}/train.jsonl" \
    --validation-manifest "${protocol_dir}/validation.jsonl" \
    --protocol-file "${protocol_dir}/protocol.json" \
    --output-dir "${output_dir}" \
    --max-steps 1024 --num-queries 32 --seed "${seed}" \
    --learning-rate 5e-5 --warmup-steps 64 --min-lr-ratio 0.1 \
    --residual-scale-cap 0.03 --auxiliary-weight "${auxiliary_weight}" --max-grad-norm 1.0 \
    --max-pixels 1003520 --max-eval-new-tokens 768 --validation-interval 256 --log-steps 10 \
    "${eval_only[@]}"
