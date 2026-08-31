#!/usr/bin/env bash
# Resume the existing BSCC P1 at 100k, select at cumulative 100k/150k/200k/250k,
# then launch the validation-selected 200k-step formal P2 without another bounded pre-run.
#SBATCH --job-name=bscc_p1c_p2_200k
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
base_model="${BSCC_GOT_MODEL:-${workspace}/models/GOT-OCR2-original}"
p1_run_id="${BSCC_P1_RUN_ID:-bscc_synth_p1_legacy_100000_20260830_v1}"
p2_run_id="${BSCC_P2_RUN_ID:-bscc_synth_p2_formal_200000_seed42_20260831_v1}"
protocol_id="${BSCC_PROTOCOL_ID:-bscc_p1_100k_to_250k_p2_200k_seed42_20260831_v1}"
p1_selection_name="validation_selection_seed42_400"
p1_run_root="${workspace}/training_runs/GOT/${p1_run_id}"
p2_run_root="${workspace}/training_runs/GOT/${p2_run_id}"
protocol_root="${workspace}/protocols/${protocol_id}"
validation_manifest="${protocol_root}/validation_400_seed42.jsonl"
validation_lock="${protocol_root}/validation_400_seed42.lock.json"
p1_selection="${p1_run_root}/p1/${p1_selection_name}/selection.json"
runner="${workspace}/ocrmodel/tools/training/run_variable_layout_a100.py"
locker="${workspace}/ocrmodel/tools/preprocessing/prepare_random_validation_subset.py"
health_checker="${workspace}/ocrmodel/tools/training/check_checkpoint_health.py"
project_root="${workspace}/ocrmodel/src/GOT-OCR-2.0"

for path in \
    "${dataset_root}/train/manifest.jsonl" \
    "${dataset_root}/validation/manifest.jsonl" \
    "${dataset_root}/test/manifest.jsonl" \
    "${mthv2_root}/train/manifest.jsonl" \
    "${base_model}/model.safetensors" \
    "${base_model}/config.json" \
    "${p1_run_root}/p1/model/checkpoint-100000/model.safetensors" \
    "${p1_run_root}/p1/model/checkpoint-100000/trainer_state.json"; do
    [[ -f "${path}" ]] || { printf 'ERROR: missing required path: %s\n' "${path}" >&2; exit 66; }
done
[[ ! -e "${p2_run_root}" ]] || { printf 'ERROR: formal P2 run already exists: %s\n' "${p2_run_root}" >&2; exit 74; }

gpu_ids="${CUDA_VISIBLE_DEVICES:?Slurm did not assign CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a assigned_gpus <<< "${gpu_ids}"
[[ "${#assigned_gpus[@]}" -eq 4 ]] || { printf 'ERROR: this workflow requires four assigned GPUs.\n' >&2; exit 75; }

mkdir -p "${workspace}/training_runs/GOT" "${workspace}/runs" "${workspace}/tmp" \
    "${workspace}/cache/huggingface" "${workspace}/cache/torch_extensions" "${protocol_root}"
export TMPDIR="${workspace}/tmp"
export HF_HOME="${workspace}/cache/huggingface"
export TORCH_EXTENSIONS_DIR="${workspace}/cache/torch_extensions"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source /home/bingxing2/apps/package/pytorch/2.0.1+cu118_cp310/env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export NCCL_P2P_DISABLE=1

python "${locker}" \
    --source-manifest "${dataset_root}/validation/manifest.jsonl" \
    --output-manifest "${validation_manifest}" \
    --metadata "${validation_lock}" \
    --page-count 400 --seed 42 --reuse-existing

python "${health_checker}" \
    "${p1_run_root}/p1/model/checkpoint-100000" \
    > "${protocol_root}/p1_checkpoint_100000.health.json"

python "${runner}" \
    --dataset-root "${dataset_root}/train" \
    --manifest "${dataset_root}/train/manifest.jsonl" \
    --validation-manifest "${validation_manifest}" \
    --validation-image-root "${dataset_root}/validation" \
    --test-manifest "${dataset_root}/test/manifest.jsonl" \
    --source-model "${base_model}" --tokenizer-model "${base_model}" \
    --stages p1 --ablation vlqa_layout_p1_p2 --layout-loss-preset layout_full \
    --layout-memory-resolution 64 --num-layout-prompt-queries 32 \
    --max-layout-records 512 --max-layout-tokens 2048 \
    --layout-decoder-layers 2 --layout-decoder-hidden-size 256 --layout-decoder-num-heads 8 \
    --layout-boundary-loss-weight 0 --layout-count-condition-strength 0 \
    --p1-max-steps 250000 --p1-checkpoint-steps 50000 \
    --p1-candidate-steps 100000,150000,200000,250000 --checkpoint-retention 5 \
    --replay-manifest "${mthv2_root}/train/manifest.jsonl" \
    --replay-image-root "${mthv2_root}/train" \
    --p1-learning-rate 1e-4 --p1-vision-learning-rate 1e-6 \
    --p1-projector-learning-rate 1e-5 --p1-layout-learning-rate 1e-4 \
    --p1-qwen-learning-rate 0 --p1-gate-learning-rate 0 --p1-lm-head-learning-rate 0 \
    --lr-scheduler-type constant --warmup-ratio 0 \
    --per-device-batch-size 1 --gradient-accumulation-steps 1 \
    --gpu-ids "${gpu_ids}" --gpu-utilization-limit 50 \
    --distributed-strategy deepspeed_zero2 --nccl-p2p-disable --seed 42 \
    --protocol-version bscc_p1_continuation_seed42_validation_v1 \
    --variant bscc_synthetic_legacy_pvld_p1_continued \
    --validation-page-count 400 \
    --p1-validation-selection-name "${p1_selection_name}" \
    --runs-root "${workspace}/training_runs/GOT" --project-root "${project_root}" \
    --run-id "${p1_run_id}" --resume-existing-run

[[ -f "${p1_selection}" ]] || { printf 'ERROR: P1 validation selection was not produced.\n' >&2; exit 70; }
selected_p1="$(jq -er '.selected.model_path' "${p1_selection}")"
python "${health_checker}" "${selected_p1}" > "${protocol_root}/p1_selected_checkpoint.health.json"

python "${runner}" \
    --dataset-root "${dataset_root}/train" \
    --manifest "${dataset_root}/train/manifest.jsonl" \
    --validation-manifest "${validation_manifest}" \
    --validation-image-root "${dataset_root}/validation" \
    --test-manifest "${dataset_root}/test/manifest.jsonl" \
    --source-model "${selected_p1}" --tokenizer-model "${selected_p1}" \
    --source-validation-selection "${p1_selection}" \
    --stages p2 --ablation vlqa_layout_p1_p2 --layout-loss-preset layout_full \
    --layout-memory-resolution 64 --num-layout-prompt-queries 32 \
    --max-layout-records 512 --max-layout-tokens 2048 \
    --layout-decoder-layers 2 --layout-decoder-hidden-size 256 --layout-decoder-num-heads 8 \
    --layout-boundary-loss-weight 0 --layout-count-condition-strength 0 \
    --p2-max-steps 200000 --p2-checkpoint-steps 50000 --checkpoint-retention 4 \
    --health-check-steps 10000 \
    --p2-learning-rate 5e-5 --p2-vision-learning-rate 5e-7 \
    --p2-projector-learning-rate 5e-6 --p2-layout-learning-rate 5e-5 \
    --p2-qwen-learning-rate 1e-6 --p2-gate-learning-rate 1e-6 --p2-lm-head-learning-rate 0 \
    --lr-scheduler-type cosine --warmup-ratio 0.001 --layout-module-fp32 \
    --per-device-batch-size 1 --gradient-accumulation-steps 1 \
    --gpu-ids "${gpu_ids}" --gpu-utilization-limit 50 \
    --distributed-strategy deepspeed_zero2 --nccl-p2p-disable --seed 42 \
    --protocol-version bscc_p2_formal_fp32_seed42_validation_v1 \
    --variant bscc_synthetic_p2_formal \
    --validation-page-count 400 \
    --runs-root "${workspace}/training_runs/GOT" --project-root "${project_root}" \
    --run-id "${p2_run_id}"

printf '{"event":"bscc_p1_continuation_p2_formal_completed","slurm_job_id":"%s","p1_run_id":"%s","p1_selection":"%s","p2_run_id":"%s","p2_steps":200000,"p2_checkpoint_steps":50000,"p2_health_check_steps":10000,"test_used_for_selection":false}\n' \
    "${SLURM_JOB_ID:-unknown}" "${p1_run_id}" "${p1_selection}" "${p2_run_id}"
