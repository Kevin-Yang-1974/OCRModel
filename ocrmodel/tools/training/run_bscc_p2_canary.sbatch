#!/usr/bin/env bash
# BSCC bounded P2 canary initialized directly from the existing 100k P1 checkpoint.
# This is an engineering canary, not a validation-selected formal P2 run.
#SBATCH --job-name=bscc_p2_canary
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=08:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -euo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
env_dir="${BSCC_GOT_ENV:-${workspace}/envs/got2-py310-cu118}"
dataset_root="${BSCC_SYNTHETIC_ROOT:-${workspace}/datasets/got_layout_pages/ancient_photo_diverse_formal_s3s4_dense_20260827_v4}"
source_model="${BSCC_P2_CANARY_SOURCE:-${workspace}/training_runs/GOT/bscc_synth_p1_legacy_100000_20260830_v1/p1/model/checkpoint-100000}"
run_id="${BSCC_P2_CANARY_RUN_ID:-bscc_p2_canary_from_p1_100000_fp32_20260831_v5}"
run_root="${workspace}/training_runs/GOT/${run_id}"
runner="${workspace}/ocrmodel/tools/training/run_variable_layout_a100.py"
project_root="${workspace}/ocrmodel/src/GOT-OCR-2.0"
source_lock="${run_root}/metadata/direct_canary_source_selection.json"

for path in \
    "${dataset_root}/train/manifest.jsonl" \
    "${dataset_root}/validation/manifest.jsonl" \
    "${dataset_root}/test/manifest.jsonl" \
    "${source_model}/model.safetensors" \
    "${source_model}/config.json"; do
    [[ -f "${path}" ]] || { printf 'ERROR: missing required path: %s\n' "${path}" >&2; exit 66; }
done
[[ ! -e "${run_root}" ]] || { printf 'ERROR: canary run already exists: %s\n' "${run_root}" >&2; exit 74; }

gpu_ids="${CUDA_VISIBLE_DEVICES:?Slurm did not assign CUDA_VISIBLE_DEVICES}"
IFS=',' read -r -a assigned_gpus <<< "${gpu_ids}"
[[ "${#assigned_gpus[@]}" -eq 1 ]] || { printf 'ERROR: P2 canary requires one assigned GPU for the BSCC bounded run.\n' >&2; exit 75; }

mkdir -p "${workspace}/training_runs/GOT" "${workspace}/runs" "${workspace}/tmp" \
    "${workspace}/cache/huggingface" "${workspace}/cache/torch_extensions" "${run_root}/metadata"
export TMPDIR="${workspace}/tmp"
export HF_HOME="${workspace}/cache/huggingface"
export TORCH_EXTENSIONS_DIR="${workspace}/cache/torch_extensions"
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
source /home/bingxing2/apps/package/pytorch/2.0.1+cu118_cp310/env.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

config_sha256="$(sha256sum "${source_model}/config.json" | awk '{print $1}')"
weights_sha256="$(sha256sum "${source_model}/model.safetensors" | awk '{print $1}')"
/usr/bin/python3 - "${source_lock}" "${source_model}" "${config_sha256}" "${weights_sha256}" <<'PY'
import json
import sys
from pathlib import Path

target, model, config_hash, weights_hash = sys.argv[1:]
payload = {
    "purpose": "layout_ablation_validation_selection",
    "selection_purpose": "p1_layout",
    "selection_split": "validation",
    "test_used_for_selection": False,
    "ablation_id": "vlqa_layout_p1_p2",
    "selection_role": "direct_checkpoint_engineering_canary",
    "formal_selection": False,
    "note": "Directly initialized from existing 100000-step P1 checkpoint; no validation selection was used.",
    "selected": {
        "model_path": str(Path(model).resolve()),
        "optimizer_step": 100000,
        "config_sha256": config_hash,
        "weights_sha256": weights_hash,
    },
}
Path(target).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

export CUDA_VISIBLE_DEVICES="${gpu_ids}"
export NCCL_P2P_DISABLE=1
validation_page_count="$(wc -l < "${dataset_root}/validation/manifest.jsonl")"

python "${runner}" \
    --dataset-root "${dataset_root}/train" \
    --manifest "${dataset_root}/train/manifest.jsonl" \
    --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
    --validation-image-root "${dataset_root}/validation" \
    --test-manifest "${dataset_root}/test/manifest.jsonl" \
    --source-model "${source_model}" --tokenizer-model "${source_model}" \
    --stages p2 --ablation vlqa_layout_p1_p2 --layout-loss-preset layout_full \
    --layout-memory-resolution 64 --num-layout-prompt-queries 32 \
    --max-layout-records 512 --max-layout-tokens 2048 \
    --layout-decoder-layers 2 --layout-decoder-hidden-size 256 --layout-decoder-num-heads 8 \
    --layout-boundary-loss-weight 0 --layout-count-condition-strength 0 \
    --p2-max-steps 1000 --p2-checkpoint-steps 500 --checkpoint-retention 3 \
    --p2-learning-rate 5e-5 --p2-vision-learning-rate 5e-7 \
    --p2-projector-learning-rate 5e-6 --p2-layout-learning-rate 5e-5 \
    --p2-qwen-learning-rate 1e-6 --p2-gate-learning-rate 1e-6 --p2-lm-head-learning-rate 0 \
    --lr-scheduler-type cosine --warmup-ratio 0.001 \
    --layout-module-fp32 \
    --source-validation-selection "${source_lock}" \
    --per-device-batch-size 1 --gradient-accumulation-steps 1 \
    --gpu-ids "${gpu_ids}" --gpu-utilization-limit 50 \
    --distributed-strategy deepspeed_zero2 --nccl-p2p-disable --seed 42 \
    --protocol-version bscc_p2_direct_checkpoint_canary_v1 \
    --variant bscc_synthetic_p2_finite_canary \
    --validation-page-count "${validation_page_count}" \
    --runs-root "${workspace}/training_runs/GOT" --project-root "${project_root}" \
    --run-id "${run_id}" \
    > "${workspace}/runs/${run_id}.train.log" 2>&1

printf '{"event":"bscc_p2_canary_completed","run_id":"%s","run_root":"%s","source_checkpoint":"%s","steps":1000,"checkpoint_steps":[500,1000],"formal_selection":false,"test_used_for_selection":false}\n' \
    "${run_id}" "${run_root}" "${source_model}"
