#!/usr/bin/env bash
#SBATCH --job-name=bscc_pvld_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -euo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
env_dir="${BSCC_GOT_ENV:-${workspace}/envs/got2-py310-cu118}"
model_dir="${BSCC_GOT_MODEL:-${workspace}/models/GOT-OCR2-original}"
smoke_mode="${BSCC_SMOKE_MODE:?set BSCC_SMOKE_MODE}"
run_id="${BSCC_RUN_ID:-bscc_${smoke_mode}_${SLURM_JOB_ID}}"
run_dir="${workspace}/runs/${run_id}"
manifest="${BSCC_SMOKE_MANIFEST:-${workspace}/samples/mthv2_train_2pages/manifest.jsonl}"
image_root="${BSCC_SMOKE_IMAGE_ROOT:-${workspace}/samples/mthv2_train_2pages}"

mkdir -p "${run_dir}" "${workspace}/tmp" "${workspace}/cache/huggingface" \
    "${workspace}/cache/torch_extensions"
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
summary="${run_dir}/summary.json"
printf 'job_id=%s\nhost=%s\nmode=%s\ncuda_visible_devices=%s\nmanifest=%s\n' \
    "${SLURM_JOB_ID}" "$(hostname)" "${smoke_mode}" \
    "${CUDA_VISIBLE_DEVICES:-}" "${manifest}" > "${run_dir}/metadata.txt"

case "${smoke_mode}" in
    cuda_bf16)
        python "${workspace}/ocrmodel/tools/training/run_bscc_smoke_component.py" \
            --mode cuda_bf16 --output "${summary}"
        ;;
    nccl)
        torchrun --standalone --nproc_per_node=2 \
            "${workspace}/ocrmodel/tools/training/run_bscc_smoke_component.py" \
            --mode nccl --output "${summary}"
        ;;
    zero2)
        torchrun --standalone --nproc_per_node=2 \
            "${workspace}/ocrmodel/tools/training/run_bscc_smoke_component.py" \
            --mode zero2 --output "${summary}"
        ;;
    pvld_p1|pvld_p2|pvld_dist)
        [[ -f "${manifest}" ]] || { printf 'ERROR: missing smoke manifest: %s\n' "${manifest}" >&2; exit 66; }
        stage=p1
        ablation=vlqa_layout_p1_p2
        max_steps=1
        if [[ "${smoke_mode}" == pvld_p2 ]]; then
            stage=p2
            ablation=vlqa_layout_direct
        elif [[ "${smoke_mode}" == pvld_dist ]]; then
            max_steps="${BSCC_PVLD_STEPS:-5}"
            (( max_steps >= 1 && max_steps <= 10 )) || { printf 'ERROR: BSCC_PVLD_STEPS must be 1..10.\n' >&2; exit 64; }
        fi
        world_size="$(python -c 'import os; print(len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")))')"
        deepspeed --num_gpus "${world_size}" --master_port "$((21000 + SLURM_JOB_ID % 20000))" \
            "${project_root}/scripts/train_GOT_layout.py" \
            --deepspeed "${project_root}/zero_config/zero2.json" \
            --model_name_or_path "${model_dir}" \
            --tokenizer_name_or_path "${model_dir}" \
            --layout_manifest "${manifest}" --layout_image_root "${image_root}" \
            --layout_split train --layout_stage "${stage}" \
            --layout_architecture pvld --ablation_id "${ablation}" \
            --layout_loss_preset layout_full --p2_train_scope adapter_projector \
            --max_regions 512 --num_layout_prompt_queries 32 \
            --max_layout_records 512 --max_layout_tokens 2048 \
            --layout_decoder_layers 2 --layout_decoder_hidden_size 256 \
            --layout_decoder_num_heads 8 --layout_boundary_loss_weight 1.0 \
            --layout_count_condition_strength 1.0 \
            --layout_writeback_mode visual_value_layout_routing \
            --layout_writeback_source layout_evidence --layout_writeback_num_heads 8 \
            --layout_writeback_gate_init 0 --datasets layout-page-jsonl \
            --conversation_version mpt --use_im_start_end True \
            --bf16 True --fp16 False --gradient_accumulation_steps 1 \
            --per_device_train_batch_size 1 --optim adamw_torch \
            --evaluation_strategy no --save_strategy no --logging_steps 1 \
            --model_max_length 2048 --gradient_checkpointing True \
            --dataloader_num_workers 0 --report_to none \
            --remove_unused_columns False --max_steps "${max_steps}" \
            --learning_rate 1e-4 --lr_scheduler_type constant --warmup_ratio 0 \
            --weight_decay 0 --object_loss_weight 1 --bbox_l1_loss_weight 5 \
            --bbox_giou_loss_weight 2 --direction_loss_weight 1 \
            --layout_loss_weight 1 --ocr_loss_weight "$([[ "${stage}" == p1 ]] && printf 0 || printf 1)" \
            --seed 42 --output_dir "${run_dir}/model" > "${run_dir}/train.log" 2>&1
        python - "${run_dir}/model/layout_training_metrics.json" "${summary}" "${smoke_mode}" <<'PY'
import json
import math
import sys
from pathlib import Path

metrics_path, output, mode = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
loss = float(metrics["train_loss"])
if int(metrics["global_step"]) < 1 or not math.isfinite(loss):
    raise RuntimeError("PVLD smoke did not produce a finite optimizer step")
payload = {
    "status": "ok", "mode": mode, "global_step": int(metrics["global_step"]),
    "train_loss": loss, "world_size": metrics["training_budget"]["world_size"],
    "layout_stage": metrics["layout_stage"], "input_granularity": "whole_page_image",
    "test_read": False,
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, separators=(",", ":")))
PY
        ;;
    *) printf 'ERROR: unknown BSCC_SMOKE_MODE=%s\n' "${smoke_mode}" >&2; exit 64 ;;
esac

touch "${run_dir}/BSCC_SMOKE_FINISHED"
printf '{"event":"bscc_smoke_completed","job_id":"%s","mode":"%s","run_dir":"%s"}\n' \
    "${SLURM_JOB_ID}" "${smoke_mode}" "${run_dir}"
