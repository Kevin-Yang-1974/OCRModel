#!/usr/bin/env bash
#SBATCH --job-name=bscc_multigpu_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -euo pipefail
workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
env_dir="${BSCC_GOT_ENV:-${workspace}/envs/got2-py310-cu118}"
mode="${BSCC_SMOKE_MODE:?set BSCC_SMOKE_MODE=nccl|zero2|pvld_dist}"
run_id="${BSCC_RUN_ID:-bscc_${mode}_${SLURM_JOB_ID}}"
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

script="${workspace}/ocrmodel/tools/training/run_bscc_smoke_component.py"
summary="${run_dir}/summary.json"
printf 'job_id=%s\nhost=%s\nmode=%s\ncuda_visible_devices=%s\n' \
    "${SLURM_JOB_ID}" "$(hostname)" "${mode}" "${CUDA_VISIBLE_DEVICES:-}" > "${run_dir}/metadata.txt"

case "${mode}" in
    nccl|zero2)
        torchrun --standalone --nproc_per_node=2 "${script}" --mode "${mode}" --output "${summary}"
        ;;
    pvld_dist)
        smoke_manifest="${BSCC_SMOKE_MANIFEST:?set BSCC_SMOKE_MANIFEST}"
        smoke_image_root="${BSCC_SMOKE_IMAGE_ROOT:?set BSCC_SMOKE_IMAGE_ROOT}"
        steps="${BSCC_PVLD_STEPS:-5}"
        (( steps >= 1 && steps <= 10 )) || { printf 'ERROR: BSCC_PVLD_STEPS must be 1..10.\n' >&2; exit 64; }
        project_root="${workspace}/ocrmodel/src/GOT-OCR-2.0"
        model_dir="${BSCC_GOT_MODEL:-${workspace}/models/GOT-OCR2-original}"
        deepspeed --num_gpus 2 --master_port "$((21000 + SLURM_JOB_ID % 20000))" \
            "${project_root}/scripts/train_GOT_layout.py" \
            --deepspeed "${project_root}/zero_config/zero2.json" \
            --model_name_or_path "${model_dir}" --tokenizer_name_or_path "${model_dir}" \
            --layout_manifest "${smoke_manifest}" --layout_image_root "${smoke_image_root}" \
            --layout_split train --layout_stage p2 --layout_architecture pvld \
            --ablation_id vlqa_layout_direct --layout_loss_preset layout_full \
            --p2_train_scope adapter_projector --max_regions 512 \
            --num_layout_prompt_queries 32 --max_layout_records 512 --max_layout_tokens 2048 \
            --layout_decoder_layers 2 --layout_decoder_hidden_size 256 --layout_decoder_num_heads 8 \
            --layout_boundary_loss_weight 1.0 --layout_count_condition_strength 1.0 \
            --pvld_use_spatial_memory True --pvld_coverage_detach True \
            --layout_writeback_mode visual_value_layout_routing --layout_writeback_source layout_evidence \
            --layout_writeback_num_heads 8 --layout_writeback_gate_init 0 \
            --datasets layout-page-jsonl --conversation_version mpt --use_im_start_end True \
            --bf16 True --fp16 False --gradient_accumulation_steps 1 --per_device_train_batch_size 1 \
            --optim adamw_torch --evaluation_strategy no --save_strategy no --logging_steps 1 \
            --model_max_length 2048 --gradient_checkpointing True --dataloader_num_workers 0 \
            --report_to none --remove_unused_columns False --max_steps "${steps}" \
            --learning_rate 5e-5 --lr_scheduler_type constant --warmup_ratio 0 --weight_decay 0 \
            --object_loss_weight 1 --bbox_l1_loss_weight 5 --bbox_giou_loss_weight 2 \
            --direction_loss_weight 1 --layout_loss_weight 1 --ocr_loss_weight 1 \
            --seed 42 --output_dir "${run_dir}/model" > "${run_dir}/train.log" 2>&1
        python - "${run_dir}/model/layout_training_metrics.json" "${summary}" <<'PY'
import json, math, sys
from pathlib import Path
metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
loss = float(metrics["train_loss"])
if int(metrics["global_step"]) < 1 or not math.isfinite(loss):
    raise RuntimeError("PVLD distributed smoke did not produce finite loss")
payload = {"status":"ok", "mode":"pvld_dist", "global_step":int(metrics["global_step"]),
           "train_loss":loss, "world_size":metrics["training_budget"]["world_size"],
           "input_granularity":"whole_page_image", "test_read":False}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
print(json.dumps(payload, separators=(",", ":")))
PY
        ;;
    *) printf 'ERROR: unsupported mode %s\n' "${mode}" >&2; exit 64 ;;
esac
touch "${run_dir}/BSCC_SMOKE_FINISHED"
printf '{"event":"bscc_multigpu_smoke_completed","job_id":"%s","mode":"%s","run_dir":"%s"}\n' "${SLURM_JOB_ID}" "${mode}" "${run_dir}"
