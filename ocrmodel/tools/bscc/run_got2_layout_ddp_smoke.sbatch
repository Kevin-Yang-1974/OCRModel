#!/usr/bin/env bash
# Bounded four-GPU preflight for the GOT2 layout-method DDP configuration.
# It exercises the first optimizer steps without opening the MTHv2 test split.
#SBATCH --job-name=got2_layout_ddp_smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=00:30:00
#SBATCH --output=/home/bingxing2/home/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
code_root="${workspace}/glm_ocr_layout_ot/code/ocrmodel"
got_source_root="${workspace}/ocrmodel/src/GOT-OCR-2.0"
glm_env="${workspace}/glm_ocr_layout_ot/envs/glm-ocr-py311-cu128"
got_env="${workspace}/envs/got2-py310-cu118"
got_model="${workspace}/models/GOT-OCR2-original"
source_lines="${workspace}/datasets/MTHv2/converted/mthv2_anandasky_gt_lines_v1"
joint_root="${workspace}/datasets/MTHv2/converted/mthv2_gtline_joint_1000_v2"
output_dir="${workspace}/runs/got2_layout_ddp_smoke_${SLURM_JOB_ID}"

module purge
export LD_LIBRARY_PATH=""
source "/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311/env.sh"
if [[ ! -f "${joint_root}/train/manifest.jsonl" ]]; then
    "${glm_env}/bin/python" "${code_root}/tools/prepare_mthv2_line_page_manifests.py" \
        --source-root "${source_lines}" --output-root "${joint_root}" \
        --train-limit 1000 --validation-limit 200 --without-test
    for split in train validation; do
        [[ -e "${joint_root}/${split}/images" ]] || ln -s "${source_lines}/${split}/images" "${joint_root}/${split}/images"
    done
fi

module purge
export LD_LIBRARY_PATH=""
source "/home/bingxing2/apps/package/pytorch/2.0.1+cu118_cp310/env.sh"
got_python_lib="${got_env}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${got_python_lib}/torch/lib:${got_python_lib}/nvidia/cuda_runtime/lib:${got_python_lib}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "${workspace}/runs"

cd "${got_source_root}"
"${got_env}/bin/torchrun" --standalone --nproc_per_node=4 \
    --master_port "$((29641 + ${SLURM_JOB_ID:-0} % 1000))" \
    scripts/train_GOT_layout.py \
    --model_name_or_path "${got_model}" --tokenizer_name_or_path "${got_model}" \
    --layout_manifest "${joint_root}/train/manifest.jsonl" \
    --layout_image_root "${joint_root}/train" --layout_split train \
    --layout_stage p2 --layout_architecture fixed_slot --max_regions 64 \
    --datasets layout-page-jsonl --conversation_version mpt --use_im_start_end True \
    --use_generic_adapter True --generic_adapter_dim 256 --generic_adapter_num_heads 8 \
    --generic_adapter_ffn_expansion 8 --layout_writeback_mode visual_value_layout_routing \
    --layout_writeback_source layout_evidence --layout_writeback_num_heads 8 \
    --layout_writeback_gate_init 0 --ocr_loss_weight 1.0 --layout_loss_weight 0.5 \
    --object_loss_weight 1.0 --bbox_l1_loss_weight 5.0 --bbox_giou_loss_weight 2.0 \
    --direction_loss_weight 1.0 --vision_learning_rate 1e-6 --projector_learning_rate 1e-5 \
    --layout_learning_rate 1e-4 --qwen_learning_rate 1e-6 --gate_learning_rate 1e-5 \
    --lm_head_learning_rate 0 --bf16 True --fp16 False --tf32 False \
    --gradient_checkpointing False --ddp_find_unused_parameters True \
    --gradient_accumulation_steps 1 --per_device_train_batch_size 1 \
    --evaluation_strategy no --save_strategy no --logging_steps 1 --save_safetensors True \
    --model_max_length 2048 --dataloader_num_workers 0 --remove_unused_columns False \
    --report_to none --max_steps 2 --learning_rate 2e-5 --warmup_steps 1 \
    --lr_scheduler_type cosine --weight_decay 0 --max_grad_norm 1.0 \
    --seed 42 --data_seed 42 --health_check_steps 1 --output_dir "${output_dir}"

printf '%s\n' GOT_DDP_SMOKE_OK
