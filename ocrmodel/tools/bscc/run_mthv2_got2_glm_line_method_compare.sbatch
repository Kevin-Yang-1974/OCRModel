#!/usr/bin/env bash
# Compare untouched GOT2/GLM-OCR with the project layout-semantic adapter on
# one deterministic MTHv2 GT-cropped line/column protocol.  The 1000-record
# test subset is opened only after both training arms finish.
#SBATCH --job-name=mthv2_gotglm_method
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/gotglm_line_method_compare"
code_root="${workspace}/glm_ocr_layout_ot/code/ocrmodel"
glm_code_root="${code_root}"
glm_env="${workspace}/glm_ocr_layout_ot/envs/glm-ocr-py311-cu128"
got_env="${workspace}/envs/got2-py310-cu118"
got_source_root="${workspace}/ocrmodel/src/GOT-OCR-2.0"
got_model="${workspace}/models/GOT-OCR2-original"
glm_model="${workspace}/models/GLM-OCR-ca5d8b3"
source_lines="${workspace}/datasets/MTHv2/converted/mthv2_anandasky_gt_lines_v1"
joint_root="${workspace}/datasets/MTHv2/converted/mthv2_gtline_joint_1000_v2"
run_id="${GOTGLM_LINE_RUN_ID:-mthv2_gotglm_line_method_1000_260920_v5}"
train_limit="${GOTGLM_LINE_TRAIN_LIMIT:-1000}"
validation_limit="${GOTGLM_LINE_VALIDATION_LIMIT:-200}"
gpu_ids="${GOTGLM_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
got_pytorch_package="/home/bingxing2/apps/package/pytorch/2.0.1+cu118_cp310"
glm_pytorch_package="/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311"
run_root="${workspace}/runs/${run_id}"
status_file="${run_root}/status.json"
launcher_log="${run_root}/launcher.log"

[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo '{"event":"compare_failed","error":"invalid_run_id"}' >&2; exit 64; }
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]] || { echo '{"event":"compare_failed","error":"exactly_four_gpus_required"}' >&2; exit 64; }
[[ -f "${source_lines}/train/manifest.jsonl" && -f "${source_lines}/validation/manifest.jsonl" ]] || { echo '{"event":"compare_failed","error":"source_line_manifests_missing"}' >&2; exit 66; }
[[ -f "${source_lines}/test/manifest_subset_1000.jsonl" ]] || { echo '{"event":"compare_failed","error":"locked_test_subset_missing"}' >&2; exit 66; }
[[ ! -e "${run_root}" ]] || { echo '{"event":"compare_failed","error":"run_already_exists"}' >&2; exit 74; }

mkdir -p "${run_root}" "${workspace}/runs" "${experiment_root}/protocols" "${experiment_root}/tmp"
exec > >(tee -a "${launcher_log}") 2>&1

stage="SETUP"
detail="starting"
write_status() {
    local state="$1"
    local message="${2:-}"
    printf '{"status":"%s","stage":"%s","message":"%s","run_id":"%s","test_used_for_selection":false}\n' \
        "${state}" "${stage}" "${message//\"/\\\"}" "${run_id}" > "${status_file}"
}
write_status running "${detail}"
trap 'rc=$?; if (( rc != 0 )); then write_status failed "return_code_${rc}" || true; fi; exit $rc' EXIT

load_glm_env() {
    module purge
    export LD_LIBRARY_PATH=""
    source "${glm_pytorch_package}/env.sh"
}

load_got_env() {
    module purge
    export LD_LIBRARY_PATH=""
    source "${got_pytorch_package}/env.sh"
}

load_glm_env
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="${experiment_root}/tmp/huggingface"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "${HF_HOME}"

stage="PREPARE_DATA"
write_status running "converting_line_manifests"
"${glm_env}/bin/python" "${code_root}/tools/prepare_mthv2_line_page_manifests.py" \
    --source-root "${source_lines}" \
    --output-root "${joint_root}" \
    --train-limit "${train_limit}" \
    --validation-limit "${validation_limit}" \
    --without-test
for split in train validation; do
    [[ -e "${joint_root}/${split}/images" ]] || ln -s "${source_lines}/${split}/images" "${joint_root}/${split}/images"
done
"${glm_env}/bin/python" "${code_root}/tools/audit_mthv2_manifest.py" \
    --train-manifest "${joint_root}/train/manifest.jsonl" \
    --validation-manifest "${joint_root}/validation/manifest.jsonl" \
    --without-test --allow-count-mismatch --num-queries 64 \
    --dataset-label MTHv2_GTLine_1000Train --protocol-label gotglm_line_method_train_v1 \
    --output "${experiment_root}/protocols/${run_id}.train_no_test.json"

stage="GOT_METHOD"
write_status running "training_got2_layout_plus_semantic_adapter"
load_got_env
got_python_lib="${got_env}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${got_python_lib}/torch/lib:${got_python_lib}/nvidia/cuda_runtime/lib:${got_python_lib}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
got_method_model="${run_root}/got_method_model"
torchrun="${got_env}/bin/torchrun"
[[ -x "${torchrun}" ]] || { echo '{"event":"compare_failed","error":"got_torchrun_missing"}' >&2; exit 66; }
cd "${got_source_root}"
"${torchrun}" --standalone --nproc_per_node=4 \
    --master_port "$((29641 + ${SLURM_JOB_ID:-0} % 1000))" \
    scripts/train_GOT_layout.py \
    --model_name_or_path "${got_model}" \
    --tokenizer_name_or_path "${got_model}" \
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
    --evaluation_strategy no --save_strategy no --logging_steps 16 --save_safetensors True \
    --model_max_length 2048 --dataloader_num_workers 0 --remove_unused_columns False \
    --report_to none --max_steps 512 --learning_rate 2e-5 --warmup_steps 32 \
    --lr_scheduler_type cosine --weight_decay 0 --max_grad_norm 1.0 \
    --seed 42 --data_seed 42 --health_check_steps 32 --output_dir "${got_method_model}"

stage="GLM_METHOD_STAGE1"
write_status running "training_glm_layout_ot_stage1"
load_glm_env
glm_stage1="${run_id}_stage1"
glm_stage2="${run_id}_stage2"
glm_protocol="${experiment_root}/protocols/${run_id}.train_no_test.json"
common_glm_args=(
    --foreground --gpu-ids "${gpu_ids}" --gpu-utilization-limit 50
    --remote-root "${workspace}/glm_ocr_layout_ot" --code-root "${glm_code_root}"
    --env-dir "${glm_env}" --model-dir "${glm_model}" --dataset-root "${joint_root}"
    --protocol-file "${glm_protocol}" --seed 42 --num-queries 64
    --max-steps 512 --lr-schedule-steps 512 --learning-rate 2.5e-5
    --decoder-adaptation lora --decoder-lora-rank 8 --decoder-lora-alpha 8
    --decoder-lora-dropout 0 --decoder-learning-rate 5e-6 --warmup-steps 32
    --min-lr-ratio 0.1 --gradient-accumulation-steps 1 --initial-residual-scale 0
    --residual-scale-cap 0.03 --auxiliary-weight 0.4 --auxiliary-weight-start 0.4
    --layout-loss-profile full --validation-interval 512 --max-eval-new-tokens 256
    # Deferred validation already leaves selection pending; the DDP runner
    # rejects combining --defer-validation with --skip-selection.
    --max-pixels 1003520 --without-test --defer-validation --allow-count-mismatch
)
bash "${code_root}/tools/training/run_glmocr_mthv2_ddp.sh" \
    "${common_glm_args[@]}" --run-id "${glm_stage1}" --mode layout_ot \
    --experiment-label "mthv2_gtline_layout_stage1"

stage="GLM_METHOD_STAGE2"
write_status running "training_glm_sem_adapter_and_residual_gate"
stage1_dir="${workspace}/glm_ocr_layout_ot/training_runs/${glm_stage1}/seed42"
[[ -f "${stage1_dir}/checkpoint-512/adapter.safetensors" ]] || { echo '{"event":"compare_failed","error":"glm_stage1_checkpoint_missing"}' >&2; exit 66; }
bash "${code_root}/tools/training/run_glmocr_mthv2_sem_adapter.sh" \
    "${common_glm_args[@]}" --run-id "${glm_stage2}" --init-checkpoint-dir "${stage1_dir}/checkpoint-512" \
    --max-steps 256 --lr-schedule-steps 256 --validation-interval 256 \
    --experiment-label "mthv2_gtline_sem_adapter_stage2"

stage="BASELINE_EVAL"
write_status running "evaluating_untouched_got_and_glm_after_training"
load_got_env
got_python_lib="${got_env}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${got_python_lib}/torch/lib:${got_python_lib}/nvidia/cuda_runtime/lib:${got_python_lib}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
"${got_env}/bin/python" "${code_root}/tools/evaluation/evaluate_mthv2_columns.py" \
    --model got --model-dir "${got_model}" --got-source-root "${got_source_root}" \
    --manifest "${source_lines}/test/manifest_subset_1000.jsonl" \
    --train-manifest "${joint_root}/train/manifest.jsonl" \
    --output-dir "${run_root}/got_baseline" --max-new-tokens 256

load_glm_env
"${glm_env}/bin/python" "${code_root}/tools/evaluation/evaluate_mthv2_columns.py" \
    --model glm --model-dir "${glm_model}" \
    --manifest "${source_lines}/test/manifest_subset_1000.jsonl" \
    --train-manifest "${joint_root}/train/manifest.jsonl" \
    --output-dir "${run_root}/glm_baseline" --max-new-tokens 256

stage="GOT_METHOD_EVAL"
write_status running "evaluating_got2_method"
cd "${workspace}"
load_got_env
got_python_lib="${got_env}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${got_python_lib}/torch/lib:${got_python_lib}/nvidia/cuda_runtime/lib:${got_python_lib}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
"${got_env}/bin/python" "${code_root}/tools/evaluation/evaluate_mthv2_columns.py" \
    --model got --model-dir "${got_method_model}" --got-source-root "${got_source_root}" \
    --manifest "${source_lines}/test/manifest_subset_1000.jsonl" \
    --train-manifest "${joint_root}/train/manifest.jsonl" \
    --output-dir "${run_root}/got_method" --max-new-tokens 256

stage="GLM_METHOD_EVAL"
write_status running "evaluating_glm_method"
stage2_dir="${workspace}/glm_ocr_layout_ot/training_runs/${glm_stage2}/seed42"
[[ -f "${stage2_dir}/checkpoint-256/adapter.safetensors" ]] || { echo '{"event":"compare_failed","error":"glm_stage2_checkpoint_missing"}' >&2; exit 66; }
"${glm_env}/bin/python" "${code_root}/tools/evaluation/evaluate_mthv2_columns.py" \
    --model glm --model-dir "${glm_model}" \
    --glm-checkpoint-dir "${stage2_dir}/checkpoint-256" --glm-metadata "${stage2_dir}/metadata.json" \
    --manifest "${source_lines}/test/manifest_subset_1000.jsonl" \
    --train-manifest "${joint_root}/train/manifest.jsonl" \
    --output-dir "${run_root}/glm_method" --max-new-tokens 256

stage="COMPLETE"
write_status complete "comparison_metrics_ready"
printf '{"event":"compare_complete","run_id":"%s","test_records":1000,"train_records":%s,"validation_records":%s,"test_used_for_selection":false,"got_baseline":"%s/got_baseline/summary.json","got_method":"%s/got_method/summary.json","glm_baseline":"%s/glm_baseline/summary.json","glm_method":"%s/glm_method/summary.json"}\n' \
    "${run_id}" "${train_limit}" "${validation_limit}" "${run_root}" "${run_root}" "${run_root}" "${run_root}"
