#!/usr/bin/env bash
# A100 decoder-LoRA pipeline: smoke -> deferred training -> parallel validation -> locked test.
# Parameter-optimized objective is L_official + 0.4 * L_layout; the natural-loop training objective is disabled.
set -Eeuo pipefail

root="${GLMOCR_A100_ROOT:-/data3/yky/yangky_ocr_models/glm_ocr_layout_ot}"
code="${GLMOCR_A100_CODE_ROOT:-${root}/code/ocrmodel}"
python="${GLMOCR_A100_ENV:-${root}/envs/glmocr_a100_py311_cu128}/bin/python"
env_dir="$(dirname "$(dirname "${python}")")"
nvidia_env="${GLMOCR_A100_NVIDIA_ENV:-/data3/yky/yangky_ocr_models/envs/anandasky}"
dataset="${GLMOCR_A100_MTHV2_ROOT:-/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1}"
model_dir="${GLMOCR_A100_MODEL:-/data3/yky/yangky_ocr_models/models/sota/glm_ocr/ca5d8b3e287e52589e37c28385d9655ee4372f9d}"
mode="${GLMOCR_A100_MODE:-geometry}"
dataset_label="${GLMOCR_A100_DATASET_LABEL:-MTHv2}"
protocol_label="${GLMOCR_A100_PROTOCOL_LABEL:-glm_ocr_mthv2_full_official_v1}"
allow_count_mismatch="${GLMOCR_A100_ALLOW_COUNT_MISMATCH:-0}"

run_id="${1:-glmocr_mthv2_decoder_lora_hpopt_20k_5gpu_a100_official_layout_260912_v1}"
[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
seed="${GLMOCR_A100_SEED:-42}"
gpu_ids="${GLMOCR_A100_GPU_IDS:-0,1,2,3,4}"
gpu_utilization_limit=50
max_steps="${GLMOCR_A100_MAX_STEPS:-20000}"
lr_schedule_steps="${GLMOCR_A100_LR_SCHEDULE_STEPS:-${max_steps}}"
warmup_steps="${GLMOCR_A100_WARMUP_STEPS:-216}"
min_lr_ratio="${GLMOCR_A100_MIN_LR_RATIO:-0.1}"
learning_rate="${GLMOCR_A100_LR:-2.5e-5}"
decoder_learning_rate="${GLMOCR_A100_DECODER_LR:-5e-6}"
auxiliary_weight=0.4
max_grad_norm=1.0
validation_interval="${GLMOCR_A100_VALIDATION_INTERVAL:-5000}"
num_queries="${GLMOCR_A100_NUM_QUERIES:-512}"
auxiliary_weight="${GLMOCR_A100_AUXILIARY_WEIGHT:-0.4}"
max_eval_new_tokens="${GLMOCR_A100_MAX_EVAL_NEW_TOKENS:-1536}"
generation_mode="${GLMOCR_A100_GENERATION_MODE:-loop_recovery}"
decoder_lora_rank=8
decoder_lora_alpha=8
decoder_lora_dropout=0
smoke_steps="${GLMOCR_A100_SMOKE_STEPS:-8}"
gpu_utilization_limit="${GLMOCR_A100_GPU_UTILIZATION_LIMIT:-50}"
global_step_offset="${GLMOCR_A100_GLOBAL_STEP_OFFSET:-0}"
init_checkpoint_dir="${GLMOCR_A100_INIT_CHECKPOINT_DIR:-}"
init_checkpoint_override_residual_scale="${GLMOCR_A100_INIT_CHECKPOINT_OVERRIDE_RESIDUAL_SCALE:-}"
init_checkpoint_allow_mode_mismatch="${GLMOCR_A100_INIT_CHECKPOINT_ALLOW_MODE_MISMATCH:-0}"
initial_residual_scale="${GLMOCR_A100_INITIAL_RESIDUAL_SCALE:-0}"
layout_loss_profile="${GLMOCR_A100_LAYOUT_LOSS_PROFILE:-full}"
free_generation_loss="${GLMOCR_A100_FREE_GENERATION_LOSS:-0}"
free_generation_loss_weight="${GLMOCR_A100_FREE_GENERATION_LOSS_WEIGHT:-0.05}"
free_generation_max_new_tokens="${GLMOCR_A100_FREE_GENERATION_MAX_NEW_TOKENS:-512}"

[[ "${allow_count_mismatch}" == "0" || "${allow_count_mismatch}" == "1" ]] || {
    printf '{"status":"failed","error":"invalid_count_mismatch_policy"}\n'; exit 64;
}
[[ "${init_checkpoint_allow_mode_mismatch}" == "0" || "${init_checkpoint_allow_mode_mismatch}" == "1" ]] || {
    printf '{"status":"failed","error":"invalid_checkpoint_mode_mismatch_flag"}\n'; exit 64;
}
if [[ -n "${init_checkpoint_override_residual_scale}" || "${init_checkpoint_allow_mode_mismatch}" == "1" ]]; then
    [[ -n "${init_checkpoint_dir}" ]] || {
        printf '{"status":"failed","error":"checkpoint_option_requires_init_checkpoint"}\n'; exit 64;
    }
fi

group="${root}/training_runs/${run_id}"
smoke_id="${run_id}_smoke"
smoke_group="${root}/training_runs/${smoke_id}"
run_dir="${group}/seed${seed}"
protocol_file="${root}/protocols/${run_id}.train_validation_no_test.json"
validation_root="${run_dir}/parallel-validation"
train_manifest="${dataset}/train/manifest.jsonl"
validation_manifest="${dataset}/validation/manifest.jsonl"

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
(( ${#gpu_array[@]} >= 1 )) || {
    printf '{"status":"failed","error":"at_least_one_gpu_required"}\n'; exit 64;
}
world_size="${#gpu_array[@]}"
declare -A seen_gpu=()
for gpu in "${gpu_array[@]}"; do
    [[ "${gpu}" =~ ^[0-9]+$ ]] || { printf '{"status":"failed","error":"invalid_gpu_id"}\n'; exit 64; }
    [[ -z "${seen_gpu[${gpu}]+present}" ]] || { printf '{"status":"failed","error":"duplicate_gpu_id"}\n'; exit 64; }
    seen_gpu[${gpu}]=1
done

steps=()
for ((step = validation_interval; step <= max_steps; step += validation_interval)); do
    steps+=("${step}")
done
(( ${#steps[@]} >= 1 )) || { printf '{"status":"failed","error":"no_checkpoint_steps"}\n'; exit 64; }
(( ${#steps[@]} <= world_size )) || {
    printf '{"status":"failed","error":"more_checkpoints_than_gpus","checkpoints":%s,"gpus":%s}\n' \
        "${#steps[@]}" "${world_size}"; exit 64;
}

[[ ! -e "${group}" && ! -e "${smoke_group}" ]] || {
    printf '{"status":"failed","error":"run_already_exists"}\n'; exit 74;
}
mkdir -p "${group}" "${root}/protocols"

phase() {
    printf '{"status":"%s","phase":"%s","run_id":"%s","updated_at":"%s"}\n' \
        "$1" "$2" "${run_id}" "$(date -u +%FT%TZ)" > "${group}/pipeline_status.json"
}
current_phase="preflight"
trap 'phase failed "${current_phase}"' ERR
phase running "${current_phase}"

export GLMOCR_DDP_TIMEOUT_SECONDS=86400
export PYTHONNOUSERSITE=1
export PYTHONPATH="${code}/src:${code}"
cuda_libraries="${env_dir}/lib/python3.11/site-packages/torch/lib"
cuda_arch="$(uname -m)"
[[ "${cuda_arch}" != "arm64" ]] || cuda_arch="aarch64"
system_cuda="/usr/local/cuda/targets/${cuda_arch}-linux/lib"
[[ ! -d "${system_cuda}" ]] || cuda_libraries="${system_cuda}:${cuda_libraries}"
for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
    lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
    [[ ! -d "${lib}" ]] || cuda_libraries="${cuda_libraries}:${lib}"
done
export LD_LIBRARY_PATH="${cuda_libraries}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

admit_validation_gpus() {
    command -v nvidia-smi >/dev/null 2>&1 || {
        printf '{"event":"glmocr_a100_decoder_lora_failed","error":"nvidia_smi_missing_before_validation"}\n' >&2
        exit 69
    }
    declare -A observed_utilization=()
    while IFS=',' read -r observed_id utilization; do
        observed_id="${observed_id//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [[ "${observed_id}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || {
            printf '{"event":"glmocr_a100_decoder_lora_failed","error":"cannot_parse_validation_gpu_utilization"}\n' >&2
            exit 69
        }
        observed_utilization["${observed_id}"]="${utilization}"
    done < <(nvidia-smi -i "${gpu_ids}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    for index in "${!steps[@]}"; do
        gpu="${gpu_array[${index}]}"
        utilization="${observed_utilization[${gpu}]-}"
        [[ "${utilization}" =~ ^[0-9]+$ ]] || {
            printf '{"event":"glmocr_a100_decoder_lora_failed","error":"validation_gpu_not_reported","gpu":"%s"}\n' "${gpu}" >&2
            exit 69
        }
        (( utilization < gpu_utilization_limit )) || {
            printf '{"event":"glmocr_a100_decoder_lora_failed","error":"validation_gpu_admission_failed","gpu":"%s","utilization":%s,"limit":%s}\n' "${gpu}" "${utilization}" "${gpu_utilization_limit}" >&2
            exit 75
        }
    done
    printf '{"event":"glmocr_a100_validation_gpu_admission_ok","gpu_ids":"%s"}\n' "${gpu_ids}"
}

common=(--seed "${seed}" --gpu-ids "${gpu_ids}" --gpu-utilization-limit "${gpu_utilization_limit}" --mode "${mode}" --num-queries "${num_queries}" --decoder-adaptation lora
    --decoder-lora-rank "${decoder_lora_rank}" --decoder-lora-alpha "${decoder_lora_alpha}"
    --decoder-lora-dropout "${decoder_lora_dropout}" --learning-rate "${learning_rate}"
    --decoder-learning-rate "${decoder_learning_rate}" --min-lr-ratio "${min_lr_ratio}"
    --global-step-offset "${global_step_offset}"
    --layout-loss-profile "${layout_loss_profile}" --auxiliary-weight "${auxiliary_weight}" \
    --auxiliary-weight-start "${auxiliary_weight}" --initial-residual-scale "${initial_residual_scale}"
    --generation-mode "${generation_mode}" --max-eval-new-tokens "${max_eval_new_tokens}"
    --free-generation-loss-weight "${free_generation_loss_weight}"
    --free-generation-max-new-tokens "${free_generation_max_new_tokens}"
    --dataset-label "${dataset_label}" --protocol-label "${protocol_label}"
    --without-test)
(( allow_count_mismatch == 1 )) && common+=(--allow-count-mismatch)
[[ -n "${init_checkpoint_dir}" ]] && common+=(--init-checkpoint-dir "${init_checkpoint_dir}")
[[ -n "${init_checkpoint_override_residual_scale}" ]] && common+=(--init-checkpoint-override-residual-scale "${init_checkpoint_override_residual_scale}")
(( init_checkpoint_allow_mode_mismatch == 1 )) && common+=(--init-checkpoint-allow-mode-mismatch)
(( free_generation_loss == 1 )) && common+=(--free-generation-loss)

current_phase="smoke"
phase running "${current_phase}"
bash "${code}/tools/training/run_glmocr_mthv2_ddp.sh" "${common[@]}" --foreground \
    --run-id "${smoke_id}" --max-steps "${smoke_steps}" --lr-schedule-steps "${smoke_steps}" \
    --warmup-steps 0 --validation-interval "$((smoke_steps + 1))" --smoke \
    --protocol-file "${root}/protocols/${smoke_id}.train_validation_no_test.json"
"${python}" - "${smoke_group}/smoke/seed${seed}/smoke_summary.json" "${auxiliary_weight}" "${free_generation_loss}" <<'PY'
import json
import math
import sys
summary = json.loads(open(sys.argv[1], encoding="utf-8").read())
expected_layout_weight = float(sys.argv[2])
free_generation_enabled = bool(int(sys.argv[3]))
if summary.get("status") != "complete" or summary.get("checkpoint_reload") is not True:
    raise SystemExit("smoke did not complete with checkpoint reload")
training = summary.get("training") or {}
if (summary.get("natural_loop_config") or {}).get("enabled") is not False:
    raise SystemExit("smoke unexpectedly enabled the natural-loop objective")
objective = training.get("loss_objective") or {}
expected_formula = (
    "L_free_generation_scaled + auxiliary_weight * L_layout"
    if free_generation_enabled
    else "L_official + auxiliary_weight * L_layout"
)
if objective.get("formula") != expected_formula:
    raise SystemExit(f"unexpected smoke loss objective: {objective}")
if not math.isclose(float(objective.get("layout_weight", -1.0)), expected_layout_weight):
    raise SystemExit(f"smoke layout loss weight is not {expected_layout_weight}")
if objective.get("extra_terms") != []:
    raise SystemExit(f"unexpected smoke extra loss terms: {objective.get('extra_terms')}")
print(json.dumps({"event": "glmocr_a100_decoder_lora_smoke_ok"}, separators=(",", ":")))
PY

if [[ "${GLMOCR_A100_STOP_AFTER_SMOKE:-0}" == "1" ]]; then
    phase complete smoke_complete
    printf '{"status":"complete","run_id":"%s","smoke_only":true,"world_size":%s}\n' \
        "${run_id}" "${world_size}"
    exit 0
fi

current_phase="training"
phase running "${current_phase}"
bash "${code}/tools/training/run_glmocr_mthv2_ddp.sh" "${common[@]}" --foreground \
    --run-id "${run_id}" --experiment-label "a100_decoder_lora_hpopt_official_layout_${mode}" \
    --max-steps "${max_steps}" --lr-schedule-steps "${lr_schedule_steps}" \
    --warmup-steps "${warmup_steps}" --validation-interval "${validation_interval}" \
    --defer-validation --protocol-file "${protocol_file}"
"${python}" - "${run_dir}/summary.json" "${run_dir}/metadata.json" "${auxiliary_weight}" "${free_generation_loss}" <<'PY'
import json
import math
import sys
summary = json.loads(open(sys.argv[1], encoding="utf-8").read())
metadata = json.loads(open(sys.argv[2], encoding="utf-8").read())
expected_layout_weight = float(sys.argv[3])
free_generation_enabled = bool(int(sys.argv[4]))
if summary.get("status") != "complete" or metadata.get("status") != "complete":
    raise SystemExit("training is not complete")
if summary.get("test_manifest_read") is not False or metadata.get("test_manifest_read") is not False:
    raise SystemExit("training protocol is not test-free")
objective = (summary.get("training") or {}).get("loss_objective") or {}
expected_formula = (
    "L_free_generation_scaled + auxiliary_weight * L_layout"
    if free_generation_enabled
    else "L_official + auxiliary_weight * L_layout"
)
if objective.get("formula") != expected_formula:
    raise SystemExit(f"unexpected training loss objective: {objective}")
if not math.isclose(float(objective.get("layout_weight", -1.0)), expected_layout_weight):
    raise SystemExit(f"training layout loss weight is not {expected_layout_weight}")
if objective.get("extra_terms") != []:
    raise SystemExit(f"unexpected training extra loss terms: {objective.get('extra_terms')}")
print(json.dumps({"event": "glmocr_a100_decoder_lora_objective_ok", "loss_objective": objective},
                 separators=(",", ":")))
PY

current_phase="parallel_validation"
phase running "${current_phase}"
mkdir -p "${validation_root}" "${group}/logs"
admit_validation_gpus
declare -a validation_pids=()
for index in "${!steps[@]}"; do
    step="${steps[${index}]}"
    gpu="${gpu_array[${index}]}"
    eval_dir="${validation_root}/step-${step}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export TMPDIR="${group}/tmp/validation-${step}"
        export HF_HOME="${TMPDIR}/huggingface"
        export TRANSFORMERS_CACHE="${HF_HOME}"
        export PYTHONPATH="${code}/src:${code}${PYTHONPATH:+:${PYTHONPATH}}"
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}"
        export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
        export CUBLAS_WORKSPACE_CONFIG=:4096:8
        export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
        mkdir -p "${TMPDIR}" "${HF_HOME}"
        cd "${code}"
        exec "${python}" -m layout_ocr.train_screen \
            --mode "${mode}" --model-path "${model_dir}" \
            --train-manifest "${train_manifest}" \
            --validation-manifest "${validation_manifest}" \
            --protocol-file "${protocol_file}" \
            --output-dir "${eval_dir}" \
            --per-device-batch-size 1 --gradient-accumulation-steps 1 \
            --max-steps "${max_steps}" --num-queries "${num_queries}" --seed "${seed}" \
            --experiment-label "a100_decoder_lora_validation_step${step}" \
            --learning-rate "${learning_rate}" --decoder-adaptation lora \
            --decoder-lora-rank "${decoder_lora_rank}" --decoder-lora-alpha "${decoder_lora_alpha}" \
            --decoder-lora-dropout "${decoder_lora_dropout}" \
            --decoder-learning-rate "${decoder_learning_rate}" \
            --warmup-steps "${warmup_steps}" --lr-schedule-steps "${lr_schedule_steps}" \
            --min-lr-ratio "${min_lr_ratio}" --residual-scale-cap 0.03 \
            --initial-residual-scale "${initial_residual_scale}" --auxiliary-weight "${auxiliary_weight}" \
            --auxiliary-weight-start "${auxiliary_weight}" \
            --auxiliary-ramp-steps 0 --gate-freeze-steps 0 --max-grad-norm "${max_grad_norm}" \
            --max-pixels 1003520 --max-eval-new-tokens "${max_eval_new_tokens}" \
            --validation-interval "${validation_interval}" --log-steps 16 \
            --adapter-precision fp32 --layout-loss-profile "${layout_loss_profile}" --query-assignment hungarian \
            --processor-mode fast --generation-mode "${generation_mode}" \
            --eval-checkpoint-dir "${run_dir}/checkpoint-${step}" --eval-only
    ) > "${group}/logs/seed${seed}.validation.step${step}.log" 2>&1 &
    validation_pids+=("$!")
done
validation_failed=0
for pid in "${validation_pids[@]}"; do
    wait "${pid}" || validation_failed=1
done
(( validation_failed == 0 )) || {
    printf '{"event":"glmocr_a100_decoder_lora_failed","error":"validation_worker_failed"}\n' >&2
    exit 1
}
"${python}" "${code}/tools/summarize_glmocr_parallel_validation.py" \
    --run-dir "${run_dir}" --validation-root "${validation_root}" \
    --group-root "${group}" --steps "$(IFS=,; echo "${steps[*]}")" \
    --expected-world-size "${world_size}" \
    --expected-layout-weight "${auxiliary_weight}" \
    --dataset-label "dunhuang_local_gazetteer_q32_v1" \
    > "${group}/logs/seed${seed}.parallel-validation.log" 2>&1

current_phase="locked_test"
phase running "${current_phase}"
test_protocol="${root}/protocols/${run_id}.test_locked.json"
if [[ ! -f "${test_protocol}" ]]; then
    test_audit_args=(
        --train-manifest "${train_manifest}" \
        --validation-manifest "${validation_manifest}" \
        --test-manifest "${dataset}/test/manifest.jsonl" \
        --num-queries "${num_queries}" --dataset-label "${dataset_label}" \
        --protocol-label "${protocol_label}" --output "${test_protocol}"
    )
    (( allow_count_mismatch == 1 )) && test_audit_args+=(--allow-count-mismatch)
    "${python}" "${code}/tools/audit_mthv2_manifest.py" "${test_audit_args[@]}" \
        > "${group}/logs/test-protocol-audit.log" 2>&1
fi
bash "${code}/tools/training/run_glmocr_mthv2_locked_test.sh" \
    --run-id "${run_id}" --seed "${seed}" --gpu-ids "${gpu_ids}" --foreground \
    --mode "${mode}" --num-queries "${num_queries}" \
    --max-eval-new-tokens "${max_eval_new_tokens}" --protocol-file "${test_protocol}"

phase complete complete
printf '{"status":"complete","run_id":"%s","world_size":%s,"max_steps":%s,"learning_rate":%s,"decoder_learning_rate":%s,"auxiliary_weight":%s,"warmup_steps":%s,"max_grad_norm":%s,"natural_loop_loss":false,"checkpoint_steps":"%s","test_used_for_selection":false}\n' \
    "${run_id}" "${world_size}" "${max_steps}" "${learning_rate}" "${decoder_learning_rate}" "${auxiliary_weight}" "${warmup_steps}" "${max_grad_norm}" "$(IFS=,; echo "${steps[*]}")"
