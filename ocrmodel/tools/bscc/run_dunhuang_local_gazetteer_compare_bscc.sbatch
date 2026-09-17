#!/usr/bin/env bash
# BSCC q32 comparison: GLMOCR geometry versus the official content-only
# baseline.  The job owns four GPUs and runs smoke, train, validation-only
# selection, and selection-locked test without reading test during train/val.
#
# The historical shared launcher names mention MTHv2; this entry supplies the
# Dunhuang/local-gazetteer q32 compatibility view explicitly and never uses
# the MTHv2 data.
#SBATCH --job-name=glmocr_q32_cmp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=240:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/glm_ocr_layout_ot"
code_root="${experiment_root}/code/ocrmodel"
env_dir="${experiment_root}/envs/glm-ocr-py311-cu128"
nvidia_env="${BSCC_NVIDIA_ENV:-${env_dir}}"
model_dir="${workspace}/models/GLM-OCR-ca5d8b3"
dataset_parent="${BSCC_Q32_DATASET_PARENT:-${workspace}/datasets}"
archive="${BSCC_Q32_ARCHIVE:-${dataset_parent}/dunhuang_local_gazetteer_q32_v1_portable.zip}"
portable_root="${BSCC_Q32_PORTABLE_ROOT:-${dataset_parent}/dunhuang_local_gazetteer_q32_v1_portable}"
dataset_root="${BSCC_Q32_DATASET_ROOT:-${dataset_parent}/dunhuang_local_gazetteer_q32_v1/glmocr_compat}"
pytorch_package="/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311"

arm_mode="${GLMOCR_BSCC_Q32_MODE:-compare}"
session="${GLMOCR_BSCC_Q32_SESSION:-glmocr_dunhuang_local_gazetteer_q32_compare_bscc_260913_v1}"
geometry_run_id="${GLMOCR_BSCC_Q32_GEOMETRY_RUN_ID:-glmocr_dunhuang_local_gazetteer_q32_geometry_2k_bscc_260913_v1}"
baseline_run_id="${GLMOCR_BSCC_Q32_BASELINE_RUN_ID:-glmocr_dunhuang_local_gazetteer_q32_official_content_only_2k_bscc_260913_v1}"
if [[ "${arm_mode}" == "geometry" ]]; then
    session="${GLMOCR_BSCC_Q32_SESSION:-glmocr_dunhuang_local_gazetteer_q32_geometry_bscc_260913_v3}"
    geometry_run_id="${GLMOCR_BSCC_Q32_RUN_ID:-glmocr_dunhuang_local_gazetteer_q32_geometry_2k_bscc_260913_v3}"
elif [[ "${arm_mode}" == "content_only" ]]; then
    session="${GLMOCR_BSCC_Q32_SESSION:-glmocr_dunhuang_local_gazetteer_q32_baseline_bscc_260913_v3}"
    baseline_run_id="${GLMOCR_BSCC_Q32_RUN_ID:-glmocr_dunhuang_local_gazetteer_q32_official_content_only_2k_bscc_260913_v3}"
fi
resume_geometry_run_id="${GLMOCR_BSCC_Q32_RESUME_GEOMETRY_RUN_ID:-}"
seed="${GLMOCR_BSCC_SEED:-42}"
gpu_ids="${GLMOCR_BSCC_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0,1,2,3}}}"
num_queries=32
max_steps="${GLMOCR_BSCC_Q32_MAX_STEPS:-2000}"
lr_schedule_steps="${GLMOCR_BSCC_Q32_LR_SCHEDULE_STEPS:-${max_steps}}"
validation_interval="${GLMOCR_BSCC_Q32_VALIDATION_INTERVAL:-500}"
checkpoint_steps_csv="${GLMOCR_BSCC_Q32_CHECKPOINT_STEPS:-500,1000,1500,2000}"
# Slurm --export uses commas as variable separators.  The submitter therefore
# uses colon-separated checkpoint values and normalizes them here.
checkpoint_steps_csv="${checkpoint_steps_csv//:/,}"
max_eval_new_tokens="${GLMOCR_BSCC_Q32_MAX_EVAL_NEW_TOKENS:-1536}"
generation_mode="${GLMOCR_BSCC_Q32_GENERATION_MODE:-loop_recovery}"
stop_after_validation="${GLMOCR_BSCC_Q32_STOP_AFTER_VALIDATION:-0}"
learning_rate=2.5e-5
decoder_learning_rate=5e-6
warmup_steps="${GLMOCR_BSCC_Q32_WARMUP_STEPS:-216}"
min_lr_ratio=0.1
gpu_utilization_limit=50
dataset_label="dunhuang_local_gazetteer_q32_v1"
protocol_label="glm_ocr_dunhuang_local_gazetteer_group_isolated_v1"
expected_test_manifest_sha256="da94dbd7e372e5e206c7c340be881e40f11eabff1b12df26668f9a983e7f782d"
expected_test_pages=59

case "${arm_mode}" in
    compare)
        active_mode=""
        active_auxiliary_weight=""
        active_run_id=""
        ;;
    geometry)
        active_mode="geometry"
        active_auxiliary_weight="0.4"
        active_run_id="${geometry_run_id}"
        resume_geometry_run_id=""
        ;;
    content_only)
        active_mode="content_only"
        active_auxiliary_weight="0.0"
        active_run_id="${baseline_run_id}"
        resume_geometry_run_id=""
        ;;
    *)
        printf '{"event":"glmocr_bscc_q32_failed","error":"invalid_arm_mode","mode":"%s"}\n' "${arm_mode}" >&2
        exit 64
        ;;
esac

python="${env_dir}/bin/python"
ddp_launcher="${code_root}/tools/training/run_glmocr_mthv2_ddp.sh"
validation_summarizer="${code_root}/tools/summarize_glmocr_parallel_validation.py"
metrics_summarizer="${code_root}/tools/summarize_glmocr_final_metrics.py"
manifest_auditor="${code_root}/tools/audit_mthv2_manifest.py"
locked_test_launcher="${code_root}/tools/training/run_glmocr_mthv2_locked_test.sh"

status_file="${workspace}/runs/${session}.status.json"
summary_file="${workspace}/runs/${session}.summary.json"
comparison_metrics_file="${workspace}/runs/${session}.metrics_summary.json"
current_phase="preflight"

IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
gpu_count="${#gpu_array[@]}"

[[ "${session}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
[[ "${geometry_run_id}" =~ ^[A-Za-z0-9_.-]+$ && "${baseline_run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
[[ -z "${resume_geometry_run_id}" || "${resume_geometry_run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
[[ -z "${resume_geometry_run_id}" || "${resume_geometry_run_id}" == "${geometry_run_id}" ]] || exit 64
[[ "${geometry_run_id}" != "${baseline_run_id}" ]] || exit 64
[[ "${seed}" =~ ^[0-9]+$ && "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]] || exit 64
[[ "${max_steps}" =~ ^[1-9][0-9]*$ && "${lr_schedule_steps}" =~ ^[1-9][0-9]*$ ]] || exit 64
[[ "${validation_interval}" =~ ^[1-9][0-9]*$ && "${checkpoint_steps_csv}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || exit 64
[[ "${max_eval_new_tokens}" =~ ^[1-9][0-9]*$ && "${stop_after_validation}" =~ ^[01]$ && "${warmup_steps}" =~ ^[0-9]+$ ]] || exit 64
[[ "${generation_mode}" == "plain" || "${generation_mode}" == "loop_recovery" ]] || exit 64
(( gpu_count == 4 )) || exit 64
declare -A seen_gpu=()
for gpu in "${gpu_array[@]}"; do
    [[ -z "${seen_gpu[${gpu}]+present}" ]] || exit 64
    seen_gpu["${gpu}"]=1
done

write_status() {
    local status="$1"
    local phase="$2"
    local run_id="${3:--}"
    local mode="${4:--}"
    mkdir -p "${workspace}/runs"
    printf '{"status":"%s","phase":"%s","run_id":"%s","mode":"%s","session":"%s","dataset":"%s","gpu_ids":"%s","world_size":4,"updated_at":"%s","test_used_for_selection":false}\n' \
        "${status}" "${phase}" "${run_id}" "${mode}" "${session}" \
        "${dataset_label}" "${gpu_ids}" "$(date -u +%FT%TZ)" > "${status_file}"
}

on_error() {
    local rc=$?
    trap - ERR
    write_status failed "${current_phase}_failed" "-" "-" || true
    exit "${rc}"
}
trap on_error ERR

prepare_dataset_view() {
    mkdir -p "${dataset_parent}"
    if [[ ! -d "${portable_root}" ]]; then
        [[ -f "${archive}" ]] || exit 66
        staging="${dataset_parent}/.dunhuang_local_gazetteer_q32_v1_extract"
        [[ ! -e "${staging}" ]] || exit 74
        mkdir -p "${staging}"
        unzip -q "${archive}" -d "${staging}"
        extracted="${staging}/dunhuang_local_gazetteer_q32_v1_portable"
        [[ -d "${extracted}" && ! -e "${portable_root}" ]] || exit 66
        mv -- "${extracted}" "${portable_root}"
    fi
    for split in train validation test; do
        [[ -f "${portable_root}/manifests/${split}.jsonl" ]] || exit 66
    done
    [[ -f "${portable_root}/protocol.json" && -d "${portable_root}/images" ]] || exit 66

    mkdir -p "${dataset_root}"
    for split in train validation test; do
        split_dir="${dataset_root}/${split}"
        manifest="${split_dir}/manifest.jsonl"
        mkdir -p "${split_dir}"
        if [[ -f "${manifest}" ]]; then
            cmp -s "${portable_root}/manifests/${split}.jsonl" "${manifest}" || exit 74
        else
            cp -- "${portable_root}/manifests/${split}.jsonl" "${manifest}"
        fi
        image_link="${split_dir}/images"
        if [[ -e "${image_link}" && ! -L "${image_link}" ]]; then
            exit 74
        fi
        if [[ ! -L "${image_link}" ]]; then
            ln -s "${portable_root}/images" "${image_link}"
        fi
        [[ "$(readlink -f -- "${image_link}")" == "$(readlink -f -- "${portable_root}/images")" ]] || exit 74
    done
}

preflight() {
    [[ -f "${pytorch_package}/env.sh" && -x "${python}" && -x "${env_dir}/bin/torchrun" ]] || exit 66
    [[ -f "${model_dir}/model.safetensors" && -f "${model_dir}/config.json" ]] || exit 66
    for path in "${ddp_launcher}" "${validation_summarizer}" "${metrics_summarizer}" "${manifest_auditor}" "${locked_test_launcher}"; do
        [[ -f "${path}" ]] || exit 66
    done
    [[ -f "${dataset_root}/train/manifest.jsonl" && -f "${dataset_root}/validation/manifest.jsonl" && -f "${dataset_root}/test/manifest.jsonl" ]] || exit 66
    [[ ! -e "${status_file}" && ! -e "${summary_file}" && ! -e "${comparison_metrics_file}" ]] || exit 74
    if [[ "${arm_mode}" != "compare" ]]; then
        [[ -n "${active_run_id}" ]] || exit 64
        for run_id in "${active_run_id}" "${active_run_id}_smoke"; do
            [[ ! -e "${experiment_root}/training_runs/${run_id}" ]] || exit 74
        done
    elif [[ -n "${resume_geometry_run_id}" ]]; then
        [[ -f "${experiment_root}/training_runs/${geometry_run_id}/seed${seed}/summary.json" ]] || exit 66
        [[ -f "${experiment_root}/training_runs/${geometry_run_id}/seed${seed}/metadata.json" ]] || exit 66
        [[ ! -e "${experiment_root}/training_runs/${geometry_run_id}/seed${seed}/selection.json" ]] || exit 74
        [[ ! -e "${experiment_root}/training_runs/${geometry_run_id}/seed${seed}/parallel-validation" ]] || exit 74
        [[ ! -e "${experiment_root}/training_runs/${baseline_run_id}" ]] || exit 74
    else
        for run_id in "${geometry_run_id}" "${geometry_run_id}_smoke" "${baseline_run_id}" "${baseline_run_id}_smoke"; do
            [[ ! -e "${experiment_root}/training_runs/${run_id}" ]] || exit 74
        done
    fi
    mkdir -p "${workspace}/runs" "${experiment_root}/protocols" "${experiment_root}/tmp"
}

source_environment() {
    source "${pytorch_package}/env.sh"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    set +u
    conda activate "${env_dir}"
    set -u
    export PYTHONNOUSERSITE=1
    export TOKENIZERS_PARALLELISM=false
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    export CUBLAS_WORKSPACE_CONFIG=:4096:8
    export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
    export GLMOCR_A100_NVIDIA_ENV="${nvidia_env}"
}

common_ddp_args() {
    local mode="$1"
    local auxiliary_weight="$2"
    local run_id="$3"
    local protocol_file="$4"
    printf '%s\n' \
        --foreground --run-id "${run_id}" --seed "${seed}" --gpu-ids "${gpu_ids}" \
        --gpu-utilization-limit "${gpu_utilization_limit}" --remote-root "${experiment_root}" \
        --code-root "${code_root}" --env-dir "${env_dir}" --model-dir "${model_dir}" \
        --dataset-root "${dataset_root}" --protocol-file "${protocol_file}" \
        --dataset-label "${dataset_label}" --protocol-label "${protocol_label}" \
        --allow-count-mismatch --mode "${mode}" --num-queries "${num_queries}" \
        --learning-rate "${learning_rate}" --decoder-adaptation lora \
        --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
        --decoder-learning-rate "${decoder_learning_rate}" --min-lr-ratio "${min_lr_ratio}" \
        --initial-residual-scale 0 --gate-freeze-steps 0 \
        --auxiliary-weight "${auxiliary_weight}" --auxiliary-weight-start "${auxiliary_weight}" \
        --auxiliary-ramp-steps 0 --layout-loss-profile full \
        --generation-mode "${generation_mode}" --max-eval-new-tokens "${max_eval_new_tokens}" \
        --log-steps 16
}

run_smoke() {
    local mode="$1"
    local auxiliary_weight="$2"
    local run_id="$3"
    local smoke_id="${run_id}_smoke"
    local protocol_file="${experiment_root}/protocols/${smoke_id}.train_validation_no_test.json"
    local log_path="${workspace}/runs/${smoke_id}.sbatch.log"
    current_phase="${mode}_smoke"
    write_status running "${current_phase}" "${smoke_id}" "${mode}"
    mapfile -t args < <(common_ddp_args "${mode}" "${auxiliary_weight}" "${smoke_id}" "${protocol_file}")
    args+=(--max-steps 8 --lr-schedule-steps 8 --warmup-steps 0 --validation-interval 9 --without-test --smoke)
    set +e
    bash "${ddp_launcher}" "${args[@]}" > "${log_path}" 2>&1
    local rc=$?
    set -e
    (( rc == 0 )) || exit "${rc}"
    "${python}" - "${experiment_root}/training_runs/${smoke_id}/smoke/seed${seed}/smoke_summary.json" "${auxiliary_weight}" <<'PY'
import json
import math
import sys

summary = json.loads(open(sys.argv[1], encoding="utf-8").read())
if summary.get("status") != "complete" or summary.get("checkpoint_reload") is not True:
    raise SystemExit("smoke did not complete with checkpoint reload")
if summary.get("test_used_for_selection") is not False:
    raise SystemExit("smoke is not test-free")
training = summary.get("training") or {}
if (training.get("natural_loop") or {}).get("enabled") is not False:
    raise SystemExit("smoke unexpectedly enabled natural-loop objective")
objective = training.get("loss_objective") or {}
if objective.get("formula") != "L_official + auxiliary_weight * L_layout":
    raise SystemExit(f"unexpected smoke objective: {objective}")
if not math.isclose(float(objective.get("layout_weight", -1.0)), float(sys.argv[2])):
    raise SystemExit("smoke layout weight mismatch")
if objective.get("extra_terms") != []:
    raise SystemExit("smoke has unexpected extra loss terms")
print(json.dumps({"event": "glmocr_bscc_q32_smoke_ok"}, separators=(",", ":")))
PY
}

run_training() {
    local mode="$1"
    local auxiliary_weight="$2"
    local run_id="$3"
    local protocol_file="${experiment_root}/protocols/${run_id}.train_validation_no_test.json"
    local log_path="${workspace}/runs/${run_id}.sbatch.log"
    current_phase="${mode}_training"
    write_status running "${current_phase}" "${run_id}" "${mode}"
    mapfile -t args < <(common_ddp_args "${mode}" "${auxiliary_weight}" "${run_id}" "${protocol_file}")
    args+=(--max-steps "${max_steps}" --lr-schedule-steps "${lr_schedule_steps}" --warmup-steps "${warmup_steps}" \
        --validation-interval "${validation_interval}" --defer-validation --without-test)
    set +e
    bash "${ddp_launcher}" "${args[@]}" > "${log_path}" 2>&1
    local rc=$?
    set -e
    (( rc == 0 )) || exit "${rc}"
    "${python}" - "${experiment_root}/training_runs/${run_id}/seed${seed}/summary.json" \
        "${experiment_root}/training_runs/${run_id}/seed${seed}/metadata.json" "${mode}" "${auxiliary_weight}" "${checkpoint_steps_csv}" "${max_steps}" "${lr_schedule_steps}" "${generation_mode}" "${max_eval_new_tokens}" "${warmup_steps}" <<'PY'
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
metadata = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
mode = sys.argv[3]
expected_weight = float(sys.argv[4])
expected_steps = [int(value) for value in sys.argv[5].split(",")]
expected_max_steps = int(sys.argv[6])
expected_lr_schedule_steps = int(sys.argv[7])
expected_generation_mode = sys.argv[8]
expected_max_eval_new_tokens = int(sys.argv[9])
expected_warmup_steps = int(sys.argv[10])
if summary.get("status") != "complete" or metadata.get("status") != "complete":
    raise SystemExit("training is not complete")
if summary.get("test_manifest_read") is not False or metadata.get("test_manifest_read") is not False:
    raise SystemExit("training is not test-free")
if summary.get("selection_pending") is not True:
    raise SystemExit("training did not defer validation selection")
if metadata.get("mode") != mode or metadata.get("world_size") != 4 or metadata.get("num_queries") != 32:
    raise SystemExit("training metadata mismatch")
if metadata.get("decoder_adaptation") != "lora":
    raise SystemExit("training did not use decoder LoRA")
training = summary.get("training") or {}
if not math.isclose(float(training.get("decoder_learning_rate", -1.0)), 5e-6):
    raise SystemExit("decoder LoRA learning rate mismatch")
if [int(step) for step in training.get("checkpoint_steps", [])] != expected_steps:
    raise SystemExit("checkpoint steps mismatch")
if metadata.get("max_steps") != expected_max_steps or metadata.get("lr_schedule_steps") != expected_lr_schedule_steps:
    raise SystemExit("training schedule mismatch")
if metadata.get("warmup_steps") != expected_warmup_steps:
    raise SystemExit("warmup schedule mismatch")
if metadata.get("generation_mode") != expected_generation_mode or metadata.get("max_eval_new_tokens") != expected_max_eval_new_tokens:
    raise SystemExit("generation configuration mismatch")
if (training.get("natural_loop") or {}).get("enabled") is not False:
    raise SystemExit("natural-loop objective is enabled")
for name in ("scheduled_sampling", "loop_escape", "continuation_head"):
    if (training.get(name) or {}).get("enabled") is not False:
        raise SystemExit(f"unexpected extra objective: {name}")
objective = training.get("loss_objective") or {}
if objective.get("formula") != "L_official + auxiliary_weight * L_layout":
    raise SystemExit(f"unexpected training objective: {objective}")
if not math.isclose(float(objective.get("layout_weight", -1.0)), expected_weight):
    raise SystemExit("training layout weight mismatch")
if objective.get("extra_terms") != []:
    raise SystemExit("training has unexpected extra loss terms")
print(json.dumps({"event": "glmocr_bscc_q32_training_ok", "mode": mode, "world_size": 4, "checkpoint_steps": expected_steps}, separators=(",", ":")))
PY
}

verify_existing_geometry() {
    local run_id="$1"
    local mode="$2"
    local auxiliary_weight="$3"
    "${python}" - "${experiment_root}/training_runs/${run_id}/seed${seed}/summary.json" \
        "${experiment_root}/training_runs/${run_id}/seed${seed}/metadata.json" "${mode}" "${auxiliary_weight}" <<'PY'
import json
import math
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
metadata = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
mode = sys.argv[3]
expected_weight = float(sys.argv[4])
if summary.get("status") != "complete" or metadata.get("status") != "complete":
    raise SystemExit("existing geometry training is not complete")
if summary.get("test_manifest_read") is not False or metadata.get("test_manifest_read") is not False:
    raise SystemExit("existing geometry training is not test-free")
if summary.get("selection_pending") is not True or summary.get("selection_performed") is not False:
    raise SystemExit("existing geometry selection state is not resumable")
if metadata.get("mode") != mode or metadata.get("world_size") != 4 or metadata.get("num_queries") != 32:
    raise SystemExit("existing geometry metadata mismatch")
if metadata.get("decoder_adaptation") != "lora":
    raise SystemExit("existing geometry did not use decoder LoRA")
training = summary.get("training") or {}
if not math.isclose(float(training.get("decoder_learning_rate", -1.0)), 5e-6):
    raise SystemExit("existing geometry decoder LoRA learning rate mismatch")
if [int(step) for step in training.get("checkpoint_steps", [])] != [500, 1000, 1500, 2000]:
    raise SystemExit("existing geometry checkpoint steps mismatch")
if any(item.get("checkpoint_finite") is not True for item in training.get("checkpoint_health", [])):
    raise SystemExit("existing geometry checkpoint health is not finite")
if (training.get("natural_loop") or {}).get("enabled") is not False:
    raise SystemExit("existing geometry enabled natural-loop objective")
for name in ("scheduled_sampling", "loop_escape", "continuation_head"):
    if (training.get(name) or {}).get("enabled") is not False:
        raise SystemExit(f"existing geometry has unexpected extra objective: {name}")
objective = training.get("loss_objective") or {}
if objective.get("formula") != "L_official + auxiliary_weight * L_layout":
    raise SystemExit(f"existing geometry objective mismatch: {objective}")
if not math.isclose(float(objective.get("layout_weight", -1.0)), expected_weight):
    raise SystemExit("existing geometry layout weight mismatch")
if objective.get("extra_terms") != []:
    raise SystemExit("existing geometry has unexpected extra objective terms")
print(json.dumps({"event": "glmocr_bscc_q32_existing_geometry_ok", "run_id": str(Path(sys.argv[1]).parents[1]), "checkpoint_steps": [500, 1000, 1500, 2000]}, separators=(",", ":")))
PY
}

admit_gpus() {
    command -v nvidia-smi >/dev/null 2>&1 || exit 69
    declare -A observed=()
    while IFS=',' read -r observed_id utilization; do
        observed_id="${observed_id//[[:space:]]/}"
        utilization="${utilization//[[:space:]]/}"
        [[ "${observed_id}" =~ ^[0-9]+$ && "${utilization}" =~ ^[0-9]+$ ]] || exit 69
        observed["${observed_id}"]="${utilization}"
    done < <(nvidia-smi -i "${gpu_ids}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)
    for gpu in "${gpu_array[@]}"; do
        [[ -n "${observed[${gpu}]+present}" ]] || exit 69
        (( observed[${gpu}] < gpu_utilization_limit )) || exit 75
    done
}

validation_cuda_library_path() {
    local torch_lib="${env_dir}/lib/python3.11/site-packages/torch/lib"
    local machine_arch="$(uname -m)"
    local cuda_target_arch="${machine_arch}"
    case "${machine_arch}" in
        x86_64) cuda_target_arch="x86_64" ;;
        aarch64|arm64) cuda_target_arch="aarch64" ;;
    esac
    local value="${torch_lib}"
    local system_cuda="/usr/local/cuda/targets/${cuda_target_arch}-linux/lib"
    [[ -d "${system_cuda}" ]] && value="${system_cuda}:${value}"
    for component in cudnn nccl cuda_nvrtc cuda_cupti cufft curand cusparse cusolver nvtx nvjitlink; do
        component_lib="${nvidia_env}/lib/python3.11/site-packages/nvidia/${component}/lib"
        [[ -d "${component_lib}" ]] && value="${value}:${component_lib}"
    done
    printf '%s\n' "${value}"
}

run_validation() {
    local mode="$1"
    local auxiliary_weight="$2"
    local run_id="$3"
    local group_root="${experiment_root}/training_runs/${run_id}"
    local run_dir="${group_root}/seed${seed}"
    local validation_root="${run_dir}/parallel-validation"
    local protocol_file="${experiment_root}/protocols/${run_id}.train_validation_no_test.json"
    local train_manifest="${dataset_root}/train/manifest.jsonl"
    local validation_manifest="${dataset_root}/validation/manifest.jsonl"
    [[ ! -e "${run_dir}/selection.json" && ! -e "${validation_root}" ]] || exit 74
    [[ -f "${protocol_file}" ]] || exit 66
    current_phase="${mode}_validation"
    write_status running "${current_phase}" "${run_id}" "${mode}"
    admit_gpus
    mkdir -p "${validation_root}" "${group_root}/logs" "${group_root}/tmp"
    local cuda_library_path
    cuda_library_path="$(validation_cuda_library_path)"
    local steps=()
    IFS=',' read -r -a steps <<< "${checkpoint_steps_csv}"
    (( ${#steps[@]} <= gpu_count )) || exit 64
    local pids=()
    for index in "${!steps[@]}"; do
        local step="${steps[${index}]}"
        local gpu="${gpu_array[${index}]}"
        local eval_dir="${validation_root}/step-${step}"
        local eval_tmp="${group_root}/tmp/validation-step${step}"
        local eval_log="${group_root}/logs/validation-step${step}.log"
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            export TMPDIR="${eval_tmp}"
            export HF_HOME="${TMPDIR}/huggingface"
            export TRANSFORMERS_CACHE="${HF_HOME}"
            export PYTHONPATH="${code_root}/src:${code_root}${PYTHONPATH:+:${PYTHONPATH}}"
            export LD_LIBRARY_PATH="${cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
            export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
            export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
            export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
            mkdir -p "${TMPDIR}" "${HF_HOME}"
            cd "${code_root}"
            exec "${python}" -m layout_ocr.train_screen \
                --mode "${mode}" --model-path "${model_dir}" \
                --train-manifest "${train_manifest}" --validation-manifest "${validation_manifest}" \
                --protocol-file "${protocol_file}" --output-dir "${eval_dir}" \
                --per-device-batch-size 1 --gradient-accumulation-steps 1 \
                --max-steps "${max_steps}" --num-queries "${num_queries}" --seed "${seed}" \
                --experiment-label "bscc_q32_validation_${mode}_step${step}" \
                --learning-rate "${learning_rate}" --decoder-adaptation lora \
                --decoder-lora-rank 8 --decoder-lora-alpha 8 --decoder-lora-dropout 0 \
                --decoder-learning-rate "${decoder_learning_rate}" \
                --warmup-steps 0 --lr-schedule-steps "${lr_schedule_steps}" --min-lr-ratio "${min_lr_ratio}" \
                --residual-scale-cap 0.03 --initial-residual-scale 0 \
                --auxiliary-weight "${auxiliary_weight}" --auxiliary-weight-start "${auxiliary_weight}" \
                --auxiliary-ramp-steps 0 --gate-freeze-steps 0 --max-grad-norm 1.0 \
                --max-pixels 1003520 --max-eval-new-tokens "${max_eval_new_tokens}" \
                --validation-interval "${validation_interval}" --log-steps 16 \
                --adapter-precision fp32 --layout-loss-profile full --query-assignment hungarian \
                --processor-mode fast --generation-mode "${generation_mode}" \
                --eval-checkpoint-dir "${run_dir}/checkpoint-${step}" --eval-only
        ) > "${eval_log}" 2>&1 &
        pids+=("$!")
    done
    local failed=0
    for pid in "${pids[@]}"; do
        wait "${pid}" || failed=1
    done
    (( failed == 0 )) || exit 1
    "${python}" "${validation_summarizer}" \
        --run-dir "${run_dir}" --validation-root "${validation_root}" --group-root "${group_root}" \
        --steps "${checkpoint_steps_csv}" --expected-world-size 4 \
        --expected-layout-weight "${auxiliary_weight}" --dataset-label "${dataset_label}" \
        > "${group_root}/logs/validation-summary.log" 2>&1
    "${python}" - "${run_dir}/selection.json" "${run_dir}/parallel_validation_summary.json" "${checkpoint_steps_csv}" <<'PY'
import json
import sys
selection = json.loads(open(sys.argv[1], encoding="utf-8").read())
parallel = json.loads(open(sys.argv[2], encoding="utf-8").read())
if selection.get("status") != "complete" or parallel.get("status") != "complete":
    raise SystemExit("validation selection is incomplete")
expected_steps = [int(value) for value in sys.argv[3].split(",")]
if selection.get("selected_step") not in expected_steps:
    raise SystemExit("invalid selected step")
if selection.get("test_manifest_read") is not False or selection.get("test_used_for_selection") is not False:
    raise SystemExit("validation selection is not test-free")
print(json.dumps({"event": "glmocr_bscc_q32_validation_ok", "selected_step": selection["selected_step"], "test_used_for_selection": False}, separators=(",", ":")))
PY
}

make_test_protocol() {
    local run_id="$1"
    local protocol_file="${experiment_root}/protocols/${run_id}.test_locked.json"
    [[ ! -e "${protocol_file}" ]] || exit 74
    "${python}" "${manifest_auditor}" \
        --train-manifest "${dataset_root}/train/manifest.jsonl" \
        --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
        --test-manifest "${dataset_root}/test/manifest.jsonl" \
        --num-queries "${num_queries}" --dataset-label "${dataset_label}" \
        --protocol-label "${protocol_label}" --allow-count-mismatch --output "${protocol_file}" \
        > "${workspace}/runs/${run_id}.test-protocol-audit.log" 2>&1
    "${python}" - "${protocol_file}" "${dataset_root}/test/manifest.jsonl" "${expected_test_manifest_sha256}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

protocol = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
test_manifest = Path(sys.argv[2])
expected_sha = sys.argv[3]
if protocol.get("dataset") != "dunhuang_local_gazetteer_q32_v1":
    raise SystemExit("wrong dataset label")
if protocol.get("protocol") != "glm_ocr_dunhuang_local_gazetteer_group_isolated_v1":
    raise SystemExit("wrong protocol label")
if protocol.get("split_pages") != {"train": 240, "validation": 80, "test": 59}:
    raise SystemExit(f"unexpected split counts: {protocol.get('split_pages')}")
if protocol.get("num_queries") != 32 or protocol.get("test_manifest_read") is not True:
    raise SystemExit("invalid test protocol boundary")
if protocol.get("test_used_for_selection") is not False:
    raise SystemExit("test protocol is not selection-locked")
digest = hashlib.sha256(test_manifest.read_bytes()).hexdigest()
if digest != expected_sha:
    raise SystemExit(f"test manifest fingerprint mismatch: {digest}")
recorded = ((protocol.get("manifest_stats") or {}).get("test") or {}).get("manifest_sha256")
if recorded != digest:
    raise SystemExit("test protocol fingerprint mismatch")
print(json.dumps({"event": "glmocr_bscc_q32_test_protocol_ok", "test_manifest_sha256": digest, "test_pages": 59}, separators=(",", ":")))
PY
}

run_test() {
    local mode="$1"
    local run_id="$2"
    local protocol_file="${experiment_root}/protocols/${run_id}.test_locked.json"
    local run_dir="${experiment_root}/training_runs/${run_id}/seed${seed}"
    local log_path="${workspace}/runs/${run_id}.locked-test.sbatch.log"
    current_phase="${mode}_locked_test"
    write_status running "${current_phase}" "${run_id}" "${mode}"
    set +e
    bash "${locked_test_launcher}" --foreground \
        --run-id "${run_id}" --seed "${seed}" --gpu-ids "${gpu_ids}" --mode "${mode}" \
        --num-queries "${num_queries}" --max-eval-new-tokens "${max_eval_new_tokens}" \
        --remote-root "${experiment_root}" --code-root "${code_root}" --env-dir "${env_dir}" \
        --model-dir "${model_dir}" --dataset-root "${dataset_root}" --protocol-file "${protocol_file}" \
        > "${log_path}" 2>&1
    local rc=$?
    set -e
    (( rc == 0 )) || exit "${rc}"
    "${python}" - "${run_dir}/locked-test/locked_test_summary.json" "${run_dir}/selection.json" "${mode}" "${gpu_ids}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
selection = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
if summary.get("status") != "complete" or summary.get("split") != "test":
    raise SystemExit("locked test is incomplete")
if summary.get("mode") != sys.argv[3] or summary.get("test_pages") != 59 or summary.get("test_pages_total") != 59:
    raise SystemExit("locked test coverage or mode mismatch")
if summary.get("test_shard_count") != 4 or summary.get("test_shard_gpu_ids") != sys.argv[4]:
    raise SystemExit("locked test did not use the four requested GPUs")
if summary.get("test_used_for_selection") is not False:
    raise SystemExit("locked test is not selection-locked")
if summary.get("selected_step") != selection.get("selected_step"):
    raise SystemExit("locked test step differs from validation selection")
if summary.get("decoder_adaptation") != "lora" or summary.get("decoder_lora_loaded") is not True:
    raise SystemExit("decoder LoRA was not loaded")
if (summary.get("metrics") or {}).get("pages") != 59:
    raise SystemExit("locked test metrics do not cover all test pages")
print(json.dumps({"event": "glmocr_bscc_q32_locked_test_ok", "selected_step": summary["selected_step"], "test_pages": 59}, separators=(",", ":")))
PY
    "${python}" "${metrics_summarizer}" \
        --manifest "${dataset_root}/test/manifest.jsonl" \
        --predictions "${run_dir}/locked-test/test_predictions.jsonl" \
        --output "${run_dir}/locked-test/final_metrics.json" --expected-pages "${expected_test_pages}" \
        > "${workspace}/runs/${run_id}.final-metrics.log" 2>&1
}

write_comparison_summary() {
    "${python}" - "${comparison_metrics_file}" \
        "${geometry_run_id}" "${baseline_run_id}" "${experiment_root}" "${dataset_root}" \
        "${session}" "${gpu_ids}" "${expected_test_manifest_sha256}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
geometry_run_id, baseline_run_id, experiment_root, dataset_root, session, gpu_ids, test_sha = sys.argv[2:]
def load(run_id):
    root = Path(experiment_root) / "training_runs" / run_id / "seed42"
    test_summary = json.loads((root / "locked-test" / "locked_test_summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "locked-test" / "final_metrics.json").read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "mode": test_summary.get("mode"),
        "selected_step": test_summary.get("selected_step"),
        "test_summary": str(root / "locked-test" / "locked_test_summary.json"),
        "metrics": metrics,
        "test_used_for_selection": test_summary.get("test_used_for_selection"),
    }
geometry = load(geometry_run_id)
baseline = load(baseline_run_id)
if geometry["metrics"].get("raw_reference_characters") != 14475 or baseline["metrics"].get("raw_reference_characters") != 14475:
    raise SystemExit("reference character count does not match RESULTS.md protocol")
payload = {
    "status": "complete",
    "session": session,
    "dataset": "dunhuang_local_gazetteer_q32_v1",
    "split": "test",
    "test_pages": 59,
    "gpu_ids": gpu_ids,
    "num_queries": 32,
    "max_eval_new_tokens": 1536,
    "test_manifest_sha256": test_sha,
    "test_used_for_selection": False,
    "external_comparison_reference": {
        "results_file": "D:/yangky/学推计划-glm-ocr/results/sota_zeroshot_dunhuang_local_q32_test_20260913_v3_max1536/RESULTS.md",
        "input": "whole_page_image + fixed OCR prompt",
        "generation": "do_sample=false, max_output_tokens=1536",
        "test_pages": 59,
    },
    "models": {
        "geometry": geometry,
        "official_content_only": baseline,
    },
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"event": "glmocr_bscc_q32_comparison_complete", "summary": str(output), "test_used_for_selection": False}, separators=(",", ":")))
PY
}

write_single_summary() {
    "${python}" - "${summary_file}" "${active_run_id}" "${active_mode}" "${experiment_root}" "${session}" "${gpu_ids}" "${expected_test_manifest_sha256}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
run_id, mode, experiment_root, session, gpu_ids, test_sha = sys.argv[2:]
root = Path(experiment_root) / "training_runs" / run_id / "seed42"
selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
test_summary = json.loads((root / "locked-test" / "locked_test_summary.json").read_text(encoding="utf-8"))
metrics = json.loads((root / "locked-test" / "final_metrics.json").read_text(encoding="utf-8"))
if selection.get("test_used_for_selection") is not False or test_summary.get("test_used_for_selection") is not False:
    raise SystemExit("single-arm pipeline is not selection-locked")
if metrics.get("raw_reference_characters") != 14475:
    raise SystemExit("reference character count does not match RESULTS.md protocol")
payload = {
    "status": "complete",
    "session": session,
    "run_id": run_id,
    "mode": mode,
    "dataset": "dunhuang_local_gazetteer_q32_v1",
    "split": "test",
    "test_pages": 59,
    "gpu_ids": gpu_ids,
    "num_queries": 32,
    "max_eval_new_tokens": 1536,
    "selected_step": selection.get("selected_step"),
    "test_manifest_sha256": test_sha,
    "test_used_for_selection": False,
    "selection": selection,
    "test_summary": test_summary,
    "metrics": metrics,
    "external_comparison_reference": {
        "results_file": "D:/yangky/学推计划-glm-ocr/results/sota_zeroshot_dunhuang_local_q32_test_20260913_v3_max1536/RESULTS.md",
        "input": "whole_page_image + fixed OCR prompt",
        "generation": "do_sample=false, max_output_tokens=1536",
        "test_pages": 59,
    },
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"event": "glmocr_bscc_q32_single_complete", "run_id": run_id, "mode": mode, "selected_step": selection.get("selected_step"), "summary": str(output), "test_used_for_selection": False}, separators=(",", ":")))
PY
}

write_validation_only_summary() {
    "${python}" - "${summary_file}" "${active_run_id}" "${active_mode}" "${experiment_root}" "${session}" "${gpu_ids}" "${max_eval_new_tokens}" "${generation_mode}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
run_id, mode, experiment_root, session, gpu_ids, max_tokens, generation_mode = sys.argv[2:]
root = Path(experiment_root) / "training_runs" / run_id / "seed42"
training = json.loads((root / "summary.json").read_text(encoding="utf-8"))
selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
parallel = json.loads((root / "parallel_validation_summary.json").read_text(encoding="utf-8"))
if selection.get("status") != "complete" or parallel.get("status") != "complete":
    raise SystemExit("validation-only summary is incomplete")
if selection.get("test_manifest_read") is not False or selection.get("test_used_for_selection") is not False:
    raise SystemExit("validation-only selection crossed the test boundary")
payload = {
    "status": "complete",
    "phase": "validation_complete",
    "session": session,
    "run_id": run_id,
    "mode": mode,
    "dataset": "dunhuang_local_gazetteer_q32_v1",
    "gpu_ids": gpu_ids,
    "num_queries": 32,
    "max_eval_new_tokens": int(max_tokens),
    "generation_mode": generation_mode,
    "selected_step": selection.get("selected_step"),
    "training": training.get("training"),
    "selection": selection,
    "parallel_validation": parallel,
    "test_manifest_read": False,
    "test_used_for_selection": False,
}
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"event": "glmocr_bscc_q32_validation_only_complete", "run_id": run_id, "mode": mode, "selected_step": selection.get("selected_step"), "summary": str(output), "test_used_for_selection": False}, separators=(",", ":")))
PY
}

prepare_dataset_view
preflight
source_environment
write_status running preflight "-" "-"

if [[ "${arm_mode}" != "compare" ]]; then
    current_phase="smoke_gate"
    run_smoke "${active_mode}" "${active_auxiliary_weight}" "${active_run_id}"
    run_training "${active_mode}" "${active_auxiliary_weight}" "${active_run_id}"
    run_validation "${active_mode}" "${active_auxiliary_weight}" "${active_run_id}"
    if (( stop_after_validation == 1 )); then
        current_phase="validation_complete"
        write_validation_only_summary
        write_status complete validation_complete "${active_run_id}" "${active_mode}"
        cat "${summary_file}"
        exit 0
    fi
    current_phase="test_protocol"
    write_status running "${current_phase}" "-" "-"
    make_test_protocol "${active_run_id}"
    run_test "${active_mode}" "${active_run_id}"
    current_phase="complete"
    write_single_summary
    write_status complete complete "${active_run_id}" "${active_mode}"
    cat "${summary_file}"
elif [[ -n "${resume_geometry_run_id}" ]]; then
    current_phase="geometry_resume_gate"
    write_status running "${current_phase}" "${geometry_run_id}" geometry
    verify_existing_geometry "${geometry_run_id}" geometry 0.4
    run_training content_only 0.0 "${baseline_run_id}"
else
    current_phase="smoke_gate"
    run_smoke geometry 0.4 "${geometry_run_id}"
    run_smoke content_only 0.0 "${baseline_run_id}"
    run_training geometry 0.4 "${geometry_run_id}"
    run_training content_only 0.0 "${baseline_run_id}"
fi

run_validation geometry 0.4 "${geometry_run_id}"
run_validation content_only 0.0 "${baseline_run_id}"

current_phase="test_protocol"
write_status running "${current_phase}" "-" "-"
make_test_protocol "${geometry_run_id}"
make_test_protocol "${baseline_run_id}"

run_test geometry "${geometry_run_id}"
run_test content_only "${baseline_run_id}"

current_phase="complete"
write_comparison_summary
write_status complete complete "-" "-"
cat "${comparison_metrics_file}"
