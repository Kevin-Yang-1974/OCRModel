#!/usr/bin/env bash
#SBATCH --job-name=glmocr_smoke4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/glm_ocr_layout_ot"
code_root="${experiment_root}/code/ocrmodel"
env_dir="${experiment_root}/envs/glm-ocr-py311-cu128"
model_dir="${workspace}/models/GLM-OCR-ca5d8b3"
dataset_root="${BSCC_MTHV2_PAGE_ROOT:-${workspace}/datasets/MTHv2/converted/mthv2_layout_page_v1_real}"
pytorch_package="/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311"
run_id="${GLMOCR_BSCC_RUN_ID:-glmocr_mthv2_decoder_lora_lr1e5_4gpu_official_layout_smoke_260912_v1}"
seed=42
gpu_ids="${GLMOCR_BSCC_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0,1,2,3}}}"
group_root="${experiment_root}/training_runs/${run_id}"
protocol_file="${experiment_root}/protocols/${run_id}.train_validation_no_test.json"
launcher_log="${workspace}/runs/${run_id}.sbatch.log"

[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]] || {
    printf '{"event":"glmocr_bscc_smoke_failed","error":"exactly_four_numeric_gpus_required","gpu_ids":"%s"}\n' "${gpu_ids}" >&2
    exit 64
}
IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
declare -A seen_gpu=()
for gpu in "${gpu_array[@]}"; do
    [[ -z "${seen_gpu[${gpu}]+present}" ]] || {
        printf '{"event":"glmocr_bscc_smoke_failed","error":"duplicate_gpu_id","gpu":"%s"}\n' "${gpu}" >&2
        exit 64
    }
    seen_gpu["${gpu}"]=1
done
[[ -f "${pytorch_package}/env.sh" && -f "${code_root}/tools/training/run_glmocr_mthv2_ddp.sh" ]] || {
    printf '{"event":"glmocr_bscc_smoke_failed","error":"missing_bscc_launcher_or_environment"}\n' >&2
    exit 66
}
[[ ! -e "${group_root}" ]] || {
    printf '{"event":"glmocr_bscc_smoke_failed","error":"run_already_exists","run_dir":"%s"}\n' "${group_root}" >&2
    exit 74
}

source "${pytorch_package}/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

mkdir -p "${workspace}/runs" "${experiment_root}/protocols" "${experiment_root}/tmp"
export TMPDIR="${experiment_root}/tmp/${run_id}"
export HF_HOME="${TMPDIR}/huggingface"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "${TMPDIR}" "${HF_HOME}"

set +e
bash "${code_root}/tools/training/run_glmocr_mthv2_ddp.sh" \
    --foreground \
    --run-id "${run_id}" \
    --seed "${seed}" \
    --gpu-ids "${gpu_ids}" \
    --gpu-utilization-limit 50 \
    --remote-root "${experiment_root}" \
    --code-root "${code_root}" \
    --env-dir "${env_dir}" \
    --model-dir "${model_dir}" \
    --dataset-root "${dataset_root}" \
    --protocol-file "${protocol_file}" \
    --experiment-label "bscc_decoder_lora_lr1e5_4gpu_smoke" \
    --max-steps 8 \
    --lr-schedule-steps 8 \
    --warmup-steps 0 \
    --validation-interval 9 \
    --decoder-adaptation lora \
    --decoder-lora-rank 8 \
    --decoder-lora-alpha 8 \
    --decoder-lora-dropout 0 \
    --decoder-learning-rate 1e-5 \
    --layout-loss-profile full \
    --max-eval-new-tokens 1536 \
    --without-test \
    --smoke \
    > "${launcher_log}" 2>&1
rc=$?
set -e
if (( rc != 0 )); then
    printf '{"event":"glmocr_bscc_smoke_failed","run_id":"%s","return_code":%s,"log":"%s"}\n' \
        "${run_id}" "${rc}" "${launcher_log}" >&2
    exit "${rc}"
fi
"${env_dir}/bin/python" - "${group_root}/smoke/seed${seed}/smoke_summary.json" <<'PY'
import json
import math
import sys

summary = json.loads(open(sys.argv[1], encoding="utf-8").read())
if summary.get("status") != "complete" or summary.get("checkpoint_reload") is not True:
    raise SystemExit("official-layout smoke did not complete with checkpoint reload")
if summary.get("test_used_for_selection") is not False:
    raise SystemExit("official-layout smoke is not test-free")
training = summary.get("training") or {}
config = summary.get("natural_loop_config") or {}
if config.get("enabled") is not False:
    raise SystemExit("official-layout smoke unexpectedly enabled natural-loop loss")
for name in ("scheduled_sampling", "loop_escape", "continuation_head"):
    if (training.get(name) or {}).get("enabled") is not False:
        raise SystemExit(f"official-layout smoke enabled extra objective: {name}")
objective = training.get("loss_objective") or {}
if objective.get("formula") != "L_official + auxiliary_weight * L_layout":
    raise SystemExit(f"unexpected smoke training objective: {objective}")
if not math.isclose(float(objective.get("layout_weight", -1.0)), 0.2):
    raise SystemExit("smoke layout loss weight is not 0.2")
if objective.get("extra_terms") != []:
    raise SystemExit(f"unexpected smoke extra loss terms: {objective.get('extra_terms')}")
print(json.dumps({
    "event": "glmocr_bscc_official_layout_smoke_ok",
    "loss_objective": objective,
}, separators=(",", ":")))
PY
printf '{"event":"glmocr_bscc_smoke_complete","run_id":"%s","world_size":4,"steps":8,"decoder_learning_rate":1e-5,"natural_loop_loss":false,"loss_objective":"L_official + 0.2 * L_layout","test_manifest_read":false,"log":"%s"}\n' \
    "${run_id}" "${launcher_log}"
