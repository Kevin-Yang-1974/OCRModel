#!/usr/bin/env bash
# MinerU2.5-Pro-2605 MTHv2 whole-page text-recognition SFT on four BSCC GPUs.
# The launch keeps the public MinerU tutorial entry-point shape:
#   python run_mineru.py sft --config <absolute yaml>
# The small one-step compatibility gate runs in the same Slurm allocation;
# only the following official-shaped invocation is the formal 5000-step run.
# No MTHv2 test manifest is opened by this script.
#SBATCH --job-name=mineru_mthv2_5k
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=120:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/glm_ocr_layout_ot"
code_root="${experiment_root}/code/ocrmodel"
env_dir="${MINERU_SFT_ENV_DIR:-${experiment_root}/envs/mineru-sft}"
model_dir="${MINERU_MODEL_ROOT:-${workspace}/models/sota/mineru2_5_pro/bff20d4ae2bf202df9f45284b4d43681555a97ed}"
dataset_root="${BSCC_MTHV2_PAGE_ROOT:-${workspace}/datasets/MTHv2/converted/mthv2_layout_page_v1_real}"
pytorch_package="/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311"
run_id="${MINERU_BSCC_RUN_ID:-mineru2_5_pro_mthv2_text_sft_4gpu_5000_260912_v1}"
seed=42
gpu_utilization_limit=50
group_root="${experiment_root}/training_runs/MinerU2.5-Pro"
run_dir="${group_root}/${run_id}"
data_dir="${experiment_root}/training_datasets/mineru2_5_pro_mthv2_v1"
protocol_file="${experiment_root}/protocols/${run_id}.train_validation_no_test.json"
config_file="${run_dir}/sft_config.yaml"
launcher_log="${workspace}/runs/${run_id}.sbatch.log"
status_file="${run_dir}/metadata/status.txt"

gpu_ids="${MINERU_BSCC_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0,1,2,3}}}"

[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    printf '{"event":"mineru_bscc_train_failed","error":"invalid_run_id"}\n' >&2
    exit 64
}
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]] || {
    printf '{"event":"mineru_bscc_train_failed","error":"exactly_four_numeric_gpus_required","gpu_ids":"%s"}\n' "${gpu_ids}" >&2
    exit 64
}
IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
declare -A seen_gpu=()
for gpu in "${gpu_array[@]}"; do
    [[ -z "${seen_gpu[${gpu}]+present}" ]] || {
        printf '{"event":"mineru_bscc_train_failed","error":"duplicate_gpu_id","gpu":"%s"}\n' "${gpu}" >&2
        exit 64
    }
    seen_gpu["${gpu}"]=1
done

[[ -x "${env_dir}/bin/python" && -x "${env_dir}/bin/torchrun" ]] || {
    printf '{"event":"mineru_bscc_train_failed","error":"missing_mineru_sft_environment","env_dir":"%s"}\n' "${env_dir}" >&2
    exit 66
}
[[ -f "${code_root}/tools/bscc/run_mineru.py" && -f "${code_root}/tools/bscc/mineru_mthv2_compat.py" ]] || {
    printf '{"event":"mineru_bscc_train_failed","error":"missing_compatibility_code"}\n' >&2
    exit 66
}
[[ -f "${model_dir}/model.safetensors" && -f "${model_dir}/config.json" ]] || {
    printf '{"event":"mineru_bscc_train_failed","error":"missing_staged_mineru_model","model_dir":"%s"}\n' "${model_dir}" >&2
    exit 66
}
[[ -f "${dataset_root}/train/manifest.jsonl" && -f "${dataset_root}/validation/manifest.jsonl" ]] || {
    printf '{"event":"mineru_bscc_train_failed","error":"missing_train_or_validation_manifest"}\n' >&2
    exit 66
}
[[ ! -e "${run_dir}" ]] || {
    printf '{"event":"mineru_bscc_train_failed","error":"run_already_exists","run_dir":"%s"}\n' "${run_dir}" >&2
    exit 74
}

mkdir -p "${workspace}/runs" "${group_root}" "${experiment_root}/protocols" "${experiment_root}/tmp"
source "${pytorch_package}/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

export TMPDIR="${experiment_root}/tmp/${run_id}"
export HF_HOME="${TMPDIR}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MINERU_COMPAT_MODE=bscc_ms_swift_forwarder
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="${code_root}:${PYTHONPATH:-}"
mkdir -p "${TMPDIR}" "${HF_HOME}"

gpu_report="$(nvidia-smi -i "${gpu_ids}" --query-gpu=index,utilization.gpu --format=csv,noheader,nounits)"
gpu_count=0
while IFS=',' read -r gpu_index gpu_util; do
    gpu_util="${gpu_util// /}"
    [[ "${gpu_util}" =~ ^[0-9]+$ ]] || {
        printf '{"event":"mineru_bscc_train_failed","error":"invalid_gpu_utilization","gpu_index":"%s","utilization":"%s"}\n' "${gpu_index}" "${gpu_util}" >&2
        exit 65
    }
    if (( gpu_util >= gpu_utilization_limit )); then
        printf '{"event":"mineru_bscc_train_failed","error":"gpu_admission_rejected","gpu_index":"%s","utilization":%s,"limit":%s}\n' "${gpu_index}" "${gpu_util}" "${gpu_utilization_limit}" >&2
        exit 75
    fi
    gpu_count=$((gpu_count + 1))
done <<< "${gpu_report}"
(( gpu_count == 4 )) || {
    printf '{"event":"mineru_bscc_train_failed","error":"gpu_admission_count_mismatch","expected":4,"observed":%s}\n' "${gpu_count}" >&2
    exit 65
}

mkdir -p "${run_dir}/metadata"
printf 'PREPARING\n' > "${status_file}"
printf '%s\n' "${gpu_report}" > "${run_dir}/metadata/gpu_admission.txt"

"${env_dir}/bin/python" "${code_root}/tools/bscc/mineru_mthv2_compat.py" \
    --dataset-root "${dataset_root}" \
    --output-dir "${data_dir}" \
    --protocol-file "${protocol_file}" \
    --model-dir "${model_dir}" \
    --config-path "${config_file}" \
    --run-output-dir "${run_dir}/checkpoints" \
    --max-steps 5000 \
    > "${run_dir}/metadata/data_prepare.json"

"${env_dir}/bin/python" - "${model_dir}" "${data_dir}/train.jsonl" "${config_file}" > "${run_dir}/metadata/model_preflight.json" <<'PY'
import json
import os
import sys

model_dir, train_jsonl, config_file = sys.argv[1:]
import torch
import transformers
import swift
import qwen_vl_utils

with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as handle:
    config = json.load(handle)
with open(train_jsonl, encoding="utf-8") as handle:
    first = json.loads(handle.readline())
assert config.get("model_type") == "qwen2_vl", config.get("model_type")
assert len(first["images"]) == 1
assert first["messages"][0]["content"].startswith("<image>")
assert os.path.isfile(first["images"][0])
print(json.dumps({
    "event": "mineru_bscc_compatibility_preflight_ok",
    "model_type": config.get("model_type"),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "swift": getattr(swift, "__version__", "unknown"),
    "config_file": config_file,
    "first_image": first["images"][0],
    "test_manifest_read": False,
}, separators=(",", ":")))
PY

printf 'TRAINING\n' > "${status_file}"
export NPROC_PER_NODE=4
export CUDA_VISIBLE_DEVICES="${gpu_ids}"
set +e
python "${code_root}/tools/bscc/run_mineru.py" sft --config "${config_file}" > "${launcher_log}" 2>&1
rc=$?
set -e
if (( rc != 0 )); then
    printf 'FAILED\n' > "${status_file}"
    printf '{"event":"mineru_bscc_train_failed","run_id":"%s","return_code":%s,"log":"%s"}\n' "${run_id}" "${rc}" "${launcher_log}" >&2
    exit "${rc}"
fi

printf 'COMPLETE\n' > "${status_file}"
printf '{"event":"mineru_bscc_train_complete","run_id":"%s","model":"MinerU2.5-Pro-2605-1.2B","world_size":4,"max_steps":5000,"train_count":2159,"validation_count":240,"task":"mthv2_whole_page_text_recognition_sft","compatibility_mode":"bscc_ms_swift_forwarder","test_manifest_read":false,"selection_pending":true,"log":"%s"}\n' "${run_id}" "${launcher_log}"
