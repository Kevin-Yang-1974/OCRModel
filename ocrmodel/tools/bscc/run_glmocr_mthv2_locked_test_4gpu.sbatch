#!/usr/bin/env bash
#SBATCH --job-name=glmocr_test4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00
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
run_id="${GLMOCR_BSCC_RUN_ID:-glmocr_mthv2_decoder_lora_lr1e5_20k_4gpu_official_layout_260912_v1}"
seed=42
group_root="${experiment_root}/training_runs/${run_id}"
run_dir="${group_root}/seed${seed}"
selection_file="${group_root}/selection.json"
test_protocol_file="${experiment_root}/protocols/${run_id}.test_locked.json"
launcher_log="${workspace}/runs/${run_id}.locked-test.sbatch.log"

gpu_ids="${GLMOCR_BSCC_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:-0,1,2,3}}}"
[[ "${gpu_ids}" =~ ^[0-9]+(,[0-9]+){3}$ ]] || {
    printf '{"event":"glmocr_bscc_test_failed","error":"exactly_four_numeric_gpus_required","gpu_ids":"%s"}\n' "${gpu_ids}" >&2
    exit 64
}
IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
declare -A seen_gpu=()
for gpu in "${gpu_array[@]}"; do
    [[ -z "${seen_gpu[${gpu}]+present}" ]] || {
        printf '{"event":"glmocr_bscc_test_failed","error":"duplicate_gpu_id","gpu":"%s"}\n' "${gpu}" >&2
        exit 64
    }
    seen_gpu["${gpu}"]=1
done

[[ -f "${pytorch_package}/env.sh" && -x "${env_dir}/bin/python" ]] || {
    printf '{"event":"glmocr_bscc_test_failed","error":"missing_bscc_python_environment"}\n' >&2
    exit 66
}
source "${pytorch_package}/env.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate "${env_dir}"
set -u

export TMPDIR="${experiment_root}/tmp/${run_id}-locked-test"
export HF_HOME="${TMPDIR}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "${workspace}/runs" "${experiment_root}/protocols" "${TMPDIR}" "${HF_HOME}"

train_manifest="${dataset_root}/train/manifest.jsonl"
validation_manifest="${dataset_root}/validation/manifest.jsonl"
test_manifest="${dataset_root}/test/manifest.jsonl"
[[ -f "${code_root}/tools/audit_mthv2_manifest.py" && -f "${code_root}/tools/training/run_glmocr_mthv2_locked_test.sh" ]] || {
    printf '{"event":"glmocr_bscc_test_failed","error":"missing_test_tools"}\n' >&2
    exit 66
}
[[ -f "${train_manifest}" && -f "${validation_manifest}" && -f "${test_manifest}" ]] || {
    printf '{"event":"glmocr_bscc_test_failed","error":"missing_test_manifests"}\n' >&2
    exit 66
}
[[ ! -e "${test_protocol_file}" ]] || {
    printf '{"event":"glmocr_bscc_test_failed","error":"test_protocol_already_exists","path":"%s"}\n' "${test_protocol_file}" >&2
    exit 74
}

"${env_dir}/bin/python" - "${run_dir}" "${selection_file}" <<'PY'
import json
import math
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
selection_path = Path(sys.argv[2])
summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
selection = json.loads(selection_path.read_text(encoding="utf-8"))
if summary.get("status") != "complete" or metadata.get("status") != "complete":
    raise SystemExit("training metadata is not complete")
if summary.get("test_manifest_read") is not False or selection.get("test_manifest_read") is not False:
    raise SystemExit("training or selection protocol is not test-free")
if summary.get("test_used_for_selection") is not False or selection.get("test_used_for_selection") is not False:
    raise SystemExit("selection is not validation-only")
if summary.get("selection_pending") is True or selection.get("status") != "complete":
    raise SystemExit("validation selection is not complete")
if metadata.get("world_size") != 4 or metadata.get("decoder_adaptation") != "lora":
    raise SystemExit("training configuration is not four-card decoder-LoRA")
if not math.isclose(float((metadata.get("decoder_lora_config") or {}).get("learning_rate")), 1e-5):
    raise SystemExit("training decoder LoRA learning rate is not 1e-5")
objective = (metadata.get("loss_objective") or {})
if objective.get("formula") != "L_official + auxiliary_weight * L_layout":
    raise SystemExit(f"unexpected training loss objective: {objective}")
if not math.isclose(float(objective.get("layout_weight", -1.0)), 0.2):
    raise SystemExit("training layout loss weight is not 0.2")
if objective.get("extra_terms") != []:
    raise SystemExit(f"unexpected extra loss terms: {objective.get('extra_terms')}")
selected_step = selection.get("selected_step")
if not isinstance(selected_step, int) or selected_step not in {5000, 10000, 15000, 20000}:
    raise SystemExit(f"invalid selected checkpoint: {selected_step!r}")
PY

"${env_dir}/bin/python" "${code_root}/tools/audit_mthv2_manifest.py" \
    --train-manifest "${train_manifest}" \
    --validation-manifest "${validation_manifest}" \
    --test-manifest "${test_manifest}" \
    --num-queries 512 \
    --output "${test_protocol_file}" \
    > "${workspace}/runs/${run_id}.test-protocol-audit.log" 2>&1

set +e
bash "${code_root}/tools/training/run_glmocr_mthv2_locked_test.sh" \
    --foreground \
    --run-id "${run_id}" \
    --seed "${seed}" \
    --gpu-ids "${gpu_ids}" \
    --remote-root "${experiment_root}" \
    --code-root "${code_root}" \
    --env-dir "${env_dir}" \
    --model-dir "${model_dir}" \
    --dataset-root "${dataset_root}" \
    --protocol-file "${test_protocol_file}" \
    > "${launcher_log}" 2>&1
rc=$?
set -e
if (( rc != 0 )); then
    printf '{"event":"glmocr_bscc_test_failed","run_id":"%s","return_code":%s,"log":"%s"}\n' \
        "${run_id}" "${rc}" "${launcher_log}" >&2
    exit "${rc}"
fi
printf '{"event":"glmocr_bscc_test_complete","run_id":"%s","world_size":4,"test_pages":800,"selection_file":"%s","test_protocol":"%s","test_used_for_selection":false,"log":"%s"}\n' \
    "${run_id}" "${selection_file}" "${test_protocol_file}" "${launcher_log}"
