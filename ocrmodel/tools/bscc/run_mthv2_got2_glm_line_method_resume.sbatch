#!/usr/bin/env bash
# Resume the comparison after the training arms have completed.  This entry
# deliberately starts at BASELINE_EVAL and never retrains or reads test before
# the two source training runs are verified.
#SBATCH --job-name=mthv2_gotglm_resume
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
code_root="${workspace}/glm_ocr_layout_ot/code/ocrmodel"
glm_env="${workspace}/glm_ocr_layout_ot/envs/glm-ocr-py311-cu128"
got_env="${workspace}/envs/got2-py310-cu118"
got_source_root="${workspace}/ocrmodel/src/GOT-OCR-2.0"
got_model="${workspace}/models/GOT-OCR2-original"
glm_model="${workspace}/models/GLM-OCR-ca5d8b3"
source_lines="${workspace}/datasets/MTHv2/converted/mthv2_anandasky_gt_lines_v1"
joint_root="${workspace}/datasets/MTHv2/converted/mthv2_gtline_joint_1000_v2"
source_run="${GOTGLM_SOURCE_RUN_ID:-mthv2_gotglm_line_method_1000_260920_v8}"
run_id="${GOTGLM_LINE_RUN_ID:-${source_run}_resume}"
source_stage1="${workspace}/glm_ocr_layout_ot/training_runs/${source_run}_stage1/seed42"
source_stage2_id="${GOTGLM_SOURCE_STAGE2_ID:-glmocr_mthv2_sem_adapter_stage2_v1}"
source_stage2="${workspace}/glm_ocr_layout_ot/training_runs/${source_stage2_id}/seed42"
run_root="${workspace}/runs/${run_id}"
status_file="${run_root}/status.json"
launcher_log="${run_root}/launcher.log"

[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo '{"event":"resume_failed","error":"invalid_run_id"}' >&2; exit 64; }
[[ ! -e "${run_root}" ]] || { echo '{"event":"resume_failed","error":"run_already_exists"}' >&2; exit 74; }
[[ -f "${source_lines}/test/manifest_subset_1000.jsonl" ]] || { echo '{"event":"resume_failed","error":"locked_test_subset_missing"}' >&2; exit 66; }
[[ -f "${joint_root}/train/manifest.jsonl" ]] || { echo '{"event":"resume_failed","error":"train_manifest_missing"}' >&2; exit 66; }
[[ -f "${got_model}/config.json" && -f "${got_model}/model.safetensors" ]] || { echo '{"event":"resume_failed","error":"got_baseline_missing"}' >&2; exit 66; }
[[ -f "${run_root%/*}/${source_run}/got_method_model/config.json" && -f "${run_root%/*}/${source_run}/got_method_model/model.safetensors" ]] || { echo '{"event":"resume_failed","error":"got_method_checkpoint_missing"}' >&2; exit 66; }
[[ -f "${source_stage1}/checkpoint-512/adapter.safetensors" ]] || { echo '{"event":"resume_failed","error":"glm_stage1_checkpoint_missing"}' >&2; exit 66; }
[[ -f "${source_stage2}/checkpoint-256/adapter.safetensors" && -f "${source_stage2}/metadata.json" ]] || { echo '{"event":"resume_failed","error":"glm_stage2_checkpoint_missing"}' >&2; exit 66; }

jq -e --arg init "${source_stage1}/checkpoint-512" \
    '.init_checkpoint_dir == $init and .experiment_label == "mthv2_gtline_sem_adapter_stage2" and .test_manifest_read == false and .test_used_for_selection == false' \
    "${source_stage2}/metadata.json" >/dev/null || {
    echo '{"event":"resume_failed","error":"glm_stage2_provenance_mismatch"}' >&2
    exit 67
}

mkdir -p "${run_root}"
exec > >(tee -a "${launcher_log}") 2>&1

stage="SETUP"
write_status() {
    local state="$1"
    local message="${2:-}"
    printf '{"status":"%s","stage":"%s","message":"%s","run_id":"%s","source_run":"%s","test_used_for_selection":false}\n' \
        "${state}" "${stage}" "${message}" "${run_id}" "${source_run}" > "${status_file}"
}
write_status running "resuming_after_completed_training"
trap 'rc=$?; if (( rc != 0 )); then write_status failed "return_code_${rc}" || true; fi; exit $rc' EXIT

load_glm_env() {
    module purge
    export LD_LIBRARY_PATH=""
    source "/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311/env.sh"
}

load_got_env() {
    module purge
    export LD_LIBRARY_PATH=""
    source "/home/bingxing2/apps/package/pytorch/2.0.1+cu118_cp310/env.sh"
}

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="${workspace}/gotglm_line_method_compare/tmp/huggingface"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "${HF_HOME}"

stage="BASELINE_EVAL"
write_status running "evaluating_untouched_got_and_glm"
load_got_env
got_python_lib="${got_env}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${got_python_lib}/torch/lib:${got_python_lib}/nvidia/cuda_runtime/lib:${got_python_lib}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
got_method_model="${workspace}/runs/${source_run}/got_method_model"
got_eval="${got_env}/bin/python"
glm_eval="${glm_env}/bin/python"

"${got_eval}" "${code_root}/tools/evaluation/evaluate_mthv2_columns.py" \
    --model got --model-dir "${got_model}" --got-source-root "${got_source_root}" \
    --manifest "${source_lines}/test/manifest_subset_1000.jsonl" \
    --train-manifest "${joint_root}/train/manifest.jsonl" \
    --output-dir "${run_root}/got_baseline" --max-new-tokens 256

load_glm_env
"${glm_eval}" "${code_root}/tools/evaluation/evaluate_mthv2_columns.py" \
    --model glm --model-dir "${glm_model}" \
    --manifest "${source_lines}/test/manifest_subset_1000.jsonl" \
    --train-manifest "${joint_root}/train/manifest.jsonl" \
    --output-dir "${run_root}/glm_baseline" --max-new-tokens 256

stage="GOT_METHOD_EVAL"
write_status running "evaluating_got2_method_from_${source_run}"
load_got_env
got_python_lib="${got_env}/lib/python3.10/site-packages"
export LD_LIBRARY_PATH="${got_python_lib}/torch/lib:${got_python_lib}/nvidia/cuda_runtime/lib:${got_python_lib}/nvidia/cusparse/lib:${LD_LIBRARY_PATH:-}"
"${got_eval}" "${code_root}/tools/evaluation/evaluate_mthv2_columns.py" \
    --model got --model-dir "${got_method_model}" --got-source-root "${got_source_root}" \
    --manifest "${source_lines}/test/manifest_subset_1000.jsonl" \
    --train-manifest "${joint_root}/train/manifest.jsonl" \
    --output-dir "${run_root}/got_method" --max-new-tokens 256

stage="GLM_METHOD_EVAL"
write_status running "evaluating_glm_method_from_${source_run}"
load_glm_env
"${glm_eval}" "${code_root}/tools/evaluation/evaluate_mthv2_columns.py" \
    --model glm --model-dir "${glm_model}" \
    --glm-checkpoint-dir "${source_stage2}/checkpoint-256" --glm-metadata "${source_stage2}/metadata.json" \
    --manifest "${source_lines}/test/manifest_subset_1000.jsonl" \
    --train-manifest "${joint_root}/train/manifest.jsonl" \
    --output-dir "${run_root}/glm_method" --max-new-tokens 256

for summary in "${run_root}/got_baseline/summary.json" "${run_root}/got_method/summary.json" "${run_root}/glm_baseline/summary.json" "${run_root}/glm_method/summary.json"; do
    jq -e '.records == 1000 and .test_used_for_selection == false' "${summary}" >/dev/null || {
        echo '{"event":"resume_failed","error":"summary_contract_failed"}' >&2
        exit 68
    }
done

printf '{"source_run":"%s","source_stage1":"%s","source_stage2":"%s","test_subset_records":1000,"test_used_for_selection":false,"got_baseline":"%s","got_method":"%s","glm_baseline":"%s","glm_method":"%s"}\n' \
    "${source_run}" "${source_stage1}" "${source_stage2}" \
    "${run_root}/got_baseline/summary.json" "${run_root}/got_method/summary.json" \
    "${run_root}/glm_baseline/summary.json" "${run_root}/glm_method/summary.json" > "${run_root}/comparison_summary.json"
stage="COMPLETE"
write_status complete "comparison_metrics_ready_from_${source_run}"
