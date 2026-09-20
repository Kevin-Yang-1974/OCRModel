#!/usr/bin/env bash
# Re-evaluate only the GLM method arm after fixing the inference-time bridge
# state.  The three already-valid comparison summaries are copied from the
# completed evaluation run; no training or duplicate baseline evaluation is
# started.
#SBATCH --job-name=mthv2_glm_eval_fix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/bingxing2/home/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
code_root="${workspace}/glm_ocr_layout_ot/code/ocrmodel"
glm_env="${workspace}/glm_ocr_layout_ot/envs/glm-ocr-py311-cu128"
glm_model="${workspace}/models/GLM-OCR-ca5d8b3"
source_lines="${workspace}/datasets/MTHv2/converted/mthv2_anandasky_gt_lines_v1"
joint_root="${workspace}/datasets/MTHv2/converted/mthv2_gtline_joint_1000_v2"
source_run="${GOTGLM_SOURCE_RUN_ID:-mthv2_gotglm_line_method_1000_260920_v8}"
parent_run="${GOTGLM_PARENT_EVAL_RUN_ID:-mthv2_gotglm_line_method_1000_260920_v8_resume2}"
run_id="${GOTGLM_LINE_RUN_ID:-mthv2_gotglm_line_method_1000_260920_v8_glmfix1}"
source_stage1="${workspace}/glm_ocr_layout_ot/training_runs/${source_run}_stage1/seed42"
source_stage2_id="${GOTGLM_SOURCE_STAGE2_ID:-glmocr_mthv2_sem_adapter_stage2_v1}"
source_stage2="${workspace}/glm_ocr_layout_ot/training_runs/${source_stage2_id}/seed42"
parent_root="${workspace}/runs/${parent_run}"
run_root="${workspace}/runs/${run_id}"
status_file="${run_root}/status.json"
launcher_log="${run_root}/launcher.log"
test_manifest="${source_lines}/test/manifest_subset_1000.jsonl"
train_manifest="${joint_root}/train/manifest.jsonl"

[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo '{"event":"eval_fix_failed","error":"invalid_run_id"}' >&2; exit 64; }
[[ ! -e "${run_root}" ]] || { echo '{"event":"eval_fix_failed","error":"run_already_exists"}' >&2; exit 74; }
[[ -f "${test_manifest}" && -f "${train_manifest}" ]] || { echo '{"event":"eval_fix_failed","error":"manifest_missing"}' >&2; exit 66; }
[[ -f "${glm_model}/config.json" && -f "${glm_model}/model.safetensors" ]] || { echo '{"event":"eval_fix_failed","error":"glm_model_missing"}' >&2; exit 66; }
[[ -f "${source_stage2}/checkpoint-256/adapter.safetensors" && -f "${source_stage2}/metadata.json" ]] || { echo '{"event":"eval_fix_failed","error":"glm_stage2_checkpoint_missing"}' >&2; exit 66; }

for summary in \
    "${parent_root}/got_baseline/summary.json" \
    "${parent_root}/got_method/summary.json" \
    "${parent_root}/glm_baseline/summary.json"; do
    jq -e '.records == 1000 and .test_used_for_selection == false and .status == "complete"' "${summary}" >/dev/null || {
        echo '{"event":"eval_fix_failed","error":"parent_summary_contract_failed"}' >&2
        exit 68
    }
done

jq -e --arg init "${source_stage1}/checkpoint-512" \
    '.init_checkpoint_dir == $init and .experiment_label == "mthv2_gtline_sem_adapter_stage2" and .test_manifest_read == false and .test_used_for_selection == false' \
    "${source_stage2}/metadata.json" >/dev/null || {
    echo '{"event":"eval_fix_failed","error":"glm_stage2_provenance_mismatch"}' >&2
    exit 67
}

mkdir -p "${run_root}/got_baseline" "${run_root}/got_method" "${run_root}/glm_baseline"
cp "${parent_root}/got_baseline/summary.json" "${run_root}/got_baseline/summary.json"
cp "${parent_root}/got_method/summary.json" "${run_root}/got_method/summary.json"
cp "${parent_root}/glm_baseline/summary.json" "${run_root}/glm_baseline/summary.json"
exec > >(tee -a "${launcher_log}") 2>&1

stage="SETUP"
write_status() {
    local state="$1"
    local message="${2:-}"
    printf '{"status":"%s","stage":"%s","message":"%s","run_id":"%s","source_run":"%s","test_used_for_selection":false}\n' \
        "${state}" "${stage}" "${message}" "${run_id}" "${source_run}" > "${status_file}"
}
write_status running "reusing_completed_training_and_baseline_evaluations"
trap 'rc=$?; if (( rc != 0 )); then write_status failed "return_code_${rc}" || true; fi; exit $rc' EXIT

load_glm_env() {
    module purge
    export LD_LIBRARY_PATH=""
    source "/home/bingxing2/apps/package/pytorch/2.8.0-cu128_cp311/env.sh"
}

stage="GLM_METHOD_EVAL"
write_status running "evaluating_glm_method_after_bridge_grid_fix"
load_glm_env
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="${workspace}/gotglm_line_method_compare/tmp/huggingface"
export TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "${HF_HOME}"

"${glm_env}/bin/python" "${code_root}/tools/evaluation/evaluate_mthv2_columns.py" \
    --model glm --model-dir "${glm_model}" \
    --glm-checkpoint-dir "${source_stage2}/checkpoint-256" --glm-metadata "${source_stage2}/metadata.json" \
    --manifest "${test_manifest}" --train-manifest "${train_manifest}" \
    --output-dir "${run_root}/glm_method" --max-new-tokens 256

for summary in \
    "${run_root}/got_baseline/summary.json" \
    "${run_root}/got_method/summary.json" \
    "${run_root}/glm_baseline/summary.json" \
    "${run_root}/glm_method/summary.json"; do
    jq -e '.records == 1000 and .test_used_for_selection == false and .status == "complete" and .errors == 0' "${summary}" >/dev/null || {
        echo '{"event":"eval_fix_failed","error":"summary_contract_failed"}' >&2
        exit 68
    }
done

printf '{"source_run":"%s","source_eval_run":"%s","source_stage1":"%s","source_stage2":"%s","test_subset_records":1000,"test_used_for_selection":false,"got_baseline":"%s","got_method":"%s","glm_baseline":"%s","glm_method":"%s"}\n' \
    "${source_run}" "${parent_run}" "${source_stage1}" "${source_stage2}" \
    "${run_root}/got_baseline/summary.json" "${run_root}/got_method/summary.json" \
    "${run_root}/glm_baseline/summary.json" "${run_root}/glm_method/summary.json" > "${run_root}/comparison_summary.json"
stage="COMPLETE"
write_status complete "glm_method_evaluation_fixed_and_complete"
