#!/usr/bin/env bash
# Submit one four-GPU BSCC baseline-attention diagnostic task.
set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${BSCC_GLM_EXPERIMENT_ROOT:-${workspace}/glm_ocr_layout_mask_routing}"
code_root="${BSCC_GLM_CODE_ROOT:-${experiment_root}/code/ocrmodel}"
run_id="${GLMOCR_BSCC_RUN_ID:-glmocr_mthv2_baseline_attention_64val_bscc_260921_v1}"
script="${code_root}/tools/bscc/run_glmocr_mthv2_baseline_attention_4gpu.sbatch"

[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || {
    printf '{"event":"glmocr_bscc_attention_submit_failed","error":"invalid_run_id"}\n' >&2
    exit 64
}
[[ -f "${script}" ]] || {
    printf '{"event":"glmocr_bscc_attention_submit_failed","error":"script_missing","path":"%s"}\n' "${script}" >&2
    exit 66
}
mkdir -p "${workspace}/runs"
job_id="$(sbatch --parsable --job-name=glmattn64 --export="ALL,GLMOCR_BSCC_RUN_ID=${run_id},GLMOCR_SOURCE_BRANCH=glm-ocr-layout-mask-routing" "${script}")"
printf '{"event":"glmocr_bscc_attention_submitted","job_id":"%s","run_id":"%s","pages":64,"world_size":4,"mode":"content_only","test_manifest_read":false,"script":"%s"}\n' \
    "${job_id}" "${run_id}" "${script}"
