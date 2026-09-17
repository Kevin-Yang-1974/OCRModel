#!/usr/bin/env bash
# Submit two independent four-GPU BSCC q32 pipelines at once:
# geometry and the official GLMOCR content-only baseline.
set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
code_root="${BSCC_GLM_CODE_ROOT:-${workspace}/glm_ocr_layout_ot/code/ocrmodel}"
script="${code_root}/tools/bscc/run_dunhuang_local_gazetteer_compare_bscc.sbatch"

geometry_session="${GLMOCR_BSCC_GEOMETRY_SESSION:-glmocr_dunhuang_local_gazetteer_q32_geometry_bscc_260913_v3}"
geometry_run_id="${GLMOCR_BSCC_GEOMETRY_RUN_ID:-glmocr_dunhuang_local_gazetteer_q32_geometry_2k_bscc_260913_v3}"
baseline_session="${GLMOCR_BSCC_BASELINE_SESSION:-glmocr_dunhuang_local_gazetteer_q32_baseline_bscc_260913_v3}"
baseline_run_id="${GLMOCR_BSCC_BASELINE_RUN_ID:-glmocr_dunhuang_local_gazetteer_q32_official_content_only_2k_bscc_260913_v3}"

[[ -f "${script}" ]] || {
    printf '{"event":"glmocr_bscc_q32_submit_failed","error":"pipeline_script_missing","path":"%s"}\n' "${script}" >&2
    exit 66
}
[[ "${geometry_session}" =~ ^[A-Za-z0-9_.-]+$ && "${baseline_session}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
[[ "${geometry_run_id}" =~ ^[A-Za-z0-9_.-]+$ && "${baseline_run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || exit 64
[[ "${geometry_session}" != "${baseline_session}" && "${geometry_run_id}" != "${baseline_run_id}" ]] || exit 64

geometry_job_id="$(sbatch --parsable --job-name=glmocr_geom_q32 \
    --export="ALL,GLMOCR_BSCC_Q32_MODE=geometry,GLMOCR_BSCC_Q32_SESSION=${geometry_session},GLMOCR_BSCC_Q32_RUN_ID=${geometry_run_id},GLMOCR_BSCC_Q32_RESUME_GEOMETRY_RUN_ID=" \
    "${script}")"
baseline_job_id="$(sbatch --parsable --job-name=glmocr_base_q32 \
    --export="ALL,GLMOCR_BSCC_Q32_MODE=content_only,GLMOCR_BSCC_Q32_SESSION=${baseline_session},GLMOCR_BSCC_Q32_RUN_ID=${baseline_run_id},GLMOCR_BSCC_Q32_RESUME_GEOMETRY_RUN_ID=" \
    "${script}")"

printf '{"event":"glmocr_bscc_q32_two_jobs_submitted","geometry":{"job_id":"%s","session":"%s","run_id":"%s","mode":"geometry","gpus_per_job":4},"baseline":{"job_id":"%s","session":"%s","run_id":"%s","mode":"content_only","gpus_per_job":4},"dataset":"dunhuang_local_gazetteer_q32_v1","test_used_for_selection":false}\n' \
    "${geometry_job_id}" "${geometry_session}" "${geometry_run_id}" \
    "${baseline_job_id}" "${baseline_session}" "${baseline_run_id}"
