#!/usr/bin/env bash
# Submit stage-A selection-locked tests for geometry and official GLMOCR.
set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
code_root="${BSCC_GLM_CODE_ROOT:-${workspace}/glm_ocr_layout_ot/code/ocrmodel}"
script="${code_root}/tools/bscc/run_dunhuang_local_gazetteer_locked_test_bscc.sbatch"

geometry_run_id="glmocr_dunhuang_local_gazetteer_q32_geometry_ref600_3ckpt_bscc_260913_v2"
baseline_run_id="glmocr_dunhuang_local_gazetteer_q32_official_content_only_ref600_3ckpt_bscc_260913_v2"
geometry_session="glmocr_dunhuang_local_gazetteer_q32_geometry_ref600_3ckpt_stagea_test_bscc_260913_v1"
baseline_session="glmocr_dunhuang_local_gazetteer_q32_baseline_ref600_3ckpt_stagea_test_bscc_260913_v1"

[[ -f "${script}" ]] || exit 66

geometry_job_id="$(sbatch --parsable --job-name=glmocr_g_stagea_test \
    --export="ALL,GLMOCR_BSCC_TEST_RUN_ID=${geometry_run_id},GLMOCR_BSCC_TEST_SESSION=${geometry_session},GLMOCR_BSCC_TEST_MODE=geometry" \
    "${script}")"
baseline_job_id="$(sbatch --parsable --job-name=glmocr_b_stagea_test \
    --export="ALL,GLMOCR_BSCC_TEST_RUN_ID=${baseline_run_id},GLMOCR_BSCC_TEST_SESSION=${baseline_session},GLMOCR_BSCC_TEST_MODE=content_only" \
    "${script}")"

printf '{"event":"glmocr_bscc_q32_stagea_locked_tests_submitted","geometry":{"job_id":"%s","run_id":"%s","mode":"geometry","selected_step":200},"baseline":{"job_id":"%s","run_id":"%s","mode":"content_only","selected_step":200},"world_size":4,"test_pages":59,"max_eval_new_tokens":1536,"test_used_for_selection":false}\n' \
    "${geometry_job_id}" "${geometry_run_id}" "${baseline_job_id}" "${baseline_run_id}"
