#!/usr/bin/env bash
# Submit the current-branch 600-step reference pair on BSCC.
# Both arms use the same four-GPU schedule, save checkpoints at 200/400/600,
# and stop after validation-only selection.
set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
code_root="${BSCC_GLM_CODE_ROOT:-${workspace}/glm_ocr_layout_ot/code/ocrmodel}"
script="${code_root}/tools/bscc/run_dunhuang_local_gazetteer_compare_bscc.sbatch"

geometry_session="glmocr_dunhuang_local_gazetteer_q32_geometry_ref600_3ckpt_bscc_260913_v2"
geometry_run_id="glmocr_dunhuang_local_gazetteer_q32_geometry_ref600_3ckpt_bscc_260913_v2"
baseline_session="glmocr_dunhuang_local_gazetteer_q32_baseline_ref600_3ckpt_bscc_260913_v2"
baseline_run_id="glmocr_dunhuang_local_gazetteer_q32_official_content_only_ref600_3ckpt_bscc_260913_v2"

[[ -f "${script}" ]] || exit 66

profile="GLMOCR_BSCC_Q32_MAX_STEPS=600,GLMOCR_BSCC_Q32_LR_SCHEDULE_STEPS=600,GLMOCR_BSCC_Q32_VALIDATION_INTERVAL=200,GLMOCR_BSCC_Q32_CHECKPOINT_STEPS=200:400:600,GLMOCR_BSCC_Q32_MAX_EVAL_NEW_TOKENS=1536,GLMOCR_BSCC_Q32_GENERATION_MODE=plain,GLMOCR_BSCC_Q32_STOP_AFTER_VALIDATION=1"

geometry_job_id="$(sbatch --parsable --job-name=glmocr_g600_3ckpt \
    --export="ALL,${profile},GLMOCR_BSCC_Q32_MODE=geometry,GLMOCR_BSCC_Q32_SESSION=${geometry_session},GLMOCR_BSCC_Q32_RUN_ID=${geometry_run_id},GLMOCR_BSCC_Q32_RESUME_GEOMETRY_RUN_ID=" \
    "${script}")"
baseline_job_id="$(sbatch --parsable --job-name=glmocr_b600_3ckpt \
    --export="ALL,${profile},GLMOCR_BSCC_Q32_MODE=content_only,GLMOCR_BSCC_Q32_SESSION=${baseline_session},GLMOCR_BSCC_Q32_RUN_ID=${baseline_run_id},GLMOCR_BSCC_Q32_RESUME_GEOMETRY_RUN_ID=" \
    "${script}")"

printf '{"event":"glmocr_bscc_q32_reference600_3ckpt_submitted","geometry":{"job_id":"%s","run_id":"%s","mode":"geometry","auxiliary_weight":0.4},"baseline":{"job_id":"%s","run_id":"%s","mode":"content_only","auxiliary_weight":0.0},"world_size":4,"max_steps":600,"lr_schedule_steps":600,"checkpoint_steps":[200,400,600],"warmup_steps":216,"max_eval_new_tokens":1536,"generation_mode":"plain","test_used_for_selection":false}\n' \
    "${geometry_job_id}" "${geometry_run_id}" "${baseline_job_id}" "${baseline_run_id}"
