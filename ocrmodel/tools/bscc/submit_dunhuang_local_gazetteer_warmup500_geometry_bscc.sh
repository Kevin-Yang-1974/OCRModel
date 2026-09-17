#!/usr/bin/env bash
# Stage B: one geometry-only warmup change after the 600-step reference pair.
# It keeps the same three checkpoints and stops after validation-only selection.
set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
code_root="${BSCC_GLM_CODE_ROOT:-${workspace}/glm_ocr_layout_ot/code/ocrmodel}"
script="${code_root}/tools/bscc/run_dunhuang_local_gazetteer_compare_bscc.sbatch"

session="glmocr_dunhuang_local_gazetteer_q32_geometry_warmup500_ref600_3ckpt_bscc_260913_v1"
run_id="${session}"

[[ -f "${script}" ]] || exit 66

profile="GLMOCR_BSCC_Q32_MAX_STEPS=600,GLMOCR_BSCC_Q32_LR_SCHEDULE_STEPS=600,GLMOCR_BSCC_Q32_VALIDATION_INTERVAL=200,GLMOCR_BSCC_Q32_CHECKPOINT_STEPS=200:400:600,GLMOCR_BSCC_Q32_WARMUP_STEPS=500,GLMOCR_BSCC_Q32_MAX_EVAL_NEW_TOKENS=1536,GLMOCR_BSCC_Q32_GENERATION_MODE=plain,GLMOCR_BSCC_Q32_STOP_AFTER_VALIDATION=1"

job_id="$(sbatch --parsable --job-name=glmocr_g600_w500_3ckpt \
    --export="ALL,${profile},GLMOCR_BSCC_Q32_MODE=geometry,GLMOCR_BSCC_Q32_SESSION=${session},GLMOCR_BSCC_Q32_RUN_ID=${run_id},GLMOCR_BSCC_Q32_RESUME_GEOMETRY_RUN_ID=" \
    "${script}")"

printf '{"event":"glmocr_bscc_q32_geometry_warmup500_3ckpt_submitted","job_id":"%s","run_id":"%s","mode":"geometry","auxiliary_weight":0.4,"world_size":4,"max_steps":600,"lr_schedule_steps":600,"checkpoint_steps":[200,400,600],"warmup_steps":500,"max_eval_new_tokens":1536,"generation_mode":"plain","test_used_for_selection":false}\n' \
    "${job_id}" "${run_id}"
