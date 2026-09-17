#!/usr/bin/env bash
# Submit one BSCC four-GPU q32 GLMOCR comparison pipeline.
set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
code_root="${BSCC_GLM_CODE_ROOT:-${workspace}/glm_ocr_layout_ot/code/ocrmodel}"
script="${code_root}/tools/bscc/run_dunhuang_local_gazetteer_compare_bscc.sbatch"

[[ -f "${script}" ]] || {
    printf '{"event":"glmocr_bscc_q32_submit_failed","error":"pipeline_script_missing","path":"%s"}\n' "${script}" >&2
    exit 66
}
sbatch --parsable "${script}"
