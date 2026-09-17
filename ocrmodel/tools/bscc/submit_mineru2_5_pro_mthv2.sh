#!/usr/bin/env bash
# Submit exactly one four-GPU MinerU2.5-Pro MTHv2 training task.
set -Eeuo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
experiment_root="${workspace}/glm_ocr_layout_ot"
run_id="${MINERU_BSCC_RUN_ID:-mineru2_5_pro_mthv2_text_sft_4gpu_5000_260912_v1}"
script="${experiment_root}/code/ocrmodel/tools/bscc/run_mineru2_5_pro_mthv2_4gpu.sbatch"

[[ "${run_id}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid MINERU_BSCC_RUN_ID" >&2; exit 64; }
[[ -f "${script}" ]] || { echo "missing synced MinerU sbatch script: ${script}" >&2; exit 66; }
mkdir -p "${workspace}/runs"

job_id="$(sbatch --parsable --export="ALL,MINERU_BSCC_RUN_ID=${run_id}" "${script}")"
printf '{"event":"mineru_bscc_submitted","job_id":"%s","run_id":"%s","world_size":4,"max_steps":5000,"queue":"%s"}\n' \
    "${job_id}" "${run_id}" "${script}"
