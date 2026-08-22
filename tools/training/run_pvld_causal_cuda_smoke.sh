#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

gpu_id="${1:-0}"
run_id="${2:-pvld_causal_cuda_smoke_$(date +%Y%m%d_%H%M%S)}"
utilization_limit="${3:-50}"
utilization="$(nvidia-smi -i "${gpu_id}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
[[ "${utilization}" =~ ^[0-9]+$ ]] || { printf 'ERROR: invalid GPU utilization.\n' >&2; exit 75; }
(( utilization < utilization_limit )) || { printf 'ERROR: GPU%s utilization=%s.\n' "${gpu_id}" "${utilization}" >&2; exit 75; }

run_root="${GOT_EVALUATION_RUNS}/${run_id}"
mkdir "${run_root}"
CUDA_VISIBLE_DEVICES="${gpu_id}" bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
    "${ocrmodel_root}/tools/training/smoke_pvld_causal_decoder_cuda.py" \
    --project-root "${ocrmodel_root}/src/GOT-OCR-2.0" \
    --output "${run_root}/summary.json" >"${run_root}/smoke.log" 2>&1 || {
        tail -n 20 "${run_root}/smoke.log" >&2
        exit 1
    }
jq -c '{status,device,loss,gradients,generation,ocr_visual_value_source,alpha_zero_exact_identity,formal_training_started,frozen_test_started}' "${run_root}/summary.json"
