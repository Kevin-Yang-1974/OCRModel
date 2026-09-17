#!/usr/bin/env bash
# Submit a zero-shot external-SOTA inference array on BSCC.
#
# Usage: run_sota_zeroshot_submit.sh <model> <split> [shard_count] [run_id] [limit]
#   model        paddleocr_vl_1_6 | mineru2_5_pro | opendoc_0_1b
#   split        validation | test
#   shard_count  number of Slurm array tasks (default 4)
#   run_id       evaluation run id (default sota_zeroshot_<date>_v1)
#   limit        optional page cap (for a bounded smoke)
#
# Test submissions require ALLOW_SOTA_TEST=1. Test is never used for selection.
set -euo pipefail

model="${1:?model required}"
split="${2:?split required}"
shard_count="${3:-4}"
run_id="${4:-sota_zeroshot_$(date +%Y%m%d)_v1}"
limit="${5:-}"

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
code_root="${workspace}/glm_ocr_layout_ot/code/ocrmodel"
[[ -d "${code_root}" ]] || { echo "code root missing: ${code_root}" >&2; exit 66; }
[[ "${split}" == "validation" || "${split}" == "test" ]] || { echo "invalid split" >&2; exit 64; }

allow_test=0
if [[ "${split}" == "test" ]]; then
    [[ "${ALLOW_SOTA_TEST:-0}" == "1" ]] || { echo "set ALLOW_SOTA_TEST=1 for test" >&2; exit 77; }
    allow_test=1
fi

# OpenDoc runs on the CPU ONNX provider unless the caller overrides it.
device="cuda"
[[ "${model}" == "opendoc_0_1b" ]] && device="${SOTA_DEVICE:-cpu}"

export SOTA_RUN_ID="${run_id}" SOTA_MODEL="${model}" SOTA_SPLIT="${split}"
export SOTA_SHARD_COUNT="${shard_count}" SOTA_ALLOW_TEST="${allow_test}" SOTA_DEVICE="${device}"
[[ -n "${limit}" ]] && export SOTA_LIMIT="${limit}"

last=$((shard_count - 1))
# agent-4 reports driver 11.6 while all healthy nodes run CUDA 12.8; a job that
# lands there fails model load. Exclude it by default (override with SOTA_EXCLUDE).
exclude_node="${SOTA_EXCLUDE:-paraai-n32-h-01-agent-4}"
job_id=$(sbatch --parsable --array="0-${last}" --exclude="${exclude_node}" --export=ALL \
    "${code_root}/tools/bscc/run_sota_zero_shot.sbatch")
printf '{"event":"sota_zeroshot_submitted","model":"%s","split":"%s","run_id":"%s","shards":%s,"limit":"%s","job_id":"%s","device":"%s","test_used_for_selection":false}\n' \
    "${model}" "${split}" "${run_id}" "${shard_count}" "${limit}" "${job_id}" "${device}"
