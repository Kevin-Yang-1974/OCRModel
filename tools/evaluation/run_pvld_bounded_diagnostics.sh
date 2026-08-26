#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

gpu_id=""
run_id="pvld_bounded_diagnostics_$(date +%Y%m%d_%H%M%S)"
c4_model=""
c5_model=""
dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1"
utilization_limit=50
timeout_minutes=45

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu-id) gpu_id="$2"; shift 2 ;;
        --run-id) run_id="$2"; shift 2 ;;
        --c4-model) c4_model="$2"; shift 2 ;;
        --c5-model) c5_model="$2"; shift 2 ;;
        --dataset-root) dataset_root="$2"; shift 2 ;;
        --gpu-utilization-limit) utilization_limit="$2"; shift 2 ;;
        --timeout-minutes) timeout_minutes="$2"; shift 2 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

[[ -n "${gpu_id}" && -n "${c4_model}" && -n "${c5_model}" ]] || {
    printf 'ERROR: --gpu-id, --c4-model and --c5-model are required.\n' >&2
    exit 64
}
[[ "${gpu_id}" != "2" ]] || {
    printf 'ERROR: GPU 2 is reserved and must not be queried or used.\n' >&2
    exit 64
}
utilization="$(nvidia-smi -i "${gpu_id}" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d '[:space:]')"
[[ "${utilization}" =~ ^[0-9]+$ ]] || { printf 'ERROR: invalid GPU utilization.\n' >&2; exit 75; }
(( utilization < utilization_limit )) || {
    printf 'ERROR: GPU%s utilization=%s; bounded diagnostics not started.\n' "${gpu_id}" "${utilization}" >&2
    exit 75
}

run_root="${GOT_EVALUATION_RUNS}/${run_id}"
mkdir "${run_root}"

run_one() {
    local label="$1" model="$2"
    local output="${run_root}/${label}/summary.json"
    mkdir "${run_root}/${label}"
    timeout --signal=TERM "${timeout_minutes}m" \
        env CUDA_VISIBLE_DEVICES="${gpu_id}" \
        bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/evaluation/diagnose_pvld_bounded.py" \
        --project-root "${ocrmodel_root}/src/GOT-OCR-2.0" \
        --model "${model}" \
        --tokenizer "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --train-manifest "${dataset_root}/train/manifest.jsonl" \
        --train-image-root "${dataset_root}/train" \
        --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
        --validation-image-root "${dataset_root}/validation" \
        --output "${output}" --label "${label}" \
        >"${run_root}/${label}/diagnostic.log" 2>&1 || {
            tail -n 20 "${run_root}/${label}/diagnostic.log" >&2
            return 1
        }
}

run_one C4 "${c4_model}"
run_one C5 "${c5_model}"
jq -s '{status:"completed",test_read:false,optimizer_steps:0,controls:.}' \
    "${run_root}/C4/summary.json" "${run_root}/C5/summary.json" \
    >"${run_root}/summary.json"
jq -c '{status,test_read,optimizer_steps,controls:[.controls[]|{label,oracle_vs_free:.oracle_vs_free.summary,ocr_routing:.ocr_routing_ablation.conditions,gradient_audit:.gradient_audit}]}' \
    "${run_root}/summary.json"

