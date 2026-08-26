#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

selection="${1:?selection.json required}"
session="${2:?tmux session required}"
gpu_ids="${3:?comma-separated GPU IDs required}"
output="${4:?new test output directory required}"
dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1"
launcher_log="${output}.launcher.log"

run_test() {
    bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
        "${ocrmodel_root}/tools/evaluation/evaluate_layout_ablation_test.py" \
        --selection "${selection}" --test-category Real-OOD \
        --test-manifest "${dataset_root}/test/manifest.jsonl" \
        --test-image-root "${dataset_root}/test" --model-kind pvld \
        --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
        --project-root "${ocrmodel_root}/src/GOT-OCR-2.0" --output-dir "${output}" \
        --max-regions 512 --max-records 0 --max-new-tokens 2048 \
        --no-repeat-ngram-size 20 --parallel-gpu-ids "${gpu_ids}" \
        --gpu-utilization-limit 50
}

if [[ "${PVLD_MULTI_TEST_INNER:-0}" == 1 ]]; then
    run_test >"${launcher_log}" 2>&1
    exit
fi

tmux has-session -t "${session}" 2>/dev/null && {
    printf 'ERROR: tmux session exists: %s\n' "${session}" >&2
    exit 73
}
[[ ! -e "${output}" && ! -e "${launcher_log}" ]] || {
    printf 'ERROR: test output or launcher log already exists: %s\n' "${output}" >&2
    exit 74
}
script_path="$(realpath "${BASH_SOURCE[0]}")"
tmux new-session -d -s "${session}" \
    "PVLD_MULTI_TEST_INNER=1 bash '${script_path}' '${selection}' '${session}' '${gpu_ids}' '${output}'"
printf '{"event":"pvld_multigpu_test_started","session":"%s","selection":"%s","gpus":"%s","output":"%s","log":"%s"}\n' \
    "${session}" "${selection}" "${gpu_ids}" "${output}" "${launcher_log}"
