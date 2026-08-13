#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_formal_layout_comparison.sh <dataset-id> --vlqa-model <p2-model-path> [extra compare_got2_vlqa.py args...]

The dataset is read from $GOT_LAYOUT_DATA/<dataset-id> and must contain:
  train/manifest.jsonl
  validation/manifest.jsonl
  test/manifest.jsonl

This wrapper compares the original GOT2 baseline with one completed P2 VLQA
checkpoint on the same formal test split. It does not train.
USAGE
}

if [[ $# -lt 1 ]]; then
    usage
    exit 64
fi

dataset_id="${1}"
shift

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
paths_env="${ocrmodel_root}/config/paths.env"
if [[ -f "${paths_env}" ]]; then
    # shellcheck source=/dev/null
    source "${paths_env}"
fi

if [[ -z "${GOT_LAYOUT_DATA:-}" ]]; then
    printf 'ERROR: GOT_LAYOUT_DATA is not set. Source config/paths.env or pass through the environment.\n' >&2
    exit 64
fi
if [[ -z "${GOT_SOURCE_MODEL:-}" ]]; then
    printf 'ERROR: GOT_SOURCE_MODEL is not set. Source config/paths.env or pass through the environment.\n' >&2
    exit 64
fi

vlqa_model=""
extra_args=()
while [[ $# -gt 0 ]]; do
    case "${1}" in
        --vlqa-model)
            vlqa_model="${2:-}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            extra_args+=("${1}")
            shift
            ;;
    esac
done

if [[ -z "${vlqa_model}" ]]; then
    printf 'ERROR: --vlqa-model is required and must point to a completed P2 checkpoint.\n' >&2
    exit 64
fi

dataset_root="${GOT_LAYOUT_DATA}/${dataset_id}"
bash "${ocrmodel_root}/tools/training/check_layout_dataset_mount.sh" --dataset-root "${dataset_root}" >/dev/null

exec bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
    "${ocrmodel_root}/tools/evaluation/compare_got2_vlqa.py" \
    --baseline-model "${GOT_SOURCE_MODEL}" \
    --vlqa-model "${vlqa_model}" \
    --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
    --train-manifest "${dataset_root}/train/manifest.jsonl" \
    --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
    --layout-manifest "${dataset_root}/test/manifest.jsonl" \
    --layout-image-root "${dataset_root}/test" \
    --layout-split test \
    "${extra_args[@]}"
