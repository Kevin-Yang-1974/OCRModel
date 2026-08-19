#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_mthv2_layout_prepare.sh [options]

Convert MTHv2 into whole-page OCR manifests with text-line layout regions and
per-page character/boundary sidecars. This command does not train a model.

Options:
  --dataset-root <path>      Default: /data3/yky/yangky_ocr_models/datasets/MTHv2
  --raw-root <path>          Default: <dataset-root>/raw/TKHMTH2200
  --split-root <path>        Default: <dataset-root>/official_splits
  --output-root <path>       Default: <dataset-root>/converted/mthv2_layout_page_v1
  --validation-ratio <x>     Default: 0.1 (drawn only from official train)
  --seed <n>                 Default: 20260818
  --copy-images              Copy images instead of absolute symlinks
  --overwrite                Remove and recreate the converted output
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2"
raw_root=""
split_root=""
output_root=""
validation_ratio="0.1"
seed="20260818"
copy_images=0
overwrite=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-root) dataset_root="${2:-}"; shift 2 ;;
        --raw-root) raw_root="${2:-}"; shift 2 ;;
        --split-root) split_root="${2:-}"; shift 2 ;;
        --output-root) output_root="${2:-}"; shift 2 ;;
        --validation-ratio) validation_ratio="${2:-}"; shift 2 ;;
        --seed) seed="${2:-}"; shift 2 ;;
        --copy-images) copy_images=1; shift ;;
        --overwrite) overwrite=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage; exit 64 ;;
    esac
done

raw_root="${raw_root:-${dataset_root}/raw/TKHMTH2200}"
split_root="${split_root:-${dataset_root}/official_splits}"
output_root="${output_root:-${dataset_root}/converted/mthv2_layout_page_v1}"

args=(
    "${script_dir}/prepare_mthv2_layout_dataset.py"
    --raw-root "${raw_root}"
    --train-list "${split_root}/train.txt"
    --test-list "${split_root}/test.txt"
    --output-root "${output_root}"
    --validation-ratio "${validation_ratio}"
    --seed "${seed}"
)
if [[ "${copy_images}" -eq 1 ]]; then
    args+=(--copy-images)
fi
if [[ "${overwrite}" -eq 1 ]]; then
    args+=(--overwrite)
fi

python3 "${args[@]}"
