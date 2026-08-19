#!/usr/bin/env bash
set -euo pipefail

dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2"
input_root="${dataset_root}/converted/mthv2_layout_page_v1"
output_root="${dataset_root}/converted/mthv2_layout_column_chunks16_v1"
max_regions="16"
margin_pixels="8"
overwrite=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-root) input_root="${2:-}"; shift 2 ;;
        --output-root) output_root="${2:-}"; shift 2 ;;
        --max-regions) max_regions="${2:-}"; shift 2 ;;
        --margin-pixels) margin_pixels="${2:-}"; shift 2 ;;
        --overwrite) overwrite=1; shift ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; exit 64 ;;
    esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
args=(
    "${script_dir}/split_mthv2_layout_chunks.py"
    --input-root "${input_root}"
    --output-root "${output_root}"
    --max-regions "${max_regions}"
    --margin-pixels "${margin_pixels}"
)
if [[ "${overwrite}" -eq 1 ]]; then
    args+=(--overwrite)
fi
python3 "${args[@]}"
