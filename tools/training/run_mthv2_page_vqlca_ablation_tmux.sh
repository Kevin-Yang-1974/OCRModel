#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
dataset_root="/data3/yky/yangky_ocr_models/datasets/MTHv2/converted/mthv2_layout_page_v1"

exec bash "${script_dir}/run_mthv2_chunk_ablation_tmux.sh" \
    "$@" \
    --dataset-root "${dataset_root}" \
    --max-regions 512 \
    --vlqa-writeback-mode vqlca
