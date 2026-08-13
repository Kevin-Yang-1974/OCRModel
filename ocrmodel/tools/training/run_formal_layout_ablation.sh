#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_formal_layout_ablation.sh <preset> <dataset-id> --p1-model <path> --p2-steps <steps> [extra run_layout_a100.py args...]

Loss-supervision presets:
  full              Full P2 VLQA: object + bbox L1/GIoU + direction + OCR.
  no-direction      Disable direction supervision; keep object + bbox + OCR.
  no-bbox           Disable bbox L1/GIoU supervision; keep object + direction + OCR.
  object-only       Keep object supervision + OCR; disable bbox and direction.
  ocr-only-adapter  Train the VLQA adapter through OCR only; disable all layout losses.

The dataset is read from $GOT_LAYOUT_DATA/<dataset-id> and must contain:
  train/manifest.jsonl
  validation/manifest.jsonl
  test/manifest.jsonl

This wrapper starts one P2 formal joint-train ablation from a completed P1
checkpoint. It does not implement equal-parameter non-VLQA structural controls;
those require separate model code and should not be conflated with these loss
ablation presets.
USAGE
}

if [[ $# -lt 2 ]]; then
    usage
    exit 64
fi

preset="${1}"
dataset_id="${2}"
shift 2

object_loss_weight="1"
bbox_l1_loss_weight="5"
bbox_giou_loss_weight="2"
direction_loss_weight="1"
layout_loss_weight="1"
case "${preset}" in
    full)
        ;;
    no-direction)
        direction_loss_weight="0"
        ;;
    no-bbox)
        bbox_l1_loss_weight="0"
        bbox_giou_loss_weight="0"
        ;;
    object-only)
        bbox_l1_loss_weight="0"
        bbox_giou_loss_weight="0"
        direction_loss_weight="0"
        ;;
    ocr-only-adapter)
        object_loss_weight="0"
        bbox_l1_loss_weight="0"
        bbox_giou_loss_weight="0"
        direction_loss_weight="0"
        layout_loss_weight="0"
        ;;
    --help|-h)
        usage
        exit 0
        ;;
    *)
        printf 'ERROR: unsupported ablation preset: %s\n' "${preset}" >&2
        usage
        exit 64
        ;;
esac

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

p1_model=""
p2_steps=""
extra_args=()
while [[ $# -gt 0 ]]; do
    case "${1}" in
        --p1-model)
            p1_model="${2:-}"
            shift 2
            ;;
        --p2-steps)
            p2_steps="${2:-}"
            shift 2
            ;;
        --object-loss-weight|--bbox-l1-loss-weight|--bbox-giou-loss-weight|--direction-loss-weight|--layout-loss-weight|--p2-ocr-loss-weight)
            printf 'ERROR: loss weights are owned by preset %s; do not pass %s.\n' "${preset}" "${1}" >&2
            exit 64
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

if [[ -z "${p1_model}" || -z "${p2_steps}" ]]; then
    printf 'ERROR: ablation requires --p1-model and --p2-steps.\n' >&2
    exit 64
fi

dataset_root="${GOT_LAYOUT_DATA}/${dataset_id}"
bash "${script_dir}/check_layout_dataset_mount.sh" --dataset-root "${dataset_root}" >/dev/null

exec bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
    "${ocrmodel_root}/tools/training/run_layout_a100.py" \
    --dataset-root "${dataset_root}/train" \
    --manifest "${dataset_root}/train/manifest.jsonl" \
    --validation-manifest "${dataset_root}/validation/manifest.jsonl" \
    --validation-image-root "${dataset_root}/validation" \
    --test-manifest "${dataset_root}/test/manifest.jsonl" \
    --test-image-root "${dataset_root}/test" \
    --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}" \
    --source-model "${p1_model}" \
    --p2-max-steps "${p2_steps}" \
    --mode joint-train \
    --object-loss-weight "${object_loss_weight}" \
    --bbox-l1-loss-weight "${bbox_l1_loss_weight}" \
    --bbox-giou-loss-weight "${bbox_giou_loss_weight}" \
    --direction-loss-weight "${direction_loss_weight}" \
    --layout-loss-weight "${layout_loss_weight}" \
    "${extra_args[@]}"
