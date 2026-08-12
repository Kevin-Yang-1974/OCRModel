#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_formal_layout_p1p2.sh pretrain <dataset-id> --p1-steps <steps> [extra run_layout_a100.py args...]
  run_formal_layout_p1p2.sh joint-train <dataset-id> --p1-model <path> --p2-steps <steps> [extra run_layout_a100.py args...]

The dataset is read from $GOT_LAYOUT_DATA/<dataset-id> and must contain:
  train/manifest.jsonl
  validation/manifest.jsonl
  test/manifest.jsonl

This wrapper only starts one formal stage. Run P2 only after P1 held-out
validation and checkpoint verification have passed.
USAGE
}

if [[ $# -lt 2 ]]; then
    usage
    exit 64
fi

mode="${1}"
dataset_id="${2}"
shift 2
case "${mode}" in
    pretrain|joint-train) ;;
    *)
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

p1_steps=""
p2_steps=""
p1_model=""
extra_args=()
while [[ $# -gt 0 ]]; do
    case "${1}" in
        --p1-steps)
            p1_steps="${2:-}"
            shift 2
            ;;
        --p2-steps)
            p2_steps="${2:-}"
            shift 2
            ;;
        --p1-model)
            p1_model="${2:-}"
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

dataset_root="${GOT_LAYOUT_DATA}/${dataset_id}"
bash "${script_dir}/check_layout_dataset_mount.sh" --dataset-root "${dataset_root}" >/dev/null

common_args=(
    "${ocrmodel_root}/tools/training/run_layout_a100.py"
    --dataset-root "${dataset_root}/train"
    --manifest "${dataset_root}/train/manifest.jsonl"
    --validation-manifest "${dataset_root}/validation/manifest.jsonl"
    --validation-image-root "${dataset_root}/validation"
    --test-manifest "${dataset_root}/test/manifest.jsonl"
    --test-image-root "${dataset_root}/test"
    --tokenizer-model "${GOT_TOKENIZER_MODEL:-${GOT_SOURCE_MODEL}}"
)

case "${mode}" in
    pretrain)
        if [[ -z "${p1_steps}" ]]; then
            printf 'ERROR: pretrain requires --p1-steps.\n' >&2
            exit 64
        fi
        exec bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
            "${common_args[@]}" \
            --source-model "${GOT_SOURCE_MODEL}" \
            --p1-max-steps "${p1_steps}" \
            --mode pretrain \
            "${extra_args[@]}"
        ;;
    joint-train)
        if [[ -z "${p1_model}" || -z "${p2_steps}" ]]; then
            printf 'ERROR: joint-train requires --p1-model and --p2-steps.\n' >&2
            exit 64
        fi
        exec bash "${ocrmodel_root}/tools/environment/run_got2.sh" \
            "${common_args[@]}" \
            --source-model "${p1_model}" \
            --p2-max-steps "${p2_steps}" \
            --mode joint-train \
            "${extra_args[@]}"
        ;;
esac
