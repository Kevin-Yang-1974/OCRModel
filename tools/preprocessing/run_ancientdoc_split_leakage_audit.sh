#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_ancientdoc_split_leakage_audit.sh [options]

Offline AncientDoc split leakage audit. The command reads converted manifests
and images only. It does not train, evaluate, or use GPU.

Options:
  --dataset-id <id>                  Default: ancientdoc_layout_260707
  --dataset-root <path>              Overrides $GOT_LAYOUT_DATA/<dataset-id>
  --output-dir <path>                Optional output directory
  --skip-image-hash                  Skip exact image SHA-256 checks
  --enable-perceptual-hash           Also run optional average-hash near-duplicate check
  --perceptual-hash-threshold <n>    Default: 4
  --max-examples <n>                 Default: 30
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
paths_env="${ocrmodel_root}/config/paths.env"
if [[ -f "${paths_env}" ]]; then
    # shellcheck source=/dev/null
    source "${paths_env}"
fi

dataset_id="ancientdoc_layout_260707"
dataset_root=""
output_dir=""
skip_image_hash=0
enable_perceptual_hash=0
perceptual_hash_threshold="4"
max_examples="30"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset-id) dataset_id="${2:-}"; shift 2 ;;
        --dataset-root) dataset_root="${2:-}"; shift 2 ;;
        --output-dir) output_dir="${2:-}"; shift 2 ;;
        --skip-image-hash) skip_image_hash=1; shift ;;
        --enable-perceptual-hash) enable_perceptual_hash=1; shift ;;
        --perceptual-hash-threshold) perceptual_hash_threshold="${2:-}"; shift 2 ;;
        --max-examples) max_examples="${2:-}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage; exit 64 ;;
    esac
done

if [[ -z "${dataset_root}" ]]; then
    if [[ -z "${GOT_LAYOUT_DATA:-}" ]]; then
        printf 'ERROR: GOT_LAYOUT_DATA must be set or --dataset-root must be passed.\n' >&2
        exit 64
    fi
    dataset_root="${GOT_LAYOUT_DATA}/${dataset_id}"
fi

args=(
    "${ocrmodel_root}/tools/preprocessing/audit_ancientdoc_split_leakage.py"
    --dataset-root "${dataset_root}"
    --perceptual-hash-threshold "${perceptual_hash_threshold}"
    --max-examples "${max_examples}"
)
if [[ -n "${output_dir}" ]]; then
    args+=(--output-dir "${output_dir}")
fi
if [[ "${skip_image_hash}" -eq 1 ]]; then
    args+=(--skip-image-hash)
fi
if [[ "${enable_perceptual_hash}" -eq 1 ]]; then
    args+=(--enable-perceptual-hash)
fi

python3 "${args[@]}"
