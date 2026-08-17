#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_ancientdoc_group_isolated_prepare.sh [options]

Prepare a new AncientDoc layout-page dataset with category/book-isolated splits.
The command reads the original AncientDoc GOT labels and writes converted
manifest/images under $GOT_LAYOUT_DATA. It does not train or evaluate a model.

Options:
  --ancientdoc-root <path>       Default: /data4/hyf/backup/GOT-OCR2.0/reference-260707/AncientDoc
  --output-id <id>               Default: ancientdoc_layout_260707_group_isolated_seed20260815
  --output-root <path>           Overrides $GOT_LAYOUT_DATA/<output-id>
  --seed <n>                     Default: 20260815
  --train-ratio <float>          Default: 0.6
  --validation-ratio <float>     Default: 0.2
  --test-ratio <float>           Default: 0.2
  --max-ratio-deviation <float>  Default: 0.03
  --copy-images                  Copy images instead of symlinking them
  --overwrite                    Remove and recreate output root
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
paths_env="${ocrmodel_root}/config/paths.env"
if [[ -f "${paths_env}" ]]; then
    # shellcheck source=/dev/null
    source "${paths_env}"
fi

ancientdoc_root="/data4/hyf/backup/GOT-OCR2.0/reference-260707/AncientDoc"
output_id="ancientdoc_layout_260707_group_isolated_seed20260815"
output_root=""
seed="20260815"
train_ratio="0.6"
validation_ratio="0.2"
test_ratio="0.2"
max_ratio_deviation="0.03"
copy_images=0
overwrite=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ancientdoc-root) ancientdoc_root="${2:-}"; shift 2 ;;
        --output-id) output_id="${2:-}"; shift 2 ;;
        --output-root) output_root="${2:-}"; shift 2 ;;
        --seed) seed="${2:-}"; shift 2 ;;
        --train-ratio) train_ratio="${2:-}"; shift 2 ;;
        --validation-ratio) validation_ratio="${2:-}"; shift 2 ;;
        --test-ratio) test_ratio="${2:-}"; shift 2 ;;
        --max-ratio-deviation) max_ratio_deviation="${2:-}"; shift 2 ;;
        --copy-images) copy_images=1; shift ;;
        --overwrite) overwrite=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage; exit 64 ;;
    esac
done

if [[ -z "${output_root}" ]]; then
    if [[ -z "${GOT_LAYOUT_DATA:-}" ]]; then
        printf 'ERROR: GOT_LAYOUT_DATA must be set or --output-root must be passed.\n' >&2
        exit 64
    fi
    output_root="${GOT_LAYOUT_DATA}/${output_id}"
fi

args=(
    "${ocrmodel_root}/tools/preprocessing/prepare_ancientdoc_group_isolated_dataset.py"
    --ancientdoc-root "${ancientdoc_root}"
    --output-root "${output_root}"
    --seed "${seed}"
    --train-ratio "${train_ratio}"
    --validation-ratio "${validation_ratio}"
    --test-ratio "${test_ratio}"
    --max-ratio-deviation "${max_ratio_deviation}"
)
if [[ "${copy_images}" -eq 0 ]]; then
    args+=(--symlink-images)
fi
if [[ "${overwrite}" -eq 1 ]]; then
    args+=(--overwrite)
fi

python3 "${args[@]}"

python3 "${ocrmodel_root}/tools/preprocessing/audit_ancientdoc_split_leakage.py" \
    --dataset-root "${output_root}" \
    --output-dir "${output_root}/audit/ancientdoc_split_leakage"

python3 "${ocrmodel_root}/tools/preprocessing/verify_ancientdoc_group_audit.py" \
    "${output_root}/audit/ancientdoc_split_leakage/split_leakage_audit.json" \
    --split-audit "${output_root}/split_audit.json" \
    --max-ratio-deviation "${max_ratio_deviation}"
