#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_ancientdoc_paired_analysis.sh --suite-root <validation-suite-root> [options]

Offline paired OCR analysis for completed AncientDoc evaluation outputs.
No model is loaded, no GPU is used, and no training or inference is started.

Options:
  --suite-root <path>          Required. Example: $GOT_EVALUATION_RUNS/ancientdoc_validation_20260814
  --left-label <label>         Default: c4
  --right-label <label>        Default: c6
  --manifest <path>            Optional explicit test manifest
  --output-dir <path>          Optional output directory
  --top-k <n>                  Default: 30
  --bootstrap-samples <n>      Default: 2000; use 0 to disable
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
paths_env="${ocrmodel_root}/config/paths.env"
if [[ -f "${paths_env}" ]]; then
    # shellcheck source=/dev/null
    source "${paths_env}"
fi

suite_root=""
left_label="c4"
right_label="c6"
manifest=""
output_dir=""
top_k="30"
bootstrap_samples="2000"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite-root) suite_root="${2:-}"; shift 2 ;;
        --left-label) left_label="${2:-}"; shift 2 ;;
        --right-label) right_label="${2:-}"; shift 2 ;;
        --manifest) manifest="${2:-}"; shift 2 ;;
        --output-dir) output_dir="${2:-}"; shift 2 ;;
        --top-k) top_k="${2:-}"; shift 2 ;;
        --bootstrap-samples) bootstrap_samples="${2:-}"; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'ERROR: unknown argument: %s\n' "$1" >&2; usage; exit 64 ;;
    esac
done

if [[ -z "${suite_root}" ]]; then
    printf 'ERROR: --suite-root is required.\n' >&2
    usage
    exit 64
fi

args=(
    "${ocrmodel_root}/tools/evaluation/analyze_ancientdoc_baseline_pages.py"
    --suite-root "${suite_root}"
    --left-label "${left_label}"
    --right-label "${right_label}"
    --top-k "${top_k}"
    --bootstrap-samples "${bootstrap_samples}"
)
if [[ -n "${manifest}" ]]; then
    args+=(--manifest "${manifest}")
fi
if [[ -n "${output_dir}" ]]; then
    args+=(--output-dir "${output_dir}")
fi

python3 "${args[@]}"
