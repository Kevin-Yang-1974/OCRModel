#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  run_formal_layout_post_analysis.sh <dataset-id> --vlqa-model <p2-model-path> --run-id <comparison-run-id> [extra comparison args...]

This wrapper performs the post-training formal test workflow for one completed
P2 VLQA checkpoint:
  1. original GOT2 vs VLQA comparison on the formal test split;
  2. offline error analysis;
  3. offline object-threshold sweep;
  4. offline slot-alignment diagnosis;
  5. analysis bundle summary.

It does not train and does not use GPU after the comparison step. The final
terminal line is the compact JSON emitted by summarize_layout_analysis_bundle.py.
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
if [[ -z "${GOT_EVALUATION_RUNS:-}" ]]; then
    printf 'ERROR: GOT_EVALUATION_RUNS is not set. Source config/paths.env or pass through the environment.\n' >&2
    exit 64
fi

vlqa_model=""
run_id=""
extra_args=()
while [[ $# -gt 0 ]]; do
    case "${1}" in
        --vlqa-model)
            vlqa_model="${2:-}"
            shift 2
            ;;
        --run-id)
            run_id="${2:-}"
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

if [[ -z "${vlqa_model}" || -z "${run_id}" ]]; then
    printf 'ERROR: post-analysis requires --vlqa-model and --run-id.\n' >&2
    exit 64
fi

dataset_root="${GOT_LAYOUT_DATA}/${dataset_id}"
manifest="${dataset_root}/test/manifest.jsonl"
comparison_root="${GOT_EVALUATION_RUNS}/${run_id}"

if [[ -e "${comparison_root}" ]]; then
    printf 'ERROR: comparison run already exists: %s\n' "${comparison_root}" >&2
    printf 'Use a new --run-id to avoid repeated test querying or accidental overwrite.\n' >&2
    exit 74
fi

bash "${ocrmodel_root}/tools/training/check_layout_dataset_mount.sh" --dataset-root "${dataset_root}" >/dev/null

bash "${ocrmodel_root}/tools/evaluation/run_formal_layout_comparison.sh" \
    "${dataset_id}" \
    --vlqa-model "${vlqa_model}" \
    --run-id "${run_id}" \
    "${extra_args[@]}"

python3 "${ocrmodel_root}/tools/evaluation/analyze_layout_comparison_errors.py" \
    --comparison-root "${comparison_root}" \
    --manifest "${manifest}" >/dev/null

python3 "${ocrmodel_root}/tools/evaluation/analyze_layout_threshold_sweep.py" \
    --comparison-root "${comparison_root}" \
    --manifest "${manifest}" >/dev/null

python3 "${ocrmodel_root}/tools/evaluation/analyze_layout_slot_alignment.py" \
    --comparison-root "${comparison_root}" \
    --manifest "${manifest}" >/dev/null

exec python3 "${ocrmodel_root}/tools/evaluation/summarize_layout_analysis_bundle.py" \
    --comparison-root "${comparison_root}"
