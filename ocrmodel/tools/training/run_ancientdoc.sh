#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
if [[ -f "${ocrmodel_root}/config/paths.env" ]]; then
    # shellcheck source=/dev/null
    source "${ocrmodel_root}/config/paths.env"
fi

command="${1:-}"
if [[ -z "${command}" || "${command}" == "--help" || "${command}" == "-h" ]]; then
    cat <<'USAGE'
Usage:
  run_ancientdoc.sh prepare [dataset options]
  run_ancientdoc.sh train-core [training options]
  run_ancientdoc.sh select-c4 [selection options]
  run_ancientdoc.sh train-replay --c4-selection <selection.json> [training options]

prepare builds a book-isolated near-60/20/20 dataset and runs leakage audits.
train-core trains C1/C4 only and stops at the C4 branch boundary.
select-c4 evaluates only C4 periodic checkpoints on validation; it never reads test.
train-replay resolves one frozen C4-best and independently trains C5/C6.
USAGE
    exit 0
fi
shift

case "${command}" in
    prepare)
        exec bash "${ocrmodel_root}/tools/preprocessing/run_ancientdoc_group_isolated_prepare.sh" "$@"
        ;;
    train-core)
        exec bash "${script_dir}/run_ancientdoc_baseline_suite.sh" --phase core "$@"
        ;;
    select-c4)
        exec python3 "${ocrmodel_root}/tools/evaluation/select_ancientdoc_c4.py" \
            --ocrmodel-root "${ocrmodel_root}" "$@"
        ;;
    train-replay)
        exec bash "${script_dir}/run_ancientdoc_baseline_suite.sh" --phase replay "$@"
        ;;
    train)
        printf 'ERROR: legacy train would implicitly use C4-final for C5/C6 and is disabled. Use train-core, select-c4, then train-replay.\n' >&2
        exit 64
        ;;
    *)
        printf 'ERROR: expected prepare, train-core, select-c4, or train-replay; got %s\n' "${command}" >&2
        exit 64
        ;;
esac
