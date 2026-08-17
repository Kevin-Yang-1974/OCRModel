#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
if [[ -f "${ocrmodel_root}/config/paths.env" ]]; then
    # shellcheck source=/dev/null
    source "${ocrmodel_root}/config/paths.env"
fi

exec python3 "${script_dir}/run_ancientdoc_evaluation.py" \
    --ocrmodel-root "${ocrmodel_root}" \
    "$@"
