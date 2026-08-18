#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
source "${ocrmodel_root}/config/paths.env"

export CUDA_VISIBLE_DEVICES=""

stamp="$(date +%Y%m%d_%H%M%S)"
log_root="${GOT_LAYOUT_DATA}/_server_synthesis_environment_logs"
log_path="${log_root}/${stamp}.log"
mkdir -p "${log_root}"

if ! timeout 1100 bash "${ocrmodel_root}/tools/environment/setup_layout_synthesis_env.sh" >"${log_path}" 2>&1; then
    tail -n 20 "${log_path}" >&2
    printf '{"event":"server_synthesis_smoke_failed","stage":"environment_setup","log":"%s"}\n' "${log_path}" >&2
    exit 1
fi

exec bash "${script_dir}/run_server_synthesis_smoke.sh"
