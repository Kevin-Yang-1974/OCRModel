#!/usr/bin/env bash
set -euo pipefail

# Compatibility alias. The canonical entry is run_ancientdoc.sh.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec bash "${script_dir}/run_ancientdoc.sh" "$@"
