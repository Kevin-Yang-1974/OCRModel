#!/usr/bin/env bash
set -euo pipefail

local_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
remote_host="${BSCC_SSH_HOST:-bscc-n32h}"
remote_root="${BSCC_GLM_CODE_ROOT:-yangky_ocr_models_bscc_proto/glm_ocr_layout_ot/code/ocrmodel}"

rsync -az --delete \
    --rsync-path="mkdir -p '${remote_root}' && rsync" \
    --exclude '.venv/' --exclude '.pytest_cache/' --exclude '.ruff_cache/' \
    --exclude '__pycache__/' --exclude '*.pyc' --exclude 'runs/' --exclude 'outputs/' \
    --exclude 'models/' --exclude 'data/' --exclude 'checkpoints/' \
    "${local_root}/" "${remote_host}:${remote_root}/"

printf '{"event":"glm_ocr_bscc_sync_complete","host":"%s","remote_root":"%s"}\n' \
    "${remote_host}" "${remote_root}"
