#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat >&2 <<'USAGE'
Usage:
  check_layout_dataset_mount.sh <dataset-id>
  check_layout_dataset_mount.sh --dataset-root <path>

Checks the server-side formal whole-page layout dataset mount/copy without
starting training. The dataset must contain train/, validation/, test/, and
split_audit.json. The default root is $GOT_LAYOUT_DATA/<dataset-id>.
USAGE
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ocrmodel_root="${OCRMODEL_ROOT:-$(cd -- "${script_dir}/../.." && pwd -P)}"
paths_env="${ocrmodel_root}/config/paths.env"
if [[ -f "${paths_env}" ]]; then
    # shellcheck source=/dev/null
    source "${paths_env}"
fi

dataset_root=""
if [[ $# -eq 1 && "${1}" != "--dataset-root" ]]; then
    if [[ -z "${GOT_LAYOUT_DATA:-}" ]]; then
        printf 'ERROR: GOT_LAYOUT_DATA is not set and --dataset-root was not provided.\n' >&2
        exit 64
    fi
    dataset_root="${GOT_LAYOUT_DATA}/${1}"
elif [[ $# -eq 2 && "${1}" == "--dataset-root" ]]; then
    dataset_root="${2}"
else
    usage
    exit 64
fi

dataset_root="$(cd -- "${dataset_root}" && pwd -P)"
required_files=(
    "train/manifest.jsonl"
    "validation/manifest.jsonl"
    "test/manifest.jsonl"
)
for relative in "${required_files[@]}"; do
    if [[ ! -f "${dataset_root}/${relative}" ]]; then
        printf 'ERROR: missing required dataset file: %s\n' "${dataset_root}/${relative}" >&2
        exit 66
    fi
done
audit_file=""
if [[ -f "${dataset_root}/split_audit.json" ]]; then
    audit_file="${dataset_root}/split_audit.json"
elif [[ -f "${dataset_root}/audit_summary.json" ]]; then
    # Diverse synthetic generation emits audit_summary.json; preserve compatibility
    # with the older AncientDoc mount contract without duplicating the large report.
    audit_file="${dataset_root}/audit_summary.json"
else
    printf 'ERROR: missing split_audit.json or audit_summary.json in %s\n' "${dataset_root}" >&2
    exit 66
fi
for split in train validation test; do
    if [[ ! -d "${dataset_root}/${split}/images" ]]; then
        printf 'ERROR: missing image directory: %s\n' "${dataset_root}/${split}/images" >&2
        exit 66
    fi
done

python3 - "$dataset_root" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = {
    "event": "layout_dataset_mount_ok",
    "dataset_root": str(root),
    "splits": {},
    "audit_status": None,
}
for split in ("train", "validation", "test"):
    manifest = root / split / "manifest.jsonl"
    count = 0
    first_image = None
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            count += 1
            if first_image is None:
                record = json.loads(line)
                first_image = record.get("image")
    if count < 1:
        raise SystemExit(f"empty manifest: {manifest}")
    if not isinstance(first_image, str) or not first_image:
        raise SystemExit(f"manifest has no relative image path: {manifest}")
    if not (root / split / first_image).exists():
        raise SystemExit(f"first manifest image is missing: {root / split / first_image}")
    summary["splits"][split] = {"manifest_records": count, "image_root": str(root / split)}

audit_path = root / "split_audit.json"
if not audit_path.is_file():
    audit_path = root / "audit_summary.json"
audit = json.loads(audit_path.read_text(encoding="utf-8"))
summary["audit_status"] = audit.get("status") or audit.get("event")
summary["audit_file"] = str(audit_path)
print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
PY
