#!/usr/bin/env bash
#SBATCH --job-name=mthv2_prepare
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log
#SBATCH --error=/home/bingxing2/home/scx9fxd/yangky_ocr_models_bscc_proto/runs/slurm_%x_%j.log

set -euo pipefail

workspace="${BSCC_OCR_WORKSPACE:-${HOME}/yangky_ocr_models_bscc_proto}"
archive="${BSCC_MTHV2_ARCHIVE:-${workspace}/datasets/incoming/TKHMTH2200.zip}"
dataset_root="${BSCC_MTHV2_ROOT:-${workspace}/datasets/MTHv2}"
raw_root="${dataset_root}/raw/TKHMTH2200"
split_root="${dataset_root}/official_splits"
output_root="${dataset_root}/converted/mthv2_layout_page_v1"
status_root="${dataset_root}/metadata"
mkdir -p "${status_root}"

[[ -f "${archive}" ]] || { printf 'ERROR: missing archive: %s\n' "${archive}" >&2; exit 66; }
[[ -f "${split_root}/train.txt" && -f "${split_root}/test.txt" ]] || {
    printf 'ERROR: official split files are missing below %s.\n' "${split_root}" >&2
    exit 66
}
[[ ! -e "${raw_root}" && ! -e "${output_root}" ]] || {
    printf 'ERROR: raw or converted MTHv2 output already exists; refusing to overwrite.\n' >&2
    exit 73
}

python3 - "${archive}" <<'PY'
import sys
from pathlib import PurePosixPath
from zipfile import ZipFile

with ZipFile(sys.argv[1]) as archive:
    names = archive.namelist()
    unsafe = [
        name for name in names
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts
    ]
    if unsafe:
        raise RuntimeError(f"Unsafe ZIP members: {unsafe[:5]}")
    expected = {"MTH1000", "MTH1200", "TKH"}
    observed = {PurePosixPath(name).parts[0] for name in names if name}
    if observed != expected:
        raise RuntimeError(f"Unexpected ZIP roots: {sorted(observed)}")
    if len(names) != len(set(names)):
        raise RuntimeError("ZIP contains duplicate member names")
PY

mkdir -p "${raw_root}"
unzip -q "${archive}" -d "${raw_root}"
python3 "${workspace}/ocrmodel/tools/preprocessing/prepare_mthv2_layout_dataset.py" \
    --raw-root "${raw_root}" \
    --train-list "${split_root}/train.txt" \
    --test-list "${split_root}/test.txt" \
    --output-root "${output_root}" \
    --validation-ratio 0.1 --seed 20260818

python3 - "${output_root}" "${status_root}/prepare_summary.json" <<'PY'
import json
import sys
from pathlib import Path

root, output = map(Path, sys.argv[1:])
counts = {}
for split in ("train", "validation", "test"):
    manifest = root / split / "manifest.jsonl"
    if not manifest.is_file():
        raise RuntimeError(f"Missing converted manifest: {manifest}")
    counts[split] = sum(1 for line in manifest.open(encoding="utf-8") if line.strip())
expected = {"train": 2159, "validation": 240, "test": 800}
if counts != expected:
    raise RuntimeError(f"Unexpected split counts: {counts}; expected {expected}")
payload = {
    "status": "ok",
    "dataset_id": "mthv2_layout_page_v1",
    "input_granularity": "whole_page_image",
    "split_counts": counts,
    "validation_from_official_train_only": True,
    "test_locked_for_selection": True,
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, separators=(",", ":")))
PY
touch "${status_root}/MTHV2_PREPARED"
