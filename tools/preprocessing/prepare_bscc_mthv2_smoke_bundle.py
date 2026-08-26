#!/usr/bin/env python3
"""Copy the first N MTHv2 train whole pages into a small BSCC smoke bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    records = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= args.pages:
                break
            record = json.loads(line)
            if record["split"] == "train" and record["input_level"] == "page":
                records.append(record)
    image_hashes = {}
    for record in records:
        relative = Path(record["image"])
        source = args.image_root / relative
        destination = args.output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        image_hashes[record["page_id"]] = {
            "path": relative.as_posix(), "sha256": sha256(destination)
        }
    manifest_path = args.output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    metadata = {
        "status": "ok", "source_split": "train", "pages": len(records),
        "input_granularity": "whole_page_image",
        "model_inputs": ["whole_page_image", "ocr_prompt"],
        "layout_metadata_as_model_input": False,
        "manifest_sha256": sha256(manifest_path), "images": image_hashes,
        "test_read": False,
    }
    (args.output_dir / "bundle.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
