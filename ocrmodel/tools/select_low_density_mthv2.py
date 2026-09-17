#!/usr/bin/env python3
"""Create a deterministic low-density MTHv2 view for Q32 experiments.

The selected records keep their original page annotations, while relative
image paths are resolved against the source manifest so the derived view can
live outside the dataset tree.  The command can be run for train/validation
before training and for test only after training, preserving the test-free
training boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_records(path: Path, split: str, max_regions: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    page_ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            page_id = str(record.get("page_id", ""))
            if not page_id or page_id in page_ids:
                raise ValueError(f"duplicate or missing page_id at {path}:{line_number}")
            record_split = record.get("split", record.get("official_split"))
            if record_split != split:
                raise ValueError(
                    f"split mismatch at {path}:{line_number}: expected {split}, got {record_split}"
                )
            regions = record.get("regions")
            if not isinstance(regions, list) or not regions:
                raise ValueError(f"missing regions at {path}:{line_number}")
            if len(regions) > max_regions:
                continue
            image = record.get("image_path") or record.get("image")
            if not image:
                raise ValueError(f"record {page_id} has no image path")
            image_path = Path(str(image))
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            copied = dict(record)
            copied["image_path"] = str(image_path.resolve())
            page_ids.add(page_id)
            records.append(copied)
    records.sort(key=lambda row: str(row["page_id"]))
    if not records:
        raise ValueError(f"no {split} records satisfy max_regions={max_regions}: {path}")
    return records


def _write_records(path: Path, records: list[dict[str, Any]]) -> str:
    serialized = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"derived manifest already exists with different content: {path}")
    else:
        path.write_text(serialized, encoding="utf-8", newline="\n")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-regions", type=int, default=24)
    parser.add_argument(
        "--splits",
        default="train,validation",
        help="comma-separated source splits to materialize",
    )
    args = parser.parse_args()
    if args.max_regions <= 0:
        parser.error("--max-regions must be positive")
    splits = tuple(item.strip() for item in args.splits.split(",") if item.strip())
    if not splits or any(split not in {"train", "validation", "test"} for split in splits):
        parser.error("--splits must contain only train, validation, or test")

    summary_path = args.output_root / "low_density_selection.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError(f"expected JSON object: {summary_path}")
        if summary.get("max_regions") != args.max_regions:
            raise ValueError("existing low-density view uses a different max_regions")
        summary["splits"] = dict(summary.get("splits") or {})
    else:
        summary = {
            "status": "ok",
            "source_root": str(args.input_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "max_regions": args.max_regions,
            "splits": {},
        }
    summary["test_manifest_materialized"] = bool(
        "test" in splits or "test" in (summary.get("splits") or {})
    )
    for split in splits:
        source = args.input_root / split / "manifest.jsonl"
        target = args.output_root / split / "manifest.jsonl"
        records = _load_records(source, split, args.max_regions)
        digest = _write_records(target, records)
        counts = [len(record["regions"]) for record in records]
        summary["splits"][split] = {
            "pages": len(records),
            "min_regions": min(counts),
            "max_regions": max(counts),
            "mean_regions": sum(counts) / len(counts),
            "manifest_sha256": digest,
        }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
