#!/usr/bin/env python3
"""Create a deterministic, validation-only MTHv2 manifest subset.

The tool reads exactly one input manifest.  In particular, it never opens the
MTHv2 test manifest, which makes it suitable for bounded mechanism checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
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
            if record.get("split", record.get("official_split")) != "validation":
                raise ValueError(f"input manifest is not validation-only at {path}:{line_number}")
            image = record.get("image_path") or record.get("image")
            if not image:
                raise ValueError(f"record {page_id} has no image or image_path: {path}:{line_number}")
            copied = dict(record)
            image_path = Path(str(image))
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            copied["image_path"] = str(image_path.resolve())
            page_ids.add(page_id)
            records.append(copied)
    if not records:
        raise ValueError(f"validation manifest is empty: {path}")
    return records


def write_subset(path: Path, records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            serialized = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(serialized)
            digest.update(serialized.encode("utf-8"))
    return digest.hexdigest()


def select_records(
    records: list[dict[str, Any]], count: int, seed: int, *, stratify_regions: bool = False
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if not stratify_regions:
        selected = rng.sample(records, count)
        selected.sort(key=lambda record: str(record["page_id"]))
        return selected
    ordered = sorted(records, key=lambda record: (len(record.get("regions", [])), str(record["page_id"])))
    buckets: list[list[dict[str, Any]]] = [[], [], []]
    for index, record in enumerate(ordered):
        buckets[min(2, index * 3 // max(1, len(ordered)))].append(record)
    quotas = [count // 3, count // 3, count - 2 * (count // 3)]
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for bucket, quota in zip(buckets, quotas):
        take = min(quota, len(bucket))
        selected.extend(rng.sample(bucket, take))
        leftovers.extend(record for record in bucket if record not in selected)
    if len(selected) < count:
        remaining = [record for record in records if record not in selected]
        selected.extend(rng.sample(remaining, count - len(selected)))
    selected.sort(key=lambda record: str(record["page_id"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-regions",
        action="store_true",
        help="sample low, middle, and high region-density pages evenly",
    )
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")

    records = sorted(load_records(args.input), key=lambda record: str(record["page_id"]))
    if args.count > len(records):
        parser.error(f"--count={args.count} exceeds available validation pages={len(records)}")
    selected = select_records(
        records, args.count, args.seed, stratify_regions=args.stratify_regions
    )
    manifest_sha256 = write_subset(args.output, selected)
    print(
        json.dumps(
            {
                "status": "ok",
                "input_manifest": str(args.input),
                "output_manifest": str(args.output),
                "pages": len(selected),
                "seed": args.seed,
                "stratify_regions": args.stratify_regions,
                "region_counts": {
                    "min": min(len(record.get("regions", [])) for record in selected),
                    "max": max(len(record.get("regions", [])) for record in selected),
                    "mean": sum(len(record.get("regions", [])) for record in selected)
                    / max(1, len(selected)),
                },
                "manifest_sha256": manifest_sha256,
                "test_manifest_read": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
