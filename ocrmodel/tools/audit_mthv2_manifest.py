#!/usr/bin/env python3
"""Audit the official full MTHv2 page manifests and emit a run protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {"train": 2159, "validation": 240, "test": 800}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(path: Path, split: str, *, num_queries: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    records: list[dict[str, Any]] = []
    page_ids: set[str] = set()
    max_regions = 0
    max_text_chars = 0
    missing_images: list[str] = []
    image_paths: dict[str, str] = {}
    image_hashes: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            page_id = str(record.get("page_id", ""))
            if not page_id or page_id in page_ids:
                raise ValueError(f"duplicate or missing page_id at {path}:{line_number}")
            page_ids.add(page_id)
            record_split = record.get("split", record.get("official_split"))
            if record_split != split:
                raise ValueError(
                    f"split mismatch at {path}:{line_number}: expected {split}, got {record_split}"
                )
            regions = record.get("regions")
            if not isinstance(regions, list) or not regions:
                raise ValueError(f"missing regions at {path}:{line_number}")
            if len(regions) > num_queries:
                raise ValueError(
                    f"{page_id} has {len(regions)} regions, exceeding num_queries={num_queries}"
                )
            page_text = record.get("page_text")
            if not isinstance(page_text, str) or not page_text:
                raise ValueError(f"missing page_text at {path}:{line_number}")
            image = record.get("image_path") or record.get("image")
            if not image:
                raise ValueError(f"missing image path at {path}:{line_number}")
            image_path = Path(str(image))
            if not image_path.is_absolute():
                image_path = path.parent / image_path
            if not image_path.is_file():
                missing_images.append(str(image_path))
            else:
                image_paths[page_id] = str(image)
                image_hashes[page_id] = sha256_file(image_path)
            max_regions = max(max_regions, len(regions))
            max_text_chars = max(max_text_chars, len(page_text))
            records.append(record)
    if missing_images:
        sample = ", ".join(missing_images[:3])
        raise FileNotFoundError(f"{len(missing_images)} missing images in {path}; sample: {sample}")
    stats = {
        "pages": len(records),
        "max_regions": max_regions,
        "max_text_chars": max_text_chars,
        "manifest_sha256": sha256_file(path),
        "image_paths": image_paths,
        "image_sha256": image_hashes,
    }
    return records, stats


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    manifests = {
        "train": args.train_manifest,
        "validation": args.validation_manifest,
    }
    if not getattr(args, "without_test", False):
        manifests["test"] = args.test_manifest
    stats: dict[str, Any] = {}
    for split, path in manifests.items():
        _, split_stats = load_split(path, split, num_queries=args.num_queries)
        expected = EXPECTED_COUNTS[split] if args.require_official_counts else None
        if expected is not None and split_stats["pages"] != expected:
            raise ValueError(
                f"{split} count mismatch: expected {expected}, got {split_stats['pages']}"
            )
        stats[split] = split_stats
    return {
        "status": "ok",
        "dataset": "MTHv2",
        "protocol": "glm_ocr_mthv2_full_official_v1",
        "split_pages": {split: stats[split]["pages"] for split in manifests},
        "max_regions": max(value["max_regions"] for value in stats.values()),
        "max_text_chars": max(value["max_text_chars"] for value in stats.values()),
        "num_queries": args.num_queries,
        "manifest_stats": stats,
        "source_group_metadata": "official_split_only; full source isolation unavailable",
        "test_manifest_read": "test" in manifests,
        "test_used_for_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument(
        "--without-test",
        action="store_true",
        help="audit only train/validation and do not open the test manifest",
    )
    parser.add_argument("--num-queries", type=int, default=512)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-count-mismatch",
        dest="require_official_counts",
        action="store_false",
        help="allow non-official page counts for bounded fixtures",
    )
    parser.set_defaults(require_official_counts=True)
    args = parser.parse_args()
    if args.num_queries <= 0:
        parser.error("--num-queries must be positive")
    if not args.without_test and args.test_manifest is None:
        parser.error("--test-manifest is required unless --without-test is set")
    protocol = build_protocol(args)
    serialized = json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(json.dumps(protocol, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
