"""Page-manifest access for the merged Dunhuang + local-gazetteer benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


DATASET_ID = "dunhuang_local_gazetteer_q32_v1"
EXPECTED_COUNTS = {"train": 240, "validation": 80, "test": 59}


def _read_records(manifest: Path) -> list[dict[str, Any]]:
    with manifest.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_manifest(
    manifest: Path,
    image_root: Path,
    *,
    split: str,
    limit: int | None = None,
    allow_test: bool = False,
) -> Iterator[dict[str, Any]]:
    if split not in EXPECTED_COUNTS:
        raise ValueError(f"Unsupported {DATASET_ID} split: {split!r}")
    if split == "test" and not allow_test:
        raise PermissionError("test is locked; pass --allow-test for an explicit zero-shot test run.")
    seen = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != split:
                raise ValueError(f"Manifest split mismatch at line {line_number}: {record.get('split')!r}")
            if record.get("input_level") != "page":
                raise ValueError("SOTA protocol requires page-level whole-page records.")
            if record.get("image") is None or record.get("page_id") is None:
                raise ValueError(f"Missing page_id/image at line {line_number}")
            if any(key in record for key in ("oracle_chunk", "chunk_index", "source_region_indices")):
                raise ValueError("Oracle chunks are forbidden in the whole-page SOTA protocol.")
            image_rel = Path(str(record["image"]).replace("\\", "/"))
            if image_rel.is_absolute():
                raise ValueError(f"Portable benchmark requires a relative image path at line {line_number}")
            image_path = (image_root / image_rel).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            copied = dict(record)
            copied["image_path"] = str(image_path)
            copied["protocol"] = {
                "model_inputs": ["whole_page_image", "ocr_prompt"],
                "bbox_as_input": False,
                "direction_as_input": False,
                "reading_order_as_input": False,
            }
            yield copied
            seen += 1
            if limit is not None and seen >= limit:
                return


def manifest_contract(manifest: Path, split: str) -> dict[str, Any]:
    if split not in EXPECTED_COUNTS:
        raise ValueError(f"Unsupported {DATASET_ID} split: {split!r}")
    records = _read_records(manifest)
    expected = EXPECTED_COUNTS[split]
    if len(records) != expected:
        raise ValueError(f"manifest has {len(records)} pages, expected {expected} for {split}")
    domain_counts: dict[str, int] = {}
    for record in records:
        domain = str(record.get("domain", "unknown"))
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return {
        "dataset_id": DATASET_ID,
        "split": split,
        "expected_count": expected,
        "manifest_count": len(records),
        "manifest_sha256": digest,
        "domain_counts": domain_counts,
        "test_used": False,
        "input_granularity": "whole_page_image",
        "model_inputs": ["whole_page_image", "ocr_prompt"],
    }
