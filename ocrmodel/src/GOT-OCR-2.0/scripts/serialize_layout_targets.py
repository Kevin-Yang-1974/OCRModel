#!/usr/bin/env python3
"""Serialize page regions into deterministic PVLD layout targets.

The input manifest is never modified.  The output manifest is a copy with
PVLD sidecar fields, which keeps Fixed-Slot manifests and experiments
reproducible and separate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


DEFAULT_TYPES = ("COLUMN", "ROW", "REGION")
VALID_DIRECTIONS = {"vertical_rtl", "vertical_ltr", "horizontal_ltr", "horizontal_rtl", "unknown"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_unit(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value!r}")
    return number


def normalized_bbox(region: dict[str, Any], index: int) -> list[float]:
    bbox = region.get("bbox", region.get("bbox_norm"))
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError(f"region {index} has no four-value normalized bbox")
    values = [finite_unit(value, f"region {index} bbox[{offset}]") for offset, value in enumerate(bbox)]
    if values[2] < values[0] or values[3] < values[1]:
        raise ValueError(f"region {index} bbox must satisfy x1>=x0 and y1>=y0")
    return values


def direction_for(record: dict[str, Any], region: dict[str, Any]) -> str:
    value = str(
        region.get(
            "direction",
            region.get("writing_direction", record.get("direction", record.get("writing_direction", "unknown"))),
        )
    ).lower()
    if value not in VALID_DIRECTIONS:
        raise ValueError(f"unsupported writing direction: {value!r}")
    return value


def canonical_order(record: dict[str, Any], regions: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    direction = str(record.get("direction", "")).lower()
    if direction not in VALID_DIRECTIONS:
        directions = {
            str(region.get("direction", region.get("writing_direction", "unknown"))).lower()
            for region in regions
        }
        direction = next(iter(directions), "unknown")
    if direction not in VALID_DIRECTIONS:
        direction = "unknown"
    indexed: list[tuple[int, dict[str, Any]]] = []
    for index, region in enumerate(regions):
        bbox = normalized_bbox(region, index)
        explicit = region.get("reading_order", region.get("order"))
        explicit_key = (0, int(explicit)) if explicit is not None else (1, 0)
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        if direction == "vertical_rtl":
            geometry = (-center_x, center_y, bbox[1], bbox[0])
        elif direction == "vertical_ltr":
            geometry = (center_x, center_y, bbox[1], bbox[0])
        elif direction == "horizontal_ltr":
            geometry = (center_y, center_x, bbox[0], bbox[1])
        elif direction == "horizontal_rtl":
            geometry = (center_y, -center_x, bbox[0], bbox[1])
        else:
            geometry = (center_y, center_x, bbox[0], bbox[1])
        indexed.append((index, {**region, "_sort": (*explicit_key, *geometry), "_bbox_norm": bbox}))
    indexed.sort(key=lambda item: item[1]["_sort"])
    ordered = []
    for output_index, (source_index, region) in enumerate(indexed):
        cleaned = dict(region)
        cleaned.pop("_sort", None)
        cleaned["source_region_index"] = source_index
        cleaned["region_index"] = output_index
        cleaned["bbox"] = cleaned.pop("_bbox_norm")
        cleaned["direction"] = direction_for(record, cleaned)
        cleaned["writing_direction"] = cleaned["direction"]
        cleaned["layout_type"] = str(cleaned.get("layout_type", cleaned.get("type", "REGION"))).upper()
        ordered.append(cleaned)
    return direction, ordered


def tokens_for_region(region: dict[str, Any]) -> list[str]:
    return ["<REGION>", "<TYPE>", str(region["layout_type"]), "</TYPE>", "</REGION>"]


def serialize_record(record: dict[str, Any], max_records: int) -> dict[str, Any]:
    regions = record.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError(f"{record.get('page_id', '<unknown>')} regions must be a list")
    direction, ordered = canonical_order(record, regions)
    if len(ordered) > max_records:
        raise ValueError(
            f"{record.get('page_id', '<unknown>')} has {len(ordered)} regions, "
            f"exceeding max_layout_records={max_records}; use oracle chunks or a larger limit."
        )
    order_values = [region.get("reading_order") for region in ordered if region.get("reading_order") is not None]
    if len(order_values) != len(set(order_values)):
        raise ValueError(f"{record.get('page_id', '<unknown>')} reading_order is not unique")
    tokens = ["<LAYOUT>"]
    for region in ordered:
        tokens.extend(tokens_for_region(region))
    tokens.append("<EOS>")
    target = dict(record)
    target["layout_target_text"] = " ".join(tokens)
    target["layout_target_tokens"] = tokens
    target["layout_regions"] = ordered
    target["layout_region_count"] = len(ordered)
    target["canonical_order"] = direction
    target["layout_target_schema"] = "pvld_target_v1"
    target["layout_source"] = str(record.get("layout_source", "page_regions"))
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--max-layout-records", type=int, default=64)
    parser.add_argument("--num-layout-prompt-queries", type=int, default=32)
    parser.add_argument("--allow-empty-pages", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_layout_records < 1 or args.num_layout_prompt_queries < 1:
        raise ValueError("max-layout-records and num-layout-prompt-queries must be positive")
    source = args.manifest.resolve()
    output = args.output_manifest.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = (args.sidecar or output.with_name("layout_targets.jsonl")).resolve()
    if sidecar.exists():
        raise FileExistsError(f"refusing to overwrite {sidecar}")
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            serialized = serialize_record(record, args.max_layout_records)
            if not serialized["layout_regions"] and not args.allow_empty_pages:
                raise ValueError(f"line {line_number} is an empty page; pass --allow-empty-pages")
            records.append(serialized)
    with output.open("x", encoding="utf-8", newline="\n") as manifest_handle, sidecar.open("x", encoding="utf-8", newline="\n") as sidecar_handle:
        for record in records:
            manifest_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            sidecar_record = {
                "page_id": record.get("page_id"),
                "layout_target_text": record["layout_target_text"],
                "layout_regions": record["layout_regions"],
                "layout_region_count": record["layout_region_count"],
                "canonical_order": record["canonical_order"],
                "layout_source": record["layout_source"],
                "source_manifest_sha256": sha256(source),
            }
            sidecar_handle.write(json.dumps(sidecar_record, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary = {
        "status": "ok",
        "schema": "pvld_target_v1",
        "input_manifest": str(source),
        "output_manifest": str(output),
        "sidecar": str(sidecar),
        "source_manifest_sha256": sha256(source),
        "records": len(records),
        "max_layout_records": args.max_layout_records,
        "num_layout_prompt_queries": args.num_layout_prompt_queries,
        "input_granularities": sorted({str(record.get("input_granularity", "whole_page_image")) for record in records}),
        "layout_sources": sorted({str(record.get("layout_source", "page_regions")) for record in records}),
    }
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
