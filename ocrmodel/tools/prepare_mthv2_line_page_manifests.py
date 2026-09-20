#!/usr/bin/env python3
"""Convert MTHv2 GT line crops into page-shaped manifests for GOT2 and GLM-OCR.

Each vertical line crop is treated as one independent page.  Character boxes
remain training-only layout supervision; OCR inference still receives only the
crop image and the OCR prompt.  The converter is deliberately deterministic so
the same train/validation/test records can be used by both backbones.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _subset(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(records) <= limit:
        return records
    # Evenly sample the source split rather than taking a contiguous page/file
    # prefix, which would overrepresent one source page.
    indices = [(i * len(records)) // limit for i in range(limit)]
    return [records[i] for i in indices]


def _normalized_box(box: list[float], width: float, height: float) -> list[float]:
    x0, y0, x1, y1 = (float(value) for value in box)
    x0 = min(1.0, max(0.0, x0 / width))
    y0 = min(1.0, max(0.0, y0 / height))
    x1 = min(1.0, max(0.0, x1 / width))
    y1 = min(1.0, max(0.0, y1 / height))
    # A few source annotations have a degenerate box after rounding.  Keep the
    # record and give it the smallest valid extent; dropping it would make the
    # test count no longer equal the locked subset count.
    x1 = max(x1, min(1.0, x0 + 1e-5))
    y1 = max(y1, min(1.0, y0 + 1e-5))
    return [x0, y0, x1, y1]


def convert_record(record: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    width, height = (float(value) for value in record["size"])
    boxes = record.get("character_boxes") or []
    regions: list[dict[str, Any]] = []
    for order, character in enumerate(boxes):
        regions.append(
            {
                "bbox": _normalized_box(character["bbox"], width, height),
                "reading_order": order,
                "writing_direction": record.get("writing_direction", "unknown"),
                "content_id": f"{record['id']}:char:{order}",
                "source_group_id": str(record.get("source_image", record["id"])),
                "type": "COLUMN_CHARACTER",
            }
        )
    if not regions:
        regions = [
            {
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "reading_order": 0,
                "writing_direction": record.get("writing_direction", "unknown"),
                "content_id": f"{record['id']}:whole_line",
                "source_group_id": str(record.get("source_image", record["id"])),
                "type": "COLUMN",
            }
        ]
    text = str(record.get("text", ""))
    return {
        "page_id": f"mthv2_line:{split}:{index}",
        "split": split,
        "input_level": "page",
        "image": str(record["image"]),
        "page_text": text,
        "conversations": [
            {"from": "human", "value": "<image>\nOCR: "},
            {"from": "gpt", "value": text},
        ],
        "layout_annotation_status": "complete",
        "regions": regions,
        "source_line_id": str(record["id"]),
        "source_line_bbox": record.get("source_line_bbox"),
        "writing_direction": record.get("writing_direction", "unknown"),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"empty manifest: {path}")
    return records


def write_split(
    source_manifest: Path,
    output_manifest: Path,
    split: str,
    limit: int,
) -> int:
    records = _subset(load_jsonl(source_manifest), limit)
    converted = [convert_record(record, split, index) for index, record in enumerate(records)]
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        for record in converted:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(converted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--validation-limit", type=int, default=0)
    parser.add_argument("--test-manifest", type=Path)
    parser.add_argument(
        "--without-test",
        action="store_true",
        help="Prepare only train and validation manifests; do not open the test manifest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    test_source = None if args.without_test else (
        args.test_manifest or (source_root / "test" / "manifest.jsonl")
    )
    split_specs = [
        ("train", source_root / "train" / "manifest.jsonl", args.train_limit),
        ("validation", source_root / "validation" / "manifest.jsonl", args.validation_limit),
    ]
    if test_source is not None:
        split_specs.append(("test", test_source, 0))
    counts: dict[str, int] = {}
    for split, source_manifest, limit in split_specs:
        output_manifest = output_root / split / "manifest.jsonl"
        counts[split] = write_split(source_manifest, output_manifest, split, limit)
    (output_root / "manifest_metadata.json").write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "source_test_manifest": (
                    str(test_source.resolve()) if test_source is not None else None
                ),
                "counts": counts,
                "test_used_for_selection": False,
                "test_manifest_prepared": test_source is not None,
                "input_granularity": "gt_cropped_vertical_line_image",
                "layout_supervision": "character_boxes_from_source_manifest",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
