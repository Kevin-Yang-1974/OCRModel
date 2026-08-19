#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from PIL import Image


SPLITS = ("train", "validation", "test")
PROMPT = "<image>\nOCR: "


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def batches(values: Sequence[Any], size: int) -> Iterable[tuple[int, Sequence[Any]]]:
    for start in range(0, len(values), size):
        yield start // size, values[start : start + size]


def dominant_axis(regions: Sequence[dict[str, Any]]) -> str:
    directions = Counter(str(region.get("writing_direction", "unknown")) for region in regions)
    vertical = directions["vertical_rtl"] + directions["vertical_ltr"]
    horizontal = directions["horizontal_ltr"] + directions["horizontal_rtl"]
    if vertical >= horizontal and vertical > 0:
        return "vertical"
    if horizontal > 0:
        return "horizontal"
    return "region_union"


def chunk_crop_box(
    regions: Sequence[dict[str, Any]],
    *,
    width: int,
    height: int,
    margin: int,
) -> tuple[list[int], str]:
    boxes = [region["bbox_px"] for region in regions]
    x0 = min(float(box[0]) for box in boxes)
    y0 = min(float(box[1]) for box in boxes)
    x1 = max(float(box[2]) for box in boxes)
    y1 = max(float(box[3]) for box in boxes)
    axis = dominant_axis(regions)
    if axis == "vertical":
        crop = [math.floor(x0 - margin), 0, math.ceil(x1 + margin), height]
    elif axis == "horizontal":
        crop = [0, math.floor(y0 - margin), width, math.ceil(y1 + margin)]
    else:
        crop = [
            math.floor(x0 - margin),
            math.floor(y0 - margin),
            math.ceil(x1 + margin),
            math.ceil(y1 + margin),
        ]
    crop[0] = max(0, crop[0])
    crop[1] = max(0, crop[1])
    crop[2] = min(width, crop[2])
    crop[3] = min(height, crop[3])
    return crop, axis


def transform_region(
    region: dict[str, Any],
    *,
    crop_box: Sequence[int],
    page_id: str,
    local_order: int,
) -> dict[str, Any]:
    crop_x0, crop_y0, crop_x1, crop_y1 = crop_box
    crop_width = crop_x1 - crop_x0
    crop_height = crop_y1 - crop_y0
    transformed = copy.deepcopy(region)
    transformed["source_region_id"] = region.get("region_id")
    transformed["source_reading_order"] = region.get("reading_order")
    # The audit schema uses source_kind=text for real textline labels; the
    # dataset-level layout_source retains the MTHv2 provenance.
    transformed["source_kind"] = "text"
    transformed["region_id"] = f"{page_id}_region_{local_order:02d}"
    transformed["reading_order"] = local_order
    transformed["polygon_px"] = [
        [round(float(point[0]) - crop_x0, 4), round(float(point[1]) - crop_y0, 4)]
        for point in region["polygon_px"]
    ]
    source_box = [float(value) for value in region["bbox_px"]]
    bbox_px = [
        max(0.0, source_box[0] - crop_x0),
        max(0.0, source_box[1] - crop_y0),
        min(float(crop_width), source_box[2] - crop_x0),
        min(float(crop_height), source_box[3] - crop_y0),
    ]
    transformed["bbox_px"] = [round(value, 4) for value in bbox_px]
    transformed["bbox"] = [
        round(bbox_px[0] / crop_width, 8),
        round(bbox_px[1] / crop_height, 8),
        round(bbox_px[2] / crop_width, 8),
        round(bbox_px[3] / crop_height, 8),
    ]
    return transformed


def save_crop(image: Image.Image, crop_box: Sequence[int], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    cropped = image.crop(tuple(crop_box))
    if target.suffix.lower() in {".jpg", ".jpeg"}:
        cropped.convert("RGB").save(target, quality=95)
    else:
        cropped.save(target)


def convert_record(
    record: dict[str, Any],
    *,
    input_split_root: Path,
    output_split_root: Path,
    max_regions: int,
    margin: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_image = input_split_root / Path(*PurePosixPath(record["image"]).parts)
    regions = list(record["regions"])
    output_records: list[dict[str, Any]] = []
    chunk_stats: list[dict[str, Any]] = []
    with Image.open(source_image) as image:
        width, height = image.size
        chunk_count = math.ceil(len(regions) / max_regions)
        for chunk_index, chunk_regions in batches(regions, max_regions):
            page_id = f"{record['page_id']}__chunk_{chunk_index:03d}"
            crop_box, crop_axis = chunk_crop_box(
                chunk_regions,
                width=width,
                height=height,
                margin=margin,
            )
            crop_width = crop_box[2] - crop_box[0]
            crop_height = crop_box[3] - crop_box[1]
            image_name = f"{page_id}{source_image.suffix.lower()}"
            image_relative = PurePosixPath("images") / image_name
            save_crop(
                image,
                crop_box,
                output_split_root / Path(*image_relative.parts),
            )
            transformed_regions = [
                transform_region(
                    region,
                    crop_box=crop_box,
                    page_id=page_id,
                    local_order=local_order,
                )
                for local_order, region in enumerate(chunk_regions)
            ]
            page_text = "".join(str(region.get("text", "")) for region in transformed_regions)
            annotation_relative = PurePosixPath("annotations") / f"{page_id}.json"
            annotation = {
                "schema_version": 1,
                "dataset": "MTHv2",
                "page_id": page_id,
                "source_page_id": record["page_id"],
                "source_annotation_file": record.get("annotation_file"),
                "chunk_index": chunk_index,
                "chunk_count": chunk_count,
                "source_crop_box_px": crop_box,
                "page_size": [crop_width, crop_height],
                "textlines": transformed_regions,
            }
            write_json(
                output_split_root / Path(*annotation_relative.parts),
                annotation,
            )

            output_record = copy.deepcopy(record)
            output_record.update(
                {
                    "layout_source": "mthv2_textline_ordered_chunk_v1",
                    "page_id": page_id,
                    "content_id": page_id,
                    "image": str(image_relative),
                    "annotation_file": str(annotation_relative),
                    "page_size": [crop_width, crop_height],
                    "page_text": page_text,
                    "regions": transformed_regions,
                    "conversations": [
                        {"from": "human", "value": PROMPT},
                        {"from": "gpt", "value": page_text},
                    ],
                    "source_page_id": record["page_id"],
                    "source_page_image": record["image"],
                    "source_page_size": [width, height],
                    "source_crop_box_px": crop_box,
                    "source_region_indices": [
                        int(region["reading_order"]) for region in chunk_regions
                    ],
                    "chunk_index": chunk_index,
                    "chunk_count": chunk_count,
                    "chunk_axis": crop_axis,
                    "max_regions_per_chunk": max_regions,
                }
            )
            output_records.append(output_record)
            chunk_stats.append(
                {
                    "regions": len(transformed_regions),
                    "width": crop_width,
                    "height": crop_height,
                    "axis": crop_axis,
                }
            )
    return output_records, chunk_stats


def percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output dataset already exists: {output_root}")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    split_summaries: dict[str, Any] = {}
    source_chunk_counts: list[int] = []
    chunk_region_counts: Counter[int] = Counter()
    crop_axis_counts: Counter[str] = Counter()
    source_pages = 0
    output_chunks = 0
    for split in SPLITS:
        input_split_root = input_root / split
        output_split_root = output_root / split
        records = load_jsonl(input_split_root / "manifest.jsonl")
        converted: list[dict[str, Any]] = []
        split_chunk_counts: list[int] = []
        for record in records:
            chunks, stats = convert_record(
                record,
                input_split_root=input_split_root,
                output_split_root=output_split_root,
                max_regions=args.max_regions,
                margin=args.margin_pixels,
            )
            converted.extend(chunks)
            split_chunk_counts.append(len(chunks))
            source_chunk_counts.append(len(chunks))
            for stat in stats:
                chunk_region_counts[stat["regions"]] += 1
                crop_axis_counts[stat["axis"]] += 1
        write_jsonl(output_split_root / "manifest.jsonl", converted)
        source_pages += len(records)
        output_chunks += len(converted)
        split_summaries[split] = {
            "source_pages": len(records),
            "output_chunks": len(converted),
            "pages_split_into_multiple_chunks": sum(count > 1 for count in split_chunk_counts),
        }

    summary = {
        "status": "ok",
        "event": "mthv2_layout_chunks_prepared",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "source_pages": source_pages,
        "output_chunks": output_chunks,
        "max_regions_per_chunk": args.max_regions,
        "margin_pixels": args.margin_pixels,
        "grouping": "contiguous_batches_in_official_textline_reading_order",
        "crop_policy": {
            "vertical": "selected_region_x_union_plus_margin_and_full_page_height",
            "horizontal": "selected_region_y_union_plus_margin_and_full_page_width",
            "unknown": "selected_region_bbox_union_plus_margin",
        },
        "splits": split_summaries,
        "chunks_per_source_page": {
            "min": min(source_chunk_counts),
            "p50": percentile(source_chunk_counts, 0.50),
            "p90": percentile(source_chunk_counts, 0.90),
            "p95": percentile(source_chunk_counts, 0.95),
            "p99": percentile(source_chunk_counts, 0.99),
            "max": max(source_chunk_counts),
        },
        "chunk_region_count_distribution": {
            str(key): value for key, value in sorted(chunk_region_counts.items())
        },
        "crop_axis_counts": dict(sorted(crop_axis_counts.items())),
        "layout_level": "textline",
        "label_interpretation": "MTHv2 textline polygons used as ordered column candidates",
    }
    write_json(output_root / "chunking_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split converted MTHv2 pages into ordered crops with at most K textline regions."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-regions", type=int, default=16)
    parser.add_argument("--margin-pixels", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.max_regions < 1:
        parser.error("--max-regions must be positive")
    if args.margin_pixels < 0:
        parser.error("--margin-pixels cannot be negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    summary = build_dataset(parse_args(argv))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "event": summary["event"],
                "output_root": summary["output_root"],
                "source_pages": summary["source_pages"],
                "output_chunks": summary["output_chunks"],
                "splits": summary["splits"],
                "chunks_per_source_page": summary["chunks_per_source_page"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
