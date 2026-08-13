#!/usr/bin/env python3
"""Offline slot-alignment diagnosis for VLQA layout predictions.

This script checks whether low ordered-slot IoU is caused by query-slot
misalignment: a target region is localized by some query, but not by the query
whose index equals the target reading-order slot.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
GOT_SCRIPT_ROOT = SCRIPT_ROOT / "src" / "GOT-OCR-2.0" / "scripts"
sys.path.insert(0, str(GOT_SCRIPT_ROOT))

from layout_validation_metrics import box_iou  # noqa: E402


class SlotAlignmentFailure(RuntimeError):
    pass


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise SlotAlignmentFailure(f"Expected JSON object at {path}:{line_number}")
            records.append(payload)
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ordered-slot IoU with best-query IoU for each target "
            "region in saved VLQA predictions."
        )
    )
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--iou-threshold", type=probability, default=0.5)
    parser.add_argument("--top-k", type=positive_int, default=30)
    return parser.parse_args(argv)


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path | None, Path]:
    if args.comparison_root is None and args.predictions is None:
        raise SlotAlignmentFailure("Pass --comparison-root or --predictions.")
    if args.comparison_root is not None and args.predictions is not None:
        raise SlotAlignmentFailure("Pass only one of --comparison-root or --predictions.")
    if args.comparison_root is not None:
        root = args.comparison_root.expanduser().resolve()
        predictions = root / "vlqa" / "layout_validation_predictions.jsonl"
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else root / "analysis" / "slot_alignment"
        )
        manifest = args.manifest.expanduser().resolve() if args.manifest else None
        return predictions, manifest, output_dir
    assert args.predictions is not None
    predictions = args.predictions.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else predictions.parent / "slot_alignment"
    )
    manifest = args.manifest.expanduser().resolve() if args.manifest else None
    return predictions, manifest, output_dir


def index_manifest(manifest: Path | None) -> dict[str, dict[str, Any]]:
    if manifest is None:
        return {}
    if not manifest.is_file():
        raise SlotAlignmentFailure(f"Manifest does not exist: {manifest}")
    indexed: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(manifest):
        page_id = str(record.get("page_id", ""))
        if page_id:
            indexed[page_id] = record
    return indexed


def prediction_boxes(record: dict[str, Any]) -> tuple[list[list[float]], list[float]]:
    predictions = record.get("layout_predictions")
    if not isinstance(predictions, list):
        raise SlotAlignmentFailure(f"Missing layout_predictions for {record.get('page_id')}")
    boxes: list[list[float]] = []
    scores: list[float] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise SlotAlignmentFailure(f"layout_predictions[{index}] is not an object.")
        box = prediction.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise SlotAlignmentFailure(f"layout_predictions[{index}].bbox_xyxy is invalid.")
        boxes.append([float(value) for value in box])
        scores.append(float(prediction.get("object_probability", 0.0)))
    return boxes, scores


def target_regions(record: dict[str, Any], manifest_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    page_id = str(record.get("page_id", ""))
    manifest_record = manifest_by_id.get(page_id, {})
    regions = manifest_record.get("regions") or record.get("regions") or []
    if not isinstance(regions, list):
        raise SlotAlignmentFailure(f"regions is invalid for page_id={page_id}")
    sorted_regions = sorted(
        [region for region in regions if isinstance(region, dict)],
        key=lambda region: int(region.get("reading_order", 0)),
    )
    return sorted_regions


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def analyze_records(
    records: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    *,
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset_counter: Counter[str] = Counter()
    page_gap_sums: dict[str, float] = {}
    for record in records:
        page_id = str(record.get("page_id", ""))
        boxes, scores = prediction_boxes(record)
        regions = target_regions(record, manifest_by_id)
        page_gap = 0.0
        for target_index, region in enumerate(regions):
            target_box = region.get("bbox")
            if not isinstance(target_box, list) or len(target_box) != 4:
                raise SlotAlignmentFailure(f"Invalid bbox for page_id={page_id}")
            ordered_iou = (
                box_iou(boxes[target_index], target_box)
                if target_index < len(boxes)
                else 0.0
            )
            best_query_index = -1
            best_iou = -1.0
            for query_index, box in enumerate(boxes):
                iou = box_iou(box, target_box)
                if iou > best_iou:
                    best_iou = iou
                    best_query_index = query_index
            offset = best_query_index - target_index if best_query_index >= 0 else None
            if offset is not None:
                offset_counter[str(offset)] += 1
            slot_gap = max(best_iou, 0.0) - ordered_iou
            page_gap += slot_gap
            rows.append(
                {
                    "page_id": page_id,
                    "target_index": target_index,
                    "target_reading_order": int(region.get("reading_order", target_index)),
                    "ordered_query_index": target_index,
                    "ordered_iou": ordered_iou,
                    "ordered_object_probability": (
                        scores[target_index] if target_index < len(scores) else None
                    ),
                    "best_query_index": best_query_index,
                    "best_iou": best_iou,
                    "best_object_probability": (
                        scores[best_query_index] if 0 <= best_query_index < len(scores) else None
                    ),
                    "best_query_offset": offset,
                    "slot_iou_gap": slot_gap,
                    "ordered_hit": ordered_iou >= iou_threshold,
                    "best_hit": best_iou >= iou_threshold,
                    "slot_misaligned_hit": (
                        best_iou >= iou_threshold
                        and ordered_iou < iou_threshold
                        and best_query_index != target_index
                    ),
                }
            )
        page_gap_sums[page_id] = page_gap
    total_targets = len(rows)
    ordered_hits = sum(bool(row["ordered_hit"]) for row in rows)
    best_hits = sum(bool(row["best_hit"]) for row in rows)
    misaligned_hits = sum(bool(row["slot_misaligned_hit"]) for row in rows)
    summary = {
        "targets": total_targets,
        "ordered_hits": ordered_hits,
        "best_hits": best_hits,
        "slot_misaligned_hits": misaligned_hits,
        "ordered_hit_rate": ordered_hits / total_targets if total_targets else None,
        "best_hit_rate": best_hits / total_targets if total_targets else None,
        "slot_misaligned_hit_rate": (
            misaligned_hits / total_targets if total_targets else None
        ),
        "mean_ordered_iou": mean(float(row["ordered_iou"]) for row in rows),
        "mean_best_iou": mean(float(row["best_iou"]) for row in rows),
        "mean_slot_iou_gap": mean(float(row["slot_iou_gap"]) for row in rows),
        "best_query_offset_counts": dict(offset_counter),
        "worst_pages_by_slot_gap": [
            {"page_id": page_id, "slot_iou_gap_sum": gap}
            for page_id, gap in sorted(
                page_gap_sums.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:20]
        ],
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return compact_json(value).replace("|", "\\|")
    return str(value).replace("|", "\\|")


def write_markdown(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]], top_k: int) -> None:
    fields = (
        "page_id",
        "target_index",
        "ordered_iou",
        "best_query_index",
        "best_iou",
        "best_query_offset",
        "slot_iou_gap",
        "slot_misaligned_hit",
    )
    lines = [
        "# VLQA slot-alignment diagnosis",
        "",
        "## Summary",
        "",
        "```json",
        compact_json(summary),
        "```",
        "",
        "## Largest slot IoU gaps",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in sorted(rows, key=lambda item: float(item["slot_iou_gap"]), reverse=True)[:top_k]:
        lines.append("| " + " | ".join(format_value(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path, manifest_path, output_dir = resolve_inputs(args)
    if not predictions_path.is_file():
        raise SlotAlignmentFailure(f"Predictions file does not exist: {predictions_path}")
    records = read_jsonl(predictions_path)
    rows, summary = analyze_records(
        records,
        index_manifest(manifest_path),
        iou_threshold=args.iou_threshold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "status": "ok",
        "predictions": str(predictions_path),
        "manifest": str(manifest_path) if manifest_path else None,
        "iou_threshold": args.iou_threshold,
        "summary": summary,
    }
    write_json(output_dir / "slot_alignment_summary.json", payload)
    write_csv(output_dir / "slot_alignment_targets.csv", rows)
    write_markdown(output_dir / "slot_alignment.md", summary, rows, args.top_k)
    return {
        "event": "layout_slot_alignment_completed",
        "targets": summary["targets"],
        "output_dir": str(output_dir),
        "summary": str(output_dir / "slot_alignment_summary.json"),
        "csv": str(output_dir / "slot_alignment_targets.csv"),
        "markdown": str(output_dir / "slot_alignment.md"),
        "diagnostics": summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(compact_json(diagnose(parse_args(argv))), flush=True)
    except Exception as exc:
        print(
            compact_json(
                {
                    "event": "layout_slot_alignment_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
