#!/usr/bin/env python3
"""Offline fixed slot-offset sweep for VLQA layout predictions.

This diagnostic tests a concrete hypothesis from slot-alignment analysis:
target reading-order slot k may align better to query k + offset than to query k.
It does not load a model or use GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[2]
GOT_SCRIPT_ROOT = SCRIPT_ROOT / "src" / "GOT-OCR-2.0" / "scripts"
sys.path.insert(0, str(GOT_SCRIPT_ROOT))

from layout_validation_metrics import box_iou  # noqa: E402


DEFAULT_OFFSETS = tuple(range(-3, 9))


class SlotOffsetSweepFailure(RuntimeError):
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
                raise SlotOffsetSweepFailure(f"Expected JSON object at {path}:{line_number}")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay ordered-slot metrics using query index target_index + offset."
    )
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--offset", action="append", type=int)
    parser.add_argument("--iou-threshold", type=probability, default=0.5)
    return parser.parse_args(argv)


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path | None, Path]:
    if args.comparison_root is None and args.predictions is None:
        raise SlotOffsetSweepFailure("Pass --comparison-root or --predictions.")
    if args.comparison_root is not None and args.predictions is not None:
        raise SlotOffsetSweepFailure("Pass only one of --comparison-root or --predictions.")
    if args.comparison_root is not None:
        root = args.comparison_root.expanduser().resolve()
        predictions = root / "vlqa" / "layout_validation_predictions.jsonl"
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else root / "analysis" / "slot_offset_sweep"
        )
        manifest = args.manifest.expanduser().resolve() if args.manifest else None
        return predictions, manifest, output_dir
    assert args.predictions is not None
    predictions = args.predictions.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else predictions.parent / "slot_offset_sweep"
    )
    manifest = args.manifest.expanduser().resolve() if args.manifest else None
    return predictions, manifest, output_dir


def index_manifest(manifest: Path | None) -> dict[str, dict[str, Any]]:
    if manifest is None:
        return {}
    if not manifest.is_file():
        raise SlotOffsetSweepFailure(f"Manifest does not exist: {manifest}")
    indexed: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(manifest):
        page_id = str(record.get("page_id", ""))
        if page_id:
            indexed[page_id] = record
    return indexed


def prediction_boxes(record: dict[str, Any]) -> list[list[float]]:
    predictions = record.get("layout_predictions")
    if not isinstance(predictions, list):
        raise SlotOffsetSweepFailure(f"Missing layout_predictions for {record.get('page_id')}")
    boxes: list[list[float]] = []
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict):
            raise SlotOffsetSweepFailure(f"layout_predictions[{index}] is not an object.")
        box = prediction.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise SlotOffsetSweepFailure(f"layout_predictions[{index}].bbox_xyxy is invalid.")
        boxes.append([float(value) for value in box])
    return boxes


def target_regions(record: dict[str, Any], manifest_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    page_id = str(record.get("page_id", ""))
    manifest_record = manifest_by_id.get(page_id, {})
    regions = manifest_record.get("regions") or record.get("regions") or []
    if not isinstance(regions, list):
        raise SlotOffsetSweepFailure(f"regions is invalid for page_id={page_id}")
    return sorted(
        [region for region in regions if isinstance(region, dict)],
        key=lambda region: int(region.get("reading_order", 0)),
    )


def evaluate_offset(
    records: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    *,
    offset: int,
    iou_threshold: float,
) -> dict[str, Any]:
    targets = 0
    valid_slots = 0
    hits = 0
    iou_sum = 0.0
    missed_by_range = 0
    for record in records:
        boxes = prediction_boxes(record)
        for target_index, region in enumerate(target_regions(record, manifest_by_id)):
            target_box = region.get("bbox")
            if not isinstance(target_box, list) or len(target_box) != 4:
                raise SlotOffsetSweepFailure(f"Invalid bbox for page_id={record.get('page_id')}")
            targets += 1
            query_index = target_index + offset
            if query_index < 0 or query_index >= len(boxes):
                missed_by_range += 1
                continue
            valid_slots += 1
            iou = box_iou(boxes[query_index], target_box)
            iou_sum += iou
            hits += int(iou >= iou_threshold)
    return {
        "offset": offset,
        "iou_threshold": iou_threshold,
        "targets": targets,
        "valid_slots": valid_slots,
        "missed_by_range": missed_by_range,
        "hit_count": hits,
        "hit_rate": hits / targets if targets else None,
        "hit_rate_on_valid_slots": hits / valid_slots if valid_slots else None,
        "mean_iou": iou_sum / targets if targets else None,
        "mean_iou_on_valid_slots": iou_sum / valid_slots if valid_slots else None,
    }


def evaluate_page_offset(
    boxes: list[list[float]],
    regions: list[dict[str, Any]],
    *,
    offset: int,
    iou_threshold: float,
) -> dict[str, Any]:
    targets = 0
    valid_slots = 0
    hits = 0
    iou_sum = 0.0
    missed_by_range = 0
    for target_index, region in enumerate(regions):
        target_box = region.get("bbox")
        if not isinstance(target_box, list) or len(target_box) != 4:
            raise SlotOffsetSweepFailure("Invalid region bbox.")
        targets += 1
        query_index = target_index + offset
        if query_index < 0 or query_index >= len(boxes):
            missed_by_range += 1
            continue
        valid_slots += 1
        iou = box_iou(boxes[query_index], target_box)
        iou_sum += iou
        hits += int(iou >= iou_threshold)
    return {
        "targets": targets,
        "valid_slots": valid_slots,
        "missed_by_range": missed_by_range,
        "hit_count": hits,
        "hit_rate": hits / targets if targets else None,
        "mean_iou": iou_sum / targets if targets else None,
        "mean_iou_on_valid_slots": iou_sum / valid_slots if valid_slots else None,
    }


def evaluate_page_oracle_offsets(
    records: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    *,
    offsets: Sequence[int],
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset_counter: Counter[str] = Counter()
    total_targets = 0
    baseline_hits = 0
    oracle_hits = 0
    baseline_iou_sum = 0.0
    oracle_iou_sum = 0.0
    for record in records:
        page_id = str(record.get("page_id", ""))
        boxes = prediction_boxes(record)
        regions = target_regions(record, manifest_by_id)
        if not regions:
            continue
        page_results = [
            {
                "offset": offset,
                **evaluate_page_offset(
                    boxes,
                    regions,
                    offset=offset,
                    iou_threshold=iou_threshold,
                ),
            }
            for offset in offsets
        ]
        baseline = next(
            (result for result in page_results if int(result["offset"]) == 0),
            None,
        )
        best = max(
            page_results,
            key=lambda row: (
                float(row.get("hit_rate") or 0.0),
                float(row.get("mean_iou") or 0.0),
                -abs(int(row["offset"])),
            ),
        )
        offset_counter[str(best["offset"])] += 1
        targets = int(best["targets"])
        total_targets += targets
        oracle_hits += int(best["hit_count"])
        oracle_iou_sum += float(best.get("mean_iou") or 0.0) * targets
        if baseline is not None:
            baseline_hits += int(baseline["hit_count"])
            baseline_iou_sum += float(baseline.get("mean_iou") or 0.0) * targets
        rows.append(
            {
                "page_id": page_id,
                "targets": targets,
                "best_offset": int(best["offset"]),
                "best_hit_count": int(best["hit_count"]),
                "best_hit_rate": best["hit_rate"],
                "best_mean_iou": best["mean_iou"],
                "offset0_hit_count": int(baseline["hit_count"]) if baseline else None,
                "offset0_hit_rate": baseline["hit_rate"] if baseline else None,
                "offset0_mean_iou": baseline["mean_iou"] if baseline else None,
                "hit_count_gain": (
                    int(best["hit_count"]) - int(baseline["hit_count"])
                    if baseline
                    else None
                ),
                "mean_iou_gain": (
                    float(best.get("mean_iou") or 0.0)
                    - float(baseline.get("mean_iou") or 0.0)
                    if baseline
                    else None
                ),
            }
        )
    summary = {
        "pages": len(rows),
        "targets": total_targets,
        "offset0_hits": baseline_hits,
        "oracle_hits": oracle_hits,
        "offset0_hit_rate": baseline_hits / total_targets if total_targets else None,
        "oracle_hit_rate": oracle_hits / total_targets if total_targets else None,
        "oracle_hit_rate_gain": (
            (oracle_hits - baseline_hits) / total_targets if total_targets else None
        ),
        "offset0_mean_iou": baseline_iou_sum / total_targets if total_targets else None,
        "oracle_mean_iou": oracle_iou_sum / total_targets if total_targets else None,
        "oracle_mean_iou_gain": (
            (oracle_iou_sum - baseline_iou_sum) / total_targets
            if total_targets
            else None
        ),
        "best_page_offset_counts": dict(offset_counter),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], best: dict[str, Any] | None) -> None:
    fields = (
        "offset",
        "hit_rate",
        "mean_iou",
        "valid_slots",
        "missed_by_range",
        "hit_rate_on_valid_slots",
        "mean_iou_on_valid_slots",
    )
    lines = [
        "# VLQA fixed slot-offset sweep",
        "",
        "## Best offset by hit_rate",
        "",
        "```json",
        compact_json(best),
        "```",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_offset(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("hit_rate") or 0.0),
            float(row.get("mean_iou") or 0.0),
            -abs(int(row["offset"])),
        ),
    )


def sweep(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path, manifest_path, output_dir = resolve_inputs(args)
    if not predictions_path.is_file():
        raise SlotOffsetSweepFailure(f"Predictions file does not exist: {predictions_path}")
    records = read_jsonl(predictions_path)
    manifest_by_id = index_manifest(manifest_path)
    offsets = sorted(set(args.offset if args.offset else DEFAULT_OFFSETS))
    rows = [
        evaluate_offset(
            records,
            manifest_by_id,
            offset=offset,
            iou_threshold=args.iou_threshold,
        )
        for offset in offsets
    ]
    best = best_offset(rows)
    page_oracle_rows, page_oracle_summary = evaluate_page_oracle_offsets(
        records,
        manifest_by_id,
        offsets=offsets,
        iou_threshold=args.iou_threshold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "status": "ok",
        "predictions": str(predictions_path),
        "manifest": str(manifest_path) if manifest_path else None,
        "offsets": offsets,
        "iou_threshold": args.iou_threshold,
        "best_by_hit_rate": best,
        "page_oracle": page_oracle_summary,
        "rows": rows,
    }
    write_json(output_dir / "slot_offset_sweep_summary.json", summary)
    write_csv(
        output_dir / "slot_offset_sweep.csv",
        rows,
        [
            "offset",
            "iou_threshold",
            "targets",
            "valid_slots",
            "missed_by_range",
            "hit_count",
            "hit_rate",
            "hit_rate_on_valid_slots",
            "mean_iou",
            "mean_iou_on_valid_slots",
        ],
    )
    write_csv(output_dir / "slot_offset_page_oracle.csv", page_oracle_rows)
    write_markdown(output_dir / "slot_offset_sweep.md", rows, best)
    return {
        "event": "layout_slot_offset_sweep_completed",
        "targets": rows[0]["targets"] if rows else 0,
        "output_dir": str(output_dir),
        "summary": str(output_dir / "slot_offset_sweep_summary.json"),
        "csv": str(output_dir / "slot_offset_sweep.csv"),
        "page_oracle_csv": str(output_dir / "slot_offset_page_oracle.csv"),
        "markdown": str(output_dir / "slot_offset_sweep.md"),
        "best_by_hit_rate": best,
        "page_oracle": page_oracle_summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(compact_json(sweep(parse_args(argv))), flush=True)
    except Exception as exc:
        print(
            compact_json(
                {
                    "event": "layout_slot_offset_sweep_failed",
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
