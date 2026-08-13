#!/usr/bin/env python3
"""Offline object-threshold sweep for VLQA layout predictions.

The evaluator stores all query object probabilities, boxes, and directions in
`layout_validation_predictions.jsonl`. This script replays those saved
predictions under multiple object thresholds, without loading a model or using
GPU, to determine whether low layout F1 is mainly a thresholding problem.
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

from layout_validation_metrics import (  # noqa: E402
    DIRECTION_LABELS,
    LayoutValidationAccumulator,
)


DEFAULT_THRESHOLDS = tuple(round(index / 10, 2) for index in range(1, 10))


class ThresholdSweepFailure(RuntimeError):
    pass


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ThresholdSweepFailure(f"Expected JSON object: {path}")
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ThresholdSweepFailure(f"Expected JSON object at {path}:{line_number}")
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
        description=(
            "Replay saved VLQA layout predictions under multiple object "
            "thresholds and report layout precision/recall/F1."
        )
    )
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--threshold",
        action="append",
        type=probability,
        help="Object threshold to evaluate; repeat to override the default 0.1..0.9 grid.",
    )
    parser.add_argument("--iou-threshold", type=probability, default=0.5)
    return parser.parse_args(argv)


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path | None, Path]:
    if args.comparison_root is None and args.predictions is None:
        raise ThresholdSweepFailure("Pass --comparison-root or --predictions.")
    if args.comparison_root is not None and args.predictions is not None:
        raise ThresholdSweepFailure("Pass only one of --comparison-root or --predictions.")
    if args.comparison_root is not None:
        root = args.comparison_root.expanduser().resolve()
        predictions = root / "vlqa" / "layout_validation_predictions.jsonl"
        output_dir = (
            args.output_dir.expanduser().resolve()
            if args.output_dir is not None
            else root / "analysis" / "threshold_sweep"
        )
        manifest = args.manifest.expanduser().resolve() if args.manifest else None
        return predictions, manifest, output_dir
    assert args.predictions is not None
    predictions = args.predictions.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else predictions.parent / "threshold_sweep"
    )
    manifest = args.manifest.expanduser().resolve() if args.manifest else None
    return predictions, manifest, output_dir


def index_manifest(manifest: Path | None) -> dict[str, dict[str, Any]]:
    if manifest is None:
        return {}
    if not manifest.is_file():
        raise ThresholdSweepFailure(f"Manifest does not exist: {manifest}")
    indexed: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(manifest):
        page_id = str(record.get("page_id", ""))
        if page_id:
            indexed[page_id] = record
    return indexed


def direction_index(label: Any) -> int:
    value = str(label)
    if value not in DIRECTION_LABELS:
        return DIRECTION_LABELS.index("unknown")
    return DIRECTION_LABELS.index(value)


def layout_inputs_for_record(
    record: dict[str, Any],
    manifest_by_id: dict[str, dict[str, Any]],
) -> tuple[list[float], list[list[float]], list[int], list[dict[str, Any]], str]:
    page_id = str(record.get("page_id", ""))
    layout_predictions = record.get("layout_predictions")
    if not isinstance(layout_predictions, list):
        raise ThresholdSweepFailure(f"Missing layout_predictions for page_id={page_id}")
    object_scores: list[float] = []
    predicted_boxes: list[list[float]] = []
    predicted_directions: list[int] = []
    for index, prediction in enumerate(layout_predictions):
        if not isinstance(prediction, dict):
            raise ThresholdSweepFailure(
                f"layout_predictions[{index}] is not an object for page_id={page_id}"
            )
        object_scores.append(float(prediction.get("object_probability", 0.0)))
        box = prediction.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            raise ThresholdSweepFailure(
                f"layout_predictions[{index}].bbox_xyxy is invalid for page_id={page_id}"
            )
        predicted_boxes.append([float(value) for value in box])
        predicted_directions.append(direction_index(prediction.get("writing_direction")))
    manifest_record = manifest_by_id.get(page_id, {})
    regions = manifest_record.get("regions") or record.get("regions") or []
    if not isinstance(regions, list):
        raise ThresholdSweepFailure(f"regions is invalid for page_id={page_id}")
    annotation_status = str(
        record.get("layout_annotation_status")
        or manifest_record.get("layout_annotation_status")
        or ("complete" if regions else "none")
    )
    return object_scores, predicted_boxes, predicted_directions, regions, annotation_status


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator) / float(denominator)


def get_number(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default


def failure_type(layout: dict[str, Any]) -> str:
    gt = int(get_number(layout.get("ground_truth_regions")))
    predicted = int(get_number(layout.get("predicted_regions")))
    matched = int(get_number(layout.get("matched_regions")))
    if gt <= 0:
        return "no_gt"
    false_negative = max(gt - matched, 0)
    false_positive = max(predicted - matched, 0)
    ordered_iou = get_number(layout.get("ordered_slot_bbox_mean_iou"))
    if false_negative == 0 and false_positive == 0 and ordered_iou >= 0.5:
        return "layout_ok"
    if false_negative > 0 and false_positive > 0:
        return "miss_and_extra"
    if false_negative > 0:
        return "missed_regions"
    if false_positive > 0:
        return "extra_regions"
    return "slot_or_bbox_low"


def evaluate_threshold(
    records: list[dict[str, Any]],
    manifest_by_id: dict[str, dict[str, Any]],
    *,
    threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    accumulator = LayoutValidationAccumulator(
        object_threshold=threshold,
        iou_threshold=iou_threshold,
    )
    failure_counts: Counter[str] = Counter()
    false_negative_regions = 0
    false_positive_regions = 0
    exact_region_count_pages = 0
    predicted_region_counts: list[int] = []
    ground_truth_region_counts: list[int] = []
    for record in records:
        object_scores, predicted_boxes, predicted_directions, regions, annotation_status = (
            layout_inputs_for_record(record, manifest_by_id)
        )
        page = accumulator.add_page(
            reference_text=str(record.get("reference_text", "")),
            predicted_text=str(record.get("predicted_text", "")),
            regions=regions,
            annotation_status=annotation_status,
            object_scores=object_scores,
            predicted_boxes=predicted_boxes,
            predicted_directions=predicted_directions,
        )
        layout = page["layout"]
        failure_counts[failure_type(layout)] += 1
        gt = int(layout["ground_truth_regions"])
        predicted = int(layout["predicted_regions"])
        matched = int(layout["matched_regions"])
        false_negative_regions += max(gt - matched, 0)
        false_positive_regions += max(predicted - matched, 0)
        exact_region_count_pages += int(gt == predicted)
        predicted_region_counts.append(predicted)
        ground_truth_region_counts.append(gt)
    summary = accumulator.summary()
    layout = summary["layout"]
    precision = layout.get("complete_region_precision")
    recall = layout.get("complete_region_recall")
    f1 = layout.get("complete_region_f1")
    return {
        "object_threshold": threshold,
        "iou_threshold": iou_threshold,
        "pages": summary["ocr"]["pages"],
        "complete_region_precision": precision,
        "complete_region_recall": recall,
        "complete_region_f1": f1,
        "complete_ground_truth_regions": layout.get("complete_ground_truth_regions"),
        "complete_predicted_regions": layout.get("complete_predicted_regions"),
        "complete_matched_regions": layout.get("complete_matched_regions"),
        "false_negative_regions": false_negative_regions,
        "false_positive_regions": false_positive_regions,
        "mean_predicted_regions_per_page": safe_ratio(
            sum(predicted_region_counts),
            len(predicted_region_counts),
        ),
        "mean_ground_truth_regions_per_page": safe_ratio(
            sum(ground_truth_region_counts),
            len(ground_truth_region_counts),
        ),
        "exact_region_count_pages": exact_region_count_pages,
        "exact_region_count_rate": safe_ratio(exact_region_count_pages, len(records)),
        "ordered_slot_object_recall": layout.get("ordered_slot_object_recall"),
        "ordered_slot_bbox_mean_iou": layout.get("ordered_slot_bbox_mean_iou"),
        "matched_bbox_mean_iou": layout.get("matched_bbox_mean_iou"),
        "matched_direction_accuracy": layout.get("matched_direction_accuracy"),
        "reading_order_pair_accuracy": layout.get("reading_order_pair_accuracy"),
        "reading_order_kendall_tau": layout.get("reading_order_kendall_tau"),
        "failure_types": dict(failure_counts),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "object_threshold",
        "iou_threshold",
        "pages",
        "complete_region_precision",
        "complete_region_recall",
        "complete_region_f1",
        "complete_ground_truth_regions",
        "complete_predicted_regions",
        "complete_matched_regions",
        "false_negative_regions",
        "false_positive_regions",
        "mean_predicted_regions_per_page",
        "mean_ground_truth_regions_per_page",
        "exact_region_count_pages",
        "exact_region_count_rate",
        "ordered_slot_object_recall",
        "ordered_slot_bbox_mean_iou",
        "matched_bbox_mean_iou",
        "matched_direction_accuracy",
        "reading_order_pair_accuracy",
        "reading_order_kendall_tau",
        "failure_types",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            row = {**row, "failure_types": compact_json(row["failure_types"])}
            writer.writerow(row)


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, dict):
        return compact_json(value).replace("|", "\\|")
    return str(value).replace("|", "\\|")


def write_markdown(path: Path, rows: list[dict[str, Any]], best: dict[str, Any] | None) -> None:
    fields = (
        "object_threshold",
        "complete_region_precision",
        "complete_region_recall",
        "complete_region_f1",
        "complete_predicted_regions",
        "false_negative_regions",
        "false_positive_regions",
        "exact_region_count_rate",
        "failure_types",
    )
    lines = [
        "# VLQA layout object-threshold sweep",
        "",
        "## Best threshold by complete_region_f1",
        "",
        "```json",
        compact_json(best),
        "```",
        "",
        "## Sweep table",
        "",
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(field)) for field in fields) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def best_by_f1(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    finite = [
        row
        for row in rows
        if isinstance(row.get("complete_region_f1"), (int, float))
        and math.isfinite(float(row["complete_region_f1"]))
    ]
    if not finite:
        return None
    return max(
        finite,
        key=lambda row: (
            float(row["complete_region_f1"]),
            float(row.get("complete_region_precision") or 0.0),
            -abs(float(row["object_threshold"]) - 0.5),
        ),
    )


def sweep(args: argparse.Namespace) -> dict[str, Any]:
    predictions_path, manifest_path, output_dir = resolve_inputs(args)
    if not predictions_path.is_file():
        raise ThresholdSweepFailure(f"Predictions file does not exist: {predictions_path}")
    thresholds = sorted(set(args.threshold if args.threshold else DEFAULT_THRESHOLDS))
    records = read_jsonl(predictions_path)
    manifest_by_id = index_manifest(manifest_path)
    rows = [
        evaluate_threshold(
            records,
            manifest_by_id,
            threshold=threshold,
            iou_threshold=args.iou_threshold,
        )
        for threshold in thresholds
    ]
    best = best_by_f1(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "status": "ok",
        "predictions": str(predictions_path),
        "manifest": str(manifest_path) if manifest_path else None,
        "thresholds": thresholds,
        "iou_threshold": args.iou_threshold,
        "best_by_complete_region_f1": best,
        "rows": rows,
    }
    write_json(output_dir / "threshold_sweep_summary.json", summary)
    write_csv(output_dir / "threshold_sweep.csv", rows)
    write_markdown(output_dir / "threshold_sweep.md", rows, best)
    return {
        "event": "layout_threshold_sweep_completed",
        "pages": rows[0]["pages"] if rows else 0,
        "output_dir": str(output_dir),
        "summary": str(output_dir / "threshold_sweep_summary.json"),
        "csv": str(output_dir / "threshold_sweep.csv"),
        "markdown": str(output_dir / "threshold_sweep.md"),
        "best_by_complete_region_f1": best,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(compact_json(sweep(parse_args(argv))), flush=True)
    except Exception as exc:
        print(
            compact_json(
                {
                    "event": "layout_threshold_sweep_failed",
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
