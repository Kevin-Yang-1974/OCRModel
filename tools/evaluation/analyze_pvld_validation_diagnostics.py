#!/usr/bin/env python3
"""Offline diagnostics for completed PVLD validation predictions."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-layout-records", type=int, default=512)
    parser.add_argument("--max-layout-tokens", type=int, default=2048)
    parser.add_argument("--duplicate-iou", type=float, default=0.9)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def count_near_duplicate_boxes(boxes: list[list[float]], threshold: float) -> int:
    if len(boxes) < 2:
        return 0
    array = np.asarray(boxes, dtype=np.float64)
    valid = (
        np.isfinite(array).all(axis=1)
        & (array[:, 2] > array[:, 0])
        & (array[:, 3] > array[:, 1])
    )
    array = array[valid]
    if len(array) < 2:
        return 0
    x1 = np.maximum(array[:, None, 0], array[None, :, 0])
    y1 = np.maximum(array[:, None, 1], array[None, :, 1])
    x2 = np.minimum(array[:, None, 2], array[None, :, 2])
    y2 = np.minimum(array[:, None, 3], array[None, :, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = (array[:, 2] - array[:, 0]) * (array[:, 3] - array[:, 1])
    union = area[:, None] + area[None, :] - intersection
    iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    np.fill_diagonal(iou, 0.0)
    return int(np.count_nonzero(np.max(iou, axis=1) >= threshold))


def bucket_name(count: int) -> str:
    if count <= 8:
        return "0-8"
    if count <= 16:
        return "9-16"
    if count <= 32:
        return "17-32"
    return ">32"


def summarize(label: str, path: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    true_counts: list[float] = []
    predicted_counts: list[float] = []
    count_errors: list[float] = []
    absolute_count_errors: list[float] = []
    page_cers: list[float] = []
    scores: list[float] = []
    layout_tokens: list[float] = []
    eos_pages = record_cap_pages = token_cap_pages = premature_eos_pages = 0
    duplicate_boxes = predicted_boxes = invalid_boxes = 0
    buckets: dict[str, list[dict[str, float]]] = {name: [] for name in ("0-8", "9-16", "17-32", ">32")}

    for row in rows:
        truth = len(row["regions"])
        predictions = row["layout_predictions"]
        predicted = int(row["num_predicted_regions"])
        eos = bool(row["generated_eos"])
        tokens = int(row["layout_tokens"])
        ocr = row["metrics"]["ocr"]
        cer = float(
            ocr.get(
                "cer",
                float(ocr.get("character_edits", 0))
                / max(1, int(ocr["reference_characters"])),
            )
        )
        error = predicted - truth
        boxes = [item["bbox"] for item in predictions]

        true_counts.append(float(truth))
        predicted_counts.append(float(predicted))
        count_errors.append(float(error))
        absolute_count_errors.append(float(abs(error)))
        page_cers.append(cer)
        layout_tokens.append(float(tokens))
        scores.extend(float(item["score"]) for item in predictions)
        eos_pages += int(eos)
        record_cap_pages += int(predicted >= args.max_layout_records)
        token_cap_pages += int((not eos) and tokens >= args.max_layout_tokens - 1)
        premature_eos_pages += int(eos and predicted < truth)
        predicted_boxes += len(boxes)
        invalid_boxes += sum(
            not all(math.isfinite(float(value)) for value in box)
            or float(box[2]) <= float(box[0])
            or float(box[3]) <= float(box[1])
            for box in boxes
        )
        duplicates = count_near_duplicate_boxes(boxes, args.duplicate_iou)
        duplicate_boxes += duplicates
        buckets[bucket_name(truth)].append(
            {"truth": float(truth), "predicted": float(predicted), "absolute_error": float(abs(error)), "cer": cer}
        )

    bucket_summary = {}
    for name, values in buckets.items():
        bucket_summary[name] = {
            "pages": len(values),
            "mean_true_regions": statistics.fmean(item["truth"] for item in values) if values else None,
            "mean_predicted_regions": statistics.fmean(item["predicted"] for item in values) if values else None,
            "count_mae": statistics.fmean(item["absolute_error"] for item in values) if values else None,
            "mean_page_cer": statistics.fmean(item["cer"] for item in values) if values else None,
        }

    pages = len(rows)
    return {
        "label": label,
        "predictions": str(path),
        "pages": pages,
        "count": {
            "true_mean": statistics.fmean(true_counts),
            "predicted_mean": statistics.fmean(predicted_counts),
            "predicted_median": percentile(predicted_counts, 50),
            "predicted_p90": percentile(predicted_counts, 90),
            "predicted_max": max(predicted_counts),
            "mean_signed_error": statistics.fmean(count_errors),
            "mae": statistics.fmean(absolute_count_errors),
            "exact_accuracy": sum(error == 0 for error in count_errors) / pages,
            "true_predicted_pearson": pearson(true_counts, predicted_counts),
        },
        "stopping": {
            "eos_success_rate": eos_pages / pages,
            "premature_eos_rate": premature_eos_pages / pages,
            "record_cap_rate": record_cap_pages / pages,
            "token_cap_rate": token_cap_pages / pages,
            "layout_tokens_mean": statistics.fmean(layout_tokens),
            "layout_tokens_p90": percentile(layout_tokens, 90),
            "layout_tokens_max": max(layout_tokens),
        },
        "confidence": {
            "score_min": min(scores) if scores else None,
            "score_median": percentile(scores, 50),
            "score_p10": percentile(scores, 10),
            "score_max": max(scores) if scores else None,
            "fraction_at_least_0_99": sum(score >= 0.99 for score in scores) / len(scores) if scores else None,
        },
        "boxes": {
            "predicted": predicted_boxes,
            "invalid": invalid_boxes,
            "invalid_rate": invalid_boxes / predicted_boxes if predicted_boxes else None,
            "near_duplicate": duplicate_boxes,
            "near_duplicate_rate": duplicate_boxes / predicted_boxes if predicted_boxes else None,
            "duplicate_iou_threshold": args.duplicate_iou,
        },
        "association": {
            "absolute_count_error_vs_page_cer_pearson": pearson(absolute_count_errors, page_cers),
            "predicted_count_vs_page_cer_pearson": pearson(predicted_counts, page_cers),
        },
        "true_region_buckets": bucket_summary,
    }


def write_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# PVLD Validation Diagnostics",
        "",
        "All diagnostics use only validation predictions from the validation-selected checkpoint.",
        "",
        "| Control | EOS | Record cap | Pred/true mean | Count MAE | Count corr. | Near-duplicate boxes | Invalid boxes | Count error/CER corr. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        count = item["count"]
        stopping = item["stopping"]
        boxes = item["boxes"]
        association = item["association"]
        lines.append(
            f"| {item['label']} | {stopping['eos_success_rate']:.4f} | {stopping['record_cap_rate']:.4f} | "
            f"{count['predicted_mean']:.2f}/{count['true_mean']:.2f} | {count['mae']:.2f} | "
            f"{count['true_predicted_pearson'] if count['true_predicted_pearson'] is not None else 'NA'} | "
            f"{boxes['near_duplicate_rate']:.4f} | {boxes['invalid_rate']:.4f} | "
            f"{association['absolute_count_error_vs_page_cer_pearson'] if association['absolute_count_error_vs_page_cer_pearson'] is not None else 'NA'} |"
        )
    lines.extend(["", "The JSON file contains score calibration and true-region-count buckets.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    inputs = []
    for value in args.prediction:
        label, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError(f"Expected LABEL=PATH, got: {value}")
        inputs.append((label, Path(raw_path)))
    summaries = [summarize(label, path, args) for label, path in inputs]
    payload = {
        "status": "ok",
        "protocol": "validation_only_offline_diagnostics",
        "max_layout_records": args.max_layout_records,
        "max_layout_tokens": args.max_layout_tokens,
        "controls": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(args.output_dir / "report.md", summaries)
    print(json.dumps({"status": "ok", "output_dir": str(args.output_dir), "controls": len(summaries)}))


if __name__ == "__main__":
    main()
