#!/usr/bin/env python3
"""Offline error analysis for a formal GOT2 baseline versus VLQA comparison.

This script is intentionally CPU-only and read-only. It consumes the files that
`compare_got2_vlqa.py` already wrote under a comparison run directory and
produces compact aggregate reports for deciding whether the next action should
be OCR ablation, layout threshold/slot diagnosis, or data inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_GROUPS = (
    "tier",
    "region_count_bin",
    "direction_signature",
    "text_length_bin",
    "layout_failure_type",
)


class AnalysisFailure(RuntimeError):
    pass


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnalysisFailure(f"Expected JSON object: {path}")
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
                raise AnalysisFailure(f"Expected JSON object at {path}:{line_number}")
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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze OCR deltas and VLQA layout failure modes from a completed "
            "formal comparison run."
        )
    )
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=positive_int, default=20)
    parser.add_argument(
        "--group-by",
        action="append",
        choices=DEFAULT_GROUPS,
        help="Repeat to override default grouping dimensions.",
    )
    return parser.parse_args(argv)


def get_number(value: Any, default: float = 0.0) -> float:
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default


def get_nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def bin_count(value: int, boundaries: Sequence[int]) -> str:
    previous = 0
    for boundary in boundaries:
        if value <= boundary:
            return f"{previous + 1}-{boundary}"
        previous = boundary
    return f">{boundaries[-1]}"


def text_length_bin(length: int) -> str:
    return bin_count(length, (50, 100, 200, 400, 800))


def region_count_bin(count: int) -> str:
    return bin_count(count, (1, 2, 4, 8, 12, 16))


def normalize_direction(value: Any) -> str:
    return str(value) if value else "unknown"


def direction_signature(regions: Sequence[dict[str, Any]]) -> str:
    directions = Counter(
        normalize_direction(region.get("writing_direction"))
        for region in regions
        if isinstance(region, dict)
    )
    if not directions:
        return "none"
    if len(directions) == 1:
        return next(iter(directions))
    return "mixed:" + "+".join(sorted(directions))


def infer_tier(record: dict[str, Any], page_id: str) -> str:
    for key in ("tier", "synthesis_tier", "layout_tier"):
        value = record.get(key)
        if value:
            return str(value)
    lowered = page_id.lower()
    for tier in ("s0-html-text", "s1-html-crop", "s2-hard"):
        compact = tier.replace("-", "_")
        if tier in lowered or compact in lowered:
            return tier
    return "unknown"


def layout_failure_type(layout: dict[str, Any]) -> str:
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


def load_comparison_inputs(
    comparison_root: Path,
    manifest_arg: Path | None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    root = comparison_root.expanduser().resolve()
    if not root.is_dir():
        raise AnalysisFailure(f"Comparison root does not exist: {root}")
    summary = read_json(root / "summary.json")
    baseline_summary = read_json(root / "baseline" / "layout_validation_metrics.json")
    vlqa_summary = read_json(root / "vlqa" / "layout_validation_metrics.json")
    baseline_predictions = read_jsonl(root / "baseline" / "layout_validation_predictions.jsonl")
    vlqa_predictions = read_jsonl(root / "vlqa" / "layout_validation_predictions.jsonl")
    manifest_path = (
        manifest_arg.expanduser().resolve()
        if manifest_arg is not None
        else Path(str(baseline_summary.get("manifest", ""))).expanduser().resolve()
    )
    if not manifest_path.is_file():
        raise AnalysisFailure(
            "Manifest path is unavailable. Pass --manifest explicitly; "
            f"resolved path was {manifest_path!s}."
        )
    manifest = read_jsonl(manifest_path)
    return summary, baseline_summary, baseline_predictions, vlqa_predictions, manifest


def index_by_page_id(records: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        page_id = str(record.get("page_id", ""))
        if not page_id:
            raise AnalysisFailure(f"{label} contains a record without page_id.")
        if page_id in indexed:
            raise AnalysisFailure(f"{label} contains duplicate page_id: {page_id}")
        indexed[page_id] = record
    return indexed


def page_analysis_rows(
    baseline_predictions: list[dict[str, Any]],
    vlqa_predictions: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_id = index_by_page_id(baseline_predictions, "baseline predictions")
    vlqa_by_id = index_by_page_id(vlqa_predictions, "vlqa predictions")
    manifest_by_id = index_by_page_id(manifest, "manifest")
    if set(baseline_by_id) != set(vlqa_by_id):
        missing_vlqa = sorted(set(baseline_by_id) - set(vlqa_by_id))[:10]
        missing_baseline = sorted(set(vlqa_by_id) - set(baseline_by_id))[:10]
        raise AnalysisFailure(
            "Baseline/VLQA page_id sets differ: "
            f"missing_vlqa={missing_vlqa}, missing_baseline={missing_baseline}"
        )

    rows: list[dict[str, Any]] = []
    for page_id in sorted(baseline_by_id):
        baseline = baseline_by_id[page_id]
        vlqa = vlqa_by_id[page_id]
        manifest_record = manifest_by_id.get(page_id, {})
        regions = manifest_record.get("regions") or vlqa.get("regions") or []
        if not isinstance(regions, list):
            regions = []
        reference_text = str(vlqa.get("reference_text") or baseline.get("reference_text") or "")
        baseline_ocr = get_nested(baseline, "metrics", "ocr") or {}
        vlqa_ocr = get_nested(vlqa, "metrics", "ocr") or {}
        layout = get_nested(vlqa, "metrics", "layout") or {}
        if not isinstance(baseline_ocr, dict) or not isinstance(vlqa_ocr, dict):
            raise AnalysisFailure(f"Missing OCR metrics for page_id={page_id}")
        if not isinstance(layout, dict):
            layout = {}
        baseline_edits = int(get_number(baseline_ocr.get("edit_distance")))
        vlqa_edits = int(get_number(vlqa_ocr.get("edit_distance")))
        delta_edits = vlqa_edits - baseline_edits
        row = {
            "page_id": page_id,
            "tier": infer_tier(manifest_record, page_id),
            "source_group_id": str(manifest_record.get("source_group_id", "unknown")),
            "region_count": len(regions),
            "region_count_bin": region_count_bin(len(regions)),
            "direction_signature": direction_signature(regions),
            "text_length": len(reference_text),
            "text_length_bin": text_length_bin(len(reference_text)),
            "baseline_edit_distance": baseline_edits,
            "vlqa_edit_distance": vlqa_edits,
            "delta_edit_distance": delta_edits,
            "baseline_cer": baseline_ocr.get("cer"),
            "vlqa_cer": vlqa_ocr.get("cer"),
            "delta_cer": (
                get_number(vlqa_ocr.get("cer")) - get_number(baseline_ocr.get("cer"))
                if baseline_ocr.get("cer") is not None and vlqa_ocr.get("cer") is not None
                else None
            ),
            "baseline_exact_match": bool(baseline_ocr.get("exact_match")),
            "vlqa_exact_match": bool(vlqa_ocr.get("exact_match")),
            "ocr_outcome": "improved" if delta_edits < 0 else "worse" if delta_edits > 0 else "same",
            "layout_failure_type": layout_failure_type(layout),
            "layout_ground_truth_regions": int(get_number(layout.get("ground_truth_regions"))),
            "layout_predicted_regions": int(get_number(layout.get("predicted_regions"))),
            "layout_matched_regions": int(get_number(layout.get("matched_regions"))),
            "layout_false_negative_regions": max(
                int(get_number(layout.get("ground_truth_regions")))
                - int(get_number(layout.get("matched_regions"))),
                0,
            ),
            "layout_false_positive_regions": max(
                int(get_number(layout.get("predicted_regions")))
                - int(get_number(layout.get("matched_regions"))),
                0,
            ),
            "layout_region_precision": layout.get("region_precision"),
            "layout_region_recall": layout.get("region_recall"),
            "layout_ordered_slot_bbox_mean_iou": layout.get("ordered_slot_bbox_mean_iou"),
            "layout_matched_bbox_mean_iou": layout.get("matched_bbox_mean_iou"),
            "layout_ordered_direction_accuracy": layout.get("ordered_direction_accuracy"),
            "layout_matched_direction_accuracy": layout.get("matched_direction_accuracy"),
        }
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pages = len(rows)
    baseline_edits = sum(int(row["baseline_edit_distance"]) for row in rows)
    vlqa_edits = sum(int(row["vlqa_edit_distance"]) for row in rows)
    reference_characters = sum(int(row["text_length"]) for row in rows)
    return {
        "pages": pages,
        "baseline_edit_distance": baseline_edits,
        "vlqa_edit_distance": vlqa_edits,
        "delta_edit_distance_vlqa_minus_baseline": vlqa_edits - baseline_edits,
        "baseline_page_cer_from_predictions": (
            baseline_edits / reference_characters if reference_characters else None
        ),
        "vlqa_page_cer_from_predictions": (
            vlqa_edits / reference_characters if reference_characters else None
        ),
        "delta_page_cer_from_predictions": (
            (vlqa_edits - baseline_edits) / reference_characters
            if reference_characters
            else None
        ),
        "baseline_exact_matches": sum(bool(row["baseline_exact_match"]) for row in rows),
        "vlqa_exact_matches": sum(bool(row["vlqa_exact_match"]) for row in rows),
        "exact_match_delta_vlqa_minus_baseline": (
            sum(bool(row["vlqa_exact_match"]) for row in rows)
            - sum(bool(row["baseline_exact_match"]) for row in rows)
        ),
        "ocr_outcomes": dict(Counter(str(row["ocr_outcome"]) for row in rows)),
        "layout_failure_types": dict(Counter(str(row["layout_failure_type"]) for row in rows)),
    }


def aggregate_group(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    aggregates: list[dict[str, Any]] = []
    for value, group_rows in sorted(grouped.items()):
        summary = summarize_rows(group_rows)
        aggregates.append(
            {
                "group_by": key,
                "group": value,
                **summary,
                "mean_layout_ordered_slot_bbox_iou": mean_optional(
                    row.get("layout_ordered_slot_bbox_mean_iou") for row in group_rows
                ),
                "mean_layout_matched_bbox_iou": mean_optional(
                    row.get("layout_matched_bbox_mean_iou") for row in group_rows
                ),
                "mean_layout_precision": mean_optional(
                    row.get("layout_region_precision") for row in group_rows
                ),
                "mean_layout_recall": mean_optional(
                    row.get("layout_region_recall") for row in group_rows
                ),
            }
        )
    return sorted(
        aggregates,
        key=lambda item: (
            -int(item["pages"]),
            str(item["group_by"]),
            str(item["group"]),
        ),
    )


def mean_optional(values: Iterable[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return sum(finite) / len(finite) if finite else None


def top_pages(rows: list[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    fields = (
        "page_id",
        "tier",
        "region_count",
        "direction_signature",
        "text_length",
        "baseline_edit_distance",
        "vlqa_edit_distance",
        "delta_edit_distance",
        "layout_failure_type",
        "layout_predicted_regions",
        "layout_matched_regions",
        "layout_false_negative_regions",
        "layout_false_positive_regions",
        "layout_ordered_slot_bbox_mean_iou",
    )

    def project(row: dict[str, Any]) -> dict[str, Any]:
        return {field: row.get(field) for field in fields}

    return {
        "largest_vlqa_improvements": [
            project(row)
            for row in sorted(rows, key=lambda item: int(item["delta_edit_distance"]))[:top_k]
            if int(row["delta_edit_distance"]) < 0
        ],
        "largest_vlqa_regressions": [
            project(row)
            for row in sorted(rows, key=lambda item: -int(item["delta_edit_distance"]))[:top_k]
            if int(row["delta_edit_distance"]) > 0
        ],
        "worst_layout_slot_iou": [
            project(row)
            for row in sorted(
                rows,
                key=lambda item: get_number(item.get("layout_ordered_slot_bbox_mean_iou"), 1.0),
            )[:top_k]
        ],
    }


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


def markdown_table(rows: list[dict[str, Any]], fields: Sequence[str], limit: int = 20) -> str:
    selected = rows[:limit]
    if not selected:
        return "_No rows._\n"
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in selected:
        body.append(
            "| "
            + " | ".join(format_markdown_value(row.get(field)) for field in fields)
            + " |"
        )
    return "\n".join([header, divider, *body]) + "\n"


def format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    groups = report["groups"]
    top = report["top_pages"]
    lines = [
        "# Layout comparison error analysis",
        "",
        "## Overview",
        "",
        "```json",
        compact_json(report["overview"]),
        "```",
        "",
        "## Group summaries",
        "",
    ]
    group_fields = (
        "group_by",
        "group",
        "pages",
        "delta_page_cer_from_predictions",
        "exact_match_delta_vlqa_minus_baseline",
        "mean_layout_ordered_slot_bbox_iou",
        "mean_layout_precision",
        "mean_layout_recall",
    )
    lines.append(markdown_table(groups, group_fields, limit=50))
    page_fields = (
        "page_id",
        "tier",
        "delta_edit_distance",
        "layout_failure_type",
        "layout_ordered_slot_bbox_mean_iou",
        "layout_false_negative_regions",
        "layout_false_positive_regions",
    )
    for title, rows in top.items():
        lines.extend(["", f"## {title}", "", markdown_table(rows, page_fields, limit=20)])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    summary, baseline_summary, baseline_predictions, vlqa_predictions, manifest = (
        load_comparison_inputs(args.comparison_root, args.manifest)
    )
    rows = page_analysis_rows(baseline_predictions, vlqa_predictions, manifest)
    group_keys = tuple(args.group_by) if args.group_by else DEFAULT_GROUPS
    groups = []
    for key in group_keys:
        groups.extend(aggregate_group(rows, key))
    report = {
        "schema_version": 1,
        "status": "ok",
        "comparison_root": str(args.comparison_root.expanduser().resolve()),
        "comparison_summary": {
            "comparison": summary.get("comparison"),
            "baseline_metrics": baseline_summary.get("metrics"),
        },
        "overview": summarize_rows(rows),
        "groups": groups,
        "top_pages": top_pages(rows, args.top_k),
    }
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else args.comparison_root.expanduser().resolve() / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "error_analysis_summary.json", report)
    write_csv(output_dir / "page_error_analysis.csv", rows)
    write_csv(output_dir / "group_error_analysis.csv", groups)
    write_markdown(output_dir / "error_analysis.md", report)
    return {
        "event": "layout_comparison_error_analysis_completed",
        "pages": report["overview"]["pages"],
        "output_dir": str(output_dir),
        "summary": str(output_dir / "error_analysis_summary.json"),
        "page_csv": str(output_dir / "page_error_analysis.csv"),
        "group_csv": str(output_dir / "group_error_analysis.csv"),
        "markdown": str(output_dir / "error_analysis.md"),
        "overview": report["overview"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(compact_json(analyze(parse_args(argv))), flush=True)
    except Exception as exc:
        print(
            compact_json(
                {
                    "event": "layout_comparison_error_analysis_failed",
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
