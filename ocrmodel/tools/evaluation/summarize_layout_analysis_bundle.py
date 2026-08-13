#!/usr/bin/env python3
"""Summarize completed layout comparison diagnostics into one compact bundle.

This script is intentionally offline, CPU-only, and read-only with respect to
model outputs. It consumes the JSON files already produced by:

- analyze_layout_comparison_errors.py
- analyze_layout_threshold_sweep.py
- analyze_layout_slot_alignment.py

and writes a compact cross-diagnostic summary for reporting the current
checkpoint state without re-running inference or re-parsing large prediction
files.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence


class AnalysisBundleFailure(RuntimeError):
    pass


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisBundleFailure(f"Required JSON file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AnalysisBundleFailure(f"Expected JSON object: {path}")
    return payload


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


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize completed layout error analysis, threshold sweep, and "
            "slot-alignment diagnostics for one comparison run."
        )
    )
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        help="Defaults to <comparison-root>/analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to <analysis-dir>/analysis_bundle.",
    )
    parser.add_argument("--top-k", type=positive_int, default=10)
    parser.add_argument("--min-group-pages", type=non_negative_int, default=5)
    return parser.parse_args(argv)


def finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def get_nested(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def safe_delta(left: Any, right: Any) -> float | None:
    left_number = finite_number(left)
    right_number = finite_number(right)
    if left_number is None or right_number is None:
        return None
    return left_number - right_number


def sort_offset_counts(counts: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, count in counts.items():
        if not isinstance(count, int):
            continue
        rows.append({"offset": str(offset), "count": count})

    def key(row: dict[str, Any]) -> tuple[int, int, str]:
        offset = str(row["offset"])
        try:
            abs_offset = abs(int(offset))
            signed_offset = int(offset)
        except ValueError:
            abs_offset = 10**9
            signed_offset = 10**9
        return (-int(row["count"]), abs_offset, str(signed_offset))

    return sorted(rows, key=key)[:limit]


def project_page(row: dict[str, Any]) -> dict[str, Any]:
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
    return {field: row.get(field) for field in fields if field in row}


def top_groups(
    groups: Iterable[dict[str, Any]],
    *,
    top_k: int,
    min_pages: int,
) -> dict[str, list[dict[str, Any]]]:
    eligible = [
        group
        for group in groups
        if isinstance(group, dict)
        and isinstance(group.get("pages"), int)
        and int(group["pages"]) >= min_pages
    ]

    def project(group: dict[str, Any]) -> dict[str, Any]:
        fields = (
            "group_by",
            "group",
            "pages",
            "delta_edit_distance_vlqa_minus_baseline",
            "delta_page_cer_from_predictions",
            "exact_match_delta_vlqa_minus_baseline",
            "ocr_outcomes",
            "layout_failure_types",
            "mean_layout_ordered_slot_bbox_iou",
            "mean_layout_matched_bbox_iou",
            "mean_layout_precision",
            "mean_layout_recall",
        )
        return {field: group.get(field) for field in fields if field in group}

    by_regression = sorted(
        eligible,
        key=lambda group: (
            -int(group.get("delta_edit_distance_vlqa_minus_baseline") or 0),
            -int(group.get("pages") or 0),
        ),
    )
    by_low_ordered_iou = sorted(
        eligible,
        key=lambda group: (
            finite_number(group.get("mean_layout_ordered_slot_bbox_iou"))
            if finite_number(group.get("mean_layout_ordered_slot_bbox_iou")) is not None
            else 10**9,
            -int(group.get("pages") or 0),
        ),
    )
    by_miss_and_extra = sorted(
        eligible,
        key=lambda group: (
            -int((group.get("layout_failure_types") or {}).get("miss_and_extra", 0))
            if isinstance(group.get("layout_failure_types"), dict)
            else 0,
            -int(group.get("pages") or 0),
        ),
    )
    return {
        "largest_group_regressions": [project(group) for group in by_regression[:top_k]],
        "lowest_group_ordered_iou": [project(group) for group in by_low_ordered_iou[:top_k]],
        "most_miss_and_extra_groups": [project(group) for group in by_miss_and_extra[:top_k]],
    }


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (dict, list)):
        return compact_json(value).replace("|", "\\|")
    return str(value).replace("|", "\\|")


def markdown_table(rows: list[dict[str, Any]], fields: Sequence[str], limit: int) -> str:
    selected = rows[:limit]
    if not selected:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(format_value(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, bundle: dict[str, Any], top_k: int) -> None:
    headline = bundle["headline"]
    residual = bundle["residual_diagnosis"]
    lines = [
        "# Layout analysis bundle",
        "",
        "## Headline",
        "",
        "```json",
        compact_json(headline),
        "```",
        "",
        "## Residual diagnosis",
        "",
        "```json",
        compact_json(residual),
        "```",
        "",
        "## Group priorities",
        "",
    ]
    group_fields = (
        "group_by",
        "group",
        "pages",
        "delta_edit_distance_vlqa_minus_baseline",
        "delta_page_cer_from_predictions",
        "mean_layout_ordered_slot_bbox_iou",
        "layout_failure_types",
    )
    for title, rows in bundle["group_priorities"].items():
        lines.extend(["", f"### {title}", "", markdown_table(rows, group_fields, top_k)])

    page_fields = (
        "page_id",
        "tier",
        "region_count",
        "delta_edit_distance",
        "layout_failure_type",
        "layout_false_negative_regions",
        "layout_false_positive_regions",
        "layout_ordered_slot_bbox_mean_iou",
    )
    lines.extend(["", "## Page priorities", ""])
    for title, rows in bundle["page_priorities"].items():
        fields = ("page_id", "slot_iou_gap_sum") if title == "worst_slot_gap_pages" else page_fields
        lines.extend(["", f"### {title}", "", markdown_table(rows, fields, top_k)])

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    comparison_root = args.comparison_root.expanduser().resolve()
    analysis_dir = (
        args.analysis_dir.expanduser().resolve()
        if args.analysis_dir is not None
        else comparison_root / "analysis"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else analysis_dir / "analysis_bundle"
    )

    comparison_summary = read_json(comparison_root / "summary.json")
    error_summary = read_json(analysis_dir / "error_analysis_summary.json")
    threshold_summary = read_json(analysis_dir / "threshold_sweep" / "threshold_sweep_summary.json")
    slot_alignment_summary = read_json(
        analysis_dir / "slot_alignment" / "slot_alignment_summary.json"
    )

    comparison = comparison_summary.get("comparison") or {}
    overview = error_summary.get("overview") or {}
    best_threshold = threshold_summary.get("best_by_complete_region_f1") or {}
    slot_diagnostics = (
        slot_alignment_summary.get("summary")
        or slot_alignment_summary.get("diagnostics")
        or {}
    )
    if not isinstance(comparison, dict):
        comparison = {}
    if not isinstance(overview, dict):
        overview = {}
    if not isinstance(best_threshold, dict):
        best_threshold = {}
    if not isinstance(slot_diagnostics, dict):
        slot_diagnostics = {}

    default_layout_f1 = get_nested(comparison, "vlqa_layout", "complete_region_f1")
    best_threshold_f1 = best_threshold.get("complete_region_f1")
    threshold_f1_gain = safe_delta(best_threshold_f1, default_layout_f1)
    ordered_hit_rate = slot_diagnostics.get("ordered_hit_rate")
    best_hit_rate = slot_diagnostics.get("best_hit_rate")

    top_pages = error_summary.get("top_pages") if isinstance(error_summary.get("top_pages"), dict) else {}
    groups = error_summary.get("groups") if isinstance(error_summary.get("groups"), list) else []
    page_priorities = {
        "largest_vlqa_regressions": [
            project_page(row)
            for row in top_pages.get("largest_vlqa_regressions", [])[: args.top_k]
            if isinstance(row, dict)
        ],
        "worst_layout_slot_iou": [
            project_page(row)
            for row in top_pages.get("worst_layout_slot_iou", [])[: args.top_k]
            if isinstance(row, dict)
        ],
        "worst_slot_gap_pages": [
            row
            for row in slot_diagnostics.get("worst_pages_by_slot_gap", [])[: args.top_k]
            if isinstance(row, dict)
        ],
    }

    headline = {
        "pages": overview.get("pages"),
        "ocr_page_cer_delta_vlqa_minus_baseline": comparison.get(
            "ocr_page_cer_delta_vlqa_minus_baseline"
        ),
        "baseline_edit_distance": overview.get("baseline_edit_distance"),
        "vlqa_edit_distance": overview.get("vlqa_edit_distance"),
        "delta_edit_distance_vlqa_minus_baseline": overview.get(
            "delta_edit_distance_vlqa_minus_baseline"
        ),
        "ocr_outcomes": overview.get("ocr_outcomes"),
        "baseline_exact_matches": overview.get("baseline_exact_matches"),
        "vlqa_exact_matches": overview.get("vlqa_exact_matches"),
        "layout_failure_types": overview.get("layout_failure_types"),
        "default_complete_region_f1": default_layout_f1,
        "best_threshold_object": best_threshold.get("object_threshold"),
        "best_threshold_complete_region_f1": best_threshold_f1,
        "threshold_f1_gain_over_default": threshold_f1_gain,
        "ordered_hit_rate": ordered_hit_rate,
        "best_hit_rate": best_hit_rate,
        "slot_misaligned_hit_rate": slot_diagnostics.get("slot_misaligned_hit_rate"),
        "mean_ordered_iou": slot_diagnostics.get("mean_ordered_iou"),
        "mean_best_iou": slot_diagnostics.get("mean_best_iou"),
    }
    residual_diagnosis = {
        "threshold_is_main_issue": (
            bool(threshold_f1_gain is not None and threshold_f1_gain >= 0.02)
        ),
        "slot_alignment_gap": safe_delta(best_hit_rate, ordered_hit_rate),
        "slot_misalignment_is_main_issue": (
            bool(
                finite_number(slot_diagnostics.get("slot_misaligned_hit_rate")) is not None
                and float(slot_diagnostics["slot_misaligned_hit_rate"]) >= 0.10
            )
        ),
        "top_query_offsets": sort_offset_counts(
            slot_diagnostics.get("best_query_offset_counts")
            if isinstance(slot_diagnostics.get("best_query_offset_counts"), dict)
            else {}
        ),
        "recommended_next_actions": [
            "inspect_group_and_page_priorities",
            "sample_remaining_miss_and_extra_pages",
            "start_same_budget_ablation_A0_A6",
            "defer_64x64_branch_until_localization_evidence_requires_it",
        ],
    }
    bundle = {
        "schema_version": 1,
        "status": "ok",
        "comparison_root": str(comparison_root),
        "analysis_dir": str(analysis_dir),
        "input_summaries": {
            "comparison_summary": str(comparison_root / "summary.json"),
            "error_analysis": str(analysis_dir / "error_analysis_summary.json"),
            "threshold_sweep": str(
                analysis_dir / "threshold_sweep" / "threshold_sweep_summary.json"
            ),
            "slot_alignment": str(
                analysis_dir / "slot_alignment" / "slot_alignment_summary.json"
            ),
        },
        "headline": headline,
        "residual_diagnosis": residual_diagnosis,
        "group_priorities": top_groups(groups, top_k=args.top_k, min_pages=args.min_group_pages),
        "page_priorities": page_priorities,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "analysis_bundle_summary.json", bundle)
    write_markdown(output_dir / "analysis_bundle.md", bundle, args.top_k)
    return {
        "event": "layout_analysis_bundle_completed",
        "pages": headline["pages"],
        "output_dir": str(output_dir),
        "summary": str(output_dir / "analysis_bundle_summary.json"),
        "markdown": str(output_dir / "analysis_bundle.md"),
        "headline": headline,
        "residual_diagnosis": residual_diagnosis,
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        print(compact_json(summarize(parse_args(argv))), flush=True)
    except Exception as exc:
        print(
            compact_json(
                {
                    "event": "layout_analysis_bundle_failed",
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
