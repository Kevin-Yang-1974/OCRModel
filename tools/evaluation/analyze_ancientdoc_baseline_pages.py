#!/usr/bin/env python3
"""Offline paired OCR analysis for completed AncientDoc evaluations.

The script is CPU-only and does not load GOT2. It reads existing evaluation
outputs and compares two saved prediction JSONL files page by page, defaulting
to C4 versus C6. Both page-level and source-group-level paired bootstrap
intervals are reported; source groups are the primary sampling units.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


DEFAULT_LEFT = "c4"
DEFAULT_RIGHT = "c6"
DEFAULT_GROUPS = (
    "source_group_id",
    "ancientdoc_split",
    "category",
    "book",
    "text_length_bin",
    "reference_page_bin",
)


class AncientDocAnalysisFailure(RuntimeError):
    pass


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AncientDocAnalysisFailure(f"Expected JSON object: {path}")
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
                raise AncientDocAnalysisFailure(
                    f"Expected JSON object at {path}:{line_number}"
                )
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


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze page-level paired OCR differences from a completed "
            "AncientDoc validation suite."
        )
    )
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--left-label", default=DEFAULT_LEFT)
    parser.add_argument("--right-label", default=DEFAULT_RIGHT)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--top-k", type=positive_int, default=30)
    parser.add_argument("--bootstrap-samples", type=nonnegative_int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    parser.add_argument(
        "--group-by",
        action="append",
        choices=DEFAULT_GROUPS,
        help="Repeat to override default grouping dimensions.",
    )
    return parser.parse_args(argv)


def resolve_predictions_path(summary_path: Path, summary: dict[str, Any]) -> Path:
    value = summary.get("predictions")
    if isinstance(value, str) and value:
        path = Path(value).expanduser()
        if path.is_file():
            return path.resolve()
        sibling = summary_path.parent / path
        if sibling.is_file():
            return sibling.resolve()
    fallback = summary_path.parent / "layout_validation_predictions.jsonl"
    if fallback.is_file():
        return fallback.resolve()
    raise AncientDocAnalysisFailure(
        f"Cannot resolve predictions path from {summary_path}; tried {value!r} and {fallback}"
    )


def resolve_manifest(
    manifest_arg: Path | None,
    summaries: Iterable[dict[str, Any]],
) -> Path:
    if manifest_arg is not None:
        manifest = manifest_arg.expanduser().resolve()
        if not manifest.is_file():
            raise AncientDocAnalysisFailure(f"Manifest does not exist: {manifest}")
        return manifest
    for summary in summaries:
        value = summary.get("manifest")
        if isinstance(value, str) and value:
            manifest = Path(value).expanduser()
            if manifest.is_file():
                return manifest.resolve()
    raise AncientDocAnalysisFailure("Manifest path is unavailable; pass --manifest explicitly.")


def index_by_page_id(records: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        page_id = record.get("page_id")
        if not isinstance(page_id, str) or not page_id:
            raise AncientDocAnalysisFailure(f"{label} contains a record without page_id.")
        if page_id in indexed:
            raise AncientDocAnalysisFailure(f"{label} contains duplicate page_id: {page_id}")
        indexed[page_id] = record
    return indexed


def safe_metric(record: dict[str, Any], key: str) -> Any:
    metrics = record.get("metrics")
    if not isinstance(metrics, dict):
        return None
    ocr = metrics.get("ocr")
    if not isinstance(ocr, dict):
        return None
    return ocr.get(key)


def as_int(value: Any, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return default


def as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def remove_whitespace(value: str) -> str:
    return "".join(character for character in value if not character.isspace())


def bin_count(value: int, boundaries: Sequence[int]) -> str:
    previous = 0
    for boundary in boundaries:
        if value <= boundary:
            return f"{previous + 1}-{boundary}"
        previous = boundary
    return f">{boundaries[-1]}"


def text_length_bin(length: int) -> str:
    return bin_count(length, (50, 100, 200, 400, 800, 1200))


def reference_page_bin(page_number: int | None) -> str:
    if page_number is None:
        return "unknown"
    return bin_count(page_number, (10, 25, 50, 100, 200))


def parse_original_image(record: dict[str, Any], page_id: str) -> dict[str, Any]:
    raw = record.get("original_image") or record.get("image") or ""
    image = str(raw)
    parts = PurePosixPath(image).parts if image else ()
    category = str(record.get("category") or "unknown")
    book = str(record.get("book") or "unknown")
    if len(parts) >= 3:
        category = parts[-3]
        book = parts[-2]
    elif len(parts) >= 2:
        category = parts[-2]
        book = parts[-1].rsplit(".", 1)[0]

    page_number = None
    for candidate in reversed(parts or (page_id,)):
        match = re.search(r"(?:page|p|页)[_\-]?(\d+)", candidate, flags=re.IGNORECASE)
        if match:
            page_number = int(match.group(1))
            break
    split_match = re.search(r"ancientdoc_split(\d+)", page_id)
    split = f"split{split_match.group(1)}" if split_match else str(record.get("split", "unknown"))
    return {
        "original_image": image,
        "category": category or "unknown",
        "book": book or "unknown",
        "reference_page_number": page_number,
        "reference_page_bin": reference_page_bin(page_number),
        "ancientdoc_split": split,
    }


def model_error_categories(reference: str, prediction: str, edits: int) -> set[str]:
    categories: set[str] = set()
    ref_len = len(reference)
    pred_len = len(prediction)
    if pred_len == 0:
        categories.add("empty_prediction")
    if ref_len > 0:
        ratio = pred_len / ref_len
        if ratio < 0.50:
            categories.add("under_generation")
        if ratio > 1.50:
            categories.add("over_generation")
        if edits / ref_len >= 1.0:
            categories.add("cer_ge_1")
    if len(remove_whitespace(prediction)) == 0 and ref_len > 0:
        categories.add("blank_after_whitespace")
    repeated_runs = re.findall(r"(.)\1{9,}", prediction)
    if repeated_runs:
        categories.add("long_repeated_character_run")
    if pred_len >= 100 and len(set(prediction)) <= max(3, pred_len // 100):
        categories.add("low_character_diversity")
    if not categories:
        categories.add("ordinary_error")
    return categories


def paired_rows(
    *,
    left_label: str,
    right_label: str,
    left_predictions: list[dict[str, Any]],
    right_predictions: list[dict[str, Any]],
    manifest: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    left_by_id = index_by_page_id(left_predictions, f"{left_label} predictions")
    right_by_id = index_by_page_id(right_predictions, f"{right_label} predictions")
    manifest_by_id = index_by_page_id(manifest, "manifest")
    if set(left_by_id) != set(right_by_id):
        missing_right = sorted(set(left_by_id) - set(right_by_id))[:20]
        missing_left = sorted(set(right_by_id) - set(left_by_id))[:20]
        raise AncientDocAnalysisFailure(
            f"Prediction page_id sets differ: missing_{right_label}={missing_right}, "
            f"missing_{left_label}={missing_left}"
        )

    rows: list[dict[str, Any]] = []
    for page_id in sorted(left_by_id):
        left = left_by_id[page_id]
        right = right_by_id[page_id]
        manifest_record = manifest_by_id.get(page_id, {})
        reference = str(
            right.get("reference_text")
            or left.get("reference_text")
            or manifest_record.get("page_text")
            or ""
        )
        left_prediction = str(left.get("predicted_text") or "")
        right_prediction = str(right.get("predicted_text") or "")
        left_edits = as_int(safe_metric(left, "edit_distance"))
        right_edits = as_int(safe_metric(right, "edit_distance"))
        left_ws_edits = as_int(safe_metric(left, "whitespace_normalized_edit_distance"))
        right_ws_edits = as_int(safe_metric(right, "whitespace_normalized_edit_distance"))
        ref_chars = as_int(safe_metric(right, "reference_characters"), default=len(reference))
        ws_ref_chars = as_int(
            safe_metric(right, "whitespace_normalized_reference_characters"),
            default=len(remove_whitespace(reference)),
        )
        delta_edits = right_edits - left_edits
        delta_ws_edits = right_ws_edits - left_ws_edits
        metadata = parse_original_image(manifest_record, page_id)
        left_categories = model_error_categories(reference, left_prediction, left_edits)
        right_categories = model_error_categories(reference, right_prediction, right_edits)
        row = {
            "page_id": page_id,
            "source_group_id": str(manifest_record.get("source_group_id", "unknown")),
            "tier": str(manifest_record.get("tier", "unknown")),
            **metadata,
            "reference_characters": ref_chars,
            "whitespace_normalized_reference_characters": ws_ref_chars,
            "text_length_bin": text_length_bin(ref_chars),
            "left_label": left_label,
            "right_label": right_label,
            "left_edit_distance": left_edits,
            "right_edit_distance": right_edits,
            "delta_edit_distance_right_minus_left": delta_edits,
            "left_cer": as_float(safe_metric(left, "cer")),
            "right_cer": as_float(safe_metric(right, "cer")),
            "delta_cer_right_minus_left": (
                delta_edits / ref_chars if ref_chars > 0 else None
            ),
            "left_whitespace_edit_distance": left_ws_edits,
            "right_whitespace_edit_distance": right_ws_edits,
            "delta_whitespace_edit_distance_right_minus_left": delta_ws_edits,
            "left_whitespace_cer": as_float(safe_metric(left, "whitespace_normalized_cer")),
            "right_whitespace_cer": as_float(safe_metric(right, "whitespace_normalized_cer")),
            "delta_whitespace_cer_right_minus_left": (
                delta_ws_edits / ws_ref_chars if ws_ref_chars > 0 else None
            ),
            "left_exact_match": bool(safe_metric(left, "exact_match")),
            "right_exact_match": bool(safe_metric(right, "exact_match")),
            "left_whitespace_exact_match": bool(
                safe_metric(left, "whitespace_normalized_exact_match")
            ),
            "right_whitespace_exact_match": bool(
                safe_metric(right, "whitespace_normalized_exact_match")
            ),
            "left_prediction_characters": len(left_prediction),
            "right_prediction_characters": len(right_prediction),
            "left_prediction_to_reference_ratio": (
                len(left_prediction) / ref_chars if ref_chars > 0 else None
            ),
            "right_prediction_to_reference_ratio": (
                len(right_prediction) / ref_chars if ref_chars > 0 else None
            ),
            "paired_outcome": (
                f"{right_label}_better"
                if delta_edits < 0
                else f"{right_label}_worse"
                if delta_edits > 0
                else "same"
            ),
            "left_error_categories": ";".join(sorted(left_categories)),
            "right_error_categories": ";".join(sorted(right_categories)),
            "new_error_categories": ";".join(sorted(right_categories - left_categories)),
            "resolved_error_categories": ";".join(sorted(left_categories - right_categories)),
        }
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, Any]], left_label: str, right_label: str) -> dict[str, Any]:
    pages = len(rows)
    ref_chars = sum(int(row["reference_characters"]) for row in rows)
    ws_ref_chars = sum(int(row["whitespace_normalized_reference_characters"]) for row in rows)
    left_edits = sum(int(row["left_edit_distance"]) for row in rows)
    right_edits = sum(int(row["right_edit_distance"]) for row in rows)
    left_ws_edits = sum(int(row["left_whitespace_edit_distance"]) for row in rows)
    right_ws_edits = sum(int(row["right_whitespace_edit_distance"]) for row in rows)
    return {
        "pages": pages,
        "reference_characters": ref_chars,
        "left_label": left_label,
        "right_label": right_label,
        "left_character_edits": left_edits,
        "right_character_edits": right_edits,
        "delta_character_edits_right_minus_left": right_edits - left_edits,
        "left_page_cer_from_predictions": left_edits / ref_chars if ref_chars else None,
        "right_page_cer_from_predictions": right_edits / ref_chars if ref_chars else None,
        "delta_page_cer_right_minus_left": (
            (right_edits - left_edits) / ref_chars if ref_chars else None
        ),
        "relative_page_cer_change_right_vs_left": (
            ((right_edits / ref_chars) - (left_edits / ref_chars)) / (left_edits / ref_chars)
            if ref_chars and left_edits
            else None
        ),
        "left_whitespace_edits": left_ws_edits,
        "right_whitespace_edits": right_ws_edits,
        "delta_whitespace_edits_right_minus_left": right_ws_edits - left_ws_edits,
        "left_whitespace_page_cer_from_predictions": (
            left_ws_edits / ws_ref_chars if ws_ref_chars else None
        ),
        "right_whitespace_page_cer_from_predictions": (
            right_ws_edits / ws_ref_chars if ws_ref_chars else None
        ),
        "delta_whitespace_page_cer_right_minus_left": (
            (right_ws_edits - left_ws_edits) / ws_ref_chars if ws_ref_chars else None
        ),
        "left_exact_matches": sum(bool(row["left_exact_match"]) for row in rows),
        "right_exact_matches": sum(bool(row["right_exact_match"]) for row in rows),
        "exact_match_delta_right_minus_left": (
            sum(bool(row["right_exact_match"]) for row in rows)
            - sum(bool(row["left_exact_match"]) for row in rows)
        ),
        "paired_outcomes": dict(Counter(str(row["paired_outcome"]) for row in rows)),
    }


def summarize_error_categories(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {
        "left_error_categories": Counter(),
        "right_error_categories": Counter(),
        "new_error_categories": Counter(),
        "resolved_error_categories": Counter(),
    }
    for row in rows:
        for field, counter in counters.items():
            values = [item for item in str(row.get(field, "")).split(";") if item]
            counter.update(values)
    return {field: dict(counter) for field, counter in counters.items()}


def mean_optional(values: Iterable[Any]) -> float | None:
    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    return sum(finite) / len(finite) if finite else None


def aggregate_group(rows: list[dict[str, Any]], key: str, left_label: str, right_label: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, "unknown"))].append(row)
    aggregates: list[dict[str, Any]] = []
    for value, group_rows in sorted(grouped.items()):
        summary = summarize_rows(group_rows, left_label, right_label)
        aggregates.append(
            {
                "group_by": key,
                "group": value,
                **summary,
                "mean_left_prediction_to_reference_ratio": mean_optional(
                    row.get("left_prediction_to_reference_ratio") for row in group_rows
                ),
                "mean_right_prediction_to_reference_ratio": mean_optional(
                    row.get("right_prediction_to_reference_ratio") for row in group_rows
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


def bootstrap_delta_cer(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any] | None:
    if samples <= 0 or not rows:
        return None
    rng = random.Random(seed)
    deltas: list[float] = []
    n = len(rows)
    for _ in range(samples):
        delta_edits = 0
        ref_chars = 0
        for _ in range(n):
            row = rows[rng.randrange(n)]
            delta_edits += int(row["delta_edit_distance_right_minus_left"])
            ref_chars += int(row["reference_characters"])
        if ref_chars:
            deltas.append(delta_edits / ref_chars)
    if not deltas:
        return None
    deltas.sort()

    def quantile(probability: float) -> float:
        index = min(len(deltas) - 1, max(0, round(probability * (len(deltas) - 1))))
        return deltas[index]

    return {
        "method": "page_paired_nonparametric_bootstrap",
        "samples": samples,
        "seed": seed,
        "delta_page_cer_right_minus_left_mean": sum(deltas) / len(deltas),
        "delta_page_cer_right_minus_left_ci95": [
            quantile(0.025),
            quantile(0.975),
        ],
        "share_delta_below_zero": sum(delta < 0 for delta in deltas) / len(deltas),
        "note": (
            "Page-level bootstrap treats pages as independent and is diagnostic only; "
            "use the source-group cluster bootstrap for cross-source uncertainty."
        ),
    }


def cluster_bootstrap_delta_cer(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
    group_key: str = "source_group_id",
) -> dict[str, Any] | None:
    if samples <= 0 or not rows:
        return None
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(group_key, "unknown"))].append(row)
    group_names = sorted(grouped)
    if not group_names:
        return None

    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(samples):
        delta_edits = 0
        ref_chars = 0
        for _ in range(len(group_names)):
            sampled_rows = grouped[group_names[rng.randrange(len(group_names))]]
            delta_edits += sum(
                int(row["delta_edit_distance_right_minus_left"])
                for row in sampled_rows
            )
            ref_chars += sum(int(row["reference_characters"]) for row in sampled_rows)
        if ref_chars:
            deltas.append(delta_edits / ref_chars)
    if not deltas:
        return None
    deltas.sort()

    def quantile(probability: float) -> float:
        index = min(len(deltas) - 1, max(0, round(probability * (len(deltas) - 1))))
        return deltas[index]

    return {
        "method": "source_group_id_paired_cluster_bootstrap",
        "sampling_unit": group_key,
        "groups": len(group_names),
        "pages": len(rows),
        "samples": samples,
        "seed": seed,
        "delta_page_cer_right_minus_left_mean": sum(deltas) / len(deltas),
        "delta_page_cer_right_minus_left_ci95": [
            quantile(0.025),
            quantile(0.975),
        ],
        "share_delta_below_zero": sum(delta < 0 for delta in deltas) / len(deltas),
        "note": (
            "Paired cluster bootstrap resamples source groups with replacement and "
            "keeps all pages from each sampled group together."
        ),
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


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(rows: list[dict[str, Any]], fields: Sequence[str], limit: int) -> str:
    selected = rows[:limit]
    if not selected:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(format_value(row.get(field)) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def top_pages(rows: list[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    fields = (
        "page_id",
        "category",
        "book",
        "reference_page_number",
        "reference_characters",
        "left_edit_distance",
        "right_edit_distance",
        "delta_edit_distance_right_minus_left",
        "left_prediction_characters",
        "right_prediction_characters",
        "paired_outcome",
        "new_error_categories",
        "resolved_error_categories",
        "original_image",
    )

    def project(row: dict[str, Any]) -> dict[str, Any]:
        return {field: row.get(field) for field in fields}

    return {
        "largest_right_improvements": [
            project(row)
            for row in sorted(rows, key=lambda item: int(item["delta_edit_distance_right_minus_left"]))[:top_k]
            if int(row["delta_edit_distance_right_minus_left"]) < 0
        ],
        "largest_right_regressions": [
            project(row)
            for row in sorted(rows, key=lambda item: -int(item["delta_edit_distance_right_minus_left"]))[:top_k]
            if int(row["delta_edit_distance_right_minus_left"]) > 0
        ],
        "worst_right_pages": [
            project(row)
            for row in sorted(rows, key=lambda item: int(item["right_edit_distance"]), reverse=True)[:top_k]
        ],
        "worst_left_pages": [
            project(row)
            for row in sorted(rows, key=lambda item: int(item["left_edit_distance"]), reverse=True)[:top_k]
        ],
    }


def write_worst_pages(path: Path, report: dict[str, Any], top_k: int) -> None:
    page_fields = (
        "page_id",
        "category",
        "book",
        "reference_page_number",
        "reference_characters",
        "left_edit_distance",
        "right_edit_distance",
        "delta_edit_distance_right_minus_left",
        "paired_outcome",
        "new_error_categories",
        "resolved_error_categories",
    )
    lines = ["# AncientDoc paired worst pages", ""]
    for title, rows in report["top_pages"].items():
        lines.extend([f"## {title}", "", markdown_table(rows, page_fields, top_k), ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_analysis_summary(path: Path, report: dict[str, Any]) -> None:
    overview = report["overview"]
    page_bootstrap = report.get("bootstrap_delta_cer")
    cluster_bootstrap = report.get("cluster_bootstrap_delta_cer")
    group_fields = (
        "group_by",
        "group",
        "pages",
        "delta_page_cer_right_minus_left",
        "relative_page_cer_change_right_vs_left",
        "right_exact_matches",
        "exact_match_delta_right_minus_left",
    )
    lines = [
        "# AncientDoc C4/C6 paired OCR analysis",
        "",
        "## Overview",
        "",
        "```json",
        compact_json(overview),
        "```",
        "",
    ]
    if cluster_bootstrap is not None:
        lines.extend(
            [
                "## Source-group cluster bootstrap",
                "",
                "```json",
                compact_json(cluster_bootstrap),
                "```",
                "",
            ]
        )
    if page_bootstrap is not None:
        lines.extend(
            [
                "## Page bootstrap (diagnostic)",
                "",
                "```json",
                compact_json(page_bootstrap),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Error categories",
            "",
            "```json",
            compact_json(report["error_categories"]),
            "```",
            "",
            "## Group comparison",
            "",
            markdown_table(report["groups"], group_fields, limit=80),
            "",
            "## Notes",
            "",
            "- This is an offline analysis of existing predictions; no model inference or training is run.",
            "- The source_group_id cluster bootstrap is the primary uncertainty interval because pages from the same source group are correlated.",
            "- The page bootstrap is retained only as a within-split diagnostic and must not replace the cluster interval.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    suite_root = args.suite_root.expanduser().resolve()
    if not suite_root.is_dir():
        raise AncientDocAnalysisFailure(f"Suite root does not exist: {suite_root}")

    left_summary_path = suite_root / args.left_label / "layout_validation_metrics.json"
    right_summary_path = suite_root / args.right_label / "layout_validation_metrics.json"
    if not left_summary_path.is_file():
        raise AncientDocAnalysisFailure(f"Missing metrics: {left_summary_path}")
    if not right_summary_path.is_file():
        raise AncientDocAnalysisFailure(f"Missing metrics: {right_summary_path}")

    left_summary = read_json(left_summary_path)
    right_summary = read_json(right_summary_path)
    left_predictions_path = resolve_predictions_path(left_summary_path, left_summary)
    right_predictions_path = resolve_predictions_path(right_summary_path, right_summary)
    manifest_path = resolve_manifest(args.manifest, (left_summary, right_summary))
    left_predictions = read_jsonl(left_predictions_path)
    right_predictions = read_jsonl(right_predictions_path)
    manifest = read_jsonl(manifest_path)

    rows = paired_rows(
        left_label=args.left_label,
        right_label=args.right_label,
        left_predictions=left_predictions,
        right_predictions=right_predictions,
        manifest=manifest,
    )
    group_keys = tuple(args.group_by) if args.group_by else DEFAULT_GROUPS
    groups: list[dict[str, Any]] = []
    for key in group_keys:
        groups.extend(aggregate_group(rows, key, args.left_label, args.right_label))

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else suite_root / "analysis" / f"{args.left_label}_vs_{args.right_label}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "schema_version": 2,
        "status": "ok",
        "suite_root": str(suite_root),
        "left_label": args.left_label,
        "right_label": args.right_label,
        "left_metrics": str(left_summary_path),
        "right_metrics": str(right_summary_path),
        "left_predictions": str(left_predictions_path),
        "right_predictions": str(right_predictions_path),
        "manifest": str(manifest_path),
        "overview": summarize_rows(rows, args.left_label, args.right_label),
        "bootstrap_delta_cer": bootstrap_delta_cer(
            rows,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ),
        "cluster_bootstrap_delta_cer": cluster_bootstrap_delta_cer(
            rows,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ),
        "error_categories": summarize_error_categories(rows),
        "groups": groups,
        "top_pages": top_pages(rows, args.top_k),
    }

    write_csv(output_dir / "page_comparison.csv", rows)
    write_csv(output_dir / "group_comparison.csv", groups)
    write_json(output_dir / "error_categories.json", report["error_categories"])
    write_json(output_dir / "summary.json", report)
    write_worst_pages(output_dir / "worst_pages.md", report, args.top_k)
    write_analysis_summary(output_dir / "analysis_summary.md", report)

    return {
        "event": "ancientdoc_paired_analysis_completed",
        "suite_root": str(suite_root),
        "left_label": args.left_label,
        "right_label": args.right_label,
        "pages": report["overview"]["pages"],
        "delta_page_cer_right_minus_left": report["overview"][
            "delta_page_cer_right_minus_left"
        ],
        "paired_outcomes": report["overview"]["paired_outcomes"],
        "output_dir": str(output_dir),
        "summary": str(output_dir / "summary.json"),
        "analysis_summary": str(output_dir / "analysis_summary.md"),
        "page_csv": str(output_dir / "page_comparison.csv"),
        "group_csv": str(output_dir / "group_comparison.csv"),
        "worst_pages": str(output_dir / "worst_pages.md"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = analyze(parse_args(argv))
    except AncientDocAnalysisFailure as exc:
        print(compact_json({"event": "ancientdoc_paired_analysis_failed", "error": str(exc)}))
        return 1
    print(compact_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
