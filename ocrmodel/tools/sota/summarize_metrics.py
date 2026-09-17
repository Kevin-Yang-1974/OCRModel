#!/usr/bin/env python3
"""Compute detailed OCR metrics from a completed SOTA prediction JSONL."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from rapidfuzz.distance import Levenshtein

    def edit_distance(reference: str, prediction: str) -> int:
        return int(Levenshtein.distance(reference, prediction))
except ImportError:
    def edit_distance(reference: str, prediction: str) -> int:
        if len(reference) < len(prediction):
            reference, prediction = prediction, reference
        previous = list(range(len(prediction) + 1))
        for row, reference_char in enumerate(reference, 1):
            current = [row]
            for column, prediction_char in enumerate(prediction, 1):
                current.append(min(current[-1] + 1, previous[column] + 1, previous[column - 1] + (reference_char != prediction_char)))
            previous = current
        return previous[-1]


def remove_whitespace(value: str) -> str:
    return "".join(character for character in value if not character.isspace())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _reference_length_bucket(length: int) -> str:
    if length < 512:
        return "<512"
    if length < 1024:
        return "512-1023"
    if length < 2048:
        return "1024-2047"
    return ">=2048"


def _metric_summary(rows: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    total_edits = total_reference = compact_edits = compact_reference = 0
    total_prediction = 0
    page_cers: list[float] = []
    normalized_distances: list[float] = []
    latencies: list[float] = []
    peak_memory: list[float] = []
    length_ratios: list[float] = []
    status_counts: dict[str, int] = {}
    exact = 0
    for truth, prediction in rows:
        reference_text = str(truth.get("page_text", ""))
        status = str(prediction.get("status"))
        predicted_text = str(prediction.get("normalized_text", "")) if status == "ok" else ""
        status_counts[status] = status_counts.get(status, 0) + 1
        edits = edit_distance(reference_text, predicted_text)
        total_edits += edits
        total_reference += len(reference_text)
        total_prediction += len(predicted_text)
        page_cers.append(edits / len(reference_text) if reference_text else float(bool(predicted_text)))
        normalized_distances.append(edits / max(len(reference_text), len(predicted_text), 1))
        length_ratios.append(len(predicted_text) / max(len(reference_text), 1))
        exact += int(reference_text == predicted_text)

        compact_truth = remove_whitespace(reference_text)
        compact_prediction = remove_whitespace(predicted_text)
        compact_edits += edit_distance(compact_truth, compact_prediction)
        compact_reference += len(compact_truth)
        runtime = prediction.get("runtime") or {}
        if isinstance(runtime.get("latency_seconds"), (int, float)):
            latencies.append(float(runtime["latency_seconds"]))
        if isinstance(runtime.get("peak_memory_mib"), (int, float)):
            peak_memory.append(float(runtime["peak_memory_mib"]))

    pages = len(rows)
    total_latency = sum(latencies)
    return {
        "pages": pages,
        "micro_page_cer": total_edits / total_reference if total_reference else None,
        "whitespace_stripped_micro_page_cer": compact_edits / compact_reference if compact_reference else None,
        "macro_page_cer": statistics.fmean(page_cers) if page_cers else None,
        "mean_normalized_edit_distance": statistics.fmean(normalized_distances) if normalized_distances else None,
        "mean_edit_distance": total_edits / pages if pages else None,
        "exact_matches": exact,
        "exact_match_rate": exact / pages if pages else None,
        "failed_pages": pages - status_counts.get("ok", 0),
        "failure_rate": (pages - status_counts.get("ok", 0)) / pages if pages else None,
        "status_counts": status_counts,
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "median_latency_seconds": statistics.median(latencies) if latencies else None,
        "p95_latency_seconds": _percentile(latencies, 0.95),
        "pages_per_second": len(latencies) / total_latency if total_latency else None,
        "peak_memory_mib": max(peak_memory) if peak_memory else None,
        "total_edit_distance": total_edits,
        "total_reference_characters": total_reference,
        "total_prediction_characters": total_prediction,
        "mean_reference_characters": total_reference / pages if pages else None,
        "mean_prediction_characters": total_prediction / pages if pages else None,
        "mean_prediction_reference_length_ratio": statistics.fmean(length_ratios) if length_ratios else None,
    }


def summarize(manifest_path: Path, predictions_path: Path, expected_pages: int | None = None) -> dict[str, Any]:
    manifest = read_jsonl(manifest_path)
    predictions = read_jsonl(predictions_path)
    if expected_pages is not None and (len(manifest) != expected_pages or len(predictions) != expected_pages):
        raise ValueError(f"incomplete result: manifest={len(manifest)} predictions={len(predictions)} expected={expected_pages}")
    by_page = {str(record["page_id"]): record for record in predictions}
    if len(by_page) != len(predictions):
        raise ValueError("duplicate page_id in predictions")

    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    by_domain: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    by_length: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for truth in manifest:
        page_id = str(truth["page_id"])
        if page_id not in by_page:
            raise ValueError(f"missing prediction for {page_id}")
        prediction = by_page[page_id]
        row = (truth, prediction)
        rows.append(row)
        domain = str(truth.get("domain", "unknown"))
        by_domain.setdefault(domain, []).append(row)
        bucket = _reference_length_bucket(len(str(truth.get("page_text", ""))))
        by_length.setdefault(bucket, []).append(row)

    result = {"schema_version": 2, **_metric_summary(rows)}
    result["domain_metrics"] = {key: _metric_summary(value) for key, value in sorted(by_domain.items())}
    result["reference_length_metrics"] = {key: _metric_summary(value) for key, value in sorted(by_length.items())}
    result["test_used_for_selection"] = False
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pages", type=int)
    args = parser.parse_args()
    result = summarize(args.manifest, args.predictions, args.expected_pages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
