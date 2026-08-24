#!/usr/bin/env python3
"""Compute unified MTHv2 OCR metrics from a completed SOTA prediction JSONL."""

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


def summarize(manifest_path: Path, predictions_path: Path, expected_pages: int | None = None) -> dict[str, Any]:
    manifest = read_jsonl(manifest_path)
    predictions = read_jsonl(predictions_path)
    if expected_pages is not None and (len(manifest) != expected_pages or len(predictions) != expected_pages):
        raise ValueError(f"incomplete result: manifest={len(manifest)} predictions={len(predictions)} expected={expected_pages}")
    by_page = {str(record["page_id"]): record for record in predictions}
    if len(by_page) != len(predictions):
        raise ValueError("duplicate page_id in predictions")

    total_edits = total_reference = compact_edits = compact_reference = 0
    page_cers: list[float] = []
    normalized_distances: list[float] = []
    latencies: list[float] = []
    peak_memory: list[float] = []
    exact = failures = 0
    for truth in manifest:
        page_id = str(truth["page_id"])
        if page_id not in by_page:
            raise ValueError(f"missing prediction for {page_id}")
        prediction = by_page[page_id]
        reference_text = str(truth.get("page_text", ""))
        predicted_text = str(prediction.get("normalized_text", "")) if prediction.get("status") == "ok" else ""
        failures += int(prediction.get("status") != "ok")
        edits = edit_distance(reference_text, predicted_text)
        total_edits += edits
        total_reference += len(reference_text)
        page_cers.append(edits / len(reference_text) if reference_text else float(bool(predicted_text)))
        normalized_distances.append(edits / max(len(reference_text), len(predicted_text), 1))
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

    pages = len(manifest)
    total_latency = sum(latencies)
    return {
        "schema_version": 1,
        "pages": pages,
        "micro_page_cer": total_edits / total_reference if total_reference else None,
        "whitespace_stripped_micro_page_cer": compact_edits / compact_reference if compact_reference else None,
        "macro_page_cer": statistics.fmean(page_cers) if page_cers else None,
        "mean_normalized_edit_distance": statistics.fmean(normalized_distances) if normalized_distances else None,
        "mean_edit_distance": total_edits / pages if pages else None,
        "exact_matches": exact,
        "exact_match_rate": exact / pages if pages else None,
        "failed_pages": failures,
        "failure_rate": failures / pages if pages else None,
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "pages_per_second": len(latencies) / total_latency if total_latency else None,
        "peak_memory_mib": max(peak_memory) if peak_memory else None,
        "total_edit_distance": total_edits,
        "total_reference_characters": total_reference,
        "test_used_for_selection": False,
    }


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
