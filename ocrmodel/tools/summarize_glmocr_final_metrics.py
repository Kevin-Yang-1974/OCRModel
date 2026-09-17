#!/usr/bin/env python3
"""Aggregate raw and whitespace-stripped OCR metrics for GLM-OCR test rows."""

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
                current.append(
                    min(
                        current[-1] + 1,
                        previous[column] + 1,
                        previous[column - 1] + (reference_char != prediction_char),
                    )
                )
            previous = current
        return previous[-1]


def remove_whitespace(value: str) -> str:
    return "".join(character for character in value if not character.isspace())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(manifest_path: Path, predictions_path: Path, expected_pages: int) -> dict[str, Any]:
    manifest = read_jsonl(manifest_path)
    predictions = read_jsonl(predictions_path)
    if len(manifest) != expected_pages or len(predictions) != expected_pages:
        raise ValueError(
            f"incomplete test rows: manifest={len(manifest)} "
            f"predictions={len(predictions)} expected={expected_pages}"
        )
    by_page = {str(row["page_id"]): row for row in predictions}
    if len(by_page) != len(predictions):
        raise ValueError("duplicate prediction page_id")
    raw_edits = 0
    raw_reference = 0
    compact_edits = 0
    compact_reference = 0
    page_cers: list[float] = []
    neds: list[float] = []
    exact_matches = 0
    compact_exact_matches = 0
    by_density: dict[str, dict[str, Any]] = {}
    for truth in manifest:
        page_id = str(truth["page_id"])
        if page_id not in by_page:
            raise ValueError(f"missing prediction for {page_id}")
        row = by_page[page_id]
        reference = str(truth.get("page_text", ""))
        prediction = str(row.get("prediction", ""))
        edits = edit_distance(reference, prediction)
        compact_reference_text = remove_whitespace(reference)
        compact_prediction_text = remove_whitespace(prediction)
        compact_edits_for_page = edit_distance(
            compact_reference_text, compact_prediction_text
        )
        raw_edits += edits
        raw_reference += len(reference)
        compact_edits += compact_edits_for_page
        compact_reference += len(compact_reference_text)
        page_cers.append(
            edits / len(reference) if reference else float(bool(prediction))
        )
        neds.append(edits / max(len(reference), len(prediction), 1))
        exact_matches += int(reference == prediction)
        compact_exact_matches += int(
            compact_reference_text == compact_prediction_text
        )
        density = str(row.get("density_bucket", "unknown"))
        bucket = by_density.setdefault(
            density,
            {"pages": 0, "edits": 0, "reference_characters": 0},
        )
        bucket["pages"] += 1
        bucket["edits"] += edits
        bucket["reference_characters"] += len(reference)
    pages = len(manifest)
    result = {
        "schema_version": 1,
        "pages": pages,
        "raw_reference_characters": raw_reference,
        "raw_edit_distance": raw_edits,
        "whitespace_stripped_reference_characters": compact_reference,
        "whitespace_stripped_edit_distance": compact_edits,
        "micro_cer": raw_edits / raw_reference if raw_reference else None,
        "whitespace_stripped_cer": (
            compact_edits / compact_reference if compact_reference else None
        ),
        "macro_cer": statistics.fmean(page_cers) if page_cers else None,
        "mean_ned": statistics.fmean(neds) if neds else None,
        "mean_edit_distance": raw_edits / pages if pages else None,
        "average_edit_distance": raw_edits / pages if pages else None,
        "exact_matches": exact_matches,
        "exact_match_rate": exact_matches / pages if pages else None,
        "whitespace_stripped_exact_matches": compact_exact_matches,
        "whitespace_stripped_exact_match_rate": (
            compact_exact_matches / pages if pages else None
        ),
        "by_density": {
            key: {
                **value,
                "micro_cer": (
                    value["edits"] / value["reference_characters"]
                    if value["reference_characters"]
                    else None
                ),
            }
            for key, value in sorted(by_density.items())
        },
        "definitions": {
            "micro_cer": "raw total Levenshtein edits / raw total reference characters",
            "whitespace_stripped_cer": "remove all Unicode whitespace from both strings before micro CER",
            "macro_cer": "mean of per-page raw CER",
            "mean_ned": "mean of per-page edits / max(raw reference length, prediction length, 1)",
            "exact_match": "raw reference string equals raw prediction string",
            "whitespace_stripped_exact_match": "exact match after removing all Unicode whitespace",
        },
        "test_used_for_selection": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pages", type=int, default=59)
    args = parser.parse_args()
    result = summarize(args.manifest, args.predictions, args.expected_pages)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

