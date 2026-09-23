#!/usr/bin/env python3
"""Offline frequency/error diagnosis for line-mask validation predictions.

Re-aggregates existing predictions with the real train-manifest character
counts, so the ``low_frequency_k*``/``r2_k*`` fields stop being the null
placeholders produced when a caller passes an empty Counter. Also splits the
substitution errors that dominate the residual CER. Reads no test data and
selects nothing; callers must pass an already-locked prediction set.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def load_records(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train_character_counts(records: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record["page_text"])
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                        help="prediction directory holding predictions-rank*.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.code_root.resolve() / "src"))
    from layout_ocr.metrics import (
        aggregate_ocr_metrics,
        levenshtein_alignment,
        levenshtein_error_counts,
    )

    train_counts = train_character_counts(load_records(args.train_manifest))
    report: dict[str, object] = {
        "train_manifest": str(args.train_manifest),
        "train_characters": sum(train_counts.values()),
        "train_unique_characters": len(train_counts),
        "test_manifest_read": False,
        "test_used_for_selection": False,
        "arms": {},
    }
    for spec in args.arm:
        name, _, directory = spec.partition("=")
        rows: list[dict] = []
        for shard in sorted(Path(directory).glob("predictions-rank*.jsonl")):
            rows.extend(load_records(shard))
        pairs = [(r["reference"], r["prediction"]) for r in rows]
        metrics = aggregate_ocr_metrics(pairs, train_counts)
        metrics["generation_limit_hits"] = sum(bool(r["limit_hit"]) for r in rows)
        metrics["eos_pages"] = len(rows) - metrics["generation_limit_hits"]
        metrics["loop_pages"] = sum(
            bool(r["repetition"].get("repeated_cycle_detected")) for r in rows
        )

        # Split substitutions by train frequency of the reference character, so
        # a frequency-driven failure mode is visible instead of averaged away.
        buckets = {"k1": (0, 1), "k3": (0, 3), "k5": (0, 5), "frequent": (5, 10**9), "unseen": (0, 0)}
        totals: Counter[str] = Counter()
        counts: Counter[str] = Counter()
        wrong_prone = Counter()
        for reference, prediction in pairs:
            reference = "".join(reference.split())
            prediction = "".join(prediction.split())
            errors = levenshtein_error_counts(reference, prediction)
            totals["edits"] += sum(errors.values())
            counts["chars"] += len(reference)
            for kind in ("insertions", "deletions", "substitutions"):
                totals[kind] += errors[kind]
            _, matches = levenshtein_alignment(reference, prediction)
            for char, hits in matches.items():
                counts[f"match_freq_{min(train_counts.get(char, 0), 10**9)}"] += hits
            for char in Counter(reference):
                bucket = "unseen" if train_counts.get(char, 0) == 0 else "frequent"
                for tag, (low, high) in buckets.items():
                    if low < train_counts.get(char, 0) <= high or (tag == "unseen" and train_counts.get(char, 0) == 0):
                        counts[f"{tag}_ref_chars"] += Counter(reference)[char]
                        counts[f"{tag}_matches"] += matches[char]
        subst_by_frequency = {
            tag: {
                "reference_characters": counts[f"{tag}_ref_chars"],
                "matched_characters": counts[f"{tag}_matches"],
                "recall": (counts[f"{tag}_matches"] / counts[f"{tag}_ref_chars"])
                if counts[f"{tag}_ref_chars"] else None,
            }
            for tag in ("k1", "k3", "k5", "frequent", "unseen")
        }

        # Which reference characters absorb the most edits, and what replaces them.
        for reference, prediction in pairs:
            reference_norm = "".join(reference.split())
            prediction_norm = "".join(prediction.split())
            rows_costs = _substitution_map(reference_norm, prediction_norm)
            for ref_char, pred_char in rows_costs:
                wrong_prone[f"{ref_char}->{pred_char}"] += 1

        report["arms"][name] = {
            "directory": directory,
            "pages": len(rows),
            "metrics": metrics,
            "edit_totals": dict(totals),
            "by_frequency": subst_by_frequency,
            "top_confusions": wrong_prone.most_common(40),
            "worst_pages": sorted(
                (
                    {
                        "page_id": r["page_id"],
                        "edits": sum(
                            levenshtein_error_counts(
                                "".join(r["reference"].split()),
                                "".join(r["prediction"].split()),
                            ).values()
                        ),
                        "reference_characters": len("".join(r["reference"].split())),
                        "generation_tokens": r["generation_tokens"],
                    }
                    for r in rows
                ),
                key=lambda entry: entry["edits"],
                reverse=True,
            )[:10],
        }
        print(json.dumps({"arm": name, "cer": metrics["cer"]}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(f"wrote {args.output}")


def _substitution_map(reference: str, prediction: str) -> list[tuple[str, str]]:
    """Levenshtein alignment restricted to substitution moves."""
    rows = len(reference) + 1
    cols = len(prediction) + 1
    costs = [[0] * cols for _ in range(rows)]
    moves = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], moves[i][0] = i, "D"
    for j in range(1, cols):
        costs[0][j], moves[0][j] = j, "I"
    for i in range(1, rows):
        for j in range(1, cols):
            substitution = costs[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1])
            deletion = costs[i - 1][j] + 1
            insertion = costs[i][j - 1] + 1
            costs[i][j], moves[i][j] = min(
                (substitution, "M"), (deletion, "D"), (insertion, "I")
            )
    pairs: list[tuple[str, str]] = []
    i, j = len(reference), len(prediction)
    while i or j:
        move = moves[i][j]
        if move == "M":
            if reference[i - 1] != prediction[j - 1]:
                pairs.append((reference[i - 1], prediction[j - 1]))
            i -= 1
            j -= 1
        elif move == "D":
            i -= 1
        else:
            j -= 1
    return pairs


if __name__ == "__main__":
    main()
