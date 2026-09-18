#!/usr/bin/env python3
"""Paired significance testing and error taxonomy for OCR prediction dumps.

Two runs scored on the same pages produce a CER difference whose noise floor is
set by the page sample, not by the metric.  A point estimate cannot separate a
real 0.002 improvement from sampling noise on 59 pages, so this tool resamples
pages jointly (paired bootstrap) and reports a confidence interval for the
difference, plus an optional taxonomy of substitution pairs.

The taxonomy matters for classical-script corpora: ``aggregate_ocr_metrics``
strips whitespace only, so a prediction that renders a character in a different
orthographic form than the reference is charged a full substitution.  Printing
the most frequent pairs makes that label noise visible instead of silently
inflating CER.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def load_predictions(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            reference = "".join(record["reference"].split())
            prediction = "".join(record["prediction"].split())
            rows.append((record.get("page_id", ""), reference, prediction))
    if not rows:
        raise ValueError(f"no prediction rows in {path}")
    return rows


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, char_b in enumerate(b, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (char_a != char_b),
            )
        previous = current
    return previous[-1]


def substitution_pairs(a: str, b: str) -> Counter[tuple[str, str]]:
    """Substitution pairs recovered from an edit-distance alignment."""
    n, m = len(a), len(b)
    distance = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        distance[i][0] = i
    for j in range(m + 1):
        distance[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )
    pairs: Counter[tuple[str, str]] = Counter()
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and distance[i][j] == distance[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] != b[j - 1]:
                pairs[(a[i - 1], b[j - 1])] += 1
            i -= 1
            j -= 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return pairs


def paired_bootstrap(
    rows_a: list[tuple[str, str, str]],
    rows_b: list[tuple[str, str, str]],
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    """CI for CER(a) - CER(b), resampling pages jointly.

    Pages are resampled as units because both runs saw the same page; treating
    characters as independent would understate the interval.
    """
    if len(rows_a) != len(rows_b):
        raise ValueError(
            f"paired comparison needs equal page counts: {len(rows_a)} vs {len(rows_b)}"
        )
    ids_a = [row[0] for row in rows_a]
    ids_b = [row[0] for row in rows_b]
    if ids_a != ids_b:
        raise ValueError("paired comparison needs the same pages in the same order")

    stats_a = [(levenshtein(r, p), len(r)) for _, r, p in rows_a]
    stats_b = [(levenshtein(r, p), len(r)) for _, r, p in rows_b]
    rng = random.Random(seed)
    page_count = len(rows_a)
    differences: list[float] = []
    for _ in range(iterations):
        index = [rng.randrange(page_count) for _ in range(page_count)]
        errors_a = sum(stats_a[i][0] for i in index)
        chars_a = sum(stats_a[i][1] for i in index)
        errors_b = sum(stats_b[i][0] for i in index)
        chars_b = sum(stats_b[i][1] for i in index)
        differences.append(errors_a / chars_a - errors_b / chars_b)
    differences.sort()
    low = differences[int(0.025 * iterations)]
    high = differences[int(0.975 * iterations)]
    share_at_or_below_zero = sum(1 for value in differences if value <= 0) / iterations
    return low, high, share_at_or_below_zero


def cer(rows: list[tuple[str, str, str]]) -> float:
    errors = sum(levenshtein(reference, prediction) for _, reference, prediction in rows)
    characters = sum(len(reference) for _, reference, _ in rows)
    return errors / max(1, characters)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-a", type=Path, required=True)
    parser.add_argument("--prediction-b", type=Path, required=True)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--substitutions",
        action="store_true",
        help="also report the most frequent substitution pairs of run A",
    )
    parser.add_argument("--top-pairs", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_a = load_predictions(args.prediction_a)
    rows_b = load_predictions(args.prediction_b)
    low, high, share = paired_bootstrap(rows_a, rows_b, args.iterations, args.seed)

    print(f"pages: {len(rows_a)}")
    print(f"{args.label_a}: CER {cer(rows_a):.6f}")
    print(f"{args.label_b}: CER {cer(rows_b):.6f}")
    print(
        f"difference ({args.label_a} - {args.label_b}): "
        f"95% CI [{low:+.6f}, {high:+.6f}], P(<=0) = {share:.3f}"
    )
    verdict = "significant" if low > 0 or high < 0 else "not significant"
    print(f"verdict: {verdict}")

    if args.substitutions:
        total: Counter[tuple[str, str]] = Counter()
        reference_chars: Counter[str] = Counter()
        for _, reference, prediction in rows_a:
            total.update(substitution_pairs(reference, prediction))
            reference_chars.update(reference)
        edits = sum(total.values())
        print(f"\nsubstitutions in {args.label_a}: {edits} across {len(total)} distinct pairs")
        for rank, ((ref_char, pred_char), count) in enumerate(total.most_common(args.top_pairs), 1):
            print(
                f"  {rank:3d}. {ref_char} -> {pred_char}  {count:5d}"
                f"   ref({ref_char})={reference_chars[ref_char]:5d}"
                f"   ref({pred_char})={reference_chars[pred_char]:5d}"
            )
        cumulative = 0
        for rank, (_, count) in enumerate(total.most_common(), 1):
            cumulative += count
            if rank in (10, 50, 100, 200, 500, 1000):
                print(f"  top {rank:4d} pairs = {cumulative / edits:.3f} of substitutions")


if __name__ == "__main__":
    main()
