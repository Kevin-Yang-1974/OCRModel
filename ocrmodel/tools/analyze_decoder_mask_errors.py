#!/usr/bin/env python3
"""Where do the errors actually fall? Error decomposition beyond a single CER.

Two arms can reach the same CER with opposite failure modes: one hallucinates
extra glyphs, the other silently drops real ones.  The Gate C screen showed
exactly that pattern, so this tool decomposes each run's per-page predictions
into:

* insertion / deletion / substitution totals, normalised per 1000 reference
  characters so runs on the same manifest stay comparable;
* the *length ratio* -- predicted characters over reference characters -- which
  separates "wrote too little" from "wrote too much";
* a deletion position profile: the share of dropped characters that fall in the
  last 20% of the reference.  Tail-heavy deletions mean the page was cut short
  (an early stop or a generation limit); deletions spread through the body mean
  content was skipped line by line.  The two need different fixes;
* generation-limit hits, which mark pages that ran out of token budget.

Reads the ``predictions.jsonl`` written by ``evaluate_decoder_mask.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _normalize(text: str) -> str:
    return "".join(text.split())


def _align_positions(reference: str, prediction: str) -> dict[str, Any]:
    """Levenshtein with the same tie-break order as ``layout_ocr.metrics``.

    Returns the error counts plus the reference positions that were deleted and
    the prediction positions that were inserted.
    """

    rows, cols = len(reference) + 1, len(prediction) + 1
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
            costs[i][j], moves[i][j] = min((substitution, "M"), (deletion, "D"), (insertion, "I"))

    insertions = deletions = substitutions = 0
    deleted_positions: list[int] = []
    inserted_positions: list[int] = []
    i, j = len(reference), len(prediction)
    while i or j:
        move = moves[i][j]
        if move == "M":
            if reference[i - 1] != prediction[j - 1]:
                substitutions += 1
            i -= 1
            j -= 1
        elif move == "D":
            deletions += 1
            deleted_positions.append(i - 1)
            i -= 1
        else:
            insertions += 1
            inserted_positions.append(j - 1)
            j -= 1
    return {
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "deleted_positions": deleted_positions,
        "inserted_positions": inserted_positions,
    }


def _summarize_run(path: Path) -> dict[str, Any]:
    if path.is_dir():
        path = path / "predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)

    totals = {"insertions": 0, "deletions": 0, "substitutions": 0}
    reference_characters = 0
    predicted_characters = 0
    deleted_total = 0
    deleted_tail = 0
    inserted_total = 0
    inserted_lead = 0
    limit_hits = 0
    pages = 0
    # Deletions split by whether the page ran out of generation budget.  The tail
    # share above under-counts truncation (a page cut at 60% loses deletions that
    # are not in the reference's last 20%), so the budget split is the honest test
    # of "how much of the dropping is just the token limit".
    deletions_limit = 0
    deletions_other = 0
    reference_limit = 0
    reference_other = 0
    worst: list[dict[str, Any]] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        reference = _normalize(entry["reference"])
        prediction = _normalize(entry["prediction"])
        stats = _align_positions(reference, prediction)
        pages += 1
        reference_characters += len(reference)
        predicted_characters += len(prediction)
        for key in totals:
            totals[key] += stats[key]

        hit_limit = bool(entry.get("generation_limit_hit", False))
        limit_hits += int(hit_limit)
        if hit_limit:
            deletions_limit += stats["deletions"]
            reference_limit += len(reference)
        else:
            deletions_other += stats["deletions"]
            reference_other += len(reference)

        # A deletion in the final 20% of the reference is read as "the page was
        # cut short"; one earlier in the body is read as skipped content.
        tail_start = int(len(reference) * 0.8)
        deleted_total += stats["deletions"]
        deleted_tail += sum(1 for pos in stats["deleted_positions"] if pos >= tail_start)
        # Insertions in the first 10% are usually a runaway preamble/repeat.
        lead_end = max(1, int(len(prediction) * 0.1))
        inserted_total += stats["insertions"]
        inserted_lead += sum(1 for pos in stats["inserted_positions"] if pos < lead_end)

        if stats["deletions"] > 0:
            worst.append(
                {
                    "page_id": entry["page_id"],
                    "reference_characters": len(reference),
                    "predicted_characters": len(prediction),
                    "deletions": stats["deletions"],
                    "insertions": stats["insertions"],
                    "generation_limit_hit": bool(entry.get("generation_limit_hit", False)),
                }
            )

    worst.sort(key=lambda row: row["deletions"], reverse=True)
    per_k = 1000.0 / max(1, reference_characters)
    return {
        "pages": pages,
        "reference_characters": reference_characters,
        "predicted_characters": predicted_characters,
        "length_ratio": predicted_characters / max(1, reference_characters),
        "insertions": totals["insertions"],
        "deletions": totals["deletions"],
        "substitutions": totals["substitutions"],
        "insertions_per_1k": totals["insertions"] * per_k,
        "deletions_per_1k": totals["deletions"] * per_k,
        "substitutions_per_1k": totals["substitutions"] * per_k,
        "deletion_tail_share": deleted_tail / max(1, deleted_total),
        "insertion_lead_share": inserted_lead / max(1, inserted_total),
        "generation_limit_hit_rate": limit_hits / max(1, pages),
        "deletions_on_limit_pages": deletions_limit,
        "deletions_on_other_pages": deletions_other,
        "deletion_share_from_limit_pages": deletions_limit / max(1, deletions_limit + deletions_other),
        "deletions_per_1k_on_limit_pages": deletions_limit * 1000.0 / max(1, reference_limit),
        "deletions_per_1k_on_other_pages": deletions_other * 1000.0 / max(1, reference_other),
        "worst_pages_by_deletions": worst[:5],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="decompose decoder-mask prediction errors")
    parser.add_argument("--run", action="append", required=True, metavar="ARM=PATH")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    runs: dict[str, Path] = {}
    for spec in args.run:
        arm, path = spec.split("=", 1)
        runs[arm.strip()] = Path(path.strip())

    results = {arm: _summarize_run(path) for arm, path in runs.items()}

    header = (
        f"{'arm':>6} {'ref_chars':>9} {'len_ratio':>9} {'ins':>7} {'del':>7} {'sub':>7} "
        f"{'del/1k':>7} {'ins/1k':>7} {'del_tail':>8} {'limit%':>7}"
    )
    print(header)
    for arm, row in results.items():
        print(
            f"{arm:>6} {row['reference_characters']:>9} {row['length_ratio']:>9.3f} "
            f"{row['insertions']:>7} {row['deletions']:>7} {row['substitutions']:>7} "
            f"{row['deletions_per_1k']:>7.1f} {row['insertions_per_1k']:>7.1f} "
            f"{row['deletion_tail_share'] * 100:>7.1f}% {row['generation_limit_hit_rate'] * 100:>6.1f}%"
        )
    print()
    print("del_tail = share of deletions in the last 20% of the reference "
          "(high => pages cut short; low => content skipped in the body)")
    print("len_ratio = predicted characters / reference characters")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"runs": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
