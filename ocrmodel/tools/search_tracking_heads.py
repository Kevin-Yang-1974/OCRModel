"""Re-choose the probed head set by the quantity the tracker is actually scored on.

Stage 0 picked the eight heads by *instantaneous* single-head accuracy on the select pages: which
heads name the right line for the step being read. That is not the quantity the arm pays for. The
routing applies the estimate one step later, so what matters is the accuracy of the line that was
*applied*, and a head that is right on the current step can still be the one that drags the
applied line off at a boundary.

Everything here is offline over a recorded probe file, so a candidate costs nothing to check. The
selection runs on the select pages and is reported on the check pages, because choosing a subset
on the pages it is then scored on is the one way this could produce a number that does not
reproduce -- the same rule stage 0 registered for the layer and head choice.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_attention_localization import align, char_lines, page_steps  # noqa: E402

DEFAULT_LAYERS = (8,)
DEFAULT_HEADS = (2, 3, 8, 10, 11, 12, 14, 15)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--heads", type=int, nargs="+", default=list(DEFAULT_HEADS))
    parser.add_argument("--select-pages", type=Path, required=True)
    parser.add_argument("--check-pages", type=Path, required=True)
    parser.add_argument("--bias", type=float, default=1.0)
    parser.add_argument("--bar", type=float, default=6.0)
    parser.add_argument("--corrected", action="store_true")
    parser.add_argument("--min-heads", type=int, default=1)
    parser.add_argument("--max-heads", type=int, default=8)
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def load_pages(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def page_inputs(report: dict[str, Any], record: dict[str, Any]) -> dict[str, Any] | None:
    """Per-step per-head distributions and the truth line of each emitted character."""

    steps = page_steps(report)
    regions = list(record.get("regions") or [])
    characters = list(record.get("characters") or [])
    if not steps or not regions or not characters:
        return None
    line_of_char = char_lines(regions, characters)
    ordered = sorted(steps)
    fragments = [steps[step]["emitted"] for step in ordered]
    generated = "".join(fragment for fragment in fragments if fragment)
    mapping = align(generated, record["page_text"])
    char_step: list[int] = []
    for step in ordered:
        char_step.extend([step] * len(steps[step]["emitted"] or ""))
    truth_of_step: dict[int, int] = {}
    for position, step in enumerate(char_step):
        if position >= len(mapping) or mapping[position] is None:
            continue
        truth = line_of_char[mapping[position]] if mapping[position] < len(line_of_char) else -1
        if truth >= 0:
            # One line per step: the first scored character's, matching the replay's convention of
            # scoring the applied line rather than each character separately.
            truth_of_step.setdefault(step, truth)
    return {
        "ordered": ordered,
        "steps": steps,
        "truth": truth_of_step,
        "num_regions": len(regions),
    }


def mean_distribution(
    heads: list[dict[str, Any]], layers, chosen: tuple[int, ...], bias: float
) -> list[float] | None:
    width = 0
    total: list[float] | None = None
    count = 0
    for row in heads:
        if row.get("layer") not in layers or row.get("head") not in chosen:
            continue
        probs = [float(value) for value in row.get("line_probs") or []]
        if not probs:
            continue
        if total is None:
            width = len(probs)
            total = [0.0] * width
        elif len(probs) != width:
            continue
        for index, value in enumerate(probs):
            total[index] += value
        count += 1
    if total is None or count == 0:
        return None
    return [value / count for value in total]


def apply_correction(probs: list[float], biased_line: int, bias: float) -> list[float]:
    if biased_line < 0 or bias == 0.0 or not 0 <= biased_line < len(probs):
        return probs
    factor = math.exp(-bias)
    adjusted = list(probs)
    adjusted[biased_line] *= factor
    total = sum(adjusted)
    return [value / total for value in adjusted] if total > 0 else adjusted


def score_page(page: dict[str, Any], chosen: tuple[int, ...], args) -> tuple[int, int]:
    """Applied-line hits and scored steps, for one subset on one page."""

    state = -1
    hits = 0
    scored = 0
    for step in page["ordered"]:
        truth = page["truth"].get(step)
        if truth is not None:
            scored += 1
            hits += int(state == truth)
        probs = mean_distribution(page["steps"][step]["heads"], args.layers, chosen, args.bias)
        if probs is None:
            state = -1
            continue
        if args.corrected:
            probs = apply_correction(probs, state, args.bias)
        best = max(range(len(probs) - 1), key=lambda index: probs[index])
        state = best if probs[best] * page["num_regions"] >= args.bar else -1
    return hits, scored


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    select_ids = load_pages(args.select_pages)
    check_ids = load_pages(args.check_pages)

    records = {}
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                records[row["page_id"]] = row

    pages: dict[str, dict[str, Any]] = {}
    with args.probe.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            report = json.loads(line)
            record = records.get(report["page_id"])
            if record is None:
                continue
            page = page_inputs(report, record)
            if page is not None:
                pages[report["page_id"]] = page

    subsets = [
        combo
        for size in range(args.min_heads, args.max_heads + 1)
        for combo in itertools.combinations(args.heads, size)
    ]
    print(f"pages {len(pages)}  select {len(select_ids & set(pages))}  "
          f"check {len(check_ids & set(pages))}  subsets {len(subsets)}  "
          f"bar {args.bar}  corrected {args.corrected}")

    results = []
    for subset in subsets:
        select_hits = select_scored = check_hits = check_scored = 0
        for page_id, page in pages.items():
            hits, scored = score_page(page, subset, args)
            if page_id in select_ids:
                select_hits += hits
                select_scored += scored
            elif page_id in check_ids:
                check_hits += hits
                check_scored += scored
        results.append(
            {
                "heads": list(subset),
                "size": len(subset),
                "select_accuracy": select_hits / select_scored if select_scored else None,
                "select_steps": select_scored,
                "check_accuracy": check_hits / check_scored if check_scored else None,
                "check_steps": check_scored,
            }
        )

    ranked = sorted(results, key=lambda row: -(row["select_accuracy"] or 0.0))
    print()
    print(f"{'heads':>28} {'sel_acc':>8} {'chk_acc':>8}")
    for row in ranked[: args.top]:
        print(f"{str(row['heads']):>28} {row['select_accuracy']:8.4f} {row['check_accuracy']:8.4f}")
    registered = next(
        (row for row in results if row["heads"] == list(args.heads)), None
    )
    if registered:
        print()
        print(f"registered set {registered['heads']}  select {registered['select_accuracy']:.4f}  "
              f"check {registered['check_accuracy']:.4f}")
    best_single = max((row for row in results if row["size"] == 1),
                      key=lambda row: row["select_accuracy"] or 0.0)
    print(f"best single head on select: {best_single['heads']}  "
          f"select {best_single['select_accuracy']:.4f}  check {best_single['check_accuracy']:.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "bar": args.bar,
                    "corrected": args.corrected,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
