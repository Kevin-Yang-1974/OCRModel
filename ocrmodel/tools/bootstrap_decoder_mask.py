#!/usr/bin/env python3
"""Paired page-level bootstrap over decoder-mask evaluation runs.

The Gate C screen ranks arms by validation CER, but a single point estimate on 64
pages says nothing about whether a difference survives page resampling.  The
plan's pre-registered gate needs the *lower bound* of a paired bootstrap CI, so
this tool consumes the per-page predictions that ``evaluate_decoder_mask.py``
writes and resamples pages with replacement.

Every run is scored on the same manifest, so the pairing is exact: page *i* of arm
A is compared against page *i* of arm B, never a different page.  The statistic is
the ratio-of-sums CER (total edits / total reference characters), which matches
``aggregate_ocr_metrics`` rather than averaging per-page error rates.

Reported per comparison:

* ``improvement`` -- ``cer_baseline - cer_arm``; positive means ``arm`` is better.
* the bootstrap percentile CI of that improvement, and the fraction of resamples
  in which the arm wins (``p_improved``).
* the relative CER reduction, checked against the plan's 3% gate.

An improvement whose CI straddles zero is *not* evidence of a difference at this
sample size.  Equivalence (e.g. B2 vs B3) is read from the CI of the paired
difference, not from the size of the point estimate.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from layout_ocr.metrics import levenshtein_error_counts  # noqa: E402


def _normalize(text: str) -> str:
    """Collapse whitespace the same way ``aggregate_ocr_metrics`` does."""

    return "".join(text.split())


def _read_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """Map ``page_id`` -> per-page record from a predictions.jsonl or its dir."""

    if path.is_dir():
        path = path / "predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    pages: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        page_id = str(entry["page_id"])
        reference = _normalize(entry["reference"])
        prediction = _normalize(entry["prediction"])
        counts = levenshtein_error_counts(reference, prediction)
        pages[page_id] = {
            "edits": counts["insertions"] + counts["deletions"] + counts["substitutions"],
            "reference_characters": len(reference),
            "generation_limit_hit": bool(entry.get("generation_limit_hit", False)),
            "generated_tokens": int(entry.get("generated_tokens", 0)),
        }
    return pages


def _read_sidecar(run_path: Path) -> dict[str, Any]:
    """Resource metrics from the run's summary.json, when it is present."""

    summary_path = (run_path if run_path.is_dir() else run_path.parent) / "summary.json"
    if not summary_path.is_file():
        return {}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    resource = summary.get("resource") or {}
    return {
        "tokens_per_second": resource.get("tokens_per_second"),
        "generation_limit_hit_rate": resource.get("generation_limit_hit_rate"),
        "cuda_peak_memory_gb": resource.get("cuda_peak_memory_gb"),
        "elapsed_seconds": resource.get("elapsed_seconds"),
    }


def _cer(edits: list[int], refs: list[int]) -> float:
    total_ref = sum(refs)
    return sum(edits) / total_ref if total_ref else 0.0


def _bootstrap_difference(
    edits_a: list[int],
    edits_b: list[int],
    refs: list[int],
    iterations: int,
    rng: random.Random,
) -> dict[str, float]:
    """Percentile CI of ``cer_b - cer_a`` over page resamples (A better if > 0)."""

    n = len(refs)
    differences: list[float] = []
    wins = 0
    for _ in range(iterations):
        index = [rng.randrange(n) for _ in range(n)]
        ea = sum(edits_a[i] for i in index)
        eb = sum(edits_b[i] for i in index)
        total_ref = sum(refs[i] for i in index)
        if total_ref == 0:
            continue
        difference = (eb - ea) / total_ref
        differences.append(difference)
        if difference > 0:
            wins += 1
    if not differences:
        return {"ci_low": None, "ci_high": None, "p_improved": None, "mean": None}
    differences.sort()
    return {
        "ci_low": differences[int(0.025 * len(differences))],
        "ci_high": differences[min(len(differences) - 1, int(0.975 * len(differences)))],
        "p_improved": wins / len(differences),
        "mean": sum(differences) / len(differences),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="paired page bootstrap over decoder-mask runs")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="ARM=PATH",
        help="repeatable; PATH is a run dir or a predictions.jsonl",
    )
    parser.add_argument("--baseline", default="B0", help="arm every other arm is compared against")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--relative-gate", type=float, default=0.03, help="plan's 3%% CER gate")
    parser.add_argument("--output", type=Path, help="write the result JSON here")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    runs: dict[str, Path] = {}
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run expects ARM=PATH, got {spec!r}")
        arm, path = spec.split("=", 1)
        runs[arm.strip()] = Path(path.strip())
    if args.baseline not in runs:
        raise SystemExit(f"baseline {args.baseline!r} was not passed via --run")

    pages = {arm: _read_predictions(path) for arm, path in runs.items()}
    sidecar = {arm: _read_sidecar(path) for arm, path in runs.items()}

    shared = set.intersection(*(set(p) for p in pages.values()))
    if not shared:
        raise SystemExit("the runs share no page ids; they were not scored on one manifest")
    page_ids = sorted(shared)

    per_run: dict[str, dict[str, Any]] = {}
    for arm in runs:
        edits = [pages[arm][p]["edits"] for p in page_ids]
        refs = [pages[arm][p]["reference_characters"] for p in page_ids]
        hits = sum(1 for p in page_ids if pages[arm][p]["generation_limit_hit"])
        per_run[arm] = {
            "pages": len(page_ids),
            "reference_characters": sum(refs),
            "character_errors": sum(edits),
            "cer": _cer(edits, refs),
            "generation_limit_hit_rate": hits / max(1, len(page_ids)),
            "exact_page_rate": sum(
                1
                for p in page_ids
                if pages[arm][p]["edits"] == 0 and pages[arm][p]["reference_characters"] > 0
            )
            / max(1, len(page_ids)),
            **sidecar[arm],
        }

    rng = random.Random(args.seed)
    comparisons: list[dict[str, Any]] = []
    arms = list(runs)
    for i, arm in enumerate(arms):
        for other in arms[i + 1 :]:
            edits_a = [pages[arm][p]["edits"] for p in page_ids]
            edits_b = [pages[other][p]["edits"] for p in page_ids]
            refs = [pages[arm][p]["reference_characters"] for p in page_ids]
            # improvement > 0 always means "the second arm of the pair is better",
            # so the pair is reported once with an explicit direction.
            stats = _bootstrap_difference(edits_b, edits_a, refs, args.iterations, rng)
            cer_a, cer_b = per_run[arm]["cer"], per_run[other]["cer"]
            relative = (cer_a - cer_b) / cer_a if cer_a else 0.0
            comparisons.append(
                {
                    "arm": arm,
                    "versus": other,
                    "cer": cer_a,
                    "cer_versus": cer_b,
                    "improvement": cer_a - cer_b,  # >0: `versus` is better
                    "relative_reduction": relative,
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "p_versus_better": stats["p_improved"],
                    "suggests_difference": bool(
                        stats["ci_low"] is not None and stats["ci_low"] > 0.0
                    ),
                    "passes_relative_gate": bool(
                        stats["ci_low"] is not None
                        and stats["ci_low"] > 0.0
                        and relative >= args.relative_gate
                    ),
                }
            )

    result = {
        "status": "complete",
        "selection_metric": "validation_cer",
        "baseline": args.baseline,
        "iterations": args.iterations,
        "seed": args.seed,
        "relative_gate": args.relative_gate,
        "pages": len(page_ids),
        "page_ids": page_ids,
        "runs": per_run,
        "comparisons": comparisons,
        "test_used_for_selection": False,
    }

    print(
        f"paired page bootstrap: {len(page_ids)} shared pages, "
        f"{args.iterations} resamples, seed {args.seed}"
    )
    print(f"{'arm':>6} {'CER':>8} {'limit%':>8} {'tok/s':>8} {'peakGB':>8}")
    for arm, row in per_run.items():
        tps = row.get("tokens_per_second")
        peak = row.get("cuda_peak_memory_gb")
        print(
            f"{arm:>6} {row['cer']:>8.4f} {row['generation_limit_hit_rate'] * 100:>7.1f}% "
            f"{(tps if tps is not None else float('nan')):>8.1f} "
            f"{(peak if peak is not None else float('nan')):>8.2f}"
        )
    print()
    print(f"{'A':>6} {'B':>6} {'CER_A':>8} {'CER_B':>8} {'improve':>9} {'CI low':>9} {'CI high':>9} {'rel%':>7} {'verdict':>10}")
    for row in comparisons:
        low = row["ci_low"]
        high = row["ci_high"]
        verdict = "DIFFERENT" if row["suggests_difference"] else "not-sig"
        if row["passes_relative_gate"]:
            verdict = "PASSES-GATE"
        print(
            f"{row['arm']:>6} {row['versus']:>6} {row['cer']:>8.4f} {row['cer_versus']:>8.4f} "
            f"{row['improvement']:>9.4f} "
            f"{(low if low is not None else float('nan')):>9.4f} "
            f"{(high if high is not None else float('nan')):>9.4f} "
            f"{row['relative_reduction'] * 100:>6.2f}% {verdict:>10}"
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
