"""Replay the tracked line source offline, from a probe file, under policies never yet run.

§8 of `docs/LAYOUT_LINE_DETECTOR_AND_PREDMAP_RESULT.md` measured that the bias raises the very
confidence the gate reads (the 8x band grows 70.8% -> 84.8% while the steps it newly admits score
84% against the natives' 97%). The gate was registered on an *unbiased* run, so this is a
measurable defect, but the fix has never been tested -- it has only ever been argued about.

The whole decision chain is a pure function of the recorded per-step, per-head line
distributions, so it can be replayed. The one thing the replay needs beyond the file is which
line was biased while a given step's attention was being read, and that is recoverable: the
routing hook reads the estimate the probe wrote on the *previous* step, so the line biased at
step ``s`` is the accepted estimate from step ``s-1``. Replaying the accepted set and getting back
the routing report's own ``accepted_fraction`` is the check that this reconstruction is right;
without it every number below is a plausible fiction.

The correction itself is exact rather than approximate. Adding ``B`` to every key of one line
multiplies that line's softmax mass by ``e^B`` before renormalisation, so dividing it back out
recovers the unbiased distribution -- and the probe stores the distribution **per head**, so each
head is corrected with its own normaliser instead of the corrected average being approximated.

Scored metric: the line the tracker actually *applied* against the true line of the characters
emitted at that step. That is the quantity the CER gain comes through, and it is not the same as
the instantaneous readout accuracy §8 reports -- the applied line is one step stale by
construction, and a policy can improve the readout while making the applied line worse.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_attention_localization import (  # noqa: E402
    align,
    char_lines,
    combine,
    page_steps,
    select,
)

# Stage 1's registered bar, in the scale-free unit the tracker uses.
DEFAULT_BAR = 6.0
# The tracked arm's bias.  Only used to invert the contamination, so a wrong value shows up as a
# correction that does not move the distribution rather than as a silent shift.
DEFAULT_BIAS = 1.0
DEFAULT_LAYERS = (8,)
DEFAULT_HEADS = (2, 3, 8, 10, 11, 12, 14, 15)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path, help="manifest.char.jsonl")
    parser.add_argument("--bias", type=float, default=DEFAULT_BIAS)
    parser.add_argument("--bar", type=float, default=DEFAULT_BAR)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--heads", type=int, nargs="+", default=list(DEFAULT_HEADS))
    parser.add_argument(
        "--bars",
        type=float,
        nargs="+",
        default=None,
        help="sweep these bars instead of --bar; every policy is reported at each",
    )
    parser.add_argument(
        "--routing-report",
        type=Path,
        default=None,
        help=(
            "the arm's routing jsonl. Replaying the recorded policy has to reproduce its own "
            "gated count, and that is the only check that the reconstruction of which line was "
            "biased at each step is right; without it every number below is a plausible fiction"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def correct_head(line_probs: list[float], biased_line: int, bias: float) -> list[float]:
    """Invert one head's contamination by the bias that was on while it was read.

    The biased line's mass was multiplied by ``e^B`` before renormalisation, so multiplying it by
    ``e^-B`` and renormalising gives back the distribution that would have been read with no bias.
    Exact for this bias shape, which is a constant on every key of the line.
    """

    if biased_line < 0 or bias == 0.0 or not 0 <= biased_line < len(line_probs):
        return line_probs
    factor = math.exp(-bias)
    adjusted = list(line_probs)
    adjusted[biased_line] *= factor
    total = sum(adjusted)
    return [value / total for value in adjusted] if total > 0 else adjusted


def readout(
    heads: list[dict[str, Any]], layers, heads_filter, aggregate: str, biased_line: int, bias: float,
    correct: bool,
) -> dict[str, Any] | None:
    """The step's one line distribution, optionally with the bias divided out first."""

    rows = select(heads, layers, heads_filter)
    if not correct:
        return combine(rows, aggregate)
    if not rows:
        return None
    width = len(rows[0].get("line_probs") or [])
    if not width:
        return None
    total = [0.0] * width
    for row in rows:
        adjusted = correct_head([float(value) for value in row["line_probs"]], biased_line, bias)
        for index, value in enumerate(adjusted):
            total[index] += value
    mean = [value / len(rows) for value in total]
    # The last slot is everything off every line; it can win the argmax but is not a line.
    best = max(range(len(mean) - 1), key=lambda index: mean[index])
    return {
        "line_probs": mean,
        "argmax_line": best,
        "top_line_mass": mean[best],
        "background_mass": mean[-1],
    }


def policy_recorded(state: int, candidate: int, confidence: float, bar: float) -> int:
    return candidate if confidence >= bar else -1


def policy_hold(state: int, candidate: int, confidence: float, bar: float) -> int:
    """Keep aiming at the last line when the estimate is not good enough to replace it.

    Dropping the line costs dose on every rejected step -- §9.1 showed a stricter bar shrinks
    ``mean_boxes_hit`` from 81.0 to 66.7, and the deletions rise with it.
    """

    return candidate if confidence >= bar else state


def policy_mono(state: int, candidate: int, confidence: float, bar: float) -> int:
    """Reading order does not go backwards, so a lower line is refused outright.

    Deployable: the line number indexes a list already sorted by the detector's reading order,
    whose pairwise accuracy is 0.937 on MTHv2 and 0.980 zero-shot on Dunhuang. This is the one
    structural prior a deployed tracker has and the recorded oracle pointer has it by
    construction -- ``synced`` never walks backwards.
    """

    if confidence < bar:
        return -1
    return candidate if candidate >= state else state


def policy_mono_hold(state: int, candidate: int, confidence: float, bar: float) -> int:
    if confidence < bar:
        return state
    return candidate if candidate >= state else state


POLICIES: dict[str, tuple[Callable[[int, int, float, float], int], bool]] = {
    # name -> (policy, whether the confidence has the bias divided out of it)
    "recorded": (policy_recorded, False),
    "corrected": (policy_recorded, True),
    "hold": (policy_hold, False),
    "mono": (policy_mono, False),
    "mono_hold": (policy_mono_hold, False),
    "corrected_hold": (policy_hold, True),
    "corrected_mono": (policy_mono, True),
}


def replay_page(
    report: dict[str, Any],
    record: dict[str, Any],
    reference: str,
    *,
    layers,
    heads,
    bias: float,
    bars: list[float],
    aggregate: str = "mean",
) -> dict[str, Any]:
    """One page, every policy, every bar."""

    steps = page_steps(report)
    regions = list(record.get("regions") or [])
    characters = list(record.get("characters") or [])
    if not steps or not regions or not characters:
        return {"page_id": report.get("page_id"), "skipped": True}
    lines = char_lines(regions, characters)
    ordered = sorted(steps)
    fragments = [steps[step]["emitted"] for step in ordered]
    generated = "".join(fragment for fragment in fragments if fragment)
    mapping = align(generated, reference)
    char_step: list[int] = []
    for step in ordered:
        char_step.extend([step] * len(steps[step]["emitted"] or ""))

    # Truth per generated character, and the step it was emitted from.  Characters with no line
    # (an insertion, or a reference position with no box) are not scored -- the same rule the
    # offline localizer uses, so the two reports are on one denominator.
    scored: list[tuple[int, int]] = []
    for position, step in enumerate(char_step):
        if position >= len(mapping) or mapping[position] is None:
            continue
        truth = lines[mapping[position]] if mapping[position] < len(lines) else -1
        if truth >= 0:
            scored.append((step, truth))

    per_policy: dict[str, dict[str, Any]] = {}
    for bar in bars:
        for name, (policy, corrected) in POLICIES.items():
            key = f"{name}@{bar:g}"
            state = -1
            applied: dict[int, int] = {}
            accepted = 0
            observations = 0
            gated_equivalent = 0
            for step in ordered:
                applied[step] = state
                biased_line = state
                # The routing's own counter, for the cross-check below: it calls a step gated
                # when no line was carried into it. Reconstructing that count from the replay is
                # what proves the lag model -- step s is biased by step s-1's estimate.
                gated_equivalent += int(biased_line < 0)
                row = readout(
                    steps[step]["heads"], layers, heads, aggregate, biased_line, bias, corrected
                )
                if row is not None:
                    observations += 1
                    confidence = float(row["top_line_mass"]) * len(regions)
                    state = policy(state, int(row["argmax_line"]), confidence, bar)
                    if state >= 0:
                        accepted += 1
            hit = sum(1 for step, truth in scored if applied.get(step, -1) == truth)
            biased_chars = sum(1 for step, _ in scored if applied.get(step, -1) >= 0)
            per_policy[key] = {
                "chars": len(scored),
                "applied_correct": hit,
                "applied_accuracy": hit / len(scored) if scored else None,
                "biased_chars": biased_chars,
                "coverage": biased_chars / len(scored) if scored else None,
                "accepted_steps": accepted,
                "observations": observations,
                "gated_equivalent": gated_equivalent,
                "accepted_fraction": accepted / observations if observations else None,
                # Accuracy restricted to the characters the policy actually biased: this is the
                # quantity the CER gain runs through, and it separates "aimed at nothing" from
                # "aimed at the wrong line".
                "accuracy_where_biased": (
                    hit / biased_chars if biased_chars else None
                ),
            }
    return {"page_id": report.get("page_id"), "policies": per_policy}


def verify_against_routing_report(
    path: Path | None, totals: dict[str, dict[str, float]], bias: float
) -> tuple[int, int, float] | None:
    """Compare the replay's gated count with the routing report's own.

    The report counts a decoding step as gated when no line was carried into it, which is what
    the replay reconstructs; agreement means the one-step lag was modelled correctly. A mismatch
    that is a constant one-per-page is the structural first step rather than an error, so the
    difference is reported as a number instead of being asserted away.
    """

    if path is None or not path.is_file():
        return None
    recorded = 0
    bar = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            recorded += int(row.get("gated_steps", 0))
            tracked = row.get("tracked") or {}
            if bar is None and tracked.get("confidence_bar") is not None:
                bar = float(tracked["confidence_bar"])
    if bar is None:
        return None
    key = f"recorded@{bar:g}"
    entry = totals.get(key)
    if entry is None:
        return None
    return recorded, int(entry["gated_equivalent"]), bar


def merge(totals: dict[str, dict[str, float]], page: dict[str, Any]) -> None:
    for key, entry in page.get("policies", {}).items():
        bucket = totals.setdefault(
            key,
            {"chars": 0, "correct": 0, "biased": 0, "accepted": 0, "observations": 0, "gated_equivalent": 0},
        )
        bucket["chars"] += entry["chars"]
        bucket["correct"] += entry["applied_correct"]
        bucket["biased"] += entry["biased_chars"]
        bucket["accepted"] += entry["accepted_steps"]
        bucket["observations"] += entry["observations"]
        bucket["gated_equivalent"] += entry["gated_equivalent"]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bars = args.bars or [args.bar]
    manifest = {}
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                manifest[row["page_id"]] = row
    totals: dict[str, dict[str, float]] = {}
    pages = 0
    skipped = 0
    with args.probe.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            report = json.loads(line)
            record = manifest.get(report["page_id"])
            if record is None:
                skipped += 1
                continue
            outcome = replay_page(
                report,
                record,
                record["page_text"],
                layers=args.layers,
                heads=args.heads,
                bias=args.bias,
                bars=bars,
            )
            if outcome.get("skipped"):
                skipped += 1
                continue
            pages += 1
            merge(totals, outcome)

    print(f"pages {pages}  skipped {skipped}  bias {args.bias}  "
          f"layers {args.layers} heads {args.heads}")
    print()
    # Sorted by coverage, not by accuracy. Every policy can buy accuracy by biasing fewer steps,
    # and §9.1 already showed that reading a bar change as a mechanism change is how a dose
    # effect gets mistaken for a better estimate -- so the curve has to be read left to right.
    header = (f"{'policy':>18} {'coverage':>9} {'acc':>8} {'biased_acc':>11} "
              f"{'accepted':>9} {'chars':>7}")
    print(header)
    for key in sorted(totals, key=lambda name: (_coverage(totals[name]), name)):
        entry = totals[key]
        print(f"{key:>18} {_coverage(entry):9.4f} {_accuracy(entry):8.4f} "
              f"{_biased_accuracy(entry):11.4f} "
              f"{_accepted(entry):9.4f} {entry['chars']:7d}")

    check = verify_against_routing_report(args.routing_report, totals, args.bias)
    if check is not None:
        recorded, replayed, bar = check
        verdict = "matches" if recorded == replayed else f"DIFFERS by {replayed - recorded:+d}"
        print()
        print(f"reconstruction check at bar {bar:g}: routing report gated {recorded}, "
              f"replay gated {replayed}  ->  {verdict}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "probe": str(args.probe),
            "bias": args.bias,
            "layers": list(args.layers),
            "heads": list(args.heads),
            "pages": pages,
            "skipped": skipped,
            "reconstruction_check": (
                None if check is None
                else {"recorded_gated": check[0], "replayed_gated": check[1], "bar": check[2]}
            ),
            "policies": {key: dict(value) for key, value in totals.items()},
        }
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


def _accuracy(entry: dict[str, float]) -> float:
    return entry["correct"] / entry["chars"] if entry["chars"] else 0.0


def _biased_accuracy(entry: dict[str, float]) -> float:
    return entry["correct"] / entry["biased"] if entry["biased"] else 0.0


def _coverage(entry: dict[str, float]) -> float:
    return entry["biased"] / entry["chars"] if entry["chars"] else 0.0


def _accepted(entry: dict[str, float]) -> float:
    return entry["accepted"] / entry["observations"] if entry["observations"] else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
