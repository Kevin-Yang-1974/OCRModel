"""Score the attention probe offline: is there a readable "where am I" signal?

Stage 1 of ``plans/LAYOUT_ATTENTION_TRACKING.md``.  The probe records, per decoding
step, where each observed head's attention fell on the page.  This turns that into
the one question the stage is allowed to answer -- *can the model's own attention
say which line it is reading* -- and nothing further.  No routing is scored here.

## What it needs, and what happens when it is missing

The line label for a step comes from the character the step emitted: the generated
text is aligned to the reference, and the aligned character's box gives its line.
That needs a manifest carrying the **character box channel** (``characters``, written
by ``tools/prepare_mthv2_char_manifest.py`` for MTHv2).  Regions alone carry only a
box, an order and a direction -- no text and no character count -- so a page with
regions but no character boxes has no line truth at all.

That is the situation for Dunhuang today, and the plan says so: the character-level
channel has to be built for vertical ancient columns first.  This tool therefore
reports the gap and scores nothing for such pages, rather than substituting the
region count or a uniform assumption for a truth it does not have.

## The alignment

The probe cannot see the sampled token, so the eval loop stamps each step with the
text that generation emitted from it (``attach_emitted``), built by cumulative decode
because one token can carry several characters and one character can span several
tokens.  Those per-step fragments are concatenated and aligned to the reference with
an edit-distance backtrace.  The alignment is what distinguishes three kinds of step,
and they are reported separately because they mean different things:

``matched``      the emitted characters correspond to reference characters, so the
                 step has a true line and can be scored.
``inserted``     the emitted characters correspond to nothing in the reference, so
                 there is no true line.  Scoring these would be scoring the model
                 against its own hallucination.
``ambiguous``    the emitted span covers more than one aligned reference position, so
                 a single line label is a choice rather than an observation.  Scored
                 on the first, and counted so the choice is visible.

The existing ``_advance`` greedy pointer is *not* used as truth here, for the reason
the plan gives: repeated characters, insertions and substitutions all make it
ambiguous, and it is the thing being judged, not a reference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# The gap between `noroute` and `bias2` was 0.034 CER, so a localization readout that
# only holds at 60% is not a base worth building a tracker on.
GATE_ACCURACY = 0.90
GATE_COVERAGE = 0.50
# A step is "confident" when the attention-weighted line share clears this.  Fixed
# before the run: choosing it after seeing the curve is the same as choosing heads by
# their answer.
DEFAULT_CONFIDENCE = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path, help="probe.jsonl from the run")
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="validation_predictions.jsonl, for the reference and the emitted text",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="the manifest the run evaluated, with the character box channel",
    )
    parser.add_argument(
        "--select-pages",
        type=Path,
        default=None,
        help=(
            "one page id per line: the pages the layers/heads were chosen on. The gate "
            "is never reported on these"
        ),
    )
    parser.add_argument(
        "--check-pages",
        type=Path,
        default=None,
        help=(
            "one page id per line: the reserved pages held out during head selection. "
            "The gate is reported here, and only here"
        ),
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument(
        "--confidence-mode",
        choices=["absolute", "uniform_multiple"],
        default="absolute",
        help=(
            "'absolute' compares top_line_mass to --confidence. 'uniform_multiple' compares "
            "top_line_mass times the page's line count, i.e. how many times above a uniform "
            "distribution over its lines the readout concentrates -- the same number means "
            "the same thing on a ten-line page and a forty-line one, which the absolute "
            "share does not. The default is 'absolute' so results reported under it stay "
            "reproducible; the stage-1 pre-registration selects uniform_multiple at 2.0"
        ),
    )
    parser.add_argument(
        "--neighbourhood",
        type=int,
        default=3,
        help=(
            "how many characters either side count as 'near' a dropped or repeated "
            "stretch when reporting the readout there"
        ),
    )
    parser.add_argument(
        "--aggregate",
        choices=["mean", "best"],
        default="mean",
        help=(
            "how the selected heads are combined into one per-step readout: 'mean' "
            "averages their line distributions, 'best' picks the head with the largest "
            "top-line mass. 'best' is reported for comparison only -- it chooses per "
            "step by the quantity being scored"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def load_pages(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def char_lines(regions: list[dict[str, Any]], characters: list[dict[str, Any]]) -> list[int]:
    """The region index owning each reference character, or ``-1`` when unbounded.

    Regions are sorted by reading order, matching ``layout_targets`` and the probe, so
    the index this returns is the same label space the probe's ``argmax_line`` is in.
    Ownership is the first box (in reading order) containing the character's centre --
    the same rule the training targets and the probe's token map both use.
    """

    ordered = sorted(regions, key=lambda item: int(item["reading_order"]))
    lines: list[int] = []
    for entry in characters:
        box = (entry or {}).get("bbox")
        if not box:
            # The manifest could not place this character (its aligner left it
            # unmatched).  Un-placeable is not a line, so it is not scored.
            lines.append(-1)
            continue
        x = (float(box[0]) + float(box[2])) / 2.0
        y = (float(box[1]) + float(box[3])) / 2.0
        found = -1
        for index, region in enumerate(ordered):
            region_box = region["bbox"]
            if (
                float(region_box[0]) <= x <= float(region_box[2])
                and float(region_box[1]) <= y <= float(region_box[3])
            ):
                found = index
                break
        lines.append(found)
    return lines


def align(generated: str, reference: str) -> list[int | None]:
    """Align the generated text to the reference; one entry per generated character.

    Each entry is the reference position that character corresponds to, or ``None``
    when the character is an insertion.  A full matrix is fine at page scale (a few
    thousand characters) and keeps the backtrace obvious.
    """

    n, m = len(generated), len(reference)
    # cost[i][j] = edit distance between generated[:i] and reference[:j]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        previous, current = cost[i - 1], cost[i]
        char = generated[i - 1]
        for j in range(1, m + 1):
            current[j] = min(
                previous[j] + 1,  # the generated character is inserted
                current[j - 1] + 1,  # a reference character is skipped (deleted)
                previous[j - 1] + (char != reference[j - 1]),  # match or substitute
            )

    mapping: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            substitute = cost[i - 1][j - 1] + (generated[i - 1] != reference[j - 1])
            if cost[i][j] == substitute:
                mapping[i - 1] = j - 1
                i, j = i - 1, j - 1
                continue
        if i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            mapping[i - 1] = None
            i -= 1
            continue
        j -= 1
    return mapping


def page_steps(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """A page's probe records by decoding step.

    Returns the *records*, not the head rows: the emitted text and the key count live
    on the step, while the per-layer/per-head readouts live in ``record["heads"]``.
    Flattening the two together loses the step-level fields, which is how the emitted
    text went missing the first time this was written.
    """

    steps: dict[int, dict[str, Any]] = {}
    for record in report.get("steps", []):
        entry = steps.setdefault(int(record["step"]), {"heads": [], "emitted": None})
        entry["heads"].extend(record.get("heads", []))
        if record.get("emitted") is not None:
            entry["emitted"] = record["emitted"]
        entry["text_keys"] = record.get("text_keys")
    return steps


def steps_for_head_selection(report: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """The per-layer/per-head readouts for each step, for the selection report."""

    return {step: entry["heads"] for step, entry in page_steps(report).items()}


def select(rows: list[dict[str, Any]], layers, heads) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if (layers is None or row["layer"] in layers) and (heads is None or row["head"] in heads)
    ]


def combine(rows: list[dict[str, Any]], aggregate: str) -> dict[str, Any] | None:
    """Reduce the selected heads to one readout for a step.

    The per-line probabilities are averaged rather than the argmax being taken per
    head first: averaging ``argmax_line`` across heads would throw away the
    distribution that says how sure the step is, and the plan's whole first revision
    was to stop collapsing to a single point before looking at the spread.
    """

    if not rows:
        return None
    if aggregate == "best":
        return max(rows, key=lambda row: row.get("top_line_mass", 0.0))
    width = len(rows[0].get("line_probs") or [])
    if not width:
        return None
    total = [0.0] * width
    for row in rows:
        for index, value in enumerate(row["line_probs"]):
            total[index] += float(value)
    count = float(len(rows))
    line_probs = [value / count for value in total]
    best = max(range(len(line_probs) - 1), key=lambda index: line_probs[index])
    return {
        "line_probs": line_probs,
        "argmax_line": best,
        "top_line_mass": line_probs[best],
        "background_mass": line_probs[-1],
        "in_line_pos": sum(float(row.get("in_line_pos", -1.0)) for row in rows) / count,
        "m_t": sum(float(row["m_t"]) for row in rows) / count,
    }


def expected_in_line_pos(region: dict[str, Any], box: list[float]) -> tuple[float, int]:
    """Where along its line the reference character sits, in the probe's convention.

    Returns the normalized coordinate and the axis it was measured on, so the caller
    can compare like with like: a vertical column is read top-to-bottom and uses
    ``y``; a horizontal line uses ``x``.  Returns ``-1.0`` when the region's box has
    no extent on that axis.
    """

    axis = 1 if region.get("writing_direction") == "vertical_rtl" else 0
    low, high = float(region["bbox"][axis]), float(region["bbox"][axis + 2])
    if high <= low:
        return -1.0, axis
    centre = (float(box[axis]) + float(box[axis + 2])) / 2.0
    return (centre - low) / (high - low), axis


def _prefix_counts(flags: list[bool]) -> list[int]:
    """Prefix sums, so a window query is two lookups instead of a scan."""

    counts = [0]
    for flag in flags:
        counts.append(counts[-1] + int(flag))
    return counts


def _in_window(counts: list[int], start: int, end: int) -> bool:
    start = max(0, start)
    end = min(len(counts) - 1, end)
    return start < end and counts[end] - counts[start] > 0


def neighbourhood_flags(
    generated: str, reference: str, mapping: list[int | None], window: int
) -> tuple[list[bool], list[bool]]:
    """Which generated positions sit near a dropped reference character or a repeat.

    The plan asks for the readout's behaviour specifically around missed and repeated
    characters, because those are the two failures a line constraint is supposed to
    address: drifting off the line shows up as a dropped or duplicated stretch, so if the
    readout is *also* least reliable exactly there, a tracker built on it inherits the
    failure rather than fixing it.

    ``dropped[j]``      reference position ``j`` that no generated character was aligned
                        to -- a character the model skipped.
    ``repeated[i]``     generated position ``i`` lies inside a run of a short substring
                        repeated back to back at least three times -- the loop the
                        training side's cycle processor targets.

    ``repeated`` deliberately does **not** use ``repeated_trigram_rate``'s notion, which
    counts any trigram occurring more than once in the page.  That is a page-level rate
    over ordinary text: in Chinese, common trigrams recur constantly, and as a per-character
    flag it marked 75% of a validation page, which measures the language rather than the
    failure.  What a line constraint is meant to answer for is a stretch the model got
    stuck on, so the test is a consecutive cycle.
    """

    mapped = {index for index in mapping if index is not None}
    dropped = [index not in mapped for index in range(len(reference))]

    repeated = [False] * len(generated)
    index = 0
    while index < len(generated):
        span = 0
        for period in range(1, 9):
            if index + period * 3 > len(generated):
                continue
            unit = generated[index : index + period]
            repeats = 1
            while (
                generated[index + repeats * period : index + (repeats + 1) * period] == unit
            ):
                repeats += 1
            if repeats >= 3:
                span = max(span, period * repeats)
        if span:
            for offset in range(span):
                repeated[index + offset] = True
            index += span
        else:
            index += 1

    dropped_counts = _prefix_counts(dropped)
    repeated_counts = _prefix_counts(repeated)
    near_dropped = [
        index < len(mapping)
        and mapping[index] is not None
        and _in_window(dropped_counts, mapping[index] - window, mapping[index] + window + 1)
        for index in range(len(generated))
    ]
    near_repeated = [
        _in_window(repeated_counts, index - window, index + window + 1)
        for index in range(len(generated))
    ]
    return near_dropped, near_repeated


def score_page(
    report: dict[str, Any],
    reference: str,
    prediction: str,
    regions: list[dict[str, Any]],
    characters: list[dict[str, Any]],
    *,
    layers,
    heads,
    aggregate: str,
    window: int = 3,
) -> dict[str, Any]:
    """Score one page's probe report against the reference it was asked to produce."""

    steps = page_steps(report)
    if not steps:
        return {"page_id": report.get("page_id"), "steps": 0, "scored": 0, "reason": "no steps"}
    lines = char_lines(regions, characters)
    ordered = sorted(regions, key=lambda item: int(item["reading_order"]))

    # The emitted text, step by step.  Step 1 is the first *decode* step: token 0 was
    # sampled from the prefill, so it belongs to no observed step.  Missing text is a
    # gap in the evidence, not a step to skip silently.
    ordered_steps = sorted(steps)
    fragments = [steps[step]["emitted"] for step in ordered_steps]
    missing_text = sum(1 for fragment in fragments if fragment is None)
    generated = "".join(fragment for fragment in fragments if fragment)
    mapping = align(generated, reference)

    # Walk the generated string back to the step that produced each character.
    char_step: list[int] = []
    for index, step in enumerate(ordered_steps):
        fragment = fragments[index] or ""
        char_step.extend([step] * len(fragment))

    near_dropped, near_repeated = neighbourhood_flags(generated, reference, mapping, window)

    scored: list[dict[str, Any]] = []
    inserted = 0
    unbounded = 0
    for position, emitted_index in enumerate(char_step):
        step = emitted_index
        reference_index = mapping[position] if position < len(mapping) else None
        readout = combine(select(steps[step]["heads"], layers, heads), aggregate)
        if readout is None:
            continue
        if reference_index is None:
            inserted += 1
            continue
        truth = lines[reference_index] if reference_index < len(lines) else -1
        if truth < 0:
            unbounded += 1
            continue
        # A span covering several reference characters has no single true line.  The
        # first is used, and the step is counted below (``ambiguous_steps``) so the
        # convention is visible rather than assumed away.
        scored.append(
            {
                "step": step,
                "layer": readout.get("layer"),
                "head": readout.get("head"),
                "truth": truth,
                "pred": int(readout["argmax_line"]),
                "confidence": float(readout["top_line_mass"]),
                # The page's line count travels with the row so the scale-free
                # confidence unit can be computed per character.
                "regions": max(1, len(ordered)),
                "background": float(readout.get("background_mass", 0.0)),
                "m_t": float(readout.get("m_t", 0.0)),
                "in_line_pos": float(readout.get("in_line_pos", -1.0)),
                "expected_pos": (
                    expected_in_line_pos(ordered[truth], characters[reference_index]["bbox"])[0]
                    if characters[reference_index].get("bbox")
                    else -1.0
                ),
                "reference_index": reference_index,
                # Which failure neighbourhood this character sits in, so the readout can
                # be reported where a line constraint would actually have to work.
                "near_dropped": near_dropped[position],
                "near_repeated": near_repeated[position],
            }
        )

    # A step whose emitted span maps to more than one reference position is ambiguous:
    # the mapping from character to step is many-to-one, so its single line label is a
    # convention.  Counted per step, not per character.
    per_step: dict[int, set[int]] = {}
    for position, step in enumerate(char_step):
        if position < len(mapping) and mapping[position] is not None:
            per_step.setdefault(step, set()).add(int(mapping[position]))
    ambiguous = sum(1 for indices in per_step.values() if len(indices) > 1)

    # End-to-end check on the step-to-token mapping.  The characters attributed to the
    # observed steps must be a *suffix* of what the model actually produced: step 1 is
    # the first decode step, and the token sampled from the prefill contributes a
    # leading piece that no step ever observes.  If the two do not line up this way,
    # the mapping is off and every character below is scored against the wrong step --
    # which produces a complete set of plausible numbers and no error at all.
    observed = "".join(fragment for fragment in fragments if fragment)
    emitted_matches = int(prediction.endswith(observed))

    return {
        "page_id": report.get("page_id"),
        "steps": len(ordered_steps),
        "steps_missing_text": missing_text,
        "scored": len(scored),
        "inserted": inserted,
        "ambiguous_steps": ambiguous,
        "unbounded_chars": unbounded,
        "generated_chars": len(generated),
        "reference_chars": len(reference),
        "prediction_chars": len(prediction),
        "emitted_matches_prediction": emitted_matches,
        "num_regions": len(ordered),
        "rows": scored,
    }


def accuracy(rows: list[dict[str, Any]]) -> float | None:
    return sum(1 for row in rows if row["pred"] == row["truth"]) / len(rows) if rows else None


def confidence_value(row: dict[str, Any], mode: str) -> float:
    """The step's confidence in the unit the mode defines.

    ``uniform_multiple`` multiplies by the page's line count because the absolute share is
    not comparable across pages: with ``N`` lines nothing exceeds ``1/N`` under a uniform
    distribution, so a fixed bar is far harder to clear on a page with more lines.  That
    showed up as coverage correlating -0.65 with the line count on the stage-1 subset.
    """

    if mode == "uniform_multiple":
        return float(row["confidence"]) * max(1, int(row.get("regions") or 1))
    return float(row["confidence"])


def curves(rows: list[dict[str, Any]], confidence: float, mode: str = "absolute") -> dict[str, Any]:
    """Overall accuracy, the high-confidence subset, and the accuracy-vs-confidence bands."""

    confident = [row for row in rows if confidence_value(row, mode) >= confidence]
    # The band edges have to follow the unit.  They used to be fixed at 0.0-1.0 in steps of
    # 0.2, which is the absolute unit's range and nothing else's: the scale-free unit has a
    # median near 10, so almost every row fell outside every band and the curve came out
    # empty while still looking like a curve.
    edges = (
        (0.0, 2.0, 4.0, 6.0, 8.0) if mode == "uniform_multiple" else (0.0, 0.2, 0.4, 0.6, 0.8)
    )
    step = edges[1] - edges[0]
    bands: dict[str, Any] = {}
    for index, low in enumerate(edges):
        high = low + step
        last = index == len(edges) - 1
        band = [
            row
            for row in rows
            if low <= confidence_value(row, mode) < (float("inf") if last else high)
        ]
        label = f"{low:.1f}+" if last else f"{low:.1f}-{high:.1f}"
        bands[label] = {"steps": len(band), "accuracy": accuracy(band)}
    line_breaks = [row for row in rows if row["row_break"]]
    within = [row for row in rows if not row["row_break"]]
    dropped = [row for row in rows if row["near_dropped"]]
    repeated = [row for row in rows if row["near_repeated"]]
    clean = [row for row in rows if not row["near_dropped"] and not row["near_repeated"]]
    errors = [
        abs(row["in_line_pos"] - row["expected_pos"])
        for row in rows
        if row["in_line_pos"] >= 0 and row["expected_pos"] >= 0 and row["truth"] == row["pred"]
    ]
    return {
        "scored": len(rows),
        "accuracy": accuracy(rows),
        "confident_steps": len(confident),
        "confident_share": len(confident) / len(rows) if rows else None,
        "confident_accuracy": accuracy(confident),
        "bands": bands,
        "at_row_break": {"chars": len(line_breaks), "accuracy": accuracy(line_breaks)},
        "within_row": {"chars": len(within), "accuracy": accuracy(within)},
        # The readout where a line constraint would have to earn its keep.  If the
        # attention is least reliable near a dropped or repeated stretch, a tracker
        # built on it inherits the failure it was meant to fix.
        "near_dropped": {
            "chars": len(dropped),
            "accuracy": accuracy(dropped),
            "mean_confidence": (
                sum(row["confidence"] for row in dropped) / len(dropped) if dropped else None
            ),
        },
        "near_repeated": {
            "chars": len(repeated),
            "accuracy": accuracy(repeated),
            "mean_confidence": (
                sum(row["confidence"] for row in repeated) / len(repeated) if repeated else None
            ),
        },
        "clean": {"chars": len(clean), "accuracy": accuracy(clean)},
        "in_line_error": sum(errors) / len(errors) if errors else None,
        "mean_visual_mass": (
            sum(row["m_t"] for row in rows) / len(rows) if rows else None
        ),
    }


def add_row_breaks(rows: list[dict[str, Any]]) -> None:
    """Mark the steps where the true line differs from the previous scored step's.

    Line changes are where a tracker has to make a decision, and they are rare enough
    that an overall accuracy can be respectable while every change is wrong.
    """

    previous: int | None = None
    for row in sorted(rows, key=lambda item: item["step"]):
        row["row_break"] = previous is not None and row["truth"] != previous
        previous = row["truth"]


def baselines(rows: list[dict[str, Any]], pages: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """The two simple baselines the plan asks for, on the same steps.

    ``stay_previous_line``  predict the previous scored step's true line.  This is the
        prior "reading mostly stays on one line", and it is the bar the attention has
        to clear to be worth anything -- on a page of long columns it is strong.
    ``uniform_scan``  spread the lines evenly over the scored steps, i.e. read the page
        at a constant rate.  This is the shape a page-level scan would have.

    Both are computed on exactly the steps the attention was scored on, so the
    comparison is paired.
    """

    ordered = sorted(rows, key=lambda item: item["step"])
    stay_hits = 0
    stay_total = 0
    previous: int | None = None
    for row in ordered:
        if previous is not None:
            stay_total += 1
            stay_hits += int(previous == row["truth"])
        previous = row["truth"]

    scan_hits = 0
    for page_id, page_rows in pages.items():
        page_rows = sorted(page_rows, key=lambda item: item["step"])
        lines = sorted({row["truth"] for row in page_rows})
        if len(lines) < 2 or len(page_rows) < 2:
            continue
        low, high = lines[0], lines[-1]
        span = high - low
        for index, row in enumerate(page_rows):
            predicted = low + round(span * index / (len(page_rows) - 1))
            scan_hits += int(predicted == row["truth"])
    return {
        "stay_previous_line": stay_hits / stay_total if stay_total else None,
        "stay_previous_line_steps": stay_total,
        "uniform_scan": scan_hits / len(rows) if rows else None,
        "uniform_scan_steps": len(rows),
    }


def summarise(
    rows: list[dict[str, Any]], confidence: float, page_rows: dict, mode: str = "absolute"
) -> dict[str, Any]:
    add_row_breaks(rows)
    base = baselines(rows, page_rows)
    result = curves(rows, confidence, mode)
    result["baselines"] = base
    best = base["stay_previous_line"]
    result["beats_stay_previous_line"] = (
        None if result["accuracy"] is None or best is None else result["accuracy"] > best
    )
    result["gate"] = {
        "accuracy_target": GATE_ACCURACY,
        "coverage_target": GATE_COVERAGE,
        "confident_accuracy": result["confident_accuracy"],
        "confident_share": result["confident_share"],
        "passes": bool(
            result["confident_accuracy"] is not None
            and result["confident_accuracy"] >= GATE_ACCURACY
            and (result["confident_share"] or 0.0) >= GATE_COVERAGE
            and result["beats_stay_previous_line"]
        ),
    }
    return result


def by_selection(pages: list[dict[str, Any]], layers, heads) -> dict[str, Any]:
    """Per-layer and per-head accuracy, which is what the layer/head choice is made on.

    Computed on the *selection* pages only by the caller.  Reporting it on the check
    pages would make the held-out set a second selection set, which is the thing the
    pre-registered split exists to prevent.
    """

    per_layer: dict[str, dict[str, int]] = {}
    per_head: dict[str, dict[str, int]] = {}
    for page in pages:
        for row in page["rows"]:
            step_rows_raw = page["raw_by_step"].get(row["step"], [])
            for candidate in step_rows_raw:
                if layers is not None and candidate["layer"] not in layers:
                    continue
                if heads is not None and candidate["head"] not in heads:
                    continue
                for bucket, key in (
                    (per_layer, str(candidate["layer"])),
                    (per_head, f"{candidate['layer']}:{candidate['head']}"),
                ):
                    entry = bucket.setdefault(key, {"steps": 0, "hits": 0})
                    entry["steps"] += 1
                    entry["hits"] += int(int(candidate["argmax_line"]) == row["truth"])
    return {
        "per_layer": {
            key: {"steps": value["steps"], "accuracy": value["hits"] / value["steps"]}
            for key, value in sorted(per_layer.items(), key=lambda item: int(item[0]))
        },
        "per_head": {
            key: {"steps": value["steps"], "accuracy": value["hits"] / value["steps"]}
            for key, value in sorted(per_head.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports = {row["page_id"]: row for row in load_jsonl(args.probe)}
    predictions = {row["page_id"]: row for row in load_jsonl(args.predictions)}
    manifest = {row["page_id"]: row for row in load_jsonl(args.manifest)}
    select_pages = load_pages(args.select_pages)
    check_pages = load_pages(args.check_pages)

    page_ids = [
        page_id
        for page_id in reports
        if page_id in predictions and page_id in manifest
    ]
    pages: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for page_id in page_ids:
        record = manifest[page_id]
        characters = record.get("characters")
        if not characters:
            # No character box channel means no line truth.  Scored as a gap, not as
            # a page where the attention happened to be wrong.
            skipped.append({"page_id": page_id, "reason": "no character box channel"})
            continue
        page = score_page(
            reports[page_id],
            predictions[page_id]["reference"],
            predictions[page_id].get("prediction", ""),
            record.get("regions") or [],
            characters,
            layers=args.layers,
            heads=args.heads,
            aggregate=args.aggregate,
            window=args.neighbourhood,
        )
        page["raw_by_step"] = steps_for_head_selection(reports[page_id])
        pages.append(page)

    def subset(ids):
        if ids is None:
            return pages
        wanted = set(ids)
        return [page for page in pages if page["page_id"] in wanted]

    report: dict[str, Any] = {
        "probe": str(args.probe),
        "predictions": str(args.predictions),
        "manifest": str(args.manifest),
        "layers": args.layers,
        "heads": args.heads,
        "aggregate": args.aggregate,
        "confidence": args.confidence,
        "confidence_mode": args.confidence_mode,
        "neighbourhood": args.neighbourhood,
        "pages_scored": len(pages),
        "pages_without_char_channel": skipped,
    }
    select_subset = subset(select_pages) if select_pages is not None else pages
    report["selection"] = by_selection(select_subset, args.layers, args.heads)
    for name, ids in (("select", select_pages), ("check", check_pages)):
        chosen = subset(ids)
        if not chosen:
            continue
        rows: list[dict[str, Any]] = []
        for page in chosen:
            rows.extend(page["rows"])
        per_page = {page["page_id"]: page["rows"] for page in chosen}
        report[name] = summarise(rows, args.confidence, per_page, args.confidence_mode)
        report[name]["pages"] = [page["page_id"] for page in chosen]
        # Per page, because a half-to-half difference can be page composition rather than
        # a property of the readout, and the plan asks for grouped statistics when pages
        # from one book are strongly correlated.  An aggregate alone cannot tell the two
        # apart.
        report[name]["per_page"] = {
            page["page_id"]: {
                "scored": page["scored"],
                "accuracy": accuracy(page["rows"]),
                "inserted": page["inserted"],
                "regions": page["num_regions"],
            }
            for page in sorted(chosen, key=lambda item: item["page_id"])
        }
        report[name]["alignment"] = {
            "scored_chars": sum(page["scored"] for page in chosen),
            "inserted_chars": sum(page["inserted"] for page in chosen),
            "ambiguous_steps": sum(page["ambiguous_steps"] for page in chosen),
            "unbounded_chars": sum(page["unbounded_chars"] for page in chosen),
            "steps_missing_text": sum(page["steps_missing_text"] for page in chosen),
            "total_steps": sum(page["steps"] for page in chosen),
            "generated_chars": sum(page["generated_chars"] for page in chosen),
            "prediction_chars": sum(page["prediction_chars"] for page in chosen),
            "pages_matching_prediction": sum(
                page["emitted_matches_prediction"] for page in chosen
            ),
        }
    if "check" not in report and "select" not in report and pages:
        # No split was given, so score everything -- but only when there is something
        # to score.  An unpaged report of zeros would read as "no signal" when it
        # actually means "no truth to compare against".
        rows = [row for page in pages for row in page["rows"]]
        per_page = {page["page_id"]: page["rows"] for page in pages}
        report["all"] = summarise(rows, args.confidence, per_page, args.confidence_mode)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
