"""Tests for the offline tracker replay.

The replay is the only place the bias-corrected confidence can be checked before it costs a GPU
run, so two things have to hold: the correction has to invert the bias exactly, and the replay has
to reconstruct which line was biased at each step. Both are properties of arithmetic here rather
than of the model, which is why they can be pinned.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from replay_tracker_offline import (  # noqa: E402
    correct_head,
    policy_hold,
    policy_mono,
    policy_mono_hold,
    policy_recorded,
    readout,
    replay_page,
)


def test_correction_inverts_the_bias_exactly():
    # The bias multiplies one line's mass by e^B before renormalising; dividing it back out has
    # to return the original distribution, not something merely close to it.
    original = [0.30, 0.20, 0.10, 0.40]
    bias, line = 1.0, 0
    factor = math.exp(bias)
    scaled = list(original)
    scaled[line] *= factor
    total = sum(scaled)
    contaminated = [value / total for value in scaled]
    recovered = correct_head(contaminated, line, bias)
    for got, want in zip(recovered, original):
        assert got == pytest.approx(want, abs=1e-12)


def test_correction_is_the_identity_without_a_biased_line():
    distribution = [0.5, 0.25, 0.25]
    assert correct_head(distribution, -1, 1.0) == distribution
    assert correct_head(distribution, 0, 0.0) == distribution


def test_correction_does_not_touch_an_out_of_range_line():
    distribution = [0.5, 0.5]
    assert correct_head(distribution, 7, 1.0) == distribution


def test_correction_lowers_the_mass_the_bias_inflated():
    # The measured defect, in miniature: the bias makes the line it aimed at look more certain.
    contaminated = [0.6, 0.2, 0.2]
    corrected = correct_head(contaminated, 0, 1.0)
    assert corrected[0] < contaminated[0]
    assert sum(corrected) == pytest.approx(1.0)


def test_policy_recorded_drops_the_line_below_the_bar():
    assert policy_recorded(3, 4, 1.0, 6.0) == -1
    assert policy_recorded(3, 4, 7.0, 6.0) == 4


def test_policy_hold_keeps_the_last_line():
    assert policy_hold(3, 4, 1.0, 6.0) == 3
    assert policy_hold(3, 4, 7.0, 6.0) == 4


def test_policy_mono_refuses_to_walk_backwards():
    # Reading order is monotone, so a lower line is refused even at high confidence.
    assert policy_mono(5, 2, 9.0, 6.0) == 5
    assert policy_mono(5, 6, 9.0, 6.0) == 6
    assert policy_mono(5, 6, 1.0, 6.0) == -1


def test_policy_mono_hold_keeps_the_line_when_a_step_is_refused():
    assert policy_mono_hold(5, 2, 9.0, 6.0) == 5
    assert policy_mono_hold(5, 2, 1.0, 6.0) == 5


def _head(layer: int, head: int, line_probs: list[float]) -> dict:
    best = max(range(len(line_probs) - 1), key=lambda index: line_probs[index])
    return {
        "layer": layer,
        "head": head,
        "line_probs": line_probs,
        "argmax_line": best,
        "top_line_mass": line_probs[best],
        "background_mass": line_probs[-1],
        "m_t": 0.4,
        "in_line_pos": 0.5,
    }


def test_readout_corrects_each_head_with_its_own_normaliser():
    # The correction is per head because the renormaliser depends on that head's own mass on the
    # biased line; applying one head's factor to a combined average would be approximate.
    heads = [_head(8, 2, [0.5, 0.25, 0.25]), _head(8, 3, [0.2, 0.7, 0.1])]
    biased = 1
    plain = readout(heads, (8,), (2, 3), "mean", biased, 1.0, False)
    corrected = readout(heads, (8,), (2, 3), "mean", biased, 1.0, True)
    assert plain is not None and corrected is not None
    # Both heads were biased on line 1, so line 1 loses mass and the others gain it.
    assert corrected["line_probs"][1] < plain["line_probs"][1]
    assert corrected["line_probs"][0] > plain["line_probs"][0]


def _page_setup():
    """One page, two stacked columns, two characters, three decoding steps."""

    regions = [
        {"reading_order": 0, "bbox": [0.0, 0.0, 1.0, 0.5], "writing_direction": "vertical_rtl"},
        {"reading_order": 1, "bbox": [0.0, 0.5, 1.0, 1.0], "writing_direction": "vertical_rtl"},
    ]
    characters = [
        {"bbox": [0.4, 0.1, 0.6, 0.4]},  # inside region 0
        {"bbox": [0.4, 0.6, 0.6, 0.9]},  # inside region 1
    ]
    record = {"page_id": "p", "regions": regions, "characters": characters, "page_text": "甲一"}
    # Step 1 sees the query that emits "甲" (on line 0), step 2 emits "一" (on line 1).
    steps = [
        {"step": 1, "heads": [_head(8, 2, [0.7, 0.1, 0.2])], "emitted": "甲"},
        {"step": 2, "heads": [_head(8, 2, [0.1, 0.7, 0.2])], "emitted": "一"},
        {"step": 3, "heads": [_head(8, 2, [0.1, 0.2, 0.7])], "emitted": ""},
    ]
    report = {"page_id": "p", "num_regions": 2, "steps": steps, "layers": [8], "heads": [2]}
    return report, record


# The bar is in units of a uniform distribution over the page's lines, so it cannot exceed the
# line count: with two lines the most concentrated readout scores 2.0. A bar of 6 on a two-line
# page is therefore unreachable, which is why these tests use 1.0.
BAR = 1.0


def test_replay_aims_the_first_step_at_nothing():
    # The state starts empty, so step 1 biases no line -- the one-step lag is structural.
    report, record = _page_setup()
    outcome = replay_page(
        report, record, record["page_text"], layers=(8,), heads=(2,), bias=1.0, bars=[BAR]
    )
    assert f"recorded@{BAR:g}" in outcome["policies"]
    entry = outcome["policies"][f"recorded@{BAR:g}"]
    # Two characters are scorable; only the second can sit on a biased step.
    assert entry["chars"] == 2
    assert entry["biased_chars"] == 1


def test_replay_applied_line_is_the_previous_step_estimate():
    # Step 2 emits the character on line 1, but the line applied at step 2 is step 1's estimate,
    # which had not moved off line 0 yet. The lag is the tracker's cost, so the replay has to
    # show it rather than quietly scoring the fresh readout.
    report, record = _page_setup()
    outcome = replay_page(
        report, record, record["page_text"], layers=(8,), heads=(2,), bias=1.0, bars=[BAR]
    )
    entry = outcome["policies"][f"recorded@{BAR:g}"]
    # Steps 1 and 2 clear the bar; step 3's readout falls on the background slot and does not.
    assert entry["accepted_steps"] == 2
    assert entry["applied_correct"] == 0
    assert entry["accuracy_where_biased"] == 0.0


def test_a_policy_that_never_accepts_reports_no_coverage():
    # A bar nothing can clear has to read as "aimed at nothing", not as a score of zero.
    report, record = _page_setup()
    outcome = replay_page(
        report, record, record["page_text"], layers=(8,), heads=(2,), bias=1.0, bars=[99.0]
    )
    entry = outcome["policies"]["recorded@99"]
    assert entry["accepted_steps"] == 0
    assert entry["coverage"] == 0.0
