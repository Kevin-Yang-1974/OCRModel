"""Tests for the share of the bias that lands on the next line.

The applied line is one decoding step stale, and the offline replay puts that at 8.5% of
characters -- nearly all of them the first three of their own line. Covering the next line aims at
the right column on those; on every other character it adds a wrong column at this share, which is
the cost and is not measurable offline. Two properties have to hold for the arm to be worth
running: the current line keeps its full bias, and the correction divides out exactly what was
added, on both lines.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from layout_ocr.attention_probe import AttentionProbe
from layout_ocr.attention_routing import AttentionRouting

IMAGE_TOKEN_ID = 3
VISUAL_COUNT = 4
GRID = torch.tensor([[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]])

# Two stacked rows of the patch grid: region 0 owns tokens 0-1, region 1 owns tokens 2-3.
REGIONS = [
    {"reading_order": 0, "bbox": [0.0, 0.0, 1.0, 0.5]},
    {"reading_order": 1, "bbox": [0.0, 0.5, 1.0, 1.0]},
]


class _Tracked:
    """The probe publishes through ``observe`` and reads ``line`` back off the same object."""

    def __init__(self, line=0):
        self.line = line

    def set_page(self):
        self.line = -1

    def observe(self, line, confidence, **_kwargs):
        pass

    def report(self):
        return {"line": self.line}


def _bridge():
    return SimpleNamespace(last_patch_positions=GRID)


def _routing(**kwargs):
    line = kwargs.pop("line", 0)
    runtime = AttentionRouting(
        _bridge(), kwargs.pop("bias", 1.0), IMAGE_TOKEN_ID, box_source="line",
        line_source="tracked", tracked=_Tracked(line), **kwargs
    )
    runtime.set_page("p", None, 1, torch.tensor([[IMAGE_TOKEN_ID] * VISUAL_COUNT]))
    runtime.regions = [dict(region) for region in REGIONS]
    # ``set_page`` clears the estimate -- a new page invalidates it -- so the line the test wants
    # is set after, the way the first accepted step of a page would set it.
    runtime.tracked.line = line
    return runtime


def test_the_current_line_keeps_its_full_bias():
    plain = _routing(next_line_scale=0.0)
    shared = _routing(next_line_scale=0.5)
    plain_mask = plain._mask_for(0, 4, torch.device("cpu"), torch.float32)
    shared_mask = shared._mask_for(0, 4, torch.device("cpu"), torch.float32)
    assert plain_mask is not None and shared_mask is not None
    # Region 0 is the first two tokens; the share must not dilute them.
    assert shared_mask[0, 0, 0, :2].tolist() == plain_mask[0, 0, 0, :2].tolist()


def test_the_next_line_carries_the_configured_share():
    runtime = _routing(next_line_scale=0.5)
    mask = runtime._mask_for(0, 4, torch.device("cpu"), torch.float32)
    assert mask is not None
    assert mask[0, 0, 0, :2].tolist() == [1.0, 1.0]
    assert mask[0, 0, 0, 2:].tolist() == [0.5, 0.5]
    assert runtime.report()["mean_next_line_hits"] == 2.0


def test_the_last_line_has_no_next_line_to_share_with():
    runtime = _routing(line=1, next_line_scale=0.5)
    mask = runtime._mask_for(0, 4, torch.device("cpu"), torch.float32)
    assert mask is not None
    assert mask[0, 0, 0, 2:].tolist() == [1.0, 1.0]
    assert mask[0, 0, 0, :2].tolist() == [0.0, 0.0]
    assert runtime.report()["mean_next_line_hits"] == 0.0


def test_a_negative_share_is_refused():
    with pytest.raises(ValueError, match="cannot be negative"):
        _routing(next_line_scale=-0.5)


def _probe(**kwargs):
    runtime = AttentionProbe(
        _bridge(),
        IMAGE_TOKEN_ID,
        layers=(0,),
        heads=(0,),
        tracked=_Tracked(0),
        **kwargs,
    )
    runtime.visual_count = VISUAL_COUNT
    runtime.visual_start = 1
    runtime._geometry[0] = (1, 1, 1, 1.0)
    runtime._regions = [dict(region) for region in REGIONS]
    runtime.num_regions = len(REGIONS)
    return runtime


def test_the_correction_divides_out_both_lines():
    """The added logit per token is B on the aimed line and B*s on the next; both come off."""

    runtime = _probe(routing_bias=1.0, correct_confidence=True, next_line_scale=0.5)
    added = runtime._bias_added_per_token(0)
    assert added is not None
    assert added.tolist() == [1.0, 1.0, 0.5, 0.5]


def test_the_correction_matches_the_routing_mask_it_inverts():
    # The two modules have to agree on what was added, or the correction is subtracting a
    # quantity the forward never applied.
    routing = _routing(next_line_scale=0.5)
    mask = routing._mask_for(0, 4, torch.device("cpu"), torch.float32)
    probe = _probe(routing_bias=1.0, correct_confidence=True, next_line_scale=0.5)
    added = probe._bias_added_per_token(0)
    assert added is not None
    assert mask is not None
    assert added.tolist() == mask[0, 0, 0].tolist()


def test_the_correction_recovers_the_distribution_the_two_line_bias_displaced():
    """Keys are set so the logits are the unbiased ones plus the routing's own added logits."""

    bias, share = 1.0, 0.5
    unbiased = [1.0, 0.0, 0.0, 0.0]
    # Head 0 reads KV head 0 with a unit query, so the key values *are* the logits.
    keys = torch.tensor(
        [[[unbiased[0] + bias], [unbiased[1] + bias], [unbiased[2] + bias * share],
          [unbiased[3] + bias * share]]]
    )
    runtime = _probe(routing_bias=bias, correct_confidence=True, next_line_scale=share)
    runtime._visual_keys[0] = keys
    runtime._prompt_text_keys[0] = torch.zeros(1, 1, 1, 1)
    runtime.owners = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    runtime.steps = 1
    runtime._observe_step(0, 1, torch.tensor([[[[1.0]]]]), None)
    row = runtime.report()["steps"][0]["heads"][0]
    reference = [math.exp(value) for value in unbiased]
    total = sum(reference)
    assert row["line_probs"][0] == pytest.approx(sum(reference[:2]) / total, rel=1e-6)
    assert row["line_probs"][1] == pytest.approx(sum(reference[2:]) / total, rel=1e-6)
