"""Tests for the tracked line state that ``gtmap_track`` is built on.

The arm exists to separate tracking error from detection error, so what has to hold is that the
state it drives is exactly what the attention said and nothing else: no reference text, no
carrying a stale line through a low-confidence step, and a lag that comes from where the hooks sit
rather than from a counter someone can get off by one.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from layout_ocr.attention_routing import AttentionRouting, install_attention_routing
from layout_ocr.attention_tracking import (
    DEFAULT_CONFIDENCE,
    TrackedLineState,
    aggregate_line_estimate,
)

IMAGE_TOKEN_ID = 3
PROMPT_LENGTH = 6
BIAS = 1.0

LINE_LEFT = {"bbox": [0.0, 0.0, 0.5, 1.0], "reading_order": 0}
LINE_RIGHT = {"bbox": [0.5, 0.0, 1.0, 1.0], "reading_order": 1}


def _bridge():
    positions = torch.tensor(
        [[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]], dtype=torch.float32
    )
    return SimpleNamespace(last_patch_positions=positions)


def _runtime(state, bias=BIAS):
    runtime = AttentionRouting(
        _bridge(), bias, IMAGE_TOKEN_ID, None, "synced", "line", "tracked", state
    )
    runtime.set_page(
        "p0",
        None,  # no character channel: the tracked source must not need one
        PROMPT_LENGTH,
        torch.tensor([[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2]]),
        regions=[dict(LINE_LEFT), dict(LINE_RIGHT)],
    )
    return runtime


def _step(runtime, position):
    kwargs = {"cache_position": torch.tensor([position])}
    runtime.hook(None, (torch.zeros(1, 1, 8),), kwargs)
    return kwargs


def _row(head, line_probs):
    return {"layer": 8, "head": head, "line_probs": line_probs}


def test_a_confident_estimate_is_kept():
    state = TrackedLineState(confidence=6.0)
    # Three regions, mass 0.5 on line 1: 0.5 * 3 = 1.5x uniform, under a bar of 6.
    state.observe(1, 1.5)
    assert state.line == -1
    assert state.gated == 1
    state.observe(1, 6.0)
    assert state.line == 1
    assert state.accepted == 1
    assert state.confidence == pytest.approx(6.0)


def test_a_low_confidence_step_drops_the_estimate_rather_than_keeping_it():
    """Carrying a stale line would keep aiming where the readout just said it is unsure."""

    state = TrackedLineState(confidence=6.0)
    state.observe(1, 9.0)
    assert state.line == 1
    state.observe(0, 1.0)
    assert state.line == -1
    assert state.gated == 1


def test_the_state_forgets_the_previous_page():
    state = TrackedLineState(confidence=6.0)
    state.observe(1, 9.0)
    state.set_page()
    assert state.line == -1
    assert state.observations == 0


def test_the_default_bar_is_the_one_stage_1_registered():
    assert DEFAULT_CONFIDENCE == 6.0
    assert TrackedLineState().threshold == 6.0


def test_the_estimate_averages_the_heads_before_taking_the_argmax():
    """Averaging the argmaxes would throw away the spread that carries the confidence."""

    # Head 0 says line 0 with most of its mass, head 1 says line 1 with a little; the mean should
    # still favour line 0 and the confidence should reflect the dilution.
    rows = [
        _row(0, [0.8, 0.1, 0.1]),
        _row(1, [0.2, 0.3, 0.5]),
    ]
    line, confidence = aggregate_line_estimate(rows, num_regions=2)
    assert line == 0
    # Mean distribution over two lines plus background: [0.5, 0.2, 0.3], so line 0 wins at 0.5.
    assert confidence == pytest.approx(0.5 * 2)


def test_the_estimate_only_uses_the_named_heads():
    rows = [
        _row(0, [0.9, 0.05, 0.05]),
        _row(7, [0.0, 0.9, 0.1]),
    ]
    line, _ = aggregate_line_estimate(rows, num_regions=2, heads=(7,))
    assert line == 1
    line, _ = aggregate_line_estimate(rows, num_regions=2, heads=(0,))
    assert line == 0


def test_the_estimate_refuses_a_row_without_a_line_distribution():
    assert aggregate_line_estimate([], num_regions=2) == (None, 0.0)
    assert aggregate_line_estimate([{"head": 0}], num_regions=2) == (None, 0.0)
    # No regions to localize against: there is no line to name.
    assert aggregate_line_estimate([_row(0, [0.5, 0.5])], num_regions=0) == (None, 0.0)


def test_background_winning_the_argmax_does_not_become_a_line():
    """The last slot is everything off every line; it can win the mean but is not a line."""

    rows = [_row(0, [0.1, 0.1, 0.8])]
    line, _ = aggregate_line_estimate(rows, num_regions=2)
    assert line == 0


def test_the_tracked_source_biases_the_line_the_state_names():
    state = TrackedLineState(confidence=6.0)
    runtime = _runtime(state)
    # Nothing has been observed yet, so the first step is left unbiased and counted as gated --
    # the estimate cannot exist before the probe has run once.
    assert "attention_mask" not in _step(runtime, PROMPT_LENGTH)
    assert runtime.gated == 1

    state.observe(1, 9.0)  # line 1 is the right half
    runtime._cache_key = None
    mask = _step(runtime, PROMPT_LENGTH + 1)["attention_mask"]
    assert mask[0, 0, 0, 1:5].tolist() == [0.0, BIAS, 0.0, BIAS]

    # The state moving without the pointer moving is the whole point: no reference text is
    # involved, and the next step follows the estimate.
    state.observe(0, 9.0)
    runtime._cache_key = None
    mask = _step(runtime, PROMPT_LENGTH + 2)["attention_mask"]
    assert mask[0, 0, 0, 1:5].tolist() == [BIAS, 0.0, BIAS, 0.0]


def test_the_tracked_source_needs_no_reference_and_no_character_boxes():
    state = TrackedLineState(confidence=6.0)
    runtime = _runtime(state)
    assert runtime.characters is None
    state.observe(0, 9.0)
    # The synced pointer is asked for and has nothing to work with, which is the intended state:
    # this arm must not be able to use the reference even accidentally.
    runtime.observe_inputs(
        None,
        (),
        {"input_ids": torch.tensor([[7]]), "cache_position": torch.tensor([PROMPT_LENGTH])},
    )
    assert runtime.position == 0


def test_a_gated_step_is_counted_apart_from_a_missing_box():
    state = TrackedLineState(confidence=6.0)
    runtime = _runtime(state)
    _step(runtime, PROMPT_LENGTH)
    assert runtime.gated == 1
    assert runtime.missing == 0
    report = runtime.report()
    assert report["gated_steps"] == 1
    assert report["line_source"] == "tracked"


def test_the_report_carries_the_gate_rate():
    state = TrackedLineState(confidence=6.0)
    state.observe(0, 9.0)
    state.observe(0, 1.0)
    state.observe(1, 9.0)
    report = state.report()
    assert report["observations"] == 3
    assert report["accepted"] == 2
    assert report["gated"] == 1
    assert report["accepted_fraction"] == pytest.approx(2 / 3)


def _predicted_runtime(state, boxes):
    """A tracked runtime whose map is the detector's boxes, with no regions at all."""

    runtime = AttentionRouting(
        _bridge(), BIAS, IMAGE_TOKEN_ID, None, "synced", "line", "tracked", state, "predicted"
    )
    runtime.set_page(
        "p0",
        None,
        PROMPT_LENGTH,
        torch.tensor([[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2]]),
        regions=None,  # no annotation anywhere on this path
        predicted_lines=boxes,
    )
    return runtime


def test_the_predicted_map_takes_the_boxes_from_the_detector():
    """Nothing in the tracked path reads the annotation once the map is predicted."""

    state = TrackedLineState(confidence=6.0)
    runtime = _predicted_runtime(state, [LINE_LEFT["bbox"], LINE_RIGHT["bbox"]])
    assert runtime.regions == []
    state.observe(1, 9.0)
    mask = _step(runtime, PROMPT_LENGTH)["attention_mask"]
    assert mask[0, 0, 0, 1:5].tolist() == [0.0, BIAS, 0.0, BIAS]
    state.observe(0, 9.0)
    runtime._cache_key = None
    mask = _step(runtime, PROMPT_LENGTH + 1)["attention_mask"]
    assert mask[0, 0, 0, 1:5].tolist() == [BIAS, 0.0, BIAS, 0.0]
    assert runtime.report()["line_map"] == "predicted"


def test_a_line_index_past_the_predicted_boxes_is_a_missing_box():
    """The detector may have found fewer lines than the tracker names."""

    state = TrackedLineState(confidence=6.0)
    runtime = _predicted_runtime(state, [LINE_LEFT["bbox"]])
    state.observe(5, 9.0)
    assert "attention_mask" not in _step(runtime, PROMPT_LENGTH)
    assert runtime.missing == 1
    assert runtime.gated == 0


def test_the_probe_expresses_its_readout_in_the_predicted_boxes():
    """The two sides must agree on what index i means, or the bias aims at another line."""

    from layout_ocr.attention_probe import BOX_MAPS, AttentionProbe

    assert BOX_MAPS == ("regions", "predicted")
    probe = AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(8,), box_map="predicted")
    probe.set_page(
        "p0",
        [{"bbox": [0.0, 0.0, 1.0, 1.0], "reading_order": 0}],  # annotation present but ignored
        PROMPT_LENGTH,
        torch.tensor([[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2]]),
        predicted_lines=[LINE_RIGHT["bbox"], LINE_LEFT["bbox"]],
    )
    assert probe.num_regions == 2
    assert [region["bbox"] for region in probe._regions] == [
        LINE_RIGHT["bbox"],
        LINE_LEFT["bbox"],
    ]
    assert probe.report()["box_map"] == "predicted"


def test_an_unknown_box_map_is_refused():
    from layout_ocr.attention_probe import AttentionProbe

    with pytest.raises(ValueError, match="box_map must be one of"):
        AttentionProbe(_bridge(), IMAGE_TOKEN_ID, layers=(8,), box_map="vibes")


def test_an_unknown_line_map_is_refused():
    with pytest.raises(ValueError, match="line_map must be one of"):
        AttentionRouting(
            _bridge(), BIAS, IMAGE_TOKEN_ID, None, "synced", "line", "tracked",
            TrackedLineState(), "vibes",
        )


def test_a_tracked_runtime_without_a_state_is_refused():
    with pytest.raises(ValueError, match="needs box_source='line'"):
        AttentionRouting(_bridge(), BIAS, IMAGE_TOKEN_ID, None, "synced", "line", "tracked", None)


def test_a_tracked_runtime_on_a_character_box_is_refused():
    """Tracking picks a line, so it is meaningless where a character is what gets biased."""

    with pytest.raises(ValueError, match="needs box_source='line'"):
        AttentionRouting(
            _bridge(), BIAS, IMAGE_TOKEN_ID, None, "synced", "char", "tracked",
            TrackedLineState(),
        )


def test_an_unknown_line_source_is_refused():
    with pytest.raises(ValueError, match="line_source must be one of"):
        AttentionRouting(_bridge(), BIAS, IMAGE_TOKEN_ID, None, "synced", "line", "telepathy")


class _FakeTextModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 8, layers: int = 2):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(16, hidden_size)
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        )


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.model = torch.nn.Module()
        self.model.language_model = _FakeTextModel()

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens


def test_installation_passes_the_state_through():
    state = TrackedLineState(confidence=6.0)
    runtime, handles = install_attention_routing(
        _FakeModel(), _bridge(), bias=BIAS, pointer="synced", box_source="line",
        line_source="tracked", tracked=state,
    )
    assert runtime.tracked is state
    assert runtime.line_source == "tracked"
    for handle in handles:
        handle.remove()


# -- the switch margin ---------------------------------------------------
#
# The bias inflates the mass of the line already in use, so it supplies hysteresis by accident:
# dividing it out raised the change rate from 0.060 to 0.123 per accepted step and deletions from
# 771 to 1066. The margin makes that stability deliberate -- and it is measured against what the
# *current* line still holds, not against the bar, because a penalty on the bar was tried first
# and does nothing: the switches that cost are not low-confidence ones.


def test_without_a_held_mass_the_margin_is_not_applied():
    # A caller that cannot supply the distribution gets the recorded behaviour rather than an
    # invented comparison.
    state = TrackedLineState(6.0, switch_bar=3.0)
    state.observe(3, 7.0, num_regions=10)
    state.observe(4, 7.0, num_regions=10)
    assert state.line == 4
    assert state.held == 0


def test_a_line_holding_less_than_the_margin_is_overruled():
    state = TrackedLineState(6.0, switch_bar=2.0)
    state.observe(3, 7.0, held_mass=0.5, num_regions=10)
    # Candidate mass 0.8 against the held 0.5 * 2 = 1.0: not enough to move.
    state.observe(4, 8.0, held_mass=0.5, num_regions=10)
    assert state.line == 3
    assert state.held == 1


def test_a_line_holding_more_than_the_margin_takes_over():
    state = TrackedLineState(6.0, switch_bar=2.0)
    state.observe(3, 7.0, held_mass=0.5, num_regions=10)
    # Candidate mass 1.2 against 1.0: it moves.
    state.observe(4, 12.0, held_mass=0.5, num_regions=10)
    assert state.line == 4


def test_the_same_line_never_needs_a_margin():
    state = TrackedLineState(6.0, switch_bar=3.0)
    state.observe(3, 7.0, held_mass=0.9, num_regions=10)
    state.observe(3, 7.0, held_mass=0.9, num_regions=10)
    assert state.line == 3
    assert state.held == 0


def test_the_first_line_of_a_page_is_never_held_back():
    state = TrackedLineState(6.0, switch_bar=3.0)
    state.observe(5, 7.0, held_mass=0.0, num_regions=10)
    assert state.line == 5


def test_a_gated_step_still_clears_the_line():
    state = TrackedLineState(6.0, switch_bar=3.0)
    state.observe(3, 7.0, held_mass=0.5, num_regions=10)
    state.observe(3, 1.0, held_mass=0.5, num_regions=10)
    assert state.line == -1
    assert state.gated == 1
    assert state.held == 0


def test_a_switch_bar_below_one_is_refused():
    with pytest.raises(ValueError, match="at least 1.0"):
        TrackedLineState(6.0, switch_bar=0.5)


def test_the_switch_bar_travels_with_the_report():
    state = TrackedLineState(6.0, switch_bar=1.5)
    assert state.report()["switch_bar"] == 1.5
    assert state.report()["held"] == 0
