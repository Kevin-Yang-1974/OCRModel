"""Tests for the join between the detector and the routing bias.

Two properties carry the whole join: the coordinate normalization, which is what lets the detector
be trained at one resolution and used at another, and the reading order, which both arms depend on
being the same order the annotation uses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from predict_lines_for_routing import normalize_boxes, order_boxes  # noqa: E402


def test_normalizing_is_invariant_to_the_resolution_used():
    """The same page box, whatever size the detector saw.

    A column at x 96..192 on a 1500-wide page is at 0.064..0.128.  Seen at half scale it lands on
    48..96 of 750, which must normalize to the same numbers -- otherwise the detector's training
    resolution would leak into the routing's coordinate space.
    """

    original = normalize_boxes(torch.tensor([[96.0, 145.0, 192.0, 1438.0]]), 1500, 3000)
    halved = normalize_boxes(torch.tensor([[48.0, 72.5, 96.0, 719.0]]), 750, 1500)
    assert original[0] == pytest.approx(halved[0])
    assert original[0] == pytest.approx([0.064, 0.048333333, 0.128, 0.479333333])


def test_normalizing_divides_each_axis_by_its_own_size():
    boxes = normalize_boxes(torch.tensor([[100.0, 300.0, 200.0, 900.0]]), 1000, 3000)
    assert boxes[0] == pytest.approx([0.1, 0.1, 0.2, 0.3])


# Vertical columns differ along x, horizontal lines along y, and the boxes below differ along
# both so that the two directions genuinely disagree.
DIAGONAL = [
    [0.1, 0.1, 0.2, 0.2],
    [0.4, 0.4, 0.5, 0.5],
    [0.7, 0.7, 0.8, 0.8],
]


def test_vertical_columns_are_ordered_right_to_left():
    ordered, scores = order_boxes(DIAGONAL, [0.5, 0.6, 0.7], "vertical_rtl")
    assert [box[0] for box in ordered] == pytest.approx([0.7, 0.4, 0.1])
    # The score travels with its box rather than staying at its original index.
    assert scores == pytest.approx([0.7, 0.6, 0.5])


def test_horizontal_lines_are_ordered_top_to_bottom():
    ordered, scores = order_boxes(DIAGONAL, [0.5, 0.6, 0.7], "horizontal_ltr")
    assert [box[1] for box in ordered] == pytest.approx([0.1, 0.4, 0.7])
    assert scores == pytest.approx([0.5, 0.6, 0.7])


def test_boxes_tied_on_the_sorting_axis_keep_their_order():
    """A tie is not an error; it means the direction does not order these two boxes."""

    tied = [[0.1, 0.0, 0.2, 1.0], [0.4, 0.0, 0.5, 1.0]]
    ordered, _ = order_boxes(tied, [0.5, 0.6], "horizontal_ltr")
    # Both have the same vertical centre, so the sort is stable and the input order survives.
    assert [box[0] for box in ordered] == pytest.approx([0.1, 0.4])


def test_ordering_an_empty_page_is_not_an_error():
    assert order_boxes([], [], "vertical_rtl") == ([], [])
