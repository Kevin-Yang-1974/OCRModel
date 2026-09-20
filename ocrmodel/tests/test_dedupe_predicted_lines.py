"""Tests for the predicted-box deduplication.

The tracked arm selects one box where the static arm biases a union, so a duplicate is a candidate
it can pick. Suppressing the wrong one of a duplicate pair -- or suppressing in reading order
rather than score order -- would silently shrink the map the tracker chooses from.
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from dedupe_predicted_lines import area, iou, keep_boxes  # noqa: E402


def test_identical_boxes_collapse_to_the_higher_scoring_one():
    boxes = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
    scores = [0.3, 0.9]
    # The survivor is index 1, and the output is in the input's order, so [1] not [0].
    assert keep_boxes(boxes, scores, 0.5) == [1]


def test_suppression_runs_in_score_order_not_reading_order():
    # Three overlapping boxes where the first in reading order is the weakest: suppressing by
    # reading order would keep it and drop the strong one.
    boxes = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
    scores = [0.1, 0.5, 0.9]
    assert keep_boxes(boxes, scores, 0.5) == [2]


def test_disjoint_columns_are_all_kept():
    # Adjacent columns of a vertical page: separated on x, so they do not suppress each other.
    boxes = [[0.0, 0.0, 0.2, 1.0], [0.3, 0.0, 0.5, 1.0], [0.6, 0.0, 0.8, 1.0]]
    scores = [0.9, 0.9, 0.9]
    assert keep_boxes(boxes, scores, 0.5) == [0, 1, 2]


def test_partial_overlap_survives_at_a_looser_bar_and_not_at_a_tighter_one():
    # Shifted by six tenths of a width: IoU 0.25, so a 0.5 bar keeps both and a 0.3 bar drops one.
    loose = [[0.10, 0.1, 0.20, 0.9], [0.16, 0.1, 0.26, 0.9]]
    assert keep_boxes(loose, [0.9, 0.9], 0.5) == [0, 1]
    assert keep_boxes(loose, [0.9, 0.9], 0.2) == [0]
    # Shifted by four tenths of a width: IoU 0.43, which a 0.3 bar does suppress.
    tighter = [[0.10, 0.1, 0.20, 0.9], [0.14, 0.1, 0.24, 0.9]]
    assert keep_boxes(tighter, [0.9, 0.9], 0.5) == [0, 1]
    assert keep_boxes(tighter, [0.9, 0.9], 0.3) == [0]


def test_iou_and_area_are_the_usual_ones():
    assert area([0.0, 0.0, 2.0, 3.0]) == 6.0
    assert iou([0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]) == 1.0
    assert iou([0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]) == 0.0
    # A box with no extent cannot divide.
    assert iou([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]) == 0.0


def test_output_order_is_the_input_order():
    # The line order is the reading order the routing consumes, so the survivors come back in the
    # order they were listed rather than in score order.
    boxes = [[0.0, 0.0, 1.0, 1.0], [0.5, 0.0, 0.6, 1.0], [0.0, 0.0, 1.0, 1.0]]
    scores = [0.2, 0.8, 0.9]
    assert keep_boxes(boxes, scores, 0.5) == [1, 2]
