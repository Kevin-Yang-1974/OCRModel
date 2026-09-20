"""Tests for the missed-line attribution.

The buckets drive which fix gets attempted next, so a bucket that absorbs its neighbours would
send the next round at the wrong mechanism. These pin the priority order and the boundary that
separates "found it, box off" from "never saw it".
"""

from __future__ import annotations

import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from diagnose_line_miss_attribution import (  # noqa: E402
    classify_extra,
    classify_missed,
    match_page,
    normalised_gt,
)

PAGE = (1000.0, 1000.0)


def test_identical_boxes_are_not_missed():
    gt = [[0.1, 0.1, 0.2, 0.9]]
    assignment, taken = match_page(gt, [list(gt[0])], 0.5)
    assert assignment == [0] and taken == {0}


def test_one_prediction_cannot_serve_two_lines():
    # Two columns, one box twice as wide as either: the first line takes it at IoU 0.5, and the
    # second line has nothing left.
    gt = [[0.10, 0.1, 0.20, 0.9], [0.25, 0.1, 0.35, 0.9]]
    pred = [[0.10, 0.1, 0.30, 0.9]]
    assignment, taken = match_page(gt, pred, 0.5)
    assert assignment[0] == 0
    assert assignment[1] is None
    # The box that took line 0 overlaps line 1, so this is a merge, not a blind spot.
    missed = classify_missed(gt[1], pred, taken, 0.5, PAGE)
    assert missed["bucket"] == "absorbed"


def test_free_box_overlapping_below_the_bar_is_box_off():
    gt = [[0.10, 0.10, 0.20, 0.90]]
    pred = [[0.16, 0.10, 0.26, 0.90]]  # shifted six tenths of a width, IoU 0.33
    assignment, taken = match_page(gt, pred, 0.5)
    assert assignment == [None]
    assert classify_missed(gt[0], pred, taken, 0.5, PAGE)["bucket"] == "box_off"


def test_box_far_larger_than_the_line_is_coarse():
    gt = [[0.30, 0.30, 0.34, 0.70]]
    pred = [[0.00, 0.00, 1.00, 1.00]]  # covers the line, IoU only the line's area share
    assignment, taken = match_page(gt, pred, 0.5)
    assert assignment == [None]
    assert classify_missed(gt[0], pred, taken, 0.5, PAGE)["bucket"] == "coarse"


def test_nothing_nearby_is_unseen():
    gt = [[0.10, 0.10, 0.20, 0.90]]
    pred = [[0.70, 0.10, 0.80, 0.90]]
    assignment, taken = match_page(gt, pred, 0.5)
    assert assignment == [None]
    assert classify_missed(gt[0], pred, taken, 0.5, PAGE)["bucket"] == "unseen"


def test_extra_box_on_a_matched_line_is_a_duplicate():
    gt = [[0.10, 0.10, 0.20, 0.90]]
    pred = [[0.10, 0.10, 0.20, 0.90], [0.11, 0.12, 0.19, 0.88]]
    assignment, taken = match_page(gt, pred, 0.5)
    assert taken == {0}
    assert classify_extra(pred[1], gt, assignment, PAGE)["bucket"] == "duplicate_of_matched"


def test_extra_box_on_empty_page_is_background():
    gt = [[0.10, 0.10, 0.20, 0.90]]
    pred = [[0.10, 0.10, 0.20, 0.90], [0.70, 0.10, 0.80, 0.90]]
    assignment, taken = match_page(gt, pred, 0.5)
    assert classify_extra(pred[1], gt, assignment, PAGE)["bucket"] == "on_background"


def test_sizes_come_back_in_pixels():
    # The tool's sizes are compared against the audit's table, which is in pixels.
    gt_box = [0.10, 0.10, 0.20, 0.90]
    missed = classify_missed(gt_box, [], set(), 0.5, (1000.0, 2000.0))
    assert missed["width"] == 100.0
    assert missed["height"] == 1600.0


def test_normalised_gt_divides_by_page_size():
    record = {"page_id": "p", "width": 2000.0, "height": 1000.0, "boxes": [[100.0, 50.0, 300.0, 250.0]]}
    assert normalised_gt(record) == [[0.05, 0.05, 0.15, 0.25]]


def test_normalised_gt_rejects_a_zero_page_size():
    record = {"page_id": "p", "width": 0.0, "height": 1000.0, "boxes": []}
    try:
        normalised_gt(record)
    except ValueError as error:
        assert "page size" in str(error)
    else:
        raise AssertionError("a zero page width must not divide silently")
