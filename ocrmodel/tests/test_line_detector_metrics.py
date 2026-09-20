"""Tests for the line detector's metrics.

The metrics decide which checkpoint gets selected, so an error in them is not a reporting
problem: it picks the wrong model. They are pure arithmetic over boxes, which means they can be
checked exactly -- and the ordering metric in particular has a convention that is easy to get
backwards, as the first draft of its own smoke check did.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from evaluate_line_detector import order_accuracy  # noqa: E402
from train_line_detector import (  # noqa: E402
    _dominant_direction,
    f1,
    iou_matrix,
    match_detections,
    resize_scale,
)


def test_iou_of_identical_boxes_is_one():
    boxes = torch.tensor([[0.0, 0.0, 10.0, 20.0], [30.0, 0.0, 40.0, 20.0]])
    assert iou_matrix(boxes, boxes.clone()).diagonal().tolist() == pytest.approx([1.0, 1.0])


def test_iou_counts_overlap_against_the_union():
    gt = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    # Half the box overlaps, but the union is 150 rather than 100: IoU is not the coverage.
    assert iou_matrix(gt, torch.tensor([[5.0, 0.0, 15.0, 10.0]])).item() == pytest.approx(1 / 3)
    # Fully contained: intersection 25, union 100.
    assert iou_matrix(gt, torch.tensor([[0.0, 0.0, 5.0, 5.0]])).item() == pytest.approx(0.25)
    assert iou_matrix(gt, torch.tensor([[20.0, 0.0, 30.0, 10.0]])).item() == pytest.approx(0.0)


def test_iou_with_no_boxes_is_empty_not_an_error():
    assert iou_matrix(torch.zeros((0, 4)), torch.tensor([[0.0, 0.0, 1.0, 1.0]])).shape == (0, 1)
    assert iou_matrix(torch.zeros((0, 4)), torch.zeros((0, 4))).shape == (0, 0)


def test_resize_keeps_the_aspect_ratio_and_respects_the_long_side():
    # A 1500x3000 scan at min_size 1200: the short side becomes 1200, so scale 0.8.
    assert resize_scale(1500, 3000, 1200, 2400) == pytest.approx(0.8)
    # The long side binds: 3000 * 0.8 = 2400 is exactly max_size, and anything more is capped.
    assert resize_scale(1500, 4000, 1200, 2400) == pytest.approx(0.6)
    # A small page is scaled up to the minimum.
    assert resize_scale(300, 600, 1200, 2400) == pytest.approx(4.0)


def _target(boxes):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32)}


def _prediction(boxes):
    return {"boxes": torch.tensor(boxes, dtype=torch.float32)}


def test_matching_counts_recall_against_the_lines_that_exist():
    gt = _target([[0.0, 0.0, 10.0, 100.0], [20.0, 0.0, 30.0, 100.0]])
    prediction = _prediction([[0.0, 0.0, 10.0, 100.0]])  # one of two found
    stats = match_detections([prediction], [gt], 0.5)
    assert stats["gt"] == 2 and stats["pred"] == 1 and stats["matched"] == 1
    assert stats["recall"] == pytest.approx(0.5)
    assert stats["precision"] == pytest.approx(1.0)


def test_a_box_spanning_two_lines_is_counted_as_merged():
    gt = _target([[0.0, 0.0, 10.0, 100.0], [12.0, 0.0, 22.0, 100.0]])
    # One prediction covering both: it matches at least one line by IoU and swallows the other.
    prediction = _prediction([[0.0, 0.0, 22.0, 100.0]])
    stats = match_detections([prediction], [gt], 0.3)
    assert stats["merged_gt"] >= 1
    assert stats["recall"] < 1.0


def test_a_page_with_no_detection_is_counted():
    gt = _target([[0.0, 0.0, 10.0, 100.0]])
    stats = match_detections([_prediction([])], [gt], 0.5)
    assert stats["recall"] == 0.0
    assert stats["empty_predictions"] == 1


def test_f1_is_zero_when_either_side_is():
    assert f1({"precision": 0.0, "recall": 0.5}) == 0.0
    assert f1({"precision": 1.0, "recall": 1.0}) == 1.0
    assert f1({"precision": 0.5, "recall": 0.5}) == pytest.approx(0.5)


# Ground truth in reading order for a vertical page: rightmost column first.
VERTICAL_GT = torch.tensor(
    [[500.0, 0.0, 600.0, 1000.0], [300.0, 0.0, 400.0, 1000.0], [100.0, 0.0, 200.0, 1000.0]]
)


def test_order_accuracy_is_one_when_geometry_reproduces_the_annotation():
    stats = order_accuracy(VERTICAL_GT, VERTICAL_GT.clone(), "vertical_rtl")
    assert stats["matched"] == 3
    assert stats["accuracy"] == pytest.approx(1.0)
    assert stats["inversions"] == 0


def test_the_order_of_the_prediction_list_carries_no_information():
    """The metric sorts the boxes by where they are, so permuting the list changes nothing.

    This is worth pinning because the obvious way to write a "contradicts the annotation" test is
    to reverse the prediction list, and that tests nothing at all.
    """

    permuted = torch.stack([VERTICAL_GT[2], VERTICAL_GT[1], VERTICAL_GT[0]])
    stats = order_accuracy(VERTICAL_GT, permuted, "vertical_rtl")
    assert stats["accuracy"] == pytest.approx(1.0)


def test_order_accuracy_falls_when_the_annotation_order_contradicts_the_direction():
    """Vertical columns are read right to left; a list ordered left to right is not that."""

    ascending_gt = torch.stack([VERTICAL_GT[2], VERTICAL_GT[1], VERTICAL_GT[0]])
    stats = order_accuracy(ascending_gt, ascending_gt.clone(), "vertical_rtl")
    assert stats["accuracy"] == pytest.approx(0.0)
    assert stats["inversions"] == 3


def test_order_accuracy_uses_the_axis_the_direction_implies():
    """A vertical page sorts along x, a horizontal one along y.

    The boxes differ along both axes, so the two directions genuinely disagree about the order.
    Boxes sharing one coordinate would tie under the other axis and score the same either way,
    which is how this test first passed while proving nothing.
    """

    diagonal = torch.tensor(
        [
            [100.0, 100.0, 200.0, 200.0],
            [300.0, 300.0, 400.0, 400.0],
            [500.0, 500.0, 600.0, 600.0],
        ]
    )
    assert order_accuracy(diagonal, diagonal.clone(), "horizontal_ltr")["accuracy"] == 1.0
    vertical = order_accuracy(diagonal, diagonal.clone(), "vertical_rtl")
    assert vertical["accuracy"] == pytest.approx(0.0)


def test_order_accuracy_needs_two_matches_to_say_anything():
    single = torch.tensor([[100.0, 0.0, 200.0, 1000.0]])
    stats = order_accuracy(VERTICAL_GT, single, "vertical_rtl")
    assert stats["matched"] == 1
    assert stats["accuracy"] is None


def test_the_dominant_direction_ignores_the_odd_box_out():
    assert _dominant_direction(["vertical_rtl", "vertical_rtl", "unknown"]) == "vertical_rtl"
    assert _dominant_direction([]) == "unknown"
