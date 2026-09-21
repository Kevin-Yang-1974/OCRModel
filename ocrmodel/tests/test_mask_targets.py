"""Tests for the per-token ground-truth masks (``mask_targets.py``).

The mask head is supervised by a rasterised character box per target token.  The
things that cannot be checked by reading the code once are: the rasteriser keeps
the best-covered cell at 1.0; the EOS / blank / missing-box token treatments stay
distinct; and the token-to-character alignment does not let one token steal
another's box.
"""

from __future__ import annotations

import pytest
import torch

from layout_ocr.mask_targets import (
    build_mask_targets,
    char_boxes,
    rasterize_box,
    token_char_spans,
)


class _Tokenizer:
    """Maps token ids to decoded strings, which is all the alignment reads."""

    def __init__(self, mapping: dict[int, str]):
        self.mapping = mapping

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.mapping.get(int(index), "") for index in ids)


def _xywh():
    """A 2x2 merged grid: four cells of size 0.5 centred at the quarters."""

    centers = torch.tensor(
        [
            [0.25, 0.25, 0.5, 0.5],
            [0.75, 0.25, 0.5, 0.5],
            [0.25, 0.75, 0.5, 0.5],
            [0.75, 0.75, 0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    return centers.unsqueeze(0)  # [1, 4, 4]


def test_rasterize_box_peaks_at_one_and_is_zero_outside():
    out = rasterize_box([0.0, 0.0, 0.5, 1.0], _xywh()[0])
    assert out.shape == (4,)
    # The left half holds the first and third cells, nothing else.
    assert out.tolist() == [1.0, 0.0, 1.0, 0.0]


def test_rasterize_box_normalizes_a_small_box():
    # A box smaller than one cell still yields a peak of exactly 1.0.
    out = rasterize_box([0.2, 0.2, 0.4, 0.4], _xywh()[0])
    assert out[0].item() == pytest.approx(1.0)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_char_boxes_distinguishes_missing_from_present():
    boxes, statuses = char_boxes(
        {
            "characters": [
                {"bbox": [0.0, 0.0, 1.0, 1.0], "alignment_status": "exact"},
                {"bbox": None, "alignment_status": "missing"},
                "not-a-dict",
            ]
        }
    )
    assert boxes[0] == [0.0, 0.0, 1.0, 1.0]
    assert boxes[1] is None and boxes[2] is None
    assert statuses == ["exact", "missing", "missing"]


def test_token_char_spans_marks_a_blank_token():
    tokenizer = _Tokenizer({0: "甲", 1: ""})
    spans, statuses, report = token_char_spans(tokenizer, "甲", [0, 1], ["exact"])
    assert statuses == ["exact", "blank"]
    assert spans == [(0, 1), None]
    assert report["mapped_tokens"] == 1
    assert report["blank_tokens"] == 1


def test_build_mask_targets_marks_eos_normal_and_blank_distinctly():
    tokenizer = _Tokenizer({10: "甲", 11: "乙", 12: ""})
    record = {
        "page_id": "p0",
        "page_text": "甲乙",
        "characters": [
            {"bbox": [0.0, 0.0, 0.5, 1.0], "alignment_status": "exact"},
            {"bbox": [0.5, 0.0, 1.0, 1.0], "alignment_status": "exact"},
        ],
    }
    targets = build_mask_targets(tokenizer, record, [10, 11, 12, 13], {13}, _xywh())
    assert targets.mask.shape == (1, 4, 4)
    # 甲 -> left half, 乙 -> right half.
    assert targets.mask[0, 0].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert targets.mask[0, 1].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert targets.spatial_valid[0].tolist() == [True, True, True, True]
    assert targets.stop_target[0].tolist() == [0.0, 0.0, 0.0, 1.0]
    # 12 decodes to empty -> blank -> valid empty mask; 13 is EOS -> stop=1.
    assert targets.alignment_status[2] == "blank"
    assert targets.mask[0, 2].sum().item() == 0.0
    assert targets.mask[0, 3].sum().item() == 0.0


def test_a_token_with_a_missing_box_is_not_supervised():
    tokenizer = _Tokenizer({10: "甲"})
    record = {
        "page_id": "p0",
        "page_text": "甲",
        "characters": [{"bbox": None, "alignment_status": "missing"}],
    }
    targets = build_mask_targets(tokenizer, record, [10], {13}, _xywh())
    assert targets.spatial_valid[0, 0].item() is False
    assert targets.mask[0, 0].sum().item() == 0.0
    assert targets.stop_target[0, 0].item() == 0.0


def test_a_placeholder_box_is_still_a_box():
    """``#`` means the location is right even though the glyph is unidentified."""

    tokenizer = _Tokenizer({10: "甲"})
    record = {
        "page_id": "p0",
        "page_text": "甲",
        "characters": [{"bbox": [0.0, 0.0, 0.5, 1.0], "alignment_status": "placeholder"}],
    }
    targets = build_mask_targets(tokenizer, record, [10], {13}, _xywh())
    assert targets.spatial_valid[0, 0].item() is True
    assert targets.mask[0, 0].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_one_token_can_cover_several_characters_union():
    """A token decoding to two characters takes the union of both boxes."""

    tokenizer = _Tokenizer({10: "甲乙"})
    record = {
        "page_id": "p0",
        "page_text": "甲乙",
        "characters": [
            {"bbox": [0.0, 0.0, 0.5, 1.0], "alignment_status": "exact"},
            {"bbox": [0.5, 0.0, 1.0, 1.0], "alignment_status": "exact"},
        ],
    }
    targets = build_mask_targets(tokenizer, record, [10], {13}, _xywh())
    assert targets.mask[0, 0].tolist() == [1.0, 1.0, 1.0, 1.0]
