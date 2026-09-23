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
    LineTargetError,
    build_mask_targets,
    char_boxes,
    rasterize_box,
    region_line_targets,
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


def _wide_grid(cells: int):
    """A single-row grid of ``cells`` equal cells, so coverage is countable."""

    width = 1.0 / cells
    return torch.tensor(
        [
            [(i + 0.5) * width, 0.5, width, 1.0]
            for i in range(cells)
        ],
        dtype=torch.float32,
    ).unsqueeze(0)


def _line_record(text: str, boxes, line_index: list[int]):
    return {
        "page_id": "p0",
        "page_text": text,
        "characters": [
            {"bbox": list(box), "alignment_status": "exact", "line_index": line}
            for box, line in zip(boxes, line_index)
        ],
    }


def _targets_for(mode: str, text: str, boxes, line_index, ids, cells=10):
    # Token ids start at 20: EOS is 13 below, and an id collision there would
    # silently turn one of the fixture's characters into a stop token.
    tokenizer = _Tokenizer({index: char for index, char in enumerate(ids, start=20)})
    record = _line_record(text, boxes, line_index)
    return build_mask_targets(
        tokenizer,
        record,
        list(range(20, 20 + len(ids))),
        {13},
        _wide_grid(cells),
        target_mode=mode,
        window_min=3,
        window_max=5,
        line_source="annotation",
        raster_mode="hard",
    )


def test_anchored_target_is_the_window_union_the_line_remainder():
    """`anchored` adds the rest of the line to the window.

    A single-character token has ``span`` length 1, so its window is
    ``max(window_min, min(window_max, 1)) == 3`` characters wide.  `anchored`
    keeps that window and adds the characters from the window's end to the end
    of the line; `line` covers the whole line.  The three shapes therefore come
    out strictly ordered on an eight-character line.
    """

    text = "abcdefgh"
    boxes = [[i / 10, 0.0, (i + 1) / 10, 1.0] for i in range(8)]
    lines = [0] * 8
    window = _targets_for("window", text, boxes, lines, list(text))
    anchored = _targets_for("anchored", text, boxes, lines, list(text))
    whole = _targets_for("line", text, boxes, lines, list(text))

    def covered(targets, token: int) -> int:
        return int((targets.mask[0, token] > 0).sum())

    # Line start: window is characters 0..2, the remainder 3..7 adds five cells.
    assert covered(window, 0) == 3
    assert covered(anchored, 0) == 8
    assert covered(whole, 0) == 8
    # Mid-line the remainder is shorter, so the same ordering holds with a gap.
    assert covered(window, 3) == 3
    assert covered(anchored, 3) == 5  # window 3..5, remainder 6..7
    assert covered(whole, 3) == 8
    # At the line tail the remainder is empty, so anchored collapses to window.
    assert covered(anchored, 7) == covered(window, 7) == 3
    # The three shapes are ordered on every token.
    for token in range(8):
        assert covered(window, token) <= covered(anchored, token) <= covered(whole, token)


def test_a_single_box_window_still_rasterizes():
    """A one-box window must not collapse the hull into a zero-area polygon.

    ``region_textline`` gives every token its line's single box, and a one-line
    fixture gives a ``line``/``anchored`` window exactly one box.  Feeding only
    the top-left and bottom-right corner of each box left the hull with two
    distinct points, which has no area: the rasteriser returned an all-zero mask
    and the token was supervised as if its line were blank.
    """

    record = _region_record([(0, ("abcd", [0.0, 0.0, 1.0, 0.5]))])
    tokenizer = _Tokenizer({index: char for index, char in enumerate("abcd", start=20)})
    for mode in ("line", "window", "anchored"):
        targets = build_mask_targets(
            tokenizer, record, list(range(20, 24)), {13}, _wide_grid(cells=4),
            target_mode=mode, line_source="region_textline", raster_mode="hard",
        )
        for token in range(4):
            assert int((targets.mask[0, token] > 0).sum()) == 4, mode

    # The same degenerate case through the char-manifest ``annotation`` path.
    boxes = [[i / 4, 0.0, (i + 1) / 4, 1.0] for i in range(4)]
    annotated = _targets_for("line", "abcd", boxes, [0] * 4, list("abcd"), cells=4)
    assert int((annotated.mask[0, 0] > 0).sum()) == 4


def test_anchored_is_a_superset_of_window_on_every_token():
    """The anchored raster must contain the window raster cell for cell."""

    text = "abcdefghij"
    boxes = [[i / 10, 0.0, (i + 1) / 10, 1.0] for i in range(10)]
    window = _targets_for("window", text, boxes, [0] * 10, list(text), cells=10)
    anchored = _targets_for("anchored", text, boxes, [0] * 10, list(text), cells=10)
    assert bool(((anchored.mask >= window.mask).all()))
    assert not bool((anchored.mask == window.mask).all())


def test_existing_modes_are_unchanged_by_the_anchored_addition():
    """`window` and `line` must keep their recorded shapes exactly."""

    text = "abcdefgh"
    boxes = [[i / 10, 0.0, (i + 1) / 10, 1.0] for i in range(8)]
    window = _targets_for("window", text, boxes, [0] * 8, list(text))
    whole = _targets_for("line", text, boxes, [0] * 8, list(text))
    assert int((window.mask[0, 0] > 0).sum()) == 3
    assert int((whole.mask[0, 0] > 0).sum()) == 8
    # Two lines on one page: the line mode must never reach the other line.
    split = _targets_for("line", "abcdefgh", boxes, [0, 0, 0, 0, 1, 1, 1, 1], list(text))
    assert int((split.mask[0, 0] > 0).sum()) == 4


def _region_record(lines, *, layout_level="textline", separator="", extra=None):
    """A line-level manifest record: regions *are* the lines, no characters.

    ``lines`` is a list of ``(reading_order, (text, box))``; the region list is
    kept in the order given, so a caller can hand over an out-of-order manifest
    and check that the reader sorts by ``reading_order``.
    """

    record = {
        "page_id": "p0",
        "page_text": "".join(text for _, (text, _) in sorted(lines)),
        "layout_level": layout_level,
        "page_text_separator": separator,
        "regions": [
            {
                "reading_order": order,
                "text": text,
                "bbox": list(box),
                "layout_level": "textline",
            }
            for order, (text, box) in lines
        ],
    }
    if extra:
        record.update(extra)
    return record


def test_region_line_targets_walk_page_text_against_reading_order():
    """Each line's characters take that line's one box, in reading order."""

    record = _region_record([
        (1, ("cd", [0.5, 0.0, 1.0, 1.0])),
        (0, ("ab", [0.0, 0.0, 0.5, 1.0])),
    ])
    assert record["page_text"] == "abcd"

    targets = region_line_targets(record)

    assert targets.granularity == "textline"
    assert targets.boxes == [
        [0.0, 0.0, 0.5, 1.0], [0.0, 0.0, 0.5, 1.0],
        [0.5, 0.0, 1.0, 1.0], [0.5, 0.0, 1.0, 1.0],
    ]
    assert targets.line_ids == [0, 0, 1, 1]
    assert targets.statuses == ["exact"] * 4
    assert targets.report["mapped_characters"] == 4
    assert targets.report["regions_without_valid_bbox"] == 0


def test_region_line_targets_refuse_a_region_set_that_stops_early():
    """A short walk is refused, not half-supervised.

    Leaving the tail unmapped would still report the page as having line
    evidence, so the locked evidence file could not tell which pages were
    actually covered by their regions.
    """

    record = _region_record([(0, ("ab", [0.0, 0.0, 0.5, 1.0]))])
    record["page_text"] = "abcd"

    with pytest.raises(LineTargetError, match="cover 2 of 4"):
        region_line_targets(record)


def test_region_line_targets_refuse_a_manifest_they_cannot_read():
    """Every unreadable manifest shape raises rather than producing a mapping."""

    mismatched = _region_record([(0, ("ab", [0.0, 0.0, 0.5, 1.0]))])
    mismatched["page_text"] = "ax"
    with pytest.raises(LineTargetError, match="does not reproduce page_text"):
        region_line_targets(mismatched)

    with pytest.raises(LineTargetError, match="layout_level"):
        region_line_targets(_region_record(
            [(0, ("ab", [0.0, 0.0, 0.5, 1.0]))], layout_level="region"
        ))

    with pytest.raises(LineTargetError, match="characters"):
        region_line_targets(_region_record(
            [(0, ("ab", [0.0, 0.0, 0.5, 1.0]))],
            extra={"characters": [{"bbox": [0.0, 0.0, 1.0, 1.0]}]},
        ))

    with pytest.raises(LineTargetError, match="separator"):
        region_line_targets(_region_record(
            [(0, ("ab", [0.0, 0.0, 0.5, 1.0]))], separator="\n"
        ))

    empty = {"page_id": "p0", "page_text": "ab", "layout_level": "textline", "regions": []}
    with pytest.raises(LineTargetError, match="no regions"):
        region_line_targets(empty)


def test_region_line_source_builds_a_whole_line_target_per_token():
    """`line` + region_textline gives each token the full hull of its own line."""

    record = _region_record([
        (0, ("abcd", [0.0, 0.0, 1.0, 0.5])),
        (1, ("efgh", [0.0, 0.5, 1.0, 1.0])),
    ])
    tokenizer = _Tokenizer({index: char for index, char in enumerate("abcdefgh", start=20)})
    targets = build_mask_targets(
        tokenizer,
        record,
        list(range(20, 28)),
        {13},
        _wide_grid(cells=8),
        target_mode="line",
        line_source="region_textline",
        raster_mode="hard",
    )

    # Line 0 spans the whole width, so every one of its tokens covers all cells.
    assert targets.window_report["box_granularity"] == "textline"
    assert targets.window_report["line_source"] == "region_textline"
    assert targets.window_report["lines"] == 2
    assert targets.window_report["region_line_report"]["mapped_characters"] == 8
    for token in range(4):
        assert int((targets.mask[0, token] > 0).sum()) == 8
    # Line 1 shares the same box in this fixture, but line membership is what
    # matters: no token is left unsupervised.
    assert bool(targets.spatial_valid[0, :8].all())


def test_a_page_whose_regions_stop_early_is_refused_by_the_target_builder():
    """The refusal reaches the caller that builds targets, not just the helper."""

    record = _region_record([(0, ("abcd", [0.0, 0.0, 1.0, 1.0]))])
    record["page_text"] = "abcdef"
    tokenizer = _Tokenizer({index: char for index, char in enumerate("abcdef", start=20)})
    with pytest.raises(LineTargetError, match="cover 4 of 6"):
        build_mask_targets(
            tokenizer, record, list(range(20, 26)), {13}, _wide_grid(cells=6),
            target_mode="line", line_source="region_textline", raster_mode="hard",
        )


def test_the_annotation_path_is_unchanged_by_the_region_source():
    """A char manifest with `line_index` still reads annotation granularity."""

    text = "abcdefgh"
    boxes = [[i / 10, 0.0, (i + 1) / 10, 1.0] for i in range(8)]
    annotated = _targets_for("line", text, boxes, [0] * 4 + [1] * 4, list(text))
    assert annotated.window_report["box_granularity"] == "character"
    assert annotated.window_report["line_source"] == "annotation"
    assert "region_line_report" not in annotated.window_report
    assert annotated.window_report["lines"] == 2

    # An unknown line_source is still refused rather than silently ignored.
    tokenizer = _Tokenizer({index: char for index, char in enumerate(text, start=20)})
    with pytest.raises(ValueError, match="line_source must be one of"):
        build_mask_targets(
            tokenizer, _line_record(text, boxes, [0] * 8), list(range(20, 28)), {13},
            _wide_grid(cells=8), target_mode="line", line_source="guessed",
        )


def test_line_mode_without_a_line_grouping_is_refused_even_under_auto():
    """``auto`` must not quietly degrade a line target into a token target.

    It previously did: a manifest with no line grouping fell through to
    per-token boxes while still being reported under a line-target flag, so a run
    could claim line supervision it never applied.  A caller that wants line
    targets now has to name the field it reads them from.
    """

    tokenizer = _Tokenizer({index: char for index, char in enumerate("甲乙", start=20)})
    record = {
        "page_id": "p0",
        "page_text": "甲乙",
        "characters": [
            {"bbox": [0.0, 0.0, 0.5, 1.0], "alignment_status": "exact"},
            {"bbox": [0.5, 0.0, 1.0, 1.0], "alignment_status": "exact"},
        ],
    }
    with pytest.raises(ValueError, match="found no line grouping"):
        build_mask_targets(
            tokenizer, record, [20, 21], {13}, _wide_grid(cells=4),
            target_mode="line", line_source="auto",
        )
    # ``token`` mode is unaffected: it never asked for a line grouping.
    targets = build_mask_targets(tokenizer, record, [20, 21], {13}, _wide_grid(cells=4))
    assert targets.line_evidence == "none"
    assert bool(targets.spatial_valid[0].all())


def test_token_fallbacks_counts_tokens_with_no_line_to_widen():
    """A line-mode token whose line never resolved is visible, not silent."""

    text = "abcdef"
    boxes = [[i / 6, 0.0, (i + 1) / 6, 1.0] for i in range(6)]
    # The first character has no line id, so its token has no line to widen to.
    lines = [None, 0, 0, 0, 0, 0]
    targets = _targets_for("line", text, boxes, lines, list(text), cells=6)
    assert targets.window_report["token_fallbacks"] == 1
    assert int((targets.mask[0, 0] > 0).sum()) > 0  # still supervised, by its own box
