"""Tests for the MTHv2 per-character box channel.

The channel has to be index-aligned with ``page_text`` and must not shift when the
geometry is locally wrong, so the tests pin down both the alignment's preference
(a missing box over a shifted one) and the file-level behaviour (a new manifest,
never a rewritten one).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from prepare_mthv2_char_manifest import (  # noqa: E402
    align,
    augment_split,
    page_characters,
)

PAGE_SIZE = [100.0, 200.0]


def _line(region_id: str, text: str, box_px, order: int, direction: str = "vertical_rtl"):
    return {
        "region_id": region_id,
        "text": text,
        "bbox_px": list(box_px),
        "bbox": [box_px[0] / PAGE_SIZE[0], box_px[1] / PAGE_SIZE[1],
                 box_px[2] / PAGE_SIZE[0], box_px[3] / PAGE_SIZE[1]],
        "reading_order": order,
        "writing_direction": direction,
    }


def _char(character: str, box_px):
    return {"character": character, "bbox_xyxy_px": list(box_px)}


def _annotation(textlines, characters):
    return {"page_size": list(PAGE_SIZE), "textlines": textlines, "characters": characters}


# Two vertical columns: reading order 0 is the right-hand one, as in the source.
RIGHT = (60.0, 0.0, 80.0, 120.0)
LEFT = (20.0, 0.0, 40.0, 120.0)


def _two_column_annotation(characters):
    return _annotation(
        [_line("line_000", "甲乙丙", RIGHT, 0), _line("line_001", "丁戊", LEFT, 1)],
        characters,
    )


def _record(page_text: str = "甲乙丙丁戊"):
    return {"page_id": "p0", "page_text": page_text, "regions": []}


def test_alignment_pairs_every_character_when_the_order_is_right():
    entries, stats = page_characters(
        _record(),
        _two_column_annotation(
            [
                _char("甲", (62.0, 8.0, 78.0, 32.0)),
                _char("乙", (62.0, 48.0, 78.0, 72.0)),
                _char("丙", (62.0, 88.0, 78.0, 112.0)),
                _char("丁", (22.0, 8.0, 38.0, 32.0)),
                _char("戊", (22.0, 48.0, 38.0, 72.0)),
            ]
        ),
    )
    assert len(entries) == 5
    assert stats["matched"] == 5
    assert stats["order_mismatches"] == 0
    assert [entry["line_index"] for entry in entries] == [0, 0, 0, 1, 1]
    # Normalised against the page, in xyxy order.
    assert entries[0]["bbox"] == [0.62, 0.04, 0.78, 0.16]


def test_a_bleed_in_box_is_dropped_rather_than_shifting_the_line():
    """The failure the alignment exists for: a box that geometry puts in the wrong line.

    ``戊`` belongs to no line's text -- it is annotated inside the first column,
    after ``乙``.  A positional pairing would give ``丙`` and ``丁`` the boxes of
    ``戊`` and ``丙``, i.e. every character after the bleed off by one, which no
    page-level average would reveal.  The alignment drops the intruder instead.
    """

    entries, stats = page_characters(
        _record("甲乙丙丁"),
        _annotation(
            [_line("line_000", "甲乙", RIGHT, 0), _line("line_001", "丙丁", LEFT, 1)],
            [
                _char("甲", (62.0, 8.0, 78.0, 32.0)),
                _char("乙", (62.0, 48.0, 78.0, 72.0)),
                _char("戊", (62.0, 88.0, 78.0, 112.0)),  # bleeds into the first column
                _char("丙", (22.0, 8.0, 38.0, 32.0)),
                _char("丁", (22.0, 48.0, 38.0, 72.0)),
            ],
        ),
    )
    assert stats["matched"] == 4
    assert [entry["line_index"] for entry in entries] == [0, 0, 1, 1]
    # ``丙`` gets its own box, not the intruder's.
    assert entries[2]["bbox"] == [0.22, 0.04, 0.38, 0.16]
    assert entries[3]["bbox"] == [0.22, 0.24, 0.38, 0.36]


def test_a_placeholder_is_paired_rather_than_skipped():
    """``#`` means "unidentified glyph"; the box is still the right box."""

    entries, stats = page_characters(
        _record("甲乙丙"),
        _annotation(
            [_line("line_000", "甲乙丙", RIGHT, 0)],
            [
                _char("甲", (62.0, 8.0, 78.0, 32.0)),
                _char("#", (62.0, 48.0, 78.0, 72.0)),
                _char("丙", (62.0, 88.0, 78.0, 112.0)),
            ],
        ),
    )
    assert stats["matched"] == 3
    assert stats["order_mismatches"] == 1
    assert entries[1]["bbox"] == [0.62, 0.24, 0.78, 0.36]


def test_a_horizontal_line_orders_left_to_right():
    entries, _ = page_characters(
        _record("甲乙丙"),
        _annotation(
            [_line("line_000", "甲乙丙", (0.0, 10.0, 90.0, 30.0), 0, "horizontal_ltr")],
            [
                _char("丙", (60.0, 12.0, 80.0, 28.0)),
                _char("甲", (10.0, 12.0, 30.0, 28.0)),
                _char("乙", (35.0, 12.0, 55.0, 28.0)),
            ],
        ),
    )
    assert [entry["bbox"][0] for entry in entries] == [0.1, 0.35, 0.6]


def test_a_page_without_boxes_keeps_the_index_alignment():
    entries, stats = page_characters(_record(), _two_column_annotation([]))
    assert len(entries) == 5
    assert stats["matched"] == 0
    assert all(entry["bbox"] is None for entry in entries)
    assert [entry["line_index"] for entry in entries] == [0, 0, 0, 1, 1]


def test_textlines_that_do_not_reproduce_page_text_are_refused():
    """A mismatch means the indices refer to a different string than the target."""

    with pytest.raises(ValueError, match="do not reproduce page_text"):
        page_characters(
            {"page_id": "p0", "page_text": "甲乙丙", "regions": []},
            _two_column_annotation([]),
        )


def test_align_only_moves_forwards():
    """Monotonicity is the invariant that keeps a local error local.

    A pairing that went backwards would let one mis-annotated box reorder the rest
    of the page, which is exactly what a positional pairing does.
    """

    chosen = [(i, j) for i, j in enumerate(align("甲乙丙丁", "甲丁")) if j is not None]
    assert chosen == [(0, 0), (3, 1)]
    indices = [j for _, j in chosen]
    assert indices == sorted(indices) and len(set(indices)) == len(indices)


def test_align_absorbs_an_extra_source_character():
    pairs = align("甲乙丙丁", "甲乙戊丙丁")
    assert pairs == [0, 1, 3, 4]


def test_align_leaves_a_target_character_unpaired_when_no_box_exists():
    pairs = align("甲乙丙丁", "甲乙丁")
    assert pairs == [0, 1, None, 2]


def _write_dataset(root: Path, split: str, records, annotations: dict[str, dict]) -> Path:
    (root / split / "annotations").mkdir(parents=True, exist_ok=True)
    (root / split / "manifest.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    for name, annotation in annotations.items():
        (root / split / "annotations" / name).write_text(
            json.dumps(annotation, ensure_ascii=False), encoding="utf-8"
        )
    return root / split / "manifest.jsonl"


def test_augment_split_writes_a_new_manifest_and_leaves_the_original(tmp_path):
    annotation = _two_column_annotation([_char("甲", (62.0, 8.0, 78.0, 32.0))])
    record = _record()
    record["annotation_file"] = "annotations/p0.json"
    manifest = _write_dataset(tmp_path, "validation", [record], {"p0.json": annotation})
    before = manifest.read_text(encoding="utf-8")

    stats = augment_split(tmp_path, tmp_path, "validation", "manifest.char.jsonl")

    assert manifest.read_text(encoding="utf-8") == before
    written = manifest.with_name("manifest.char.jsonl")
    assert written.is_file()
    augmented = json.loads(written.read_text(encoding="utf-8").splitlines()[0])
    assert len(augmented["characters"]) == 5
    assert augmented["char_source"] == "mthv2_official_char_annotation"
    assert augmented["char_stats"]["matched"] == 1
    # Every original field survives: the eval path still reads ``regions``.
    assert augmented["page_text"] == record["page_text"]
    assert stats["pages"] == 1


def test_a_subset_dataset_needs_its_parent_as_the_annotation_root(tmp_path):
    """A sparse subset stores no annotations of its own."""

    parent = tmp_path / "parent"
    subset = tmp_path / "subset"
    annotation = _two_column_annotation([_char("甲", (62.0, 8.0, 78.0, 32.0))])
    record = _record()
    record["annotation_file"] = "annotations/p0.json"
    _write_dataset(parent, "validation", [record], {"p0.json": annotation})
    _write_dataset(subset, "validation", [dict(record)], {})

    with pytest.raises(FileNotFoundError):
        augment_split(subset, subset, "validation", "manifest.char.jsonl")

    stats = augment_split(subset, parent, "validation", "manifest.char.jsonl")
    assert stats["matched"] == 1
