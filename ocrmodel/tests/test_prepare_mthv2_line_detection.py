"""Tests for the line-detection index builder.

The filtering rules are the part worth pinning: they decide what the detector is ever shown,
and a rule that quietly discards thin columns would look exactly like a detector that cannot
learn them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from prepare_mthv2_line_detection import (  # noqa: E402
    ALLOWED_SPLITS,
    main,
    page_rows,
)


def _record(**overrides):
    record = {
        "page_id": "p0",
        "image_path": "/images/p0.jpg",
        "page_size": [1500, 3000],
        "regions": [
            {
                "bbox_px": [1202.0, 145.0, 1294.0, 1438.0],
                "reading_order": 0,
                "writing_direction": "vertical_rtl",
                "valid": True,
            },
            {
                "bbox_px": [1300.0, 145.0, 1392.0, 1438.0],
                "reading_order": 1,
                "writing_direction": "vertical_rtl",
                "valid": True,
            },
        ],
    }
    record.update(overrides)
    return record


def test_a_page_becomes_one_row_of_boxes():
    row, stats = page_rows(_record(), min_side=4.0)
    assert row["page_id"] == "p0"
    assert row["width"] == 1500 and row["height"] == 3000
    assert len(row["boxes"]) == 2
    assert row["boxes"][0] == [1202.0, 145.0, 1294.0, 1438.0]
    # Reading order and direction travel with the boxes: the detection audit asks for order
    # accuracy and per-direction accuracy, and they cannot be recovered later.
    assert row["reading_order"] == [0, 1]
    assert row["writing_direction"] == ["vertical_rtl", "vertical_rtl"]
    assert stats["kept"] == 2


def test_a_sliver_is_dropped_and_counted():
    """A one-pixel column is an annotation artifact; training on it teaches slivers."""

    record = _record(regions=[
        {"bbox_px": [10.0, 10.0, 1390.0, 20.0], "reading_order": 0, "valid": True},
        {"bbox_px": [10.0, 30.0, 12.0, 1430.0], "reading_order": 1, "valid": True},  # 2px wide
    ])
    row, stats = page_rows(record, min_side=4.0)
    assert len(row["boxes"]) == 1
    assert stats["degenerate"] == 1
    assert stats["kept"] == 1


def test_an_invalid_region_is_dropped_and_counted():
    record = _record(regions=[
        {"bbox_px": [10.0, 10.0, 100.0, 200.0], "reading_order": 0, "valid": False},
        {"bbox_px": [10.0, 10.0, 100.0, 200.0], "reading_order": 1, "valid": True},
    ])
    row, stats = page_rows(record, min_side=4.0)
    assert len(row["boxes"]) == 1
    assert stats["invalid"] == 1


def test_a_page_with_no_usable_box_is_skipped():
    assert page_rows(_record(regions=[]), min_side=4.0) == (None, {
        "kept": 0, "degenerate": 0, "invalid": 0, "no_box_field": 0,
    })
    assert page_rows(_record(page_size=None), min_side=4.0)[0] is None
    row, _ = page_rows(_record(regions=[
        {"bbox_px": [10.0, 10.0, 12.0, 12.0], "reading_order": 0, "valid": True},
    ]), min_side=4.0)
    assert row is None


def test_a_region_without_a_box_field_is_counted_not_guessed():
    record = _record(regions=[{"reading_order": 0, "valid": True}])
    row, stats = page_rows(record, min_side=4.0)
    assert row is None
    assert stats["no_box_field"] == 1


def test_the_test_split_is_refused(tmp_path):
    """Rule 10 locks it, and the refusal belongs in the code rather than in a note."""

    assert "test" not in ALLOWED_SPLITS
    with pytest.raises(SystemExit, match="refusing splits"):
        main(["--manifest-dir", str(tmp_path), "--output-dir", str(tmp_path / "out"),
              "--splits", "test"])


def _write_manifest(root, split, records):
    directory = root / split
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.char.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_the_index_is_written_and_missing_images_are_reported(tmp_path, capsys):
    image = tmp_path / "p0.jpg"
    image.write_bytes(b"not really a jpeg")
    present = _record(page_id="p0", image_path=str(image))
    absent = _record(page_id="p1", image_path=str(tmp_path / "nope.jpg"))
    _write_manifest(tmp_path / "data", "train", [present, absent])

    code = main(["--manifest-dir", str(tmp_path / "data"), "--output-dir", str(tmp_path / "out"),
                 "--splits", "train"])
    assert code == 0
    index = tmp_path / "out" / "lines_train.jsonl"
    rows = [
        json.loads(line)
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # The page whose image is missing is left out rather than indexed to a path that would
    # crash training much later.
    assert [row["page_id"] for row in rows] == ["p0"]
    report = json.loads(
        (tmp_path / "out" / "line_detection_index_report.json").read_text(encoding="utf-8")
    )
    assert report["splits"]["train"]["missing_image_count"] == 1
    assert report["splits"]["train"]["boxes"] == 2
    assert "WARNING" in capsys.readouterr().out
