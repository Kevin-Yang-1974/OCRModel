from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "GOT-OCR-2.0"
    / "scripts"
    / "serialize_layout_targets.py"
)
SPEC = importlib.util.spec_from_file_location("serialize_layout_targets_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_canonical_order_and_mthv2_source_are_preserved() -> None:
    record = {
        "page_id": "p1",
        "layout_source": "mthv2_ordered_textline",
        "direction": "vertical_rtl",
        "regions": [
            {"bbox": [0.7, 0.1, 0.9, 0.9], "reading_order": 0, "writing_direction": "vertical_rtl", "layout_type": "COLUMN"},
            {"bbox": [0.1, 0.1, 0.3, 0.9], "reading_order": 1, "writing_direction": "vertical_rtl", "layout_type": "COLUMN"},
        ],
    }
    output = module.serialize_record(record, max_records=64)
    assert output["layout_source"] == "mthv2_ordered_textline"
    assert output["layout_region_count"] == 2
    assert output["layout_target_tokens"][0] == "<LAYOUT>"
    assert output["layout_target_tokens"][-1] == "<EOS>"
    assert output["layout_regions"][0]["bbox"][0] == pytest.approx(0.7)


def test_over_limit_is_rejected_instead_of_silent_truncation() -> None:
    record = {"page_id": "p1", "regions": [{"bbox": [0.0, 0.0, 0.1, 0.1]}] * 2}
    with pytest.raises(ValueError, match="exceeding max_layout_records"):
        module.serialize_record(record, max_records=1)

