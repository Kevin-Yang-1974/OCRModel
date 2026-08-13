from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluation"
    / "analyze_layout_slot_alignment.py"
)
SPEC = importlib.util.spec_from_file_location("layout_slot_alignment_under_test", MODULE_PATH)
slot_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = slot_module
SPEC.loader.exec_module(slot_module)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class LayoutSlotAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.run_root = self.tmp_path / "compare"
        regions = [
            {
                "bbox": [0.1, 0.1, 0.4, 0.4],
                "reading_order": 0,
                "writing_direction": "vertical_rtl",
            },
            {
                "bbox": [0.55, 0.1, 0.85, 0.4],
                "reading_order": 1,
                "writing_direction": "vertical_rtl",
            },
        ]
        write_jsonl(self.tmp_path / "manifest.jsonl", [{"page_id": "page_001", "regions": regions}])
        write_jsonl(
            self.run_root / "vlqa" / "layout_validation_predictions.jsonl",
            [
                {
                    "page_id": "page_001",
                    "regions": regions,
                    "layout_predictions": [
                        {
                            "query_index": 0,
                            "object_probability": 0.9,
                            "bbox_xyxy": [0.55, 0.1, 0.85, 0.4],
                            "writing_direction": "vertical_rtl",
                        },
                        {
                            "query_index": 1,
                            "object_probability": 0.8,
                            "bbox_xyxy": [0.1, 0.1, 0.4, 0.4],
                            "writing_direction": "vertical_rtl",
                        },
                    ],
                }
            ],
        )

    def test_slot_alignment_detects_swapped_queries(self) -> None:
        payload = slot_module.diagnose(
            slot_module.parse_args(
                [
                    "--comparison-root",
                    str(self.run_root),
                    "--manifest",
                    str(self.tmp_path / "manifest.jsonl"),
                ]
            )
        )
        self.assertEqual(payload["event"], "layout_slot_alignment_completed")
        diagnostics = payload["diagnostics"]
        self.assertEqual(diagnostics["targets"], 2)
        self.assertEqual(diagnostics["ordered_hits"], 0)
        self.assertEqual(diagnostics["best_hits"], 2)
        self.assertEqual(diagnostics["slot_misaligned_hits"], 2)
        self.assertGreater(diagnostics["mean_slot_iou_gap"], 0.9)
        output_dir = Path(payload["output_dir"])
        self.assertTrue((output_dir / "slot_alignment_summary.json").is_file())
        self.assertTrue((output_dir / "slot_alignment_targets.csv").is_file())
        self.assertTrue((output_dir / "slot_alignment.md").is_file())


if __name__ == "__main__":
    unittest.main()
