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
    / "analyze_layout_threshold_sweep.py"
)
SPEC = importlib.util.spec_from_file_location("layout_threshold_sweep_under_test", MODULE_PATH)
sweep_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = sweep_module
SPEC.loader.exec_module(sweep_module)


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


class LayoutThresholdSweepTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.run_root = self.tmp_path / "compare"
        manifest = [
            {
                "page_id": "page_001",
                "regions": [
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
                ],
            }
        ]
        predictions = [
            {
                "page_id": "page_001",
                "reference_text": "甲乙",
                "predicted_text": "甲乙",
                "layout_annotation_status": "complete",
                "regions": manifest[0]["regions"],
                "layout_predictions": [
                    {
                        "query_index": 0,
                        "object_probability": 0.9,
                        "bbox_xyxy": [0.1, 0.1, 0.4, 0.4],
                        "writing_direction": "vertical_rtl",
                    },
                    {
                        "query_index": 1,
                        "object_probability": 0.8,
                        "bbox_xyxy": [0.55, 0.1, 0.85, 0.4],
                        "writing_direction": "vertical_rtl",
                    },
                    {
                        "query_index": 2,
                        "object_probability": 0.3,
                        "bbox_xyxy": [0.1, 0.55, 0.4, 0.85],
                        "writing_direction": "vertical_rtl",
                    },
                ],
            }
        ]
        write_jsonl(self.tmp_path / "manifest.jsonl", manifest)
        write_jsonl(
            self.run_root / "vlqa" / "layout_validation_predictions.jsonl",
            predictions,
        )

    def test_threshold_sweep_identifies_higher_threshold_as_better(self) -> None:
        payload = sweep_module.sweep(
            sweep_module.parse_args(
                [
                    "--comparison-root",
                    str(self.run_root),
                    "--manifest",
                    str(self.tmp_path / "manifest.jsonl"),
                    "--threshold",
                    "0.2",
                    "--threshold",
                    "0.5",
                ]
            )
        )
        self.assertEqual(payload["event"], "layout_threshold_sweep_completed")
        self.assertEqual(payload["pages"], 1)
        self.assertEqual(payload["best_by_complete_region_f1"]["object_threshold"], 0.5)
        output_dir = Path(payload["output_dir"])
        self.assertTrue((output_dir / "threshold_sweep_summary.json").is_file())
        summary = json.loads(
            (output_dir / "threshold_sweep_summary.json").read_text(encoding="utf-8")
        )
        low, high = summary["rows"]
        self.assertEqual(low["complete_predicted_regions"], 3)
        self.assertEqual(high["complete_predicted_regions"], 2)
        self.assertLess(low["complete_region_f1"], high["complete_region_f1"])


if __name__ == "__main__":
    unittest.main()
