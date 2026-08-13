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
    / "analyze_layout_comparison_errors.py"
)
SPEC = importlib.util.spec_from_file_location("layout_error_analysis_under_test", MODULE_PATH)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def prediction(
    page_id: str,
    edits: int,
    *,
    exact: bool = False,
    layout: dict[str, object] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "page_id": page_id,
        "reference_text": "甲乙丙丁" * 10,
        "predicted_text": "",
        "metrics": {
            "ocr": {
                "edit_distance": edits,
                "reference_characters": 40,
                "cer": edits / 40,
                "exact_match": exact,
                "whitespace_normalized_edit_distance": edits,
                "whitespace_normalized_reference_characters": 40,
                "whitespace_normalized_cer": edits / 40,
                "whitespace_normalized_exact_match": exact,
            }
        },
    }
    if layout is not None:
        record["metrics"]["layout"] = layout  # type: ignore[index]
    return record


class LayoutComparisonErrorAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.run_root = self.tmp_path / "compare"
        write_json(
            self.run_root / "summary.json",
            {
                "status": "ok",
                "comparison": {
                    "ocr_page_cer_delta_vlqa_minus_baseline": -0.1,
                },
            },
        )
        write_json(
            self.run_root / "baseline" / "layout_validation_metrics.json",
            {
                "status": "ok",
                "manifest": str(self.tmp_path / "manifest.jsonl"),
                "metrics": {"ocr": {"page_cer": 0.5}},
            },
        )
        write_json(
            self.run_root / "vlqa" / "layout_validation_metrics.json",
            {
                "status": "ok",
                "manifest": str(self.tmp_path / "manifest.jsonl"),
                "metrics": {"ocr": {"page_cer": 0.25}},
            },
        )
        manifest = [
            {
                "page_id": "page_s0_001",
                "tier": "s0-html-text",
                "source_group_id": "source_a",
                "regions": [
                    {"writing_direction": "vertical_rtl"},
                    {"writing_direction": "vertical_rtl"},
                ],
            },
            {
                "page_id": "page_s2_002",
                "tier": "s2-hard",
                "source_group_id": "source_b",
                "regions": [{"writing_direction": "horizontal_ltr"} for _ in range(6)],
            },
        ]
        write_jsonl(self.tmp_path / "manifest.jsonl", manifest)
        write_jsonl(
            self.run_root / "baseline" / "layout_validation_predictions.jsonl",
            [
                prediction("page_s0_001", 20),
                prediction("page_s2_002", 5, exact=False),
            ],
        )
        write_jsonl(
            self.run_root / "vlqa" / "layout_validation_predictions.jsonl",
            [
                prediction(
                    "page_s0_001",
                    8,
                    layout={
                        "ground_truth_regions": 2,
                        "predicted_regions": 2,
                        "matched_regions": 2,
                        "region_precision": 1.0,
                        "region_recall": 1.0,
                        "ordered_slot_bbox_mean_iou": 0.8,
                        "matched_bbox_mean_iou": 0.8,
                    },
                ),
                prediction(
                    "page_s2_002",
                    12,
                    layout={
                        "ground_truth_regions": 6,
                        "predicted_regions": 8,
                        "matched_regions": 3,
                        "region_precision": 0.375,
                        "region_recall": 0.5,
                        "ordered_slot_bbox_mean_iou": 0.2,
                        "matched_bbox_mean_iou": 0.7,
                    },
                ),
            ],
        )

    def test_analysis_writes_summary_csv_and_markdown(self) -> None:
        payload = analysis.analyze(
            analysis.parse_args(
                [
                    "--comparison-root",
                    str(self.run_root),
                    "--manifest",
                    str(self.tmp_path / "manifest.jsonl"),
                    "--top-k",
                    "2",
                ]
            )
        )
        self.assertEqual(payload["event"], "layout_comparison_error_analysis_completed")
        self.assertEqual(payload["pages"], 2)
        self.assertEqual(payload["overview"]["delta_edit_distance_vlqa_minus_baseline"], -5)
        output_dir = Path(payload["output_dir"])
        self.assertTrue((output_dir / "error_analysis_summary.json").is_file())
        self.assertTrue((output_dir / "page_error_analysis.csv").is_file())
        self.assertTrue((output_dir / "group_error_analysis.csv").is_file())
        self.assertTrue((output_dir / "error_analysis.md").is_file())
        summary = json.loads((output_dir / "error_analysis_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["overview"]["ocr_outcomes"]["improved"], 1)
        self.assertEqual(summary["overview"]["ocr_outcomes"]["worse"], 1)
        groups = {(row["group_by"], row["group"]) for row in summary["groups"]}
        self.assertIn(("tier", "s0-html-text"), groups)
        self.assertIn(("layout_failure_type", "miss_and_extra"), groups)


if __name__ == "__main__":
    unittest.main()
