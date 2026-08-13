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
    / "summarize_layout_analysis_bundle.py"
)
SPEC = importlib.util.spec_from_file_location("layout_analysis_bundle_under_test", MODULE_PATH)
bundle_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = bundle_module
SPEC.loader.exec_module(bundle_module)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


class LayoutAnalysisBundleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.run_root = self.tmp_path / "compare"
        analysis_dir = self.run_root / "analysis"
        write_json(
            self.run_root / "summary.json",
            {
                "comparison": {
                    "ocr_page_cer_delta_vlqa_minus_baseline": -0.26,
                    "vlqa_layout": {
                        "complete_region_f1": 0.787,
                    },
                }
            },
        )
        write_json(
            analysis_dir / "error_analysis_summary.json",
            {
                "overview": {
                    "pages": 300,
                    "baseline_edit_distance": 9556,
                    "vlqa_edit_distance": 2026,
                    "delta_edit_distance_vlqa_minus_baseline": -7530,
                    "baseline_exact_matches": 11,
                    "vlqa_exact_matches": 75,
                    "ocr_outcomes": {"improved": 264, "same": 16, "worse": 20},
                    "layout_failure_types": {
                        "miss_and_extra": 169,
                        "layout_ok": 124,
                    },
                },
                "groups": [
                    {
                        "group_by": "tier",
                        "group": "s2-hard",
                        "pages": 100,
                        "delta_edit_distance_vlqa_minus_baseline": -1000,
                        "delta_page_cer_from_predictions": -0.1,
                        "layout_failure_types": {"miss_and_extra": 70},
                        "mean_layout_ordered_slot_bbox_iou": 0.55,
                    },
                    {
                        "group_by": "layout_failure_type",
                        "group": "miss_and_extra",
                        "pages": 169,
                        "delta_edit_distance_vlqa_minus_baseline": -2000,
                        "delta_page_cer_from_predictions": -0.12,
                        "layout_failure_types": {"miss_and_extra": 169},
                        "mean_layout_ordered_slot_bbox_iou": 0.49,
                    },
                ],
                "top_pages": {
                    "largest_vlqa_regressions": [
                        {
                            "page_id": "p_bad",
                            "tier": "s2-hard",
                            "region_count": 8,
                            "delta_edit_distance": 10,
                            "layout_failure_type": "miss_and_extra",
                        }
                    ],
                    "worst_layout_slot_iou": [
                        {
                            "page_id": "p_iou",
                            "tier": "s1-html-crop",
                            "region_count": 6,
                            "delta_edit_distance": -2,
                            "layout_failure_type": "miss_and_extra",
                            "layout_ordered_slot_bbox_mean_iou": 0.1,
                        }
                    ],
                },
            },
        )
        write_json(
            analysis_dir / "threshold_sweep" / "threshold_sweep_summary.json",
            {
                "best_by_complete_region_f1": {
                    "object_threshold": 0.3,
                    "complete_region_f1": 0.789,
                }
            },
        )
        write_json(
            analysis_dir / "slot_alignment" / "slot_alignment_summary.json",
            {
                "summary": {
                    "targets": 1752,
                    "ordered_hit_rate": 0.77,
                    "best_hit_rate": 0.806,
                    "slot_misaligned_hit_rate": 0.036,
                    "mean_ordered_iou": 0.648,
                    "mean_best_iou": 0.682,
                    "best_query_offset_counts": {"0": 1494, "1": 91, "-1": 52},
                    "worst_pages_by_slot_gap": [
                        {"page_id": "p_gap", "slot_iou_gap_sum": 4.4}
                    ],
                }
            },
        )

    def test_bundle_summarizes_existing_diagnostics(self) -> None:
        payload = bundle_module.summarize(
            bundle_module.parse_args(
                [
                    "--comparison-root",
                    str(self.run_root),
                    "--top-k",
                    "2",
                ]
            )
        )
        self.assertEqual(payload["event"], "layout_analysis_bundle_completed")
        self.assertEqual(payload["pages"], 300)
        headline = payload["headline"]
        self.assertEqual(headline["ocr_outcomes"]["improved"], 264)
        self.assertAlmostEqual(headline["threshold_f1_gain_over_default"], 0.002)
        residual = payload["residual_diagnosis"]
        self.assertFalse(residual["threshold_is_main_issue"])
        self.assertFalse(residual["slot_misalignment_is_main_issue"])
        self.assertEqual(residual["top_query_offsets"][0], {"offset": "0", "count": 1494})
        output_dir = Path(payload["output_dir"])
        self.assertTrue((output_dir / "analysis_bundle_summary.json").is_file())
        self.assertTrue((output_dir / "analysis_bundle.md").is_file())
        summary = json.loads(
            (output_dir / "analysis_bundle_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            summary["page_priorities"]["worst_slot_gap_pages"][0]["page_id"],
            "p_gap",
        )


if __name__ == "__main__":
    unittest.main()
