from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "GOT-OCR-2.0"
    / "scripts"
    / "layout_validation_metrics.py"
)
SPEC = importlib.util.spec_from_file_location("layout_validation_metrics_under_test", MODULE_PATH)
metrics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = metrics
SPEC.loader.exec_module(metrics)


def region(order: int, box: list[float], direction: str) -> dict[str, object]:
    return {
        "reading_order": order,
        "bbox": box,
        "writing_direction": direction,
    }


class LayoutValidationMetricTests(unittest.TestCase):
    def test_unicode_levenshtein_and_whitespace(self) -> None:
        self.assertEqual(metrics.levenshtein_distance("甲乙丙", "甲丁丙"), 1)
        self.assertEqual(metrics.remove_whitespace("甲 \n\t乙"), "甲乙")

    def test_ocr_only_accumulator_matches_page_metric_schema(self) -> None:
        accumulator = metrics.OCRValidationAccumulator()
        page = accumulator.add_page("甲 乙", "甲乙")
        self.assertEqual(page["edit_distance"], 1)
        self.assertTrue(page["whitespace_normalized_exact_match"])
        summary = accumulator.summary()
        self.assertEqual(summary["pages"], 1)
        self.assertEqual(summary["page_cer"], 1 / 3)
        self.assertEqual(summary["whitespace_normalized_page_cer"], 0.0)

    def test_spatial_matching_exposes_reversed_query_order(self) -> None:
        page = metrics.evaluate_page(
            reference_text="甲乙",
            predicted_text="甲乙",
            regions=[
                region(0, [0.0, 0.0, 0.4, 1.0], "vertical_rtl"),
                region(1, [0.6, 0.0, 1.0, 1.0], "horizontal_ltr"),
            ],
            annotation_status="complete",
            object_scores=[0.99, 0.99],
            predicted_boxes=[
                [0.6, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.4, 1.0],
            ],
            predicted_directions=[0, 2],
            object_threshold=0.5,
            iou_threshold=0.5,
        )
        self.assertEqual(page["layout"]["matched_regions"], 2)
        self.assertEqual(page["layout"]["region_recall"], 1.0)
        self.assertEqual(page["layout"]["ordered_slot_bbox_mean_iou"], 0.0)
        self.assertEqual(page["layout"]["matched_direction_accuracy"], 1.0)
        self.assertEqual(page["layout"]["reading_order_pair_accuracy"], 0.0)
        self.assertEqual(page["layout"]["reading_order_kendall_tau"], -1.0)

    def test_accumulator_reports_micro_ocr_and_complete_region_metrics(self) -> None:
        accumulator = metrics.LayoutValidationAccumulator(
            object_threshold=0.5,
            iou_threshold=0.5,
        )
        accumulator.add_page(
            reference_text="甲乙",
            predicted_text="甲丙",
            regions=[region(0, [0.1, 0.1, 0.9, 0.9], "horizontal_rtl")],
            annotation_status="complete",
            object_scores=[0.9, 0.8],
            predicted_boxes=[
                [0.1, 0.1, 0.9, 0.9],
                [0.0, 0.0, 0.05, 0.05],
            ],
            predicted_directions=[1, 4],
        )
        accumulator.add_page(
            reference_text="丁",
            predicted_text="丁",
            regions=[region(0, [0.2, 0.2, 0.8, 0.8], "vertical_ltr")],
            annotation_status="complete",
            object_scores=[0.1, 0.1],
            predicted_boxes=[
                [0.2, 0.2, 0.8, 0.8],
                [0.0, 0.0, 0.05, 0.05],
            ],
            predicted_directions=[3, 4],
        )
        summary = accumulator.summary()
        self.assertAlmostEqual(summary["ocr"]["page_cer"], 1 / 3)
        self.assertEqual(summary["ocr"]["page_exact_match_rate"], 0.5)
        self.assertEqual(summary["layout"]["region_recall"], 0.5)
        self.assertEqual(summary["layout"]["complete_region_precision"], 0.5)
        self.assertEqual(summary["layout"]["complete_region_recall"], 0.5)
        self.assertEqual(summary["layout"]["ordered_slot_bbox_mean_iou"], 1.0)
        self.assertEqual(summary["layout"]["ordered_direction_accuracy"], 1.0)

    def test_partial_annotations_do_not_define_precision(self) -> None:
        page = metrics.evaluate_page(
            reference_text="甲",
            predicted_text="甲",
            regions=[region(0, [0.1, 0.1, 0.4, 0.4], "unknown")],
            annotation_status="partial",
            object_scores=[0.9, 0.9],
            predicted_boxes=[
                [0.1, 0.1, 0.4, 0.4],
                [0.6, 0.6, 0.9, 0.9],
            ],
            predicted_directions=[4, 4],
            object_threshold=0.5,
            iou_threshold=0.5,
        )
        self.assertEqual(page["layout"]["region_recall"], 1.0)
        self.assertIsNone(page["layout"]["region_precision"])

    def test_degenerate_prediction_is_counted_but_not_matched(self) -> None:
        accumulator = metrics.LayoutValidationAccumulator()
        page = accumulator.add_page(
            reference_text="x",
            predicted_text="x",
            regions=[region(0, [0.1, 0.1, 0.4, 0.4], "unknown")],
            annotation_status="complete",
            object_scores=[0.9, 0.9],
            predicted_boxes=[
                [0.1, 0.1, 0.4, 0.4],
                [0.5, 0.2, 0.5, 0.7],
            ],
            predicted_directions=[4, 4],
        )
        self.assertEqual(page["layout"]["predicted_regions"], 2)
        self.assertEqual(page["layout"]["invalid_predicted_boxes"], 1)
        self.assertEqual(page["layout"]["matched_regions"], 1)
        self.assertEqual(page["layout"]["region_precision"], 0.5)
        self.assertEqual(
            accumulator.summary()["layout"]["invalid_predicted_boxes"], 1
        )

    def test_duplicate_count_and_true_count_buckets_are_aggregated(self) -> None:
        accumulator = metrics.LayoutValidationAccumulator(object_threshold=0.0)
        predictions = [
            [0.1, 0.1, 0.4, 0.4],
            [0.1, 0.1, 0.4, 0.4],
        ]
        page = accumulator.add_page(
            reference_text="x",
            predicted_text="x",
            regions=[region(0, [0.1, 0.1, 0.4, 0.4], "unknown")],
            annotation_status="complete",
            object_scores=[0.8, 0.3],
            predicted_boxes=predictions,
            predicted_directions=[4, 4],
        )
        self.assertEqual(page["layout"]["duplicate_regions_iou_0_9"], 2)
        summary = accumulator.summary()["layout"]
        self.assertEqual(summary["region_count_mae"], 1.0)
        self.assertEqual(summary["region_count_exact_accuracy"], 0.0)
        self.assertEqual(summary["duplicate_region_rate_iou_0_9"], 1.0)
        self.assertEqual(summary["true_region_count_buckets"]["0-8"]["pages"], 1)


if __name__ == "__main__":
    unittest.main()
