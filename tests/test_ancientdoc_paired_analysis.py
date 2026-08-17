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
    / "analyze_ancientdoc_baseline_pages.py"
)
SPEC = importlib.util.spec_from_file_location("ancientdoc_paired_analysis_under_test", MODULE_PATH)
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
    reference: str,
    predicted: str,
    edits: int,
    *,
    exact: bool = False,
) -> dict[str, object]:
    return {
        "page_id": page_id,
        "reference_text": reference,
        "predicted_text": predicted,
        "metrics": {
            "ocr": {
                "edit_distance": edits,
                "reference_characters": len(reference),
                "cer": edits / len(reference),
                "exact_match": exact,
                "whitespace_normalized_edit_distance": edits,
                "whitespace_normalized_reference_characters": len(reference),
                "whitespace_normalized_cer": edits / len(reference),
                "whitespace_normalized_exact_match": exact,
            }
        },
    }


class AncientDocPairedAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.suite_root = self.tmp_path / "ancientdoc_validation"
        self.manifest = self.tmp_path / "manifest.jsonl"
        write_jsonl(
            self.manifest,
            [
                {
                    "page_id": "ancientdoc_split5_000001_imgs_catA_bookA_page_7",
                    "source_group_id": "ancientdoc_book_a",
                    "tier": "real-ancientdoc",
                    "original_image": "imgs/catA/bookA/page_7.png",
                    "page_text": "abcd" * 10,
                    "regions": [],
                },
                {
                    "page_id": "ancientdoc_split5_000002_imgs_catB_bookB_page_42",
                    "source_group_id": "ancientdoc_book_b",
                    "tier": "real-ancientdoc",
                    "original_image": "imgs/catB/bookB/page_42.png",
                    "page_text": "wxyz" * 10,
                    "regions": [],
                },
            ],
        )
        c4_predictions = self.suite_root / "c4" / "layout_validation_predictions.jsonl"
        c6_predictions = self.suite_root / "c6" / "layout_validation_predictions.jsonl"
        write_jsonl(
            c4_predictions,
            [
                prediction(
                    "ancientdoc_split5_000001_imgs_catA_bookA_page_7",
                    "abcd" * 10,
                    "",
                    40,
                ),
                prediction(
                    "ancientdoc_split5_000002_imgs_catB_bookB_page_42",
                    "wxyz" * 10,
                    "wxyz" * 10,
                    0,
                    exact=True,
                ),
            ],
        )
        write_jsonl(
            c6_predictions,
            [
                prediction(
                    "ancientdoc_split5_000001_imgs_catA_bookA_page_7",
                    "abcd" * 10,
                    "abcd" * 8,
                    8,
                ),
                prediction(
                    "ancientdoc_split5_000002_imgs_catB_bookB_page_42",
                    "wxyz" * 10,
                    "",
                    40,
                ),
            ],
        )
        write_json(
            self.suite_root / "c4" / "layout_validation_metrics.json",
            {
                "status": "ok",
                "manifest": str(self.manifest),
                "predictions": str(c4_predictions),
                "metrics": {"ocr": {"page_cer": 0.5}},
            },
        )
        write_json(
            self.suite_root / "c6" / "layout_validation_metrics.json",
            {
                "status": "ok",
                "manifest": str(self.manifest),
                "predictions": str(c6_predictions),
                "metrics": {"ocr": {"page_cer": 0.6}},
            },
        )

    def test_analysis_writes_expected_outputs(self) -> None:
        payload = analysis.analyze(
            analysis.parse_args(
                [
                    "--suite-root",
                    str(self.suite_root),
                    "--bootstrap-samples",
                    "100",
                    "--top-k",
                    "2",
                ]
            )
        )
        self.assertEqual(payload["event"], "ancientdoc_paired_analysis_completed")
        self.assertEqual(payload["pages"], 2)
        self.assertEqual(payload["paired_outcomes"]["c6_better"], 1)
        self.assertEqual(payload["paired_outcomes"]["c6_worse"], 1)
        self.assertEqual(payload["delta_page_cer_right_minus_left"], 0.1)
        output_dir = Path(payload["output_dir"])
        self.assertTrue((output_dir / "summary.json").is_file())
        self.assertTrue((output_dir / "page_comparison.csv").is_file())
        self.assertTrue((output_dir / "group_comparison.csv").is_file())
        self.assertTrue((output_dir / "error_categories.json").is_file())
        self.assertTrue((output_dir / "worst_pages.md").is_file())
        self.assertTrue((output_dir / "analysis_summary.md").is_file())
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["cluster_bootstrap_delta_cer"]["groups"], 2)
        self.assertEqual(
            summary["cluster_bootstrap_delta_cer"]["method"],
            "source_group_id_paired_cluster_bootstrap",
        )
        self.assertEqual(summary["error_categories"]["right_error_categories"]["empty_prediction"], 1)
        groups = {(row["group_by"], row["group"]) for row in summary["groups"]}
        self.assertIn(("category", "catA"), groups)
        self.assertIn(("book", "bookB"), groups)
        analysis_summary = (output_dir / "analysis_summary.md").read_text(encoding="utf-8")
        self.assertIn("Source-group cluster bootstrap", analysis_summary)


if __name__ == "__main__":
    unittest.main()
