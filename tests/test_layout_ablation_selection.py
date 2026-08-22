from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "evaluation" / "select_layout_ablation_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("layout_ablation_selection_under_test", MODULE_PATH)
selection = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = selection
SPEC.loader.exec_module(selection)

class LayoutAblationSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def model(self, path: Path, step: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(f"weights-{step}".encode())

    def test_discovers_periodic_and_final_checkpoints(self) -> None:
        self.model(self.root, 8000)
        (self.root / "layout_training_metrics.json").write_text(
            json.dumps({"global_step": 8000, "ablation_id": "vlqa_layout_direct"}), encoding="utf-8"
        )
        self.model(self.root / "checkpoint-2000", 2000)
        self.model(self.root / "checkpoint-4000", 4000)
        self.assertEqual(
            [step for step, _ in selection.discover_candidates(
                self.root, expected_ablation="vlqa_layout_direct"
            )],
            [2000, 4000, 8000],
        )

    def test_selection_uses_page_cer_then_whitespace_then_step(self) -> None:
        candidates = [
            {"optimizer_step": 4000, "validation_metrics": {"page_cer": 0.2, "whitespace_normalized_page_cer": 0.1}},
            {"optimizer_step": 2000, "validation_metrics": {"page_cer": 0.2, "whitespace_normalized_page_cer": 0.1}},
            {"optimizer_step": 8000, "validation_metrics": {"page_cer": 0.3, "whitespace_normalized_page_cer": 0.05}},
        ]
        self.assertEqual(selection.select_best(candidates)["optimizer_step"], 2000)

    def test_p1_selection_uses_validation_layout_rank_not_ocr(self) -> None:
        def candidate(step: int, stopping: float, count_mae: float, f1: float) -> dict:
            return {
                "optimizer_step": step,
                "validation_metrics": {
                    "eos_success_rate": 1.0 - stopping,
                    "premature_eos_rate": 0.0,
                    "token_cap_rate": 0.0,
                    "record_cap_rate": 0.0,
                    "region_count_mae": count_mae,
                    "complete_region_f1": f1,
                    "matched_bbox_mean_iou": 0.5,
                    "ordered_bbox_mean_iou": 0.4,
                    "duplicate_region_rate_iou_0_9": 0.1,
                    "region_count_exact_accuracy": 0.2,
                },
            }
        candidates = [
            candidate(12000, 0.2, 10.0, 0.5),
            candidate(9000, 0.1, 12.0, 0.4),
        ]
        selected = selection.select_best(candidates, "p1_layout")
        self.assertEqual(selected["optimizer_step"], 9000)
        self.assertNotIn("page_cer", selected["validation_metrics"])

    def test_normalizes_current_evaluator_metric_names(self) -> None:
        normalized = selection.normalize_ocr_metrics({
            "page_cer": 0.2,
            "whitespace_normalized_page_cer": 0.1,
            "character_edits": 20,
            "reference_characters": 100,
            "page_exact_matches": 3,
            "pages": 10,
        })
        self.assertEqual(normalized, {
            "page_cer": 0.2,
            "whitespace_normalized_page_cer": 0.1,
            "total_edit_distance": 20,
            "total_reference_characters": 100,
            "exact_matches": 3,
            "pages": 10,
        })

    def test_normalizes_legacy_metric_names(self) -> None:
        normalized = selection.normalize_ocr_metrics({
            "page_cer": 0.2,
            "whitespace_normalized_page_cer": 0.1,
            "total_edit_distance": 20,
            "total_reference_characters": 100,
            "exact_matches": 3,
            "pages": 10,
        })
        self.assertEqual(normalized["total_edit_distance"], 20)
        self.assertEqual(normalized["total_reference_characters"], 100)
        self.assertEqual(normalized["exact_matches"], 3)

    def test_normalization_reports_missing_metric_aliases(self) -> None:
        with self.assertRaisesRegex(KeyError, "total_edit_distance or character_edits"):
            selection.normalize_ocr_metrics({
                "page_cer": 0.2,
                "whitespace_normalized_page_cer": 0.1,
                "reference_characters": 100,
                "page_exact_matches": 3,
                "pages": 10,
            })

    def resumable_summary(self, model: Path, manifest: Path) -> dict:
        return {
            "status": "ok",
            "model": str(model.resolve()),
            "model_kind": "baseline",
            "manifest": str(manifest.resolve()),
            "split": "validation",
            "inference_failures": 0,
            "input_protocol": {
                "model_inputs": ["whole_page_image", "ocr_prompt"],
                "layout_metadata_as_model_input": False,
            },
            "decoding": {
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": 2048,
                "no_repeat_ngram_size": 20,
            },
            "metrics": {"ocr": {"page_cer": 0.2}},
        }

    def test_loads_matching_resumable_candidate_summary(self) -> None:
        model = self.root / "checkpoint-2000"
        manifest = self.root / "validation.jsonl"
        summary_path = self.root / "summary.json"
        summary_path.write_text(
            json.dumps(self.resumable_summary(model, manifest)), encoding="utf-8"
        )
        loaded = selection.load_resumable_candidate_summary(
            summary_path,
            model=model,
            model_kind="baseline",
            validation_manifest=manifest,
            max_new_tokens=2048,
            no_repeat_ngram_size=20,
        )
        self.assertEqual(loaded["model"], str(model.resolve()))

    def test_rejects_resumable_candidate_with_mismatched_protocol(self) -> None:
        model = self.root / "checkpoint-2000"
        manifest = self.root / "validation.jsonl"
        payload = self.resumable_summary(model, manifest)
        payload["input_protocol"]["layout_metadata_as_model_input"] = True
        summary_path = self.root / "summary.json"
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "input_protocol"):
            selection.load_resumable_candidate_summary(
                summary_path,
                model=model,
                model_kind="baseline",
                validation_manifest=manifest,
                max_new_tokens=2048,
                no_repeat_ngram_size=20,
            )

    def test_zero_shot_is_step_zero_without_training_metrics(self) -> None:
        self.model(self.root, 0)
        self.assertEqual(selection.discover_candidates(self.root, zero_shot=True), [(0, self.root.resolve())])

    def test_selection_accepts_explicit_physical_gpu(self) -> None:
        args = selection.parse_args([
            "--ablation", "projector_only",
            "--model-root", str(self.root),
            "--model-kind", "baseline",
            "--tokenizer-model", str(self.root),
            "--validation-manifest", str(self.root / "validation.jsonl"),
            "--output-dir", str(self.root / "selection"),
            "--project-root", str(self.root),
            "--gpu-id", "3",
        ])
        self.assertEqual(args.gpu_id, "3")

if __name__ == "__main__":
    unittest.main()
