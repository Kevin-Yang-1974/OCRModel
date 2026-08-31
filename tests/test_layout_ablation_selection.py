from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def test_final_model_wins_over_same_step_checkpoint_with_different_hash(self) -> None:
        self.model(self.root, 7500)
        (self.root / "layout_training_metrics.json").write_text(
            json.dumps({"global_step": 7500, "ablation_id": "vlqa_layout_p1_p2"}),
            encoding="utf-8",
        )
        duplicate_step = self.root / "checkpoint-7500"
        self.model(duplicate_step, 7501)
        candidates = selection.discover_candidates(
            self.root, expected_ablation="vlqa_layout_p1_p2"
        )
        self.assertEqual(candidates, [(7500, self.root.resolve())])

    def test_keeps_different_steps_with_same_base_weights_hash(self) -> None:
        self.model(self.root, 12000)
        (self.root / "layout_training_metrics.json").write_text(
            json.dumps({"global_step": 12000, "ablation_id": "vlqa_layout_p1_p2"}), encoding="utf-8"
        )
        for step in (4000, 8000):
            checkpoint = self.root / f"checkpoint-{step}"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(b"same-base-weights")
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        final_checkpoint = self.root / "checkpoint-12000"
        final_checkpoint.mkdir()
        (final_checkpoint / "model.safetensors").write_bytes(b"same-base-weights")
        (final_checkpoint / "config.json").write_text("{}", encoding="utf-8")
        candidates = selection.discover_candidates(
            self.root,
            expected_ablation="vlqa_layout_p1_p2",
            candidate_steps={4000, 8000, 12000},
            prefer_periodic_checkpoint=True,
        )
        self.assertEqual([step for step, _ in candidates], [4000, 8000, 12000])

    def test_resume_preserves_partial_candidate_and_uses_retry_directory(self) -> None:
        output = self.root / "selection"
        partial = output / "step-00007500"
        partial.mkdir(parents=True)
        (partial / "evaluator.log").write_text("failed\n", encoding="utf-8")
        candidate_dir, summary_path = selection.candidate_output_dir(
            output, 7500, resume=True
        )
        self.assertEqual(candidate_dir, output / "step-00007500-retry-01")
        self.assertEqual(summary_path, candidate_dir / "layout_validation_metrics.json")
        self.assertEqual((partial / "evaluator.log").read_text(encoding="utf-8"), "failed\n")

    def test_resume_reuses_completed_retry_candidate(self) -> None:
        output = self.root / "selection"
        partial = output / "step-00007500"
        retry = output / "step-00007500-retry-01"
        partial.mkdir(parents=True)
        retry.mkdir(parents=True)
        (partial / "evaluator.log").write_text("failed\n", encoding="utf-8")
        metrics = retry / "layout_validation_metrics.json"
        metrics.write_text("{}", encoding="utf-8")
        candidate_dir, summary_path = selection.candidate_output_dir(
            output, 7500, resume=True
        )
        self.assertEqual(candidate_dir, retry)
        self.assertEqual(summary_path, metrics)

    def test_non_resume_refuses_partial_candidate(self) -> None:
        output = self.root / "selection"
        partial = output / "step-00007500"
        partial.mkdir(parents=True)
        (partial / "evaluator.log").write_text("failed\n", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "use --resume"):
            selection.candidate_output_dir(output, 7500, resume=False)

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

    def test_parallel_gpu_ids_are_explicit_unique_and_ordered(self) -> None:
        self.assertEqual(selection.parse_gpu_ids("3,1,4", "0"), ("3", "1", "4"))
        with self.assertRaisesRegex(ValueError, "unique"):
            selection.parse_gpu_ids("1,1", "0")
        with self.assertRaisesRegex(ValueError, "numeric"):
            selection.parse_gpu_ids("1,gpu2", "0")

    def test_gpu_admission_queries_only_the_explicit_set_once(self) -> None:
        with mock.patch.object(
            selection, "gpu_utilization", side_effect=[3, 4, 5]
        ) as utilization:
            observed = selection.require_gpus_free(("1", "2", "4"), 50)
        self.assertEqual(observed, {"1": 3, "2": 4, "4": 5})
        self.assertEqual(
            [call.args[0] for call in utilization.call_args_list], ["1", "2", "4"]
        )

    def test_worker_queue_is_sequential_and_preserves_candidate_order(self) -> None:
        candidates = [
            (step, self.root / f"model-{step}", self.root / f"out-{step}",
             self.root / f"out-{step}" / "summary.json", [str(step)])
            for step in (1000, 3000, 5000)
        ]
        observed = []

        def fake_evaluate(candidate, *, gpu_id, project_root):
            observed.append((candidate[0], gpu_id, project_root))
            return candidate[0], candidate[1], candidate[3]

        with mock.patch.object(selection, "evaluate_candidate", side_effect=fake_evaluate):
            results = selection.evaluate_worker_queue(
                candidates, gpu_id="2", project_root=self.root
            )
        self.assertEqual([item[0] for item in results], [1000, 3000, 5000])
        self.assertEqual(observed, [
            (1000, "2", self.root),
            (3000, "2", self.root),
            (5000, "2", self.root),
        ])

if __name__ == "__main__":
    unittest.main()
