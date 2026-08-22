from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "evaluation" / "evaluate_layout_ablation_test.py"
SPEC = importlib.util.spec_from_file_location("layout_ablation_test_under_test", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

class LayoutAblationTestRunnerTests(unittest.TestCase):
    def test_mock_evaluator_writes_compact_provenance_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            (model / "model.safetensors").write_bytes(b"weights")
            manifest = root / "manifest.jsonl"
            manifest.write_text("{}\n", encoding="utf-8")
            selection = root / "selection.json"
            selection.write_text(json.dumps({
                "purpose": "layout_ablation_validation_selection",
                "test_used_for_selection": False,
                "locked_object_threshold": 0.0,
                "ablation_id": "projector_only",
                "selected": {
                    "optimizer_step": 2000,
                    "model_path": str(model),
                    "config_sha256": runner.sha256(model / "config.json"),
                    "weights_sha256": runner.sha256(model / "model.safetensors"),
                },
            }), encoding="utf-8")
            output = root / "test-output"
            evaluator_environments = []

            def fake_evaluator(command, **kwargs):
                evaluator_environments.append(kwargs["env"])
                evaluator_output = Path(command[command.index("--output-dir") + 1])
                evaluator_output.mkdir(parents=True)
                (evaluator_output / "layout_validation_metrics.json").write_text(
                    json.dumps({
                        "input_protocol": {
                            "model_inputs": ["whole_page_image", "ocr_prompt"],
                            "layout_metadata_as_model_input": False,
                        },
                        "inference_failures": 0,
                        "metrics": {"ocr": {
                            "page_cer": 0.2,
                            "whitespace_normalized_page_cer": 0.1,
                            "total_edit_distance": 2,
                            "total_reference_characters": 10,
                            "exact_matches": 1,
                            "pages": 2,
                        }, "layout": None},
                    }), encoding="utf-8"
                )
                return SimpleNamespace(returncode=0)

            argv = [
                "--selection", str(selection), "--test-category", "Synthetic-ID",
                "--test-manifest", str(manifest), "--model-kind", "baseline",
                "--tokenizer-model", str(model), "--project-root", str(root),
                "--output-dir", str(output), "--gpu-id", "3",
            ]
            with mock.patch.object(runner, "require_gpu_free"), mock.patch.object(
                runner.subprocess, "run", side_effect=fake_evaluator
            ):
                self.assertEqual(runner.main(argv), 0)
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["selected_step"], 2000)
            self.assertEqual(summary["input_granularity"], "whole_page_image")
            self.assertFalse(summary["input_protocol"]["layout_metadata_as_model_input"])
            self.assertEqual(summary["metrics"]["ocr"]["total_reference_characters"], 10)
            self.assertEqual(summary["inference_physical_gpu"], "3")
            self.assertEqual(evaluator_environments[0]["CUDA_VISIBLE_DEVICES"], "3")
            self.assertEqual(summary["locked_object_threshold"], 0.0)
            self.assertEqual(summary["test_threshold_source"], "validation_selection")

if __name__ == "__main__":
    unittest.main()
