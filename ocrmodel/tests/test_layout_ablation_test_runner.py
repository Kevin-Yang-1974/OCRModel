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
PROJECT_ROOT = Path(__file__).resolve().parents[1] / "src" / "GOT-OCR-2.0"
SPEC = importlib.util.spec_from_file_location("layout_ablation_test_under_test", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

class LayoutAblationTestRunnerTests(unittest.TestCase):
    def test_parallel_gpu_ids_are_unique_and_explicit(self) -> None:
        self.assertEqual(runner.parse_gpu_ids("0,1,3", "4"), ("0", "1", "3"))
        with self.assertRaisesRegex(ValueError, "unique"):
            runner.parse_gpu_ids("0,0", "4")

    def test_manifest_shards_are_disjoint_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "test.jsonl"
            records = [{"page_id": f"p{index}"} for index in range(8)]
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            loaded, shard_paths = runner.prepare_manifest_shards(manifest, root / "run", 3)
            shards = [runner.read_jsonl(path) for path in shard_paths]
            self.assertEqual(loaded, records)
            self.assertEqual([len(shard) for shard in shards], [2, 3, 3])
            self.assertEqual([record for shard in shards for record in shard], records)

    def test_parallel_merge_restores_manifest_order_and_recomputes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "test.jsonl"
            manifest_records = [{"page_id": page_id} for page_id in ("p0", "p1", "p2")]
            manifest.write_text(
                "".join(json.dumps(record) + "\n" for record in manifest_records),
                encoding="utf-8",
            )

            def prediction(page_id: str, reference: str, predicted: str) -> dict:
                return {
                    "page_id": page_id,
                    "image": f"{page_id}.jpg",
                    "reference_text": reference,
                    "predicted_text": predicted,
                    "layout_annotation_status": "complete",
                    "regions": [{
                        "reading_order": 0,
                        "bbox": [0.1, 0.1, 0.9, 0.9],
                        "writing_direction": "vertical_rtl",
                    }],
                    "layout_predictions": [{
                        "region_index": 0,
                        "type": "region",
                        "bbox": [0.1, 0.1, 0.9, 0.9],
                        "direction": "vertical_rtl",
                        "score": 0.8,
                        "region_token_probability": 0.8,
                    }],
                    "generated_eos": True,
                    "truncated_by_max_layout_tokens": False,
                    "stopped_by_max_layout_records": False,
                    "num_layout_tokens": 12,
                }

            runtime = {
                "dtype": "bfloat16",
                "layout_forward_seconds": 2.0,
                "ocr_generation_seconds": 3.0,
                "evaluation_loop_seconds": 6.0,
                "generation_token_slots": 20,
                "peak_cuda_memory_bytes": 100,
            }
            shard_records = [
                [prediction("p0", "ab", "ab"), prediction("p2", "d", "x")],
                [prediction("p1", "c", "c")],
            ]
            shard_outputs = []
            for index, records in enumerate(shard_records):
                output = root / f"shard-{index}"
                output.mkdir()
                (output / "layout_validation_predictions.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )
                (output / "layout_validation_metrics.json").write_text(
                    json.dumps({
                        "input_protocol": {
                            "model_inputs": ["whole_page_image", "ocr_prompt"],
                            "layout_metadata_as_model_input": False,
                        },
                        "layout": {},
                        "runtime": runtime,
                        "metrics": {},
                    }),
                    encoding="utf-8",
                )
                shard_outputs.append(output)

            merged = runner.merge_shard_evaluations(
                project_root=PROJECT_ROOT,
                manifest=manifest,
                manifest_records=manifest_records,
                shard_outputs=shard_outputs,
                output=root / "merged",
                model_kind="pvld",
                object_threshold=0.0,
            )
            predictions = runner.read_jsonl(
                root / "merged" / "evaluation" / "layout_validation_predictions.jsonl"
            )
            self.assertEqual([record["page_id"] for record in predictions], ["p0", "p1", "p2"])
            self.assertEqual(merged["metrics"]["ocr"]["character_edits"], 1)
            self.assertEqual(merged["metrics"]["ocr"]["reference_characters"], 4)
            self.assertEqual(merged["metrics"]["ocr"]["page_cer"], 0.25)
            self.assertEqual(merged["metrics"]["layout"]["complete_matched_regions"], 3)
            self.assertEqual(merged["layout"]["eos_success_rate"], 1.0)

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
