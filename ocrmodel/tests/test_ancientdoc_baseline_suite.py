from __future__ import annotations

import argparse
import importlib.util
import ast
import json
import re
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "tools" / "training" / "run_got2_page_ocr_a100.py"
SPEC = importlib.util.spec_from_file_location("got2_page_ocr_runner_under_test", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(RUNNER_PATH.parent))
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

AUDIT_VERIFIER_PATH = ROOT / "tools" / "preprocessing" / "verify_ancientdoc_group_audit.py"
AUDIT_SPEC = importlib.util.spec_from_file_location(
    "ancientdoc_group_audit_verifier_under_test", AUDIT_VERIFIER_PATH
)
assert AUDIT_SPEC is not None and AUDIT_SPEC.loader is not None
audit_verifier = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(audit_verifier)

EVALUATION_PATH = ROOT / "tools" / "evaluation" / "run_ancientdoc_evaluation.py"
EVALUATION_SPEC = importlib.util.spec_from_file_location(
    "ancientdoc_evaluation_under_test", EVALUATION_PATH
)
assert EVALUATION_SPEC is not None and EVALUATION_SPEC.loader is not None
evaluation = importlib.util.module_from_spec(EVALUATION_SPEC)
sys.modules[EVALUATION_SPEC.name] = evaluation
EVALUATION_SPEC.loader.exec_module(evaluation)

CONTRACT_PATH = ROOT / "tools" / "training" / "c4_selection_contract.py"
CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "c4_selection_contract_under_test", CONTRACT_PATH
)
assert CONTRACT_SPEC is not None and CONTRACT_SPEC.loader is not None
contract = importlib.util.module_from_spec(CONTRACT_SPEC)
sys.modules[CONTRACT_SPEC.name] = contract
CONTRACT_SPEC.loader.exec_module(contract)

SELECT_C4_PATH = ROOT / "tools" / "evaluation" / "select_ancientdoc_c4.py"
SELECT_C4_SPEC = importlib.util.spec_from_file_location(
    "select_ancientdoc_c4_under_test", SELECT_C4_PATH
)
assert SELECT_C4_SPEC is not None and SELECT_C4_SPEC.loader is not None
select_c4 = importlib.util.module_from_spec(SELECT_C4_SPEC)
sys.modules[SELECT_C4_SPEC.name] = select_c4
SELECT_C4_SPEC.loader.exec_module(select_c4)


class AncientDocBaselineSuiteTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)

    @staticmethod
    def write_model(path: Path, weight_bytes: bytes) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(
            json.dumps({"use_vlqa": True, "vlqa_num_queries": 16}),
            encoding="utf-8",
        )
        tensor_names = (
            "model.layers.0.self_attn.q_proj.weight",
            "model.mm_projector_vary.weight",
            "model.layout_adapter.object_head.weight",
        )
        data = (weight_bytes * 12)[:12].ljust(12, b"0")
        header = json.dumps(
            {
                name: {
                    "dtype": "U8",
                    "shape": [4],
                    "data_offsets": [index * 4, (index + 1) * 4],
                }
                for index, name in enumerate(tensor_names)
            },
            separators=(",", ":"),
        ).encode("utf-8")
        (path / "model.safetensors").write_bytes(
            struct.pack("<Q", len(header)) + header + data
        )

    def make_c4_run(self) -> Path:
        run = self.tmp_path / "c4-run"
        model = run / "p2" / "model"
        self.write_model(model, b"step-12000")
        (model / "layout_training_metrics.json").write_text(
            json.dumps({"layout_stage": "p2", "global_step": 12000}),
            encoding="utf-8",
        )
        for step in (4000, 2000, 12000):
            checkpoint = model / f"checkpoint-{step}"
            weights = b"step-12000" if step == 12000 else f"step-{step}".encode()
            self.write_model(checkpoint, weights)
            (checkpoint / "trainer_state.json").write_text(
                json.dumps({"global_step": step}), encoding="utf-8"
            )
        (run / "LAYOUT_A100_FINISHED").touch()
        return run

    def write_c4_selection(self, run: Path, selected_step: int = 4000) -> Path:
        model = run / "p2" / "model" / f"checkpoint-{selected_step}"
        provenance = contract.checkpoint_provenance(model)
        trainer_state = model / "trainer_state.json"
        provenance.update(
            {
                "trainer_state_path": str(trainer_state),
                "trainer_state_sha256": contract.file_sha256(trainer_state),
            }
        )
        selection = self.tmp_path / "selection.json"
        candidate = {
            "candidate_id": f"step-{selected_step:06d}",
            "optimizer_step": selected_step,
            "model_path": str(model.resolve()),
            "validation_metrics": {
                "pages": 516,
                "page_cer": 0.8,
                "whitespace_normalized_page_cer": 0.7,
                "page_exact_matches": 2,
            },
            "provenance": provenance,
        }
        selection.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "purpose": "c4_checkpoint_selection",
                    "c4_run_root": str(run.resolve()),
                    "dataset_root": str(self.tmp_path / "dataset"),
                    "manifest": str(self.tmp_path / "dataset" / "validation" / "manifest.jsonl"),
                    "selection_split": "validation",
                    "test_used_for_selection": False,
                    "evaluator": {
                        "prompt": "OCR: ",
                        "decoding": "greedy",
                        "max_new_tokens": 2048,
                        "no_repeat_ngram_size": 20,
                        "batch_size": 1,
                        "layout_metadata_as_model_input": False,
                        "max_records": 0,
                    },
                    "candidates": [candidate],
                    "selected": candidate,
                }
            ),
            encoding="utf-8",
        )
        return selection

    def test_c1_runner_defaults_to_12000_steps_and_selected_gpu(self) -> None:
        args = runner.parse_args(
            [
                "--dataset-root", str(self.tmp_path / "train"),
                "--manifest", str(self.tmp_path / "train" / "manifest.jsonl"),
                "--validation-manifest", str(self.tmp_path / "validation" / "manifest.jsonl"),
                "--test-manifest", str(self.tmp_path / "test" / "manifest.jsonl"),
                "--run-id", "c1_test",
                "--gpu-id", "4",
            ]
        )
        self.assertEqual(args.max_steps, 12000)
        self.assertEqual(args.gpu_id, "4")
        self.assertEqual(args.train_scope, "decoder_projector")
        self.assertEqual(args.learning_rate, 2e-5)
        self.assertEqual(args.checkpoint_steps, 2000)
        self.assertEqual(args.lr_scheduler_type, "cosine")

    def test_c1_training_excludes_layout_targets_and_records_budget(self) -> None:
        source = (ROOT / "src" / "GOT-OCR-2.0" / "scripts" / "train_GOT_page_ocr.py").read_text(encoding="utf-8")
        self.assertIn("include_layout_targets=False", source)
        self.assertIn('"use_vlqa": False', source)
        self.assertIn('"strict_equal_parameter_control_vs_c4": False', source)
        self.assertIn('"training_budget": budget', source)
        self.assertIn('"layout_metadata_as_model_input": False', source)

    def test_training_suite_has_explicit_core_and_replay_phases(self) -> None:
        source = (ROOT / "tools" / "training" / "run_ancientdoc_baseline_suite.sh").read_text(encoding="utf-8")
        self.assertIn('phase=""', source)
        self.assertIn('"${phase}" == "core"', source)
        self.assertIn("replay phase requires --c4-selection", source)
        self.assertIn("c4_selection_contract.py", source)
        self.assertIn("run_replay_branch c5_vlqa_ocr_replay 0", source)
        self.assertIn("run_replay_branch c6_vlqa_layout_replay 1", source)
        self.assertEqual(source.count('--source-model "${c4_model}"'), 1)
        self.assertNotIn('source-model "${GOT_TRAINING_RUNS}/${run_id}', source)
        self.assertIn("--skip-post-training-validation", source)
        self.assertNotIn("run_ancientdoc_validation_suite.sh", source)
        self.assertIn("verify_ancientdoc_group_audit.py", source)

        wrapper = (ROOT / "tools" / "training" / "run_ancientdoc.sh").read_text(encoding="utf-8")
        for command in ("train-core", "select-c4", "train-replay"):
            self.assertIn(command, wrapper)
        self.assertIn("legacy train", wrapper)

    def test_group_audit_verifier_reads_persisted_nested_schema(self) -> None:
        audit_path = self.tmp_path / "split_leakage_audit.json"
        checks = {
            name: {"cross_split_value_count": 0}
            for name in audit_verifier.REQUIRED_CHECKS
        }
        audit_path.write_text(
            json.dumps({"schema_version": 1, "status": "ok", "checks": checks}),
            encoding="utf-8",
        )
        self.assertEqual(
            audit_verifier.validate_audit(audit_path),
            {name: 0 for name in audit_verifier.REQUIRED_CHECKS},
        )

    def test_group_audit_verifier_rejects_nonzero_nested_count(self) -> None:
        audit_path = self.tmp_path / "split_leakage_audit.json"
        checks = {
            name: {"cross_split_value_count": 0}
            for name in audit_verifier.REQUIRED_CHECKS
        }
        checks["book_key"]["cross_split_value_count"] = 1
        audit_path.write_text(
            json.dumps({"schema_version": 1, "status": "ok", "checks": checks}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "checks.book_key"):
            audit_verifier.validate_audit(audit_path)

    def test_evaluation_suite_has_dynamic_queue_and_required_deltas(self) -> None:
        source = (ROOT / "tools" / "evaluation" / "run_ancientdoc_evaluation.py").read_text(encoding="utf-8")
        for label in (
            "c0_got2_zero_shot",
            "c1_got2_ocr_only",
            '"c4"',
            '"c5"',
            '"c6"',
        ):
            self.assertIn(label, source)
        for delta in (
            "c1_minus_c0",
            "c4_minus_c1",
            "c5_minus_c4",
            "c6_minus_c4",
            "c6_minus_c5",
        ):
            self.assertIn(delta, source)
        self.assertIn("free_gpus.append(gpu)", source)
        self.assertIn('"CUDA_VISIBLE_DEVICES": gpu', source)
        self.assertIn("discover_candidates", source)
        self.assertIn("select_best", source)
        self.assertNotIn("wave_count", source)

    def test_evaluation_python_compiles_and_shell_is_thin(self) -> None:
        source = (ROOT / "tools" / "evaluation" / "run_ancientdoc_evaluation.py").read_text(encoding="utf-8")
        ast.parse(source)
        wrapper = (ROOT / "tools" / "evaluation" / "run_ancientdoc.sh").read_text(encoding="utf-8")
        self.assertIn("run_ancientdoc_evaluation.py", wrapper)
        self.assertLess(len(wrapper.splitlines()), 20)

    def test_checkpoint_discovery_and_validation_selection(self) -> None:
        model = self.tmp_path / "model"
        for step in (2000, 4000):
            checkpoint = model / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "model.safetensors").write_bytes(b"weights")
        candidates = evaluation.discover_candidates(model)
        self.assertEqual([step for _, step in candidates], [2000, 4000])

        results = []
        for step, cer in ((2000, 0.7), (4000, 0.8)):
            results.append(
                {
                    "_label": "c1_got2_ocr_only",
                    "_step": step,
                    "model": str(model / f"checkpoint-{step}"),
                    "_summary_path": f"step-{step}/summary.json",
                    "metrics": {
                        "ocr": {
                            "page_cer": cer,
                            "whitespace_normalized_page_cer": cer,
                        }
                    },
                }
            )
        selected = evaluation.select_best(results, "c1_got2_ocr_only")
        self.assertEqual(selected["step"], 2000)

    def test_c4_discovery_sorts_steps_and_deduplicates_final(self) -> None:
        run = self.make_c4_run()
        candidates, excluded = select_c4.discover_c4_candidates(run)
        self.assertEqual(
            [candidate.optimizer_step for candidate in candidates],
            [2000, 4000, 12000],
        )
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["optimizer_step"], 12000)
        self.assertEqual(
            excluded[0]["duplicate_of"],
            str((run / "p2" / "model" / "checkpoint-12000").resolve()),
        )

    def test_c4_discovery_rejects_incomplete_checkpoint(self) -> None:
        run = self.make_c4_run()
        (run / "p2" / "model" / "checkpoint-4000" / "model.safetensors").unlink()
        with self.assertRaises((FileNotFoundError, ValueError)):
            select_c4.discover_c4_candidates(run)

    def test_c4_discovery_rejects_corrupt_or_partial_safetensors(self) -> None:
        run = self.make_c4_run()
        weights = run / "p2" / "model" / "checkpoint-4000" / "model.safetensors"
        weights.write_bytes(b"not-a-safetensors-checkpoint")
        with self.assertRaisesRegex(ValueError, "safetensors"):
            select_c4.discover_c4_candidates(run)

    def test_c4_selection_rule_uses_cer_whitespace_then_earlier_step(self) -> None:
        def item(step: int, cer: float, whitespace: float) -> dict[str, object]:
            return {
                "optimizer_step": step,
                "model_path": f"checkpoint-{step}",
                "validation_metrics": {
                    "page_cer": cer,
                    "whitespace_normalized_page_cer": whitespace,
                },
            }

        selected = select_c4.select_candidate(
            [
                item(2000, 0.81, 0.60),
                item(4000, 0.80, 0.72),
                item(6000, 0.80, 0.70),
                item(8000, 0.80, 0.70),
            ]
        )
        self.assertEqual(selected["optimizer_step"], 6000)

    def test_c4_selection_contract_parses_and_verifies_hashes(self) -> None:
        run = self.make_c4_run()
        selection = self.write_c4_selection(run)
        resolved = contract.load_c4_selection(selection)
        self.assertEqual(resolved["selected_step"], 4000)
        self.assertEqual(
            Path(resolved["selected_model_path"]),
            (run / "p2" / "model" / "checkpoint-4000").resolve(),
        )
        self.write_model(run / "p2" / "model" / "checkpoint-4000", b"changed")
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            contract.load_c4_selection(selection)

    def test_c4_selector_is_validation_only_and_uses_dynamic_queue(self) -> None:
        source = SELECT_C4_PATH.read_text(encoding="utf-8")
        self.assertIn('split="validation"', source)
        self.assertNotIn('dataset_root / "test"', source)
        self.assertIn("evaluation.run_queue", source)
        self.assertIn('"test_used_for_selection": False', source)
        self.assertIn('args.batch_size = 1', source)
        self.assertIn('args.max_new_tokens = 2048', source)
        self.assertIn('args.no_repeat_ngram_size = 20', source)

    def test_final_evaluation_requires_frozen_c4_selection(self) -> None:
        source = EVALUATION_PATH.read_text(encoding="utf-8")
        self.assertIn("--c4-selection", source)
        self.assertIn("validate_replay_branch_consistency", source)
        self.assertIn("Final selection C4 differs", source)
        self.assertIn("optimizer_state_initialization", source)

    def test_training_suite_jsonl_resolves_all_trained_models(self) -> None:
        summary = self.tmp_path / "suite_summary.jsonl"
        records = (
            ("c1_got2_ocr_only", "c1"),
            ("c4_vlqa_ocr_only", "c4"),
            ("c5_vlqa_ocr_replay", "c5"),
            ("c6_vlqa_layout_replay", "c6"),
        )
        summary.write_text(
            "".join(
                json.dumps(
                    {
                        "event": "baseline_finished",
                        "baseline": baseline,
                        "model": str(self.tmp_path / model),
                    }
                )
                + "\n"
                for baseline, model in records
            ),
            encoding="utf-8",
        )
        args = types.SimpleNamespace(
            c0_model=self.tmp_path / "c0",
            training_suite=summary,
            c1_model=None,
            c4_model=None,
            c5_model=None,
            c6_model=None,
            phase="select",
        )
        models = evaluation.training_model_args(
            args,
            {"selected_model_path": str(self.tmp_path / "c4-selected")},
        )
        self.assertEqual(set(models), set(evaluation.LABELS))
        self.assertEqual(models["c6"], (self.tmp_path / "c6").resolve())
        self.assertEqual(models["c4"], (self.tmp_path / "c4-selected").resolve())

    def test_queue_reuse_keeps_job_identity_for_shared_model_path(self) -> None:
        model = self.tmp_path / "shared-model"
        jobs = []
        for label, step in (("c4", 2000), ("c5", 4000)):
            output = self.tmp_path / label
            output.mkdir()
            (output / "layout_validation_metrics.json").write_text(
                json.dumps({"status": "ok", "model": str(model), "metrics": {"ocr": {"page_cer": 1.0}}}),
                encoding="utf-8",
            )
            jobs.append(
                evaluation.EvaluationJob(
                    label=label,
                    model_kind="vlqa",
                    model_path=model,
                    output_dir=output,
                    split="validation",
                    step=step,
                )
            )
        args = types.SimpleNamespace(gpu_ids=["0"])
        results = evaluation.run_queue(
            args,
            jobs,
            self.tmp_path,
            self.tmp_path / "queue.jsonl",
        )
        self.assertEqual(
            [(item["_label"], item["_step"]) for item in results],
            [("c4", 2000), ("c5", 4000)],
        )

    def test_ratio_deviation_must_be_nonnegative_and_finite(self) -> None:
        self.assertEqual(evaluation.nonnegative_finite_float("0.03"), 0.03)
        for value in ("-0.1", "nan", "inf"):
            with self.assertRaises(argparse.ArgumentTypeError):
                evaluation.nonnegative_finite_float(value)


if __name__ == "__main__":
    unittest.main()
