from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "training" / "run_layout_a100.py"
)
SPEC = importlib.util.spec_from_file_location("layout_a100_runner_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def make_args(tmp_path: Path, *extra: str):
    return runner.parse_args(
        [
            "--dataset-root",
            str(tmp_path),
            "--runs-root",
            str(tmp_path / "runs"),
            *extra,
        ]
    )


def option_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


class LayoutA100RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)

    def test_formal_ablation_wrapper_declares_supported_presets(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "training"
            / "run_formal_layout_ablation.sh"
        )
        source = script.read_text(encoding="utf-8")
        self.assertIn("no-direction)", source)
        self.assertIn("no-bbox)", source)
        self.assertIn("object-only)", source)
        self.assertIn("ocr-only-adapter)", source)
        self.assertIn("--direction-loss-weight", source)
        self.assertIn("--bbox-l1-loss-weight", source)

    def formal_ablation_args(self, ablation: str, *extra: str):
        return make_args(
            self.tmp_path,
            "--mode", "ablation", "--ablation", ablation,
            "--validation-manifest", str(self.tmp_path / "validation" / "manifest.jsonl"),
            "--test-manifest", str(self.tmp_path / "test" / "manifest.jsonl"),
            *extra,
        )

    def test_a0_to_a5_stage_contracts(self) -> None:
        a0 = runner.resolve_settings(self.formal_ablation_args("got2_zero_shot"))
        self.assertEqual(a0.stages, ())
        self.assertEqual(a0.validation_model_kind, "baseline")
        for ablation, kind in (
            ("projector_only", "baseline"),
            ("generic_adapter_projector", "generic"),
            ("vlqa_ocr_only", "vlqa"),
            ("vlqa_layout_direct", "vlqa"),
        ):
            settings = runner.resolve_settings(
                self.formal_ablation_args(ablation, "--p2-max-steps", "8000")
            )
            self.assertEqual(settings.stages, ("p2",))
            self.assertEqual(settings.validation_model_kind, kind)
        a5 = runner.resolve_settings(self.formal_ablation_args(
            "vlqa_layout_p1_p2", "--p1-max-steps", "4000", "--p2-max-steps", "8000"
        ))
        self.assertEqual(a5.stages, ("p1", "p2"))

    def test_ablation_training_command_carries_strict_contract(self) -> None:
        settings = runner.resolve_settings(self.formal_ablation_args(
            "vlqa_ocr_only", "--p2-max-steps", "8000", "--checkpoint-steps", "3000"
        ))
        command = runner.build_training_command(
            settings, stage="p2", source_model=self.tmp_path / "original",
            output_dir=self.tmp_path / "p2", master_port=23456,
        )
        self.assertEqual(option_value(command, "--ablation_id"), "vlqa_ocr_only")
        self.assertEqual(option_value(command, "--layout_loss_preset"), "layout_none")
        self.assertEqual(option_value(command, "--layout_loss_weight"), "0")
        self.assertEqual(option_value(command, "--save_total_limit"), "2")

    def test_direct_groups_reject_p1_and_a5_requires_p1(self) -> None:
        with self.assertRaisesRegex(runner.RunFailure, "direct P2"):
            runner.resolve_settings(self.formal_ablation_args(
                "vlqa_layout_direct", "--p1-max-steps", "1", "--p2-max-steps", "2"
            ))
        with self.assertRaisesRegex(runner.RunFailure, "A5 requires"):
            runner.resolve_settings(self.formal_ablation_args(
                "vlqa_layout_p1_p2", "--p2-max-steps", "2"
            ))

    def test_smoke_is_fixed_to_one_record_and_one_step(self) -> None:
        settings = runner.resolve_settings(make_args(self.tmp_path))
        self.assertEqual(settings.mode, "smoke")
        self.assertEqual(settings.p1_max_steps, 1)
        self.assertEqual(settings.p2_max_steps, 1)
        self.assertEqual(settings.p1_max_records, 1)
        self.assertEqual(settings.p2_max_records, 1)
        self.assertEqual(settings.stages, ("p1", "p2"))

        command = runner.build_training_command(
            settings,
            stage="p2",
            source_model=self.tmp_path / "p1-model",
            output_dir=self.tmp_path / "p2-model",
            master_port=23456,
        )
        self.assertEqual(option_value(command, "--layout_stage"), "p2")
        self.assertEqual(
            option_value(command, "--model_name_or_path"),
            str(self.tmp_path / "p1-model"),
        )
        self.assertEqual(option_value(command, "--ocr_loss_weight"), "1")
        self.assertEqual(
            option_value(command, "--tokenizer_name_or_path"),
            str(settings.tokenizer_model),
        )
        self.assertEqual(option_value(command, "--max_steps"), "1")
        self.assertEqual(option_value(command, "--max_train_records"), "1")

    def test_selected_physical_gpu_is_exposed_and_recorded(self) -> None:
        settings = runner.resolve_settings(
            make_args(self.tmp_path, "--gpu-id", "3")
        )
        self.assertEqual(settings.gpu_id, "3")
        self.assertEqual(settings.physical_gpu_ids, ("3",))
        self.assertEqual(
            runner.training_environment(settings)["CUDA_VISIBLE_DEVICES"], "3"
        )

    def test_multiple_physical_gpus_are_exposed_in_declared_order(self) -> None:
        settings = runner.resolve_settings(
            make_args(self.tmp_path, "--gpu-ids", "3,1,4")
        )
        self.assertEqual(settings.gpu_id, "3,1,4")
        self.assertEqual(settings.physical_gpu_ids, ("3", "1", "4"))
        self.assertEqual(
            runner.training_environment(settings)["CUDA_VISIBLE_DEVICES"], "3,1,4"
        )
        self.assertEqual(
            runner.component_smoke_environment(settings)["CUDA_VISIBLE_DEVICES"], "3"
        )

    def test_component_smoke_subprocess_is_restricted_to_first_selected_gpu(self) -> None:
        settings = runner.resolve_settings(
            make_args(
                self.tmp_path,
                "--gpu-ids",
                "3,1",
                "--run-id",
                "component-smoke-env-test",
            )
        )
        (settings.run_root / "metadata").mkdir(parents=True)
        context = runner.RunContext(settings=settings, status={}, summary={})
        payload = {
            "status": "ok",
            "loss_dtype": "torch.float32",
            "object_logit_abs_max": 0.0,
            "direction_logit_abs_max": 0.0,
            "bbox_logit_abs_max": 0.0,
        }
        completed = runner.subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr=""
        )
        with mock.patch.object(
            runner.subprocess, "run", return_value=completed
        ) as run:
            self.assertEqual(runner.run_component_smoke(context), payload)
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")

    def test_gpu_flags_are_mutually_exclusive_and_ids_are_unique(self) -> None:
        with self.assertRaisesRegex(runner.RunFailure, "mutually exclusive"):
            runner.resolve_settings(
                make_args(self.tmp_path, "--gpu-id", "0", "--gpu-ids", "0,1")
            )
        with self.assertRaisesRegex(runner.RunFailure, "unique comma-separated"):
            runner.resolve_settings(make_args(self.tmp_path, "--gpu-ids", "0,0"))

    def test_gpu_locks_are_scoped_per_physical_card(self) -> None:
        source = Path(runner.__file__).read_text(encoding="utf-8")
        self.assertIn("def acquire_gpu_locks", source)
        self.assertIn('.layout_a100.gpu-{gpu_id}.lock', source)
        self.assertNotIn('(runs_root / ".layout_a100.lock")', source)

    def test_parallel_ablation_launchers_keep_gpu_modes_distinct(self) -> None:
        tools = Path(__file__).resolve().parents[1] / "tools" / "training"
        for name in (
            "run_layout_ablation_suite.sh",
            "run_layout_ablation_smoke.sh",
        ):
            source = (tools / name).read_text(encoding="utf-8")
            self.assertIn("--parallel-gpu-ids", source)
            self.assertIn("--gpu-ids", source)
            self.assertIn("count must equal --ablations count", source)
            self.assertIn("GPU%s_BUSY", source)
            self.assertIn("tail -n 20", source)
        suite = (tools / "run_layout_ablation_suite.sh").read_text(encoding="utf-8")
        self.assertIn('run_ablation "$group" "$group_gpu"', suite)
        self.assertIn('--gpu-id "$inference_gpu_id"', suite)
        smoke = (tools / "run_layout_ablation_smoke.sh").read_text(encoding="utf-8")
        self.assertIn('run_group "$group" "${selected_gpu_array[$index]}"', smoke)
        self.assertIn('(metrics.get("training_budget") or {}).get("world_size")', smoke)

    def test_gpu_id_must_be_single_numeric_id(self) -> None:
        with self.assertRaisesRegex(runner.RunFailure, "one physical numeric GPU"):
            runner.resolve_settings(make_args(self.tmp_path, "--gpu-id", "0,1"))

    def test_smoke_rejects_longer_run(self) -> None:
        with self.assertRaisesRegex(runner.RunFailure, "Smoke is fixed"):
            runner.resolve_settings(
                make_args(self.tmp_path, "--p1-max-steps", "2")
            )

    def test_pilot_requires_unlock_and_explicit_steps(self) -> None:
        with self.assertRaisesRegex(runner.RunFailure, "allow-unvalidated-pilot"):
            runner.resolve_settings(make_args(self.tmp_path, "--mode", "pilot"))
        with self.assertRaisesRegex(runner.RunFailure, "explicit"):
            runner.resolve_settings(
                make_args(
                    self.tmp_path,
                    "--mode",
                    "pilot",
                    "--allow-unvalidated-pilot",
                )
            )

        settings = runner.resolve_settings(
            make_args(
                self.tmp_path,
                "--mode",
                "pilot",
                "--allow-unvalidated-pilot",
                "--p1-max-steps",
                "10",
                "--p2-max-steps",
                "20",
            )
        )
        self.assertEqual(settings.p1_max_steps, 10)
        self.assertEqual(settings.p2_max_steps, 20)
        self.assertEqual(settings.p1_max_records, 0)
        self.assertEqual(settings.p2_max_records, 0)

    def test_overfit_is_fixed_to_two_records_and_p1_only(self) -> None:
        settings = runner.resolve_settings(
            make_args(self.tmp_path, "--mode", "overfit")
        )
        self.assertEqual(settings.stages, ("p1",))
        self.assertEqual(settings.p1_max_steps, runner.OVERFIT_P1_STEPS)
        self.assertEqual(settings.p1_max_records, runner.OVERFIT_RECORDS)
        self.assertEqual(settings.p2_max_steps, 0)
        self.assertEqual(settings.p2_max_records, 0)

        command = runner.build_training_command(
            settings,
            stage="p1",
            source_model=self.tmp_path / "source-model",
            output_dir=self.tmp_path / "p1-model",
            master_port=23456,
        )
        self.assertEqual(option_value(command, "--max_steps"), "1000")
        self.assertEqual(option_value(command, "--max_train_records"), "2")
        with self.assertRaisesRegex(runner.RunFailure, "Overfit is fixed"):
            runner.resolve_settings(
                make_args(
                    self.tmp_path,
                    "--mode",
                    "overfit",
                    "--p1-max-steps",
                    "10",
                )
            )

    def test_validate_mode_has_no_training_stages(self) -> None:
        settings = runner.resolve_settings(
            make_args(
                self.tmp_path,
                "--mode",
                "validate",
                "--layout-split",
                "validation",
                "--validation-max-records",
                "7",
                "--validation-object-threshold",
                "0.6",
            )
        )
        self.assertEqual(settings.mode, "validate")
        self.assertEqual(settings.stages, ())
        self.assertEqual(settings.p1_max_steps, 0)
        self.assertEqual(settings.validation_max_records, 7)
        self.assertEqual(settings.validation_object_threshold, 0.6)
        with self.assertRaisesRegex(runner.RunFailure, "does not train"):
            runner.resolve_settings(
                make_args(
                    self.tmp_path,
                    "--mode",
                    "validate",
                    "--p1-max-steps",
                    "3",
                )
            )

    def test_formal_pretrain_is_p1_only_and_requires_held_out_manifests(self) -> None:
        with self.assertRaisesRegex(runner.RunFailure, "validation-manifest"):
            runner.resolve_settings(
                make_args(
                    self.tmp_path,
                    "--mode",
                    "pretrain",
                    "--p1-max-steps",
                    "1200",
                )
            )

        validation_manifest = self.tmp_path / "validation" / "manifest.jsonl"
        test_manifest = self.tmp_path / "test" / "manifest.jsonl"
        settings = runner.resolve_settings(
            make_args(
                self.tmp_path,
                "--mode",
                "pretrain",
                "--p1-max-steps",
                "1200",
                "--validation-manifest",
                str(validation_manifest),
                "--test-manifest",
                str(test_manifest),
            )
        )
        self.assertEqual(settings.stages, ("p1",))
        self.assertEqual(settings.p1_max_steps, 1200)
        self.assertEqual(settings.p2_max_steps, 0)
        self.assertEqual(settings.validation_manifest, validation_manifest.resolve())
        self.assertEqual(settings.test_manifest, test_manifest.resolve())
        self.assertEqual(
            settings.audit_manifests,
            (
                (self.tmp_path / "manifest.jsonl").resolve(),
                validation_manifest.resolve(),
                test_manifest.resolve(),
            ),
        )
        with self.assertRaisesRegex(runner.RunFailure, "pairwise distinct"):
            runner.resolve_settings(
                make_args(
                    self.tmp_path,
                    "--mode",
                    "pretrain",
                    "--p1-max-steps",
                    "1200",
                    "--validation-manifest",
                    str(validation_manifest),
                    "--test-manifest",
                    str(test_manifest),
                    "--test-split",
                    "train",
                )
            )

    def test_formal_joint_train_is_p2_only_from_a_p1_checkpoint(self) -> None:
        validation_manifest = self.tmp_path / "validation" / "manifest.jsonl"
        test_manifest = self.tmp_path / "test" / "manifest.jsonl"
        settings = runner.resolve_settings(
            make_args(
                self.tmp_path,
                "--mode",
                "joint-train",
                "--source-model",
                str(self.tmp_path / "p1-model"),
                "--p2-max-steps",
                "2400",
                "--validation-manifest",
                str(validation_manifest),
                "--test-manifest",
                str(test_manifest),
            )
        )
        self.assertEqual(settings.stages, ("p2",))
        self.assertEqual(settings.p1_max_steps, 0)
        self.assertEqual(settings.p2_max_steps, 2400)
        command = runner.build_training_command(
            settings,
            stage="p2",
            source_model=settings.source_model,
            output_dir=self.tmp_path / "p2-model",
            master_port=23456,
        )
        self.assertEqual(option_value(command, "--layout_stage"), "p2")
        self.assertEqual(option_value(command, "--ocr_loss_weight"), "1")
        self.assertEqual(option_value(command, "--max_steps"), "2400")

        with self.assertRaisesRegex(runner.RunFailure, "P2-only"):
            runner.resolve_settings(
                make_args(
                    self.tmp_path,
                    "--mode",
                    "joint-train",
                    "--p1-max-steps",
                    "1",
                    "--p2-max-steps",
                    "2400",
                    "--validation-manifest",
                    str(validation_manifest),
                    "--test-manifest",
                    str(test_manifest),
                )
            )

    def test_training_command_passes_ablation_loss_weights(self) -> None:
        settings = runner.resolve_settings(
            make_args(
                self.tmp_path,
                "--mode",
                "pilot",
                "--allow-unvalidated-pilot",
                "--p1-max-steps",
                "10",
                "--p2-max-steps",
                "20",
                "--object-loss-weight",
                "0.5",
                "--bbox-l1-loss-weight",
                "0",
                "--bbox-giou-loss-weight",
                "0",
                "--direction-loss-weight",
                "0",
                "--layout-loss-weight",
                "0.25",
                "--p2-ocr-loss-weight",
                "0.75",
            )
        )
        command = runner.build_training_command(
            settings,
            stage="p2",
            source_model=self.tmp_path / "p1-model",
            output_dir=self.tmp_path / "p2-model",
            master_port=23456,
        )
        self.assertEqual(option_value(command, "--object_loss_weight"), "0.5")
        self.assertEqual(option_value(command, "--bbox_l1_loss_weight"), "0")
        self.assertEqual(option_value(command, "--bbox_giou_loss_weight"), "0")
        self.assertEqual(option_value(command, "--direction_loss_weight"), "0")
        self.assertEqual(option_value(command, "--layout_loss_weight"), "0.25")
        self.assertEqual(option_value(command, "--ocr_loss_weight"), "0.75")

    def test_overfit_assessment_is_explicit_and_bounded(self) -> None:
        tail_mean = {
            "object_loss": 0.01,
            "bbox_l1_loss": 0.01,
            "bbox_giou_loss": 0.02,
            "direction_loss": 0.01,
            "object_accuracy": 1.0,
            "bbox_mean_iou": 0.95,
            "direction_accuracy": 1.0,
        }
        metrics = {
            "dataset_examples": 2,
            "layout_adapter_initialization": "fresh_explicit_reset",
            "source_layout_tensor_count": 0,
            "expected_layout_tensor_count": 45,
            "layout_adapter_parameter_abs_max": 1.0,
            "diagnostics": {
                "log_count": runner.OVERFIT_P1_STEPS,
                "tail_window": 20,
                "first": dict(tail_mean),
                "last": dict(tail_mean),
                "tail_mean": tail_mean,
                "tail_min": dict(tail_mean),
                "tail_max": dict(tail_mean),
            },
        }
        passed = runner.assess_p1_overfit(metrics)
        self.assertEqual(passed["status"], "pass")
        self.assertTrue(all(passed["criteria"].values()))
        self.assertEqual(passed["initialization"]["mode"], "fresh_explicit_reset")
        self.assertEqual(passed["initialization"]["parameter_abs_max"], 1.0)
        self.assertEqual(passed["observed_first"]["bbox_mean_iou"], 0.95)
        self.assertEqual(passed["observed_last"]["bbox_mean_iou"], 0.95)
        self.assertEqual(passed["bbox_tail_range"]["bbox_mean_iou_max"], 0.95)

        tail_mean["bbox_mean_iou"] = 0.5
        failed = runner.assess_p1_overfit(metrics)
        self.assertEqual(failed["status"], "fail")
        self.assertFalse(failed["criteria"]["bbox_mean_iou"])

    def test_stage_metrics_require_fp32_component_diagnostics(self) -> None:
        metrics = {
            "layout_stage": "p1",
            "global_step": 1,
            "train_loss": 2.0,
            "first_batch_bbox_shape": [1, 16, 4],
            "first_batch_supervised_tokens": 0,
            "layout_loss_compute_dtype": "float32",
            "layout_adapter_initialization": "fresh_explicit_reset",
            "source_layout_tensor_count": 0,
            "expected_layout_tensor_count": 45,
            "layout_adapter_parameter_abs_max": 1.0,
            "diagnostics": {
                "log_count": 1,
                "tail_mean": {
                    "layout_loss": 2.0,
                    "object_loss": 0.1,
                    "bbox_l1_loss": 0.1,
                    "bbox_giou_loss": 0.1,
                    "direction_loss": 0.1,
                    "object_accuracy": 0.5,
                    "bbox_mean_iou": 0.2,
                    "direction_accuracy": 0.5,
                    "query_abs_max": 2.0,
                    "prediction_query_abs_max": 2.0,
                    "bbox_logit_abs_max": 0.1,
                },
            },
        }
        runner.validate_stage_metrics(
            metrics,
            stage="p1",
            expected_steps=1,
            max_regions=16,
        )
        metrics["layout_loss_compute_dtype"] = "bfloat16"
        with self.assertRaisesRegex(runner.RunFailure, "FP32"):
            runner.validate_stage_metrics(
                metrics,
                stage="p1",
                expected_steps=1,
                max_regions=16,
            )

        metrics["layout_loss_compute_dtype"] = "float32"
        metrics["layout_adapter_initialization"] = "checkpoint_loaded"
        metrics["source_layout_tensor_count"] = 45
        with self.assertRaisesRegex(runner.RunFailure, "fresh VLQA"):
            runner.validate_stage_metrics(
                metrics,
                stage="p1",
                expected_steps=1,
                max_regions=16,
            )

    def test_component_smoke_guards_initial_logit_scale(self) -> None:
        self.assertIn('losses.loss.dtype != torch.float32', runner.COMPONENT_SMOKE)
        self.assertIn('value >= 10.0', runner.COMPONENT_SMOKE)
        self.assertIn('"bbox_logit_abs_max"', runner.COMPONENT_SMOKE)

    def test_synthetic_source_selection_verifies_validation_path_and_hashes(self) -> None:
        model = self.tmp_path / "synthetic" / "checkpoint-2000"
        model.mkdir(parents=True)
        (model / "config.json").write_text(
            json.dumps({"use_vlqa": True}), encoding="utf-8"
        )
        (model / "model.safetensors").write_bytes(b"selected-weights")
        selection = self.tmp_path / "selection.json"
        selection.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "purpose": "layout_ablation_validation_selection",
                    "ablation_id": "vlqa_layout_p1_p2",
                    "selection_split": "validation",
                    "test_used_for_selection": False,
                    "selected": {
                        "optimizer_step": 2000,
                        "model_path": str(model.resolve()),
                        "config_sha256": runner.file_sha256(model / "config.json"),
                        "weights_sha256": runner.file_sha256(
                            model / "model.safetensors"
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )
        payload = runner.load_source_selection(selection, expected_model=model)
        self.assertEqual(payload["optimizer_step"], 2000)
        self.assertEqual(payload["ablation_id"], "vlqa_layout_p1_p2")

        (model / "model.safetensors").write_bytes(b"mutated")
        with self.assertRaisesRegex(ValueError, "weights SHA-256"):
            runner.load_source_selection(selection, expected_model=model)

    def test_diverse_protocol_launcher_keeps_selection_and_test_separate(self) -> None:
        launcher = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "training"
            / "run_diverse_synthetic_ancientdoc.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--source-selection", Path(runner.__file__).read_text(encoding="utf-8"))
        self.assertIn("--selection-only", launcher)
        self.assertIn("--primary-per-replay 7", launcher)
        self.assertIn("--phase select", launcher)
        self.assertIn("--phase test", launcher)


if __name__ == "__main__":
    unittest.main()
