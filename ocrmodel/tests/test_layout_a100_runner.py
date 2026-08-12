from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

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
        self.assertEqual(option_value(command, "--max_steps"), "1")
        self.assertEqual(option_value(command, "--max_train_records"), "1")

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


if __name__ == "__main__":
    unittest.main()
