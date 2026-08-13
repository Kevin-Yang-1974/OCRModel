from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "evaluation"
    / "compare_got2_vlqa.py"
)
SPEC = importlib.util.spec_from_file_location("layout_comparison_under_test", MODULE_PATH)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


class LayoutComparisonRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)

    def parse(self, *extra: str):
        return comparison.parse_args(
            [
                "--baseline-model",
                str(self.tmp_path / "baseline"),
                "--vlqa-model",
                str(self.tmp_path / "vlqa"),
                "--layout-manifest",
                str(self.tmp_path / "test" / "manifest.jsonl"),
                "--layout-split",
                "test",
                "--output-root",
                str(self.tmp_path / "runs"),
                *extra,
            ]
        )

    def test_formal_comparison_defaults_to_strict_test_protocol(self) -> None:
        args = self.parse(
            "--train-manifest",
            str(self.tmp_path / "train" / "manifest.jsonl"),
            "--validation-manifest",
            str(self.tmp_path / "validation" / "manifest.jsonl"),
        )
        resolved = comparison.resolve_args(args)
        self.assertEqual(resolved["split"], "test")
        self.assertEqual(resolved["train_split"], "train")
        self.assertEqual(resolved["validation_split"], "validation")
        self.assertEqual(resolved["baseline_model"], (self.tmp_path / "baseline").resolve())
        self.assertEqual(resolved["vlqa_model"], (self.tmp_path / "vlqa").resolve())
        self.assertEqual(len(resolved["audit_manifests"]), 3)
        self.assertFalse(args.skip_audit)

    def test_formal_test_rejects_incomplete_cross_split_audit(self) -> None:
        with self.assertRaisesRegex(comparison.ComparisonFailure, "train-manifest"):
            comparison.resolve_args(self.parse())
        with self.assertRaisesRegex(comparison.ComparisonFailure, "pairwise distinct"):
            comparison.resolve_args(
                self.parse(
                    "--train-manifest",
                    str(self.tmp_path / "train" / "manifest.jsonl"),
                    "--validation-manifest",
                    str(self.tmp_path / "validation" / "manifest.jsonl"),
                    "--train-split",
                    "test",
                )
            )

    def test_non_test_split_requires_explicit_diagnostic_unlock(self) -> None:
        with self.assertRaisesRegex(comparison.ComparisonFailure, "requires --layout-split test"):
            comparison.resolve_args(
                comparison.parse_args(
                    [
                        "--baseline-model",
                        str(self.tmp_path / "baseline"),
                        "--vlqa-model",
                        str(self.tmp_path / "vlqa"),
                        "--layout-manifest",
                        str(self.tmp_path / "validation" / "manifest.jsonl"),
                        "--layout-split",
                        "validation",
                    ]
                )
            )

        args = comparison.parse_args(
            [
                "--baseline-model",
                str(self.tmp_path / "baseline"),
                "--vlqa-model",
                str(self.tmp_path / "vlqa"),
                "--layout-manifest",
                str(self.tmp_path / "validation" / "manifest.jsonl"),
                "--layout-split",
                "validation",
                "--allow-non-test-split",
            ]
        )
        self.assertEqual(comparison.resolve_args(args)["split"], "validation")

    def test_evaluator_command_requires_p2_for_vlqa_only(self) -> None:
        source = MODULE_PATH.read_text(encoding="utf-8")
        self.assertIn('if model_kind == "vlqa":', source)
        self.assertIn('("--require-vlqa-stage", "p2")', source)
        self.assertIn('"--model-kind",', source)
        self.assertIn('"baseline"', source)

    def test_post_analysis_wrapper_runs_expected_pipeline(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "tools"
            / "evaluation"
            / "run_formal_layout_post_analysis.sh"
        )
        source = script.read_text(encoding="utf-8")
        self.assertIn("run_formal_layout_comparison.sh", source)
        self.assertIn("analyze_layout_comparison_errors.py", source)
        self.assertIn("analyze_layout_threshold_sweep.py", source)
        self.assertIn("analyze_layout_slot_alignment.py", source)
        self.assertIn("summarize_layout_analysis_bundle.py", source)
        self.assertIn("comparison run already exists", source)
        self.assertIn("exit 74", source)


if __name__ == "__main__":
    unittest.main()
