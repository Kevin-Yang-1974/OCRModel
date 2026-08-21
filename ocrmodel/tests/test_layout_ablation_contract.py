from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "src" / "GOT-OCR-2.0" / "scripts" / "layout_ablation_contract.py"
SPEC = importlib.util.spec_from_file_location("layout_ablation_contract_under_test", CONTRACT_PATH)
contract = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = contract
SPEC.loader.exec_module(contract)

class LayoutAblationContractTests(unittest.TestCase):
    def test_non_vlqa_training_does_not_collate_layout_targets(self) -> None:
        source = (
            Path(__file__).resolve().parents[1]
            / "src" / "GOT-OCR-2.0" / "scripts" / "train_GOT_layout.py"
        ).read_text(encoding="utf-8")
        self.assertIn("or model.get_model().variable_layout_adapter is not None", source)
        self.assertIn('"layout_bbox_mask" in first_sample or variable_layout', source)

    def test_vqlca_mode_is_persisted_and_training_generation_share_adapter(self) -> None:
        training_source = (
            ROOT / "src" / "GOT-OCR-2.0" / "scripts" / "train_GOT_layout.py"
        ).read_text(encoding="utf-8")
        model_source = (
            ROOT / "src" / "GOT-OCR-2.0" / "GOT" / "model" / "GOT_ocr_2_0.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "config.layout_writeback_mode = args.layout_writeback_mode",
            training_source,
        )
        self.assertIn(
            '"legacy_shared_loaded_vqlca_writeback_reset"',
            training_source,
        )
        self.assertEqual(model_source.count("layout_output = self.layout_adapter("), 1)
        self.assertIn("writeback_mode=config.layout_writeback_mode", model_source)
        self.assertIn("layout_writeback_source", training_source)
        self.assertIn("visual_value_layout_routing", training_source)
        self.assertIn("layout_memory=cnn_feature", model_source)

    def test_whole_page_protocol_does_not_pass_layout_metadata_to_adapter(self) -> None:
        model_source = (
            ROOT / "src" / "GOT-OCR-2.0" / "GOT" / "model" / "GOT_ocr_2_0.py"
        ).read_text(encoding="utf-8")
        call = model_source.split("layout_output = self.layout_adapter(", 1)[1].split(")", 1)[0]
        self.assertIn("image_feature", call)
        for forbidden in ("bbox", "direction", "reading_order"):
            self.assertNotIn(forbidden, call)

    def test_all_required_groups_are_declared(self) -> None:
        self.assertEqual(len(contract.ABLATION_IDS), 6)
        self.assertFalse(contract.ABLATIONS["got2_zero_shot"].training_allowed)
        self.assertTrue(contract.ABLATIONS["generic_adapter_projector"].use_generic_adapter)
        self.assertTrue(contract.ABLATIONS["vlqa_layout_p1_p2"].p1_required)

    def test_loss_presets_are_exact(self) -> None:
        self.assertEqual(contract.LOSS_PRESETS["layout_none"]["layout"], 0.0)
        self.assertEqual(contract.LOSS_PRESETS["object_bbox"]["direction_order"], 0.0)
        self.assertGreater(contract.LOSS_PRESETS["layout_full"]["bbox_l1"], 0.0)

    def test_a3_requires_zero_layout_losses(self) -> None:
        weights = contract.loss_weights_for("vlqa_ocr_only", "p2", "layout_none", 1.0)
        self.assertEqual(weights["ocr"], 1.0)
        self.assertEqual(weights["layout"], 0.0)
        with self.assertRaisesRegex(ValueError, "layout_none"):
            contract.loss_weights_for("vlqa_ocr_only", "p2", "layout_full", 1.0)

    def test_a4_and_a5_require_layout_supervision(self) -> None:
        with self.assertRaisesRegex(ValueError, "enabled layout loss"):
            contract.loss_weights_for("vlqa_layout_direct", "p2", "layout_none", 1.0)
        with self.assertRaisesRegex(ValueError, "OCR loss weight 0"):
            contract.loss_weights_for("vlqa_layout_p1_p2", "p1", "layout_full", 1.0)

    def test_checkpoint_source_contract(self) -> None:
        original = {"use_vlqa": False, "use_generic_adapter": False}
        contract.assert_source_protocol("vlqa_layout_direct", "p2", original, None)
        with self.assertRaisesRegex(ValueError, "original GOT2"):
            contract.assert_source_protocol(
                "vlqa_layout_direct", "p2", {"use_vlqa": True}, {"layout_stage": "p1"}
            )
        contract.assert_source_protocol(
            "vlqa_layout_p1_p2", "p2", {"use_vlqa": True},
            {"layout_stage": "p1", "ablation_id": "vlqa_layout_p1_p2"},
        )
        with self.assertRaisesRegex(ValueError, "same A5 run"):
            contract.assert_source_protocol("vlqa_layout_p1_p2", "p2", original, None)

    def test_actual_parameter_counts_define_each_scope(self) -> None:
        def report(*trainable: str):
            return {
                name: {"trainable": 100 if name in trainable else 0}
                for name in ("vary_vit", "qwen", "mm_projector_vary", "generic_adapter", "vlqa")
            }
        cases = {
            ("got2_zero_shot", "p2"): (),
            ("projector_only", "p2"): ("mm_projector_vary",),
            ("generic_adapter_projector", "p2"): ("mm_projector_vary", "generic_adapter"),
            ("vlqa_ocr_only", "p2"): ("mm_projector_vary", "vlqa"),
            ("vlqa_layout_direct", "p2"): ("mm_projector_vary", "vlqa"),
            ("vlqa_layout_p1_p2", "p1"): ("vlqa",),
            ("vlqa_layout_p1_p2", "p2"): ("mm_projector_vary", "vlqa"),
        }
        for (ablation, stage), names in cases.items():
            contract.assert_parameter_report(ablation, stage, report(*names))
        with self.assertRaisesRegex(ValueError, "frozen Vary ViT"):
            contract.assert_parameter_report("projector_only", "p2", report("vary_vit", "mm_projector_vary"))

if __name__ == "__main__":
    unittest.main()
