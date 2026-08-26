from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src" / "GOT-OCR-2.0" / "GOT" / "model"
TRAIN = ROOT / "src" / "GOT-OCR-2.0" / "scripts" / "train_GOT_layout.py"
DATASET = ROOT / "src" / "GOT-OCR-2.0" / "scripts" / "layout_page_dataset.py"
COMMON = ROOT / "tools" / "preprocessing" / "synthetic_layout_common.py"
SMOKE = ROOT / "tools" / "training" / "smoke_visual_memory.py"


class VisualReplayProtocolTest(unittest.TestCase):
    def test_vary_exposes_intermediate_without_changing_default_forward(self) -> None:
        source = (MODEL / "vision_encoder" / "vary_b.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        encoder = next(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == "ImageEncoderViT")
        methods = {node.name for node in encoder.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("forward_features", methods)
        self.assertIn("forward", methods)
        self.assertIn("return_intermediate", source)
        self.assertIn('"layout_memory_64"', source)
        self.assertIn('"ocr_features_16"', source)

    def test_got_forward_keeps_qwen_at_256_tokens_and_removes_training_no_grad(self) -> None:
        source = (MODEL / "GOT_ocr_2_0.py").read_text(encoding="utf-8")
        self.assertIn("layout_memory_resolution", source)
        self.assertIn("high_resolution_padding_mask", source)
        self.assertIn("ignore_mismatched_sizes", (TRAIN).read_text(encoding="utf-8"))
        self.assertNotIn("with torch.set_grad_enabled(False):", source)
        self.assertIn("return_intermediate=self.variable_layout_adapter is not None", source)

    def test_replay_protocol_is_registered(self) -> None:
        source = TRAIN.read_text(encoding="utf-8")
        self.assertIn("primary_per_replay: int = field(default=7)", source)
        self.assertIn("replay_ocr_loss_weight: float = field(default=0.25)", source)
        self.assertIn("vision_learning_rate", source)
        self.assertIn("projector_learning_rate", source)
        self.assertIn("layout_learning_rate", source)
        self.assertIn("qwen_learning_rate", source)
        self.assertIn("gate_learning_rate", source)
        self.assertIn('layout_stage in {"p2", "p3"}', source)

    def test_dataset_marks_replay_and_keeps_replay_ocr(self) -> None:
        source = DATASET.read_text(encoding="utf-8")
        self.assertIn('source_kind: str = "primary"', source)
        self.assertIn('source_kind="replay"', source)
        self.assertIn('supervise_ocr=True,', source)
        self.assertIn("replay_sample_mask", source)
        self.assertIn("primary_per_replay: int = 7", source)

    def test_high_difficulty_tiers_and_count_buckets_exist(self) -> None:
        source = COMMON.read_text(encoding="utf-8")
        for tier in ("s3-ancient-hard", "s4-mixed"):
            self.assertIn(tier, source)
        for bucket in ("33-64", "65-128", ">128"):
            self.assertIn(bucket, source)

    def test_bounded_memory_smoke_covers_backward_reload_and_ddp_contract(self) -> None:
        source = SMOKE.read_text(encoding="utf-8")
        for token in ("return_intermediate=True", "loss.backward()", "load_state_dict", "all_reduce"):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main()
