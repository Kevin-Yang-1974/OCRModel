"""Small CPU contract checks; also usable in the remote environment before smoke."""
from __future__ import annotations

import ast
import math
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from layout_ocr.aligned_recovery import align_prefix, recovery_target, rebuild_inputs, recovery_losses, collect_rollout


class RecoveryTests(unittest.TestCase):
    block = list(range(10, 18))
    eos = {99}

    def test_recover_without_positional_shift(self):
        target = recovery_target(self.block * 3, self.block + [40, 41, 99], self.eos)
        self.assertTrue(target["accepted"])
        self.assertEqual(target["suffix"], [40, 41, 99])
        self.assertEqual(target["prefix"], self.block * 3)

    def test_legal_repetition_is_not_penalized(self):
        target = recovery_target(self.block * 3, self.block * 3 + [40, 99], self.eos)
        self.assertFalse(target["accepted"])

    def test_early_cycle_is_seen(self):
        target = recovery_target(self.block * 3 + list(range(200, 400)), self.block + [40, 99], self.eos)
        self.assertTrue(target["accepted"])
        self.assertEqual(target["cycle"]["end"], 24)

    def test_overlong_prefix_can_end(self):
        target = recovery_target(self.block * 3, self.block + [99], self.eos)
        self.assertTrue(target["accepted"])
        self.assertEqual(target["suffix"], [99])

    def test_unread_content_does_not_end(self):
        target = recovery_target(self.block * 3, self.block + list(range(30, 60)) + [99], self.eos)
        self.assertTrue(target["accepted"])
        self.assertEqual(target["suffix"], list(range(30, 46)))

    def test_deleted_content_prevents_end(self):
        prefix = list(range(50, 90)) + self.block * 3
        gold = list(range(50, 90)) + [1] + self.block + [99]
        target = recovery_target(prefix, gold, self.eos)
        self.assertFalse(target["accepted"])
        self.assertEqual(target["reason"], "unsafe_end")

    def test_rebuilt_shapes_and_causal_boundary(self):
        inputs = {"input_ids": torch.tensor([[1, 2, 10, 99]]),
                  "labels": torch.tensor([[-100, -100, 10, 99]]),
                  "attention_mask": torch.ones(1, 4, dtype=torch.long),
                  "mm_token_type_ids": torch.tensor([[1, 0, 0, 0]]),
                  "position_ids": torch.arange(4)[None], "pixel_values": torch.ones(2, 3)}
        rebuilt = rebuild_inputs(inputs, 2, self.block * 3, [40, 99])
        self.assertEqual(rebuilt["input_ids"].shape, (1, 28))
        self.assertEqual(rebuilt["attention_mask"].shape, (1, 28))
        self.assertEqual(rebuilt["mm_token_type_ids"].shape, (1, 28))
        self.assertIs(rebuilt["pixel_values"], inputs["pixel_values"])
        self.assertNotIn("position_ids", rebuilt)
        self.assertEqual(int(rebuilt["labels"][0, 26]), 40)
        rollout = {"accepted": True, "detected": True, "prefix_inputs": rebuilt, "boundary": 25,
                   "negative": 10, "gold_next": 40, "suffix": [40, 99]}
        logits = torch.zeros(1, 28, 100, requires_grad=True)
        losses = recovery_losses(logits, rollout, self.eos)
        losses["loss"].backward()
        self.assertLess(float(logits.grad[0, 25, 40]), 0)
        self.assertGreater(float(logits.grad[0, 25, 10]), 0)
        self.assertLess(float(logits.grad[0, 26, 99]), 0)
        self.assertEqual(float(logits.grad[:, :25].abs().sum()), 0)
        self.assertEqual(float(losses["end_tokens"]), 1)

    def test_zero_sample_has_finite_zero_gradient(self):
        logits = torch.randn(1, 5, 100, requires_grad=True)
        rollout = {"accepted": False, "prefix_inputs": {"labels": torch.ones(1, 5, dtype=torch.long)}}
        losses = recovery_losses(logits, rollout, self.eos)
        losses["loss"].backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_wrong_cycle_is_detected_and_targets_loop_token(self):
        gold = list(range(1000, 1100))
        wrong = list(range(10, 18))
        generated = gold[:20] + wrong * 3
        target = recovery_target(generated, gold, self.eos)
        self.assertTrue(target["detected"])
        self.assertFalse(target["accepted"])
        self.assertEqual(target["negative"], wrong[0])
        self.assertEqual(target["prefix"], generated)
        self.assertIn("gold_next", target)

    def test_wrong_cycle_after_long_read_is_recovered(self):
        gold = list(range(1000, 1100))
        wrong = list(range(10, 18))
        generated = gold[:40] + wrong * 3
        target = recovery_target(generated, gold, self.eos)
        self.assertTrue(target["accepted"])
        self.assertEqual(target["endpoint"], 40)
        self.assertEqual(target["suffix"], gold[40:56])
        self.assertEqual(target["prefix"], generated)

    def test_detected_rejected_loop_still_gets_ul(self):
        inputs = {"input_ids": torch.tensor([[1, 2, 10, 10, 10]]),
                  "labels": torch.full((1, 5), -100, dtype=torch.long),
                  "attention_mask": torch.ones(1, 5, dtype=torch.long),
                  "mm_token_type_ids": torch.tensor([[1, 0, 0, 0, 0]])}
        logits = torch.zeros(1, 5, 100, requires_grad=True)
        rollout = {"accepted": False, "detected": True, "reason": "ambiguous_endpoint",
                   "prefix_inputs": inputs, "boundary": 3, "negative": 10,
                   "gold_next": 40, "suffix": []}
        losses = recovery_losses(logits, rollout, self.eos)
        losses["loss"].backward()
        self.assertGreater(float(losses["active_tokens"]), 0)
        self.assertGreater(float(logits.grad[0, 3, 10]), 0)
        self.assertEqual(float(logits.grad[:, :3].abs().sum()), 0)

    def test_rejected_loop_skips_ul_when_token_is_gold_next(self):
        inputs = {"input_ids": torch.tensor([[1, 2, 10]]),
                  "labels": torch.full((1, 3), -100, dtype=torch.long)}
        logits = torch.zeros(1, 3, 100, requires_grad=True)
        rollout = {"accepted": False, "detected": True, "prefix_inputs": inputs,
                   "boundary": 2, "negative": 10, "gold_next": 10, "suffix": []}
        losses = recovery_losses(logits, rollout, self.eos)
        losses["loss"].backward()
        self.assertEqual(float(logits.grad.abs().sum()), 0)
        self.assertEqual(float(losses["active_tokens"]), 0.0)

    def test_semiglobal_alignment(self):
        result = align_prefix(self.block, self.block + [40, 41])
        self.assertTrue(result["unique"])
        self.assertEqual(result["endpoint"], 8)
        self.assertEqual(result["deletions"], 0)

    def test_learning_rate_floor(self):
        source = Path(__file__).resolve().parents[1] / "src/layout_ocr/train_screen.py"
        tree = ast.parse(source.read_text(encoding="utf-8-sig"))
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "learning_rate_at_step")
        namespace = {"math": math}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
        for peak in (5e-5, 1e-6):
            values = [namespace[fn.name](step, peak_learning_rate=peak, warmup_steps=102,
                                         max_steps=1024, min_lr_ratio=0.5) for step in range(102, 1025)]
            self.assertAlmostEqual(values[0], peak)
            self.assertAlmostEqual(values[-1], peak * 0.5)
            self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_rollout_restores_modes_cache_and_image_rope(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.child = torch.nn.Linear(1, 1)
                self.config = SimpleNamespace(use_cache=False, max_position_embeddings=4096)
                self.rope_deltas = torch.tensor([123])

            def generate(model, **kwargs):
                self.assertFalse(model.training)
                self.assertFalse(torch.is_grad_enabled())
                self.assertEqual(kwargs["input_ids"].tolist(), [[1, 2]])
                self.assertNotIn("labels", kwargs)
                model.rope_deltas = torch.tensor([456])
                return torch.tensor([[1, 2] + self.block * 3])

        model = Model()
        model.train()
        model.child.eval()
        inputs = {"input_ids": torch.tensor([[1, 2] + self.block + [40, 99]]),
                  "labels": torch.tensor([[-100, -100] + self.block + [40, 99]])}
        rollout = collect_rollout(model, inputs, self.eos)
        self.assertTrue(rollout["accepted"])
        self.assertTrue(model.training)
        self.assertFalse(model.child.training)
        self.assertFalse(model.config.use_cache)
        self.assertEqual(model.rope_deltas.item(), 123)


if __name__ == "__main__":
    unittest.main(verbosity=2)
