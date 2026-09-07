import json
import os
import subprocess
from pathlib import Path

import pytest
import torch
from torch import nn

from layout_ocr import LayoutAdapterConfig, PreMergeLayoutAdapter
from layout_ocr.glm_bridge import LayoutAwarePatchMerger
from layout_ocr.train_screen import (
    clone_module_state,
    learning_rate_at_step,
    load_adapter_checkpoint,
    module_state_matches,
    save_adapter_checkpoint,
)


def test_warmup_cosine_schedule_endpoints() -> None:
    values = {
        step: learning_rate_at_step(
            step,
            peak_learning_rate=5e-5,
            warmup_steps=64,
            max_steps=1024,
            min_lr_ratio=0.1,
        )
        for step in (0, 64, 1024)
    }
    assert values[0] == pytest.approx(0.0)
    assert values[64] == pytest.approx(5e-5)
    assert values[1024] == pytest.approx(5e-6)


def test_checkpoint_reload_preserves_capped_gate_semantics(tmp_path: Path) -> None:
    config = LayoutAdapterConfig(
        hidden_size=8,
        num_queries=2,
        num_heads=2,
        mode="attention",
        max_residual_scale=0.03,
    )
    source = PreMergeLayoutAdapter(config).eval()
    with torch.no_grad():
        source.content_gate.fill_(0.5)
    bridge = LayoutAwarePatchMerger(nn.Identity(), source, spatial_merge_size=1)
    tokens = torch.randn(1, 4, 8)
    expected = source(tokens).merged_tokens

    save_adapter_checkpoint(tmp_path, bridge, step=256)
    restored = PreMergeLayoutAdapter(config).eval()
    restored_bridge = LayoutAwarePatchMerger(nn.Identity(), restored, spatial_merge_size=1)
    load_adapter_checkpoint(tmp_path, restored_bridge)
    actual = restored(tokens).merged_tokens

    torch.testing.assert_close(actual, expected)
    assert float(restored.effective_residual_scale().detach()) == pytest.approx(0.03)


def test_eval_only_state_guard_detects_updates() -> None:
    module = nn.Linear(3, 2)
    before = clone_module_state(module)
    module.eval()
    with torch.inference_mode():
        module(torch.ones(1, 3))
    assert module_state_matches(module, before)
    with torch.no_grad():
        module.bias.add_(1)
    assert not module_state_matches(module, before)


def test_slurm_array_maps_six_unique_runs_and_one_baseline() -> None:
    script = Path(__file__).parents[1] / "tools" / "bscc" / "run_mechanism_screen.sbatch"
    mappings = []
    for task_id in range(7):
        environment = {
            **os.environ,
            "SLURM_ARRAY_TASK_ID": str(task_id),
            "GLM_OCR_PRINT_TASK_MAP": "1",
        }
        completed = subprocess.run(
            ["bash", str(script)],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        mappings.append(json.loads(completed.stdout))

    trained = [row for row in mappings if not row["eval_only"]]
    assert len(trained) == 6
    assert {
        (row["mode"], row["seed"], row["auxiliary_weight"]) for row in trained
    } == {
        (mode, seed, 0.2)
        for mode in ("attention", "geometry")
        for seed in (42, 43, 44)
    }
    assert len({row["output_dir"] for row in mappings}) == 7
    assert mappings[-1]["mode"] == "content_only"
    assert mappings[-1]["eval_only"] is True
