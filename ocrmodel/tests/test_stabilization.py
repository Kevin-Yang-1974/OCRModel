import argparse
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
    configure_deterministic_execution,
    diagnostic_triage,
    learning_rate_at_step,
    load_adapter_checkpoint,
    module_state_matches,
    parse_step_list,
    processor_reproducibility_report,
    save_adapter_checkpoint,
)


def test_deterministic_execution_pins_reported_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    report = configure_deterministic_execution()
    assert report["strict_deterministic_algorithms"] is True
    assert report["cublas_workspace_config"] == ":4096:8"
    assert report["sdp_backend"] in {"math", "unavailable"}
    assert report["flash_sdp"] is False
    assert report["memory_efficient_sdp"] is False
    assert report["math_sdp"] is True


def test_processor_report_records_explicit_mode_and_resource_hash(tmp_path: Path) -> None:
    class DummyImageProcessor:
        backend = "torchvision"
        size = {"longest_edge": 100}

    class DummyProcessor:
        image_processor = DummyImageProcessor()

    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    report = processor_reproducibility_report(DummyProcessor(), "fast", tmp_path)
    assert report["requested_mode"] == "fast"
    assert report["requested_use_fast"] is True
    assert report["backend"] == "torchvision"
    assert report["resource_hashes"]["preprocessor_config.json"]


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


def test_diagnostic_step_parser_deduplicates_and_sorts() -> None:
    assert parse_step_list("128, 0,64,64") == (0, 64, 128)
    assert parse_step_list("") == ()
    with pytest.raises(argparse.ArgumentTypeError):
        parse_step_list("-1")


def test_diagnostic_triage_flags_invalid_query_mass_rise() -> None:
    points = []
    for step, cer, invalid_mass, residual in (
        (0, 0.30, 0.20, 0.00),
        (64, 0.20, 0.25, 0.01),
        (128, 0.40, 0.40, 0.02),
    ):
        points.append(
            {
                "step": step,
                "validation": {
                    "cer": cer,
                    "invalid_query_fusion_mass": invalid_mass,
                    "residual_relative_norm": residual,
                    "effective_residual_scale": 0.02,
                    "adapter_dtypes": [],
                    "layout_loss_dtypes": [],
                },
                "training": {
                    "adapter_dtypes": [],
                    "loss_dtypes": {key: "float32" for key in (
                        "layout_box",
                        "layout_order",
                        "layout_direction",
                        "layout_assignment",
                        "transport_entropy",
                    )},
                },
            }
        )
    triage = diagnostic_triage(points)
    assert triage["rules"]["query_mask_or_no_object"] is True


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
    assert all(row["processor_mode"] == "fast" for row in mappings)


def test_geometry_diagnostic_launcher_is_single_fixed_run() -> None:
    script = Path(__file__).parents[1] / "tools" / "bscc" / "run_geometry_diagnostic.sbatch"
    environment = {**os.environ, "GLM_OCR_PRINT_TASK_MAP": "1"}
    completed = subprocess.run(
        ["bash", str(script)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    mapping = json.loads(completed.stdout)
    assert mapping["mode"] == "geometry"
    assert mapping["seed"] == 42
    assert mapping["auxiliary_weight"] == 0.2
    assert mapping["max_steps"] == 128
    assert mapping["diagnostic_steps"] == [0, 64, 128]
    assert mapping["reads_test"] is False


def test_geometry_diagnostic_launcher_supports_fp32_256_points() -> None:
    script = Path(__file__).parents[1] / "tools" / "bscc" / "run_geometry_diagnostic.sbatch"
    environment = {
        **os.environ,
        "GLM_OCR_PRINT_TASK_MAP": "1",
        "GLM_OCR_DIAGNOSTIC_ID": "geometry_seed42_steps256_fp32_v3",
        "GLM_OCR_ADAPTER_PRECISION": "fp32",
        "GLM_OCR_MAX_STEPS": "256",
        "GLM_OCR_LR_SCHEDULE_STEPS": "128",
        "GLM_OCR_DIAGNOSTIC_STEPS": "0,64,128,256",
        "GLM_OCR_DIAGNOSTIC_STEPS_JSON": "[0,64,128,256]",
    }
    completed = subprocess.run(
        ["bash", str(script)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    mapping = json.loads(completed.stdout)
    assert mapping["max_steps"] == 256
    assert mapping["lr_schedule_steps"] == 128
    assert mapping["diagnostic_steps"] == [0, 64, 128, 256]
    assert mapping["adapter_precision"] == "fp32"
    assert mapping["processor_mode"] == "fast"
    assert mapping["reads_test"] is False


def test_architecture_comparison_launcher_maps_four_groups_at_256() -> None:
    script = Path(__file__).parents[1] / "tools" / "bscc" / "run_architecture_comparison.sbatch"
    mappings = []
    for task_id in range(10):
        environment = {
            **os.environ,
            "SLURM_ARRAY_TASK_ID": str(task_id),
            "GLM_OCR_PRINT_TASK_MAP": "1",
            "GLM_OCR_ARCHITECTURE_ID": "architecture_256_hungarian_fp32_lr128_v1",
            "GLM_OCR_MAX_STEPS": "256",
            "GLM_OCR_LR_SCHEDULE_STEPS": "128",
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
    baseline = [row for row in mappings if row["eval_only"]]
    assert len(trained) == 9
    assert len(baseline) == 1
    assert {
        (row["mode"], row["seed"], row["auxiliary_weight"])
        for row in trained
    } == {
        (mode, seed, 0.2)
        for mode in ("attention", "geometry", "layout_ot")
        for seed in (42, 43, 44)
    }
    assert baseline[0]["mode"] == "content_only"
    assert baseline[0]["seed"] == 42
    assert baseline[0]["auxiliary_weight"] == 0.0
    assert all(row["max_steps"] == 256 for row in mappings)
    assert all(row["lr_schedule_steps"] == 128 for row in mappings)
    assert all(row["query_assignment"] == "hungarian" for row in mappings)
    assert all(row["processor_mode"] == "fast" for row in mappings)
    assert all(row["reads_test"] is False for row in mappings)
