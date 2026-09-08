import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "tools" / "summarize_architecture_comparison.py"
    spec = importlib.util.spec_from_file_location("summarize_architecture_comparison", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(step: int, cer: float) -> dict[str, object]:
    return {
        "step": step,
        "cer": cer,
        "exact_page_rate": 0.0,
        "generation_limit_hit_rate": 0.05,
        "generation_eos_hit_rate": 0.95,
        "generation_mean_new_tokens": 370.0,
        "test_used_for_selection": False,
        "checkpoint_health": {"parameters_finite": True, "checkpoint_finite": True},
    }


def _write_run_root(root: Path, *, max_steps: int, checkpoint_steps: tuple[int, ...]) -> None:
    metric_row = {
        "step": 1,
        "ocr_loss": 1.0,
        "auxiliary_loss": 1.0,
        "total_loss": 1.2,
        "gradient_norm": 0.5,
        "learning_rate": 1e-6,
        "raw_content_gate": 0.0,
        "effective_residual_scale": 0.0,
        "parameters_finite": True,
    }
    for mode in ("attention", "geometry", "layout_ot"):
        for seed in (42, 43, 44):
            run_dir = root / f"seed{seed}" / f"{mode}_aux0.2"
            run_dir.mkdir(parents=True)
            metadata = {
                "mode": mode,
                "seed": seed,
                "max_steps": max_steps,
                "lr_schedule_steps": 128,
                "auxiliary_weight": 0.2,
                "adapter_precision": "fp32",
                "layout_loss_profile": "full",
                "query_assignment": "hungarian",
                "test_used_for_selection": False,
            }
            (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            candidates = [
                _candidate(step, 0.21 - step / 100000.0 + seed / 10000000.0)
                for step in checkpoint_steps
            ]
            summary = {
                "status": "complete",
                "mode": mode,
                "seed": seed,
                "test_used_for_selection": False,
                "training": {"steps": max_steps, "lr_schedule_steps": 128},
                "selection_candidates": candidates,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_dir / "train_metrics.jsonl").write_text(
                json.dumps(metric_row) + "\n", encoding="utf-8"
            )
    baseline = root / "content_only_eval"
    baseline.mkdir()
    metadata = {
        "mode": "content_only",
        "seed": 42,
        "max_steps": max_steps,
        "lr_schedule_steps": 128,
        "auxiliary_weight": 0.0,
        "adapter_precision": "fp32",
        "layout_loss_profile": "full",
        "query_assignment": "hungarian",
        "test_used_for_selection": False,
    }
    (baseline / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (baseline / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "mode": "content_only",
                "eval_only": True,
                "training_updates": 0,
                "parameters_unchanged": True,
                "validation": _candidate(0, 0.25),
                "test_used_for_selection": False,
            }
        ),
        encoding="utf-8",
    )


def test_dynamic_1024_checkpoint_summary(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    checkpoint_steps = (64, 128, 192, 256, 512, 768, 1024)
    _write_run_root(tmp_path, max_steps=1024, checkpoint_steps=checkpoint_steps)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_architecture_comparison.py",
            str(tmp_path),
            "--max-steps",
            "1024",
            "--lr-schedule-steps",
            "128",
            "--checkpoint-steps",
            ",".join(map(str, checkpoint_steps)),
        ],
    )

    module.main()

    output = json.loads((tmp_path / "architecture_comparison.json").read_text(encoding="utf-8"))
    assert output["config"]["max_steps"] == 1024
    assert output["config"]["checkpoint_steps"] == list(checkpoint_steps)
    assert output["config"]["identity_diagnostic_step"] == 0
    assert len(output["aggregates"]) == 3 * len(checkpoint_steps)
    assert all(result["long_horizon_effective"] for result in output["mode_results"])
    assert output["test_used_for_selection"] is False


def test_default_summary_keeps_256_protocol(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    checkpoint_steps = (64, 128, 192, 256)
    _write_run_root(tmp_path, max_steps=256, checkpoint_steps=checkpoint_steps)
    monkeypatch.setattr(
        sys,
        "argv",
        ["summarize_architecture_comparison.py", str(tmp_path)],
    )

    module.main()

    output = json.loads((tmp_path / "architecture_comparison.json").read_text(encoding="utf-8"))
    assert output["config"]["max_steps"] == 256
    assert output["config"]["checkpoint_steps"] == list(checkpoint_steps)
    assert all(result["long_horizon_effective"] is False for result in output["mode_results"])


def test_checkpoint_step_parser_rejects_identity_and_duplicates() -> None:
    module = _module()
    assert module.parse_checkpoint_steps("64, 128 256") == (64, 128, 256)
    with pytest.raises(module.argparse.ArgumentTypeError):
        module.parse_checkpoint_steps("0,64")
    with pytest.raises(module.argparse.ArgumentTypeError):
        module.parse_checkpoint_steps("64,64")
