import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "tools" / "summarize_confirmation.py"
    spec = importlib.util.spec_from_file_location("summarize_confirmation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validation(cer: float) -> dict[str, object]:
    return {
        "cer": cer,
        "exact_page_rate": 0.0,
        "low_frequency_k1_recall": 0.8,
        "low_frequency_k3_recall": 0.7,
        "low_frequency_k5_recall": 0.6,
        "generation_limit_hit_rate": 0.05,
        "test_used_for_selection": False,
        "checkpoint_health": {"parameters_finite": True, "checkpoint_finite": True},
    }


def _build_run_root(root: Path) -> None:
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
    for mode in ("attention", "geometry"):
        for seed in (42, 43, 44):
            run_dir = root / f"seed{seed}" / f"{mode}_aux0.2"
            run_dir.mkdir(parents=True)
            summary = {
                "status": "complete",
                "mode": mode,
                "seed": seed,
                "auxiliary_weight": 0.2,
                "test_used_for_selection": False,
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            candidates = [
                {"step": step, **_validation(0.2 + (step - 256) / 100000)}
                for step in (256, 512, 768, 1024)
            ]
            (run_dir / "selection.json").write_text(
                json.dumps({"candidates": candidates}), encoding="utf-8"
            )
            (run_dir / "train_metrics.jsonl").write_text(
                json.dumps(metric_row) + "\n", encoding="utf-8"
            )
    baseline = root / "content_only_eval"
    baseline.mkdir()
    (baseline / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "mode": "content_only",
                "eval_only": True,
                "training_updates": 0,
                "parameters_unchanged": True,
                "validation": _validation(0.25),
                "test_used_for_selection": False,
            }
        ),
        encoding="utf-8",
    )


def test_confirmation_selects_mode_and_step_without_test(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    _build_run_root(tmp_path)
    monkeypatch.setattr(sys, "argv", ["summarize_confirmation.py", str(tmp_path)])
    module.main()
    selection = json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))
    assert selection["selected_mode"] == "attention"
    assert selection["selected_step"] == 256
    assert selection["selected_std_cer"] == pytest.approx(0.0)
    assert selection["test_used_for_selection"] is False
    assert selection["stability"]["passed"] is True
    assert len(selection["paired_geometry_minus_attention"]) == 12


def test_confirmation_rejects_nonfinite_checkpoint(tmp_path: Path) -> None:
    module = _module()
    _build_run_root(tmp_path)
    run_dir = tmp_path / "seed42" / "attention_aux0.2"
    selection_path = run_dir / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection["candidates"][0]["checkpoint_health"]["checkpoint_finite"] = False
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint finite check failed"):
        module.read_training_run(tmp_path, 42, "attention")
