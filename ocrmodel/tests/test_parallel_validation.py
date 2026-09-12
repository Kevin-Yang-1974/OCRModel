import importlib.util
import json
import sys
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools/summarize_glmocr_parallel_validation.py"
    spec = importlib.util.spec_from_file_location("summarize_glmocr_parallel_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _metadata() -> dict:
    return {
        "status": "complete",
        "mode": "geometry",
        "num_queries": 512,
        "adapter_precision": "fp32",
        "layout_loss_profile": "full",
        "query_assignment": "hungarian",
        "processor_mode": "fast",
        "decoder_adaptation": "lora",
        "decoder_lora_config": {
            "rank": 8,
            "alpha": 8.0,
            "dropout": 0.0,
            "learning_rate": 1e-5,
        },
        "generation_mode": "loop_recovery",
        "max_eval_new_tokens": 1536,
        "test_manifest_read": False,
        "test_used_for_selection": False,
        "model_path": "/models/GLM-OCR-ca5d8b3",
        "code_sha256": "code-sha",
        "distributed_strategy": "ddp",
        "world_size": 4,
        "global_batch_size": 4,
        "seed": 42,
    }


def test_parallel_validation_selects_earlier_step_on_cer_tie(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    run_dir = tmp_path / "training_runs" / "run" / "seed42"
    group_root = run_dir.parent
    validation_root = run_dir / "parallel-validation"
    run_dir.mkdir(parents=True)
    (run_dir / "COMPLETED").touch()
    metadata = _metadata()
    _write_json(run_dir / "metadata.json", metadata)
    _write_json(
        run_dir / "summary.json",
        {
            "status": "complete",
            "test_manifest_read": False,
            "test_used_for_selection": False,
            "selection_pending": True,
            "training": {
                "checkpoint_steps": [5000, 10000, 15000, 20000],
                "natural_loop": {"enabled": False},
                "loss_objective": {
                    "formula": "L_official + auxiliary_weight * L_layout",
                    "layout_weight": 0.2,
                    "extra_terms": [],
                },
                "scheduled_sampling": {"enabled": False},
                "loop_escape": {"enabled": False},
                "continuation_head": {"enabled": False},
            },
        },
    )
    for step in (5000, 10000, 15000, 20000):
        checkpoint = run_dir / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "adapter.safetensors").touch()
        (checkpoint / "decoder_lora.safetensors").touch()
        _write_json(
            checkpoint / "checkpoint_health.json",
            {
                "step": step,
                "parameters_finite": True,
                "checkpoint_finite": True,
                "decoder_lora": {
                    "decoder_lora": {"parameters_finite": True},
                    "checkpoint_finite": True,
                },
            },
        )
    for step, cer in ((5000, 0.2), (10000, 0.1), (15000, 0.1), (20000, 0.3)):
        eval_dir = validation_root / f"step-{step}"
        eval_dir.mkdir(parents=True)
        (eval_dir / "COMPLETED").touch()
        eval_metadata = dict(metadata)
        eval_metadata["rank"] = 0
        _write_json(eval_dir / "metadata.json", eval_metadata)
        _write_json(
            eval_dir / "summary.json",
            {
                "status": "complete",
                "eval_only": True,
                "parameters_unchanged": True,
                "decoder_adaptation": "lora",
                "decoder_lora_loaded": True,
                "decoder_lora_finite": {
                    "enabled": True,
                    "parameters_finite": True,
                },
                "test_used_for_selection": False,
                "validation": {"cer": cer, "pages": 240, "test_used_for_selection": False},
            },
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_glmocr_parallel_validation.py",
            "--run-dir",
            str(run_dir),
            "--validation-root",
            str(validation_root),
            "--group-root",
            str(group_root),
        ],
    )
    module.main()

    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    group_selection = json.loads((group_root / "selection.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert selection["selected_step"] == 10000
    assert group_selection["selected_step"] == 10000
    assert selection["test_manifest_read"] is False
    assert selection["test_used_for_selection"] is False
    assert summary["selection_pending"] is False
    assert summary["validation_evaluated"] is True
