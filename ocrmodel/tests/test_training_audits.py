from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("variable_layout_a100_audit_under_test", ROOT / "tools/training/run_variable_layout_a100.py")
trainer_module = load_module(
    "trainer_vit_fixlr_audit_under_test",
    ROOT / "src/GOT-OCR-2.0/GOT/train/trainer_vit_fixlr.py",
)
checkpoint_module = load_module(
    "check_checkpoint_health_under_test",
    ROOT / "tools/training/check_checkpoint_health.py",
)


def test_nonfinite_metric_paths_reports_nested_values_only() -> None:
    paths = runner.nonfinite_metric_paths(
        {"train_loss": 1.0, "diagnostics": {"last": {"ocr_loss": float("nan")}}}
    )
    assert paths == ["metrics.diagnostics.last.ocr_loss"]


def test_runner_classifies_nonfinite_training_from_bounded_log_tail(tmp_path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text("prefix\nNon-finite training gradient at optimizer_step=7\n", encoding="utf-8")
    assert runner.classify_training_failure(log_path) == "nonfinite_training"


def test_runner_classifies_other_training_failures(tmp_path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text("RuntimeError: out of memory\n", encoding="utf-8")
    assert runner.classify_training_failure(log_path) == "failed"


def test_checkpoint_health_requires_weight_file(tmp_path) -> None:
    try:
        checkpoint_module.inspect_checkpoint(tmp_path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("checkpoint without weights must be rejected")


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = torch.nn.Parameter(torch.ones(3))
        self.frozen = torch.nn.Parameter(torch.ones(2), requires_grad=False)


def test_optimizer_audit_rejects_frozen_parameter_and_reports_groups() -> None:
    model = TinyModel()
    trainer = object.__new__(trainer_module.GOTTrainer)
    optimizer = torch.optim.AdamW(
        [{"params": [model.trainable], "lr": 1e-4, "weight_decay": 0.0, "group_name": "qwen_nodecay"}]
    )
    report = trainer._audit_optimizer_groups(optimizer, model)
    assert report["group_count"] == 1
    assert report["optimizer_parameter_count"] == 1
    assert report["frozen_parameters_in_optimizer"] == []
    assert report["groups"][0]["group_name"] == "qwen_nodecay"


def test_tied_parameter_identity_is_explicitly_distinguishable() -> None:
    embedding = torch.nn.Embedding(4, 3)
    head = torch.nn.Linear(3, 4, bias=False)
    head.weight = embedding.weight
    assert head.weight is embedding.weight
    assert head.weight.data_ptr() == embedding.weight.data_ptr()


def test_training_entry_has_hard_nonfinite_and_scope_audits() -> None:
    source = (ROOT / "src/GOT-OCR-2.0/scripts/train_GOT_layout.py").read_text(encoding="utf-8")
    assert "audit_lm_head_tying" in source
    assert "Non-finite training loss" in source
    assert "Non-finite training gradient" in source
    assert "Non-finite training output" in source
    assert "Non-finite optimizer state" in source
    assert "_assert_finite_model_parameters" in (
        ROOT / "src/GOT-OCR-2.0/GOT/train/trainer_vit_fixlr.py"
    ).read_text(encoding="utf-8")
