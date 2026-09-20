"""Tests that the attention probe is actually reachable from the run entry points.

The probe module can be perfect and still be useless if nothing installs it, or if the
flag the launcher passes is not the flag the parser knows.  These tests go through the
real parser and the real hook installation rather than grepping the files, so a renamed
flag or a dropped install call fails here instead of producing a run that quietly
recorded nothing -- which would read as "there is no signal" and close the direction.

The launcher is checked as text: it cannot be executed here, but the wiring it must
express is finite and every item in it has been wrong at least once in this project.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from layout_ocr.attention_probe import DEFAULT_LAYERS, probe_layers

LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "training"
    / "run_glmocr_layout_attention_probe_a100.sh"
)
ROUTING_LAUNCHER = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "training"
    / "run_glmocr_layout_routing_eval_a100.sh"
)


# The parser's own mandatory arguments, supplied so the probe flags are the only thing
# under test here.
_REQUIRED = [
    "--mode", "geometry",
    "--model-path", "/model",
    "--train-manifest", "/train.jsonl",
    "--validation-manifest", "/validation.jsonl",
    "--protocol-file", "/protocol.json",
    "--output-dir", "/out",
]


def _parse(monkeypatch, argv):
    """Run the real argument parser over a command line."""

    from layout_ocr import train_screen

    monkeypatch.setattr(sys, "argv", ["train_screen", *_REQUIRED, *argv])
    return train_screen.parse_args()


def test_the_probe_flag_defaults_to_off(monkeypatch):
    """An unset flag must leave every existing run exactly as it was."""

    args = _parse(monkeypatch, [])
    assert args.layout_attention_probe is False
    assert args.layout_attention_probe_layers is None
    assert args.layout_attention_probe_heads is None


def test_the_probe_flag_is_accepted_with_layers_and_heads(monkeypatch):
    args = _parse(
        monkeypatch,
        [
            "--layout-attention-probe",
            "--layout-attention-probe-layers", "0,4,8,12",
            "--layout-attention-probe-heads", "1,3",
        ],
    )
    assert args.layout_attention_probe is True
    assert args.layout_attention_probe_layers == (0, 4, 8, 12)
    assert args.layout_attention_probe_heads == (1, 3)


def test_a_malformed_layer_list_is_refused_at_parse_time(monkeypatch):
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--layout-attention-probe", "--layout-attention-probe-layers", "0,x"])
    with pytest.raises(SystemExit):
        _parse(monkeypatch, ["--layout-attention-probe", "--layout-attention-probe-layers", "-1"])


def test_the_probe_and_the_router_are_separate_flags(monkeypatch):
    """The probe must be usable with no routing installed: that is the whole stage."""

    args = _parse(monkeypatch, ["--layout-attention-probe"])
    assert args.layout_routing_bias is None


def test_the_layer_default_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("GLMOCR_ATTENTION_PROBE_LAYERS", "2,6")
    assert probe_layers() == (2, 6)
    monkeypatch.delenv("GLMOCR_ATTENTION_PROBE_LAYERS")
    assert probe_layers() == DEFAULT_LAYERS


def test_installation_leaves_the_model_forward_unchanged(monkeypatch):
    """The probe's hooks must not write into the layer kwargs."""

    import torch
    from torch import nn

    from layout_ocr.attention_probe import install_attention_probe

    calls = []

    class _Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 2
            self.num_key_value_heads = 1
            self.head_dim = 2
            self.scaling = 2**-0.5
            self.q_proj = nn.Linear(4, 4, bias=False)
            self.k_proj = nn.Linear(4, 2, bias=False)

        def forward(self, hidden_states, position_embeddings=None, attention_mask=None):
            calls.append({"mask": attention_mask, "embeds": position_embeddings})
            return hidden_states

    class _TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)
            self.layers = nn.ModuleList([nn.Module()])
            self.layers[0].self_attn = _Attention()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            from types import SimpleNamespace

            self.config = SimpleNamespace(image_token_id=7)
            self.model = nn.Module()
            self.model.language_model = _TextModel()

        def get_input_embeddings(self):
            return self.model.language_model.embed_tokens

    model = _Model()
    _, handles = install_attention_probe(model, _bridge_for_wiring(), layers=(0,))
    attention = model.model.language_model.layers[0].self_attn
    mask = torch.zeros(1, 1, 1, 4)
    embeds = (torch.ones(1, 4, 2), torch.zeros(1, 4, 2))
    attention(torch.randn(1, 4, 4), position_embeddings=embeds, attention_mask=mask)
    # Same objects reached the module: the probe observed without rewriting anything.
    assert calls[0]["mask"] is mask
    assert calls[0]["embeds"] is embeds
    for handle in handles:
        handle.remove()


def _bridge_for_wiring():
    from types import SimpleNamespace

    import torch

    return SimpleNamespace(last_patch_positions=torch.zeros(1, 2, 2))


def _launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def _probe_report(**overrides):
    report = {
        "page_id": "p0",
        "layers": [0, 4],
        "heads": None,
        "visual_tokens": 4,
        "visual_start": 1,
        "num_regions": 2,
        "decoding_steps": 3,
        "grid_missing_steps": 0,
        "emitted_missing_steps": 1,
        "emitted_join_mismatch": 0,
        "layer_geometry": {"0": {"num_heads": 4, "num_kv_heads": 2, "head_dim": 2, "scaling": 1.0}},
        "transform_failed": {},
        "steps": [
            {"step": 1, "text_keys": 3, "heads": [{"layer": 0, "head": 0, "m_t": 0.5}]},
            {"step": 2, "text_keys": 4, "heads": [{"layer": 4, "head": 1, "m_t": 0.25}]},
        ],
    }
    report.update(overrides)
    return report


def test_the_summary_carries_every_key_the_launcher_reads():
    """The summary producer and the launcher are two files apart.

    This is the contract that broke on the first real run: the launcher read a key the
    evaluation summary did not carry, so the report crashed *after* both arms had
    already been paid for.  Nothing was lost, but the run ended with a traceback
    instead of a verdict.
    """

    from layout_ocr.train_screen import _probe_summary

    summary = _probe_summary([_probe_report()])
    # Exactly the keys run_glmocr_layout_attention_probe_a100.sh indexes.  Adding a
    # read there means adding it here too, which is the point of naming them.
    for key in (
        "pages",
        "pages_with_steps",
        "layers",
        "heads",
        "visual_tokens",
        "decoding_steps",
        "grid_missing_steps",
        "emitted_missing_steps",
        "emitted_join_mismatch",
        "mean_visual_mass",
        "transform_failed",
        "layer_geometry",
    ):
        assert key in summary, f"the launcher reads {key!r} but the summary never writes it"
    assert summary["decoding_steps"] == 3
    assert summary["emitted_missing_steps"] == 1
    assert summary["pages_with_steps"] == 1


def test_the_summary_is_absent_rather_than_empty_when_the_probe_is_off():
    """A run without the probe must not report zero steps as if it had observed none."""

    from layout_ocr.train_screen import _probe_summary

    assert _probe_summary([]) is None


def test_the_summary_reports_which_layers_failed_their_transform():
    """A failed transform means those layers' numbers do not exist, not that they are low."""

    from layout_ocr.train_screen import _probe_summary

    summary = _probe_summary([_probe_report(transform_failed={"4": "RuntimeError: no embeds"})])
    assert summary["transform_failed"] == {"p0": {"4": "RuntimeError: no embeds"}}


def test_the_launcher_runs_a_no_probe_control(tmp_path):
    """Without a control that installs nothing, a probe cost has no baseline."""

    text = _launcher_text()
    assert "noroute:no" in text
    assert "probe_only:yes" in text
    # The control must not be handed the flag, and the probe arm must be.
    assert "probe_flags=(--layout-attention-probe" in text


def test_the_launcher_sets_the_probe_report_path_for_both_arms():
    """An empty file on the control arm is the evidence that it had no probe."""

    text = _launcher_text()
    assert 'export GLMOCR_ATTENTION_PROBE="${out}.attention.jsonl"' in text


def test_the_launcher_compares_token_streams_before_reading_any_localization():
    """The stage gate is bit-identity, and it has to be checked first."""

    text = _launcher_text()
    assert '"prediction"] != b[page]["prediction"]' in text
    assert "定位结论一律不读" in text


def test_the_launcher_fixes_the_page_counts_per_stage():
    text = _launcher_text()
    assert "0) page_count=4" in text
    assert "1) page_count=28" in text
    assert "不按实验收益挑页" in text


def test_the_launcher_records_the_latency_budget_before_the_numbers():
    text = _launcher_text()
    assert "LATENCY_BUDGET = 0.20" in text
    assert "待测目标" in text


def test_the_launcher_does_not_read_test():
    """The development subset comes from validation; test stays unread."""

    text = _launcher_text()
    assert "validation/manifest.char.jsonl" in text
    assert "test/manifest" not in text
    assert "--validation-manifest" in text
    # The protocol is named for what it excludes, so the exclusion is visible here.
    assert "train_validation_no_test.json" in text


def test_the_launcher_mirrors_the_routing_launcher_conventions():
    """Same environment setup and same arm-launch discipline as the routing loader."""

    probe_text = _launcher_text()
    routing_text = ROUTING_LAUNCHER.read_text(encoding="utf-8")
    for line in (
        "export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "tmux kill-session -t",
        "train_screen.py refuses to start when --output-dir already exists",
    ):
        assert line in probe_text
        assert line in routing_text
