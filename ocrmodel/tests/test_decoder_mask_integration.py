"""Tests for the wiring between the mask head and the decoder (``decoder_mask_model.py``).

The off-by-one between the query positions and the emitted token is the part that
cannot be checked by reading the code once: the last prompt position predicts the
first target token, so the first character's bias must fire inside the prefill.
These tests pin that indexing down, along with the geometry, the once-per-page key
projection, and the bias injection into a causal mask.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from layout_ocr.decoder_mask_model import (
    DecoderMaskRuntime,
    enable_eager_backend,
    install_decoder_mask_router,
)
from layout_ocr.decoder_mask_router import DecoderMaskConfig
from layout_ocr.mask_targets import MaskTargets

IMAGE_TOKEN_ID = 3


class _FakeTextModel(nn.Module):
    def __init__(self, hidden: int = 16, layers: int = 3):
        super().__init__()
        self.hidden_size = hidden
        self.embed_tokens = nn.Embedding(32, hidden)
        self.layers = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            image_token_id=IMAGE_TOKEN_ID,
            text_config=SimpleNamespace(_attn_implementation="sdpa"),
            _attn_implementation="sdpa",
        )
        self.model = nn.Module()
        self.model.visual = SimpleNamespace(spatial_merge_size=2)
        self.model.language_model = _FakeTextModel()

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens


def _install(**config_overrides):
    model = _FakeModel()
    config = DecoderMaskConfig(hidden_size=16, router_dim=8, split_layer=1, **config_overrides)
    runtime = install_decoder_mask_router(model, config, IMAGE_TOKEN_ID, 2)
    return model, runtime


def _targets(n_tokens: int, n_cells: int = 4) -> MaskTargets:
    return MaskTargets(
        mask=torch.zeros(1, n_tokens, n_cells),
        spatial_valid=torch.ones(1, n_tokens, dtype=torch.bool),
        stop_target=torch.zeros(1, n_tokens),
        char_spans=[(i, i + 1) for i in range(n_tokens)],
        alignment_status=["exact"] * n_tokens,
        alignment_report={},
    )


def test_install_registers_the_router_and_hooks():
    model, runtime = _install()
    text_model = model.model.language_model
    assert hasattr(text_model, "decoder_mask_router")
    assert runtime._first_bias_layer is text_model.layers[1]
    # One visual-capture pre-hook on the text model, plus one per biased layer.
    assert len(runtime.handles) == 1 + (len(text_model.layers) - 1)
    for handle in runtime.handles:
        handle.remove()


def test_enable_eager_backend_sets_the_implementation():
    model = _FakeModel()
    enable_eager_backend(model)
    assert model.config._attn_implementation == "eager"
    assert model.config.text_config._attn_implementation == "eager"


def test_set_page_generation_uses_the_last_prompt_position():
    model, runtime = _install()
    # Four image tokens then two text tokens: L=6, all prompt.
    input_ids = torch.tensor([[3, 3, 3, 3, 1, 2]])
    runtime.set_page(torch.tensor([[1, 4, 4]]), input_ids, 6, None)
    assert runtime.image_positions.tolist() == [0, 1, 2, 3]
    assert runtime.query_positions.tolist() == [5]
    assert runtime.spatial_shape == (2, 2)


def test_set_page_training_shifts_queries_to_predict_the_next_token():
    model, runtime = _install()
    # L=9: prompt=6 (positions 0..5), target=3 (positions 6..8).
    input_ids = torch.tensor([[3, 3, 3, 3, 1, 2, 4, 5, 6]])
    runtime.set_page(torch.tensor([[1, 4, 4]]), input_ids, 6, _targets(3))
    # q = P-1+t: the last prompt position predicts the first target token.
    assert runtime.query_positions.tolist() == [5, 6, 7]


def test_capture_visual_projects_the_keys_once():
    model, runtime = _install()
    input_ids = torch.tensor([[3, 3, 3, 3, 1, 2]])
    runtime.set_page(torch.tensor([[1, 4, 4]]), input_ids, 6, None)
    runtime.capture_visual(None, (), {"inputs_embeds": torch.randn(1, 6, 16)})
    assert runtime.keys.shape == (1, 4, 8)
    keys = runtime.keys
    # A decode step re-enters with only the new token; the keys are reused.
    runtime.capture_visual(None, (), {"inputs_embeds": torch.randn(1, 1, 16)})
    assert runtime.keys is keys


def test_build_bias_scatters_beta_times_the_mask():
    model, runtime = _install()
    runtime.image_positions = torch.tensor([0, 1, 2, 3])
    runtime.query_positions = torch.tensor([5, 6, 7])
    runtime.set_bias_strength(2.0)
    mask = torch.zeros(1, 3, 4)
    mask[0, 0, 0] = 0.5
    mask[0, 1, 1] = 1.0
    hidden = torch.zeros(1, 9, 16)
    bias = runtime._build_bias(mask, hidden, {})
    assert bias.shape == (1, 1, 9, 9)
    assert bias[0, 0, 5, 0].item() == pytest.approx(1.0)  # beta * 0.5
    assert bias[0, 0, 6, 1].item() == pytest.approx(2.0)  # beta * 1.0
    assert bias[0, 0, 7, 0].item() == 0.0


def test_layer_hook_injects_the_bias_into_a_causal_mask():
    model, runtime = _install()
    input_ids = torch.tensor([[3, 3, 3, 3, 1, 2, 4, 5, 6]])
    runtime.set_page(torch.tensor([[1, 4, 4]]), input_ids, 6, _targets(3))
    runtime.set_bias_strength(2.0)
    runtime.capture_visual(None, (), {"inputs_embeds": torch.randn(1, 9, 16)})
    causal = torch.ones(1, 1, 9, 9, dtype=torch.bool).tril()
    kwargs = {"hidden_states": torch.randn(1, 9, 16), "attention_mask": causal}
    runtime.layer_hook(runtime._first_bias_layer, (), kwargs)
    assert runtime.last_mask.shape == (1, 3, 4)
    assert runtime._bias is not None
    assert kwargs["attention_mask"].dtype == torch.float32
    assert kwargs["attention_mask"].shape == (1, 1, 9, 9)
    # The causal structure survives: a later key is still masked out.
    assert kwargs["attention_mask"][0, 0, 0, 5].item() == float("-inf")


def test_set_bias_strength_rejects_a_negative_value():
    model, runtime = _install()
    with pytest.raises(ValueError):
        runtime.set_bias_strength(-1.0)
