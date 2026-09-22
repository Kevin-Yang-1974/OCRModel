import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from layout_ocr.attention_routing import AttentionRouting
from layout_ocr.decoder_mask_router import DecoderMaskConfig
from layout_ocr.window_mask_routing import (
    FirstLayerWindowRuntime,
    WindowRouting,
    WindowRoutingProfile,
)


class Tokenizer:
    def decode(self, ids, **kwargs):
        return "".join({10: "a", 11: "b", 12: "X"}.get(int(i), "") for i in ids)


def route_fixture():
    state = SimpleNamespace(image_token_id=3, last_mask=None)
    route = WindowRouting(state, Tokenizer(), WindowRoutingProfile())
    route.set_page("p", None, 6, torch.tensor([[1, 3, 3, 3, 3, 2]]))
    return route, state


def test_prediction_cached_before_first_layer_and_shared_across_all_layers():
    route, head = route_fixture()
    head.last_mask = torch.tensor([[[0.1, 0.9, 0.4, 0.7]]])
    first = {"hidden_states": torch.randn(1, 1, 8), "cache_position": torch.tensor([6])}
    route.hook(None, (), first)
    head.last_mask = 1 - head.last_mask  # first-layer post-hook produces next mask
    last = {"hidden_states": first["hidden_states"], "cache_position": torch.tensor([6])}
    route.hook(None, (), last)
    assert torch.equal(first["attention_mask"], last["attention_mask"])
    assert first["attention_mask"][0, 0, 0, 1:5].tolist() == [0, 1, 0, 1]
    next_step = {"hidden_states": first["hidden_states"], "cache_position": torch.tensor([7])}
    route.hook(None, (), next_step)
    assert next_step["attention_mask"][0, 0, 0, 1:5].tolist() == [1, 0, 1, 0]


def test_prefill_is_unchanged_and_padding_preserved():
    route, head = route_fixture()
    head.last_mask = torch.ones(1, 1, 4)
    prefill = {"hidden_states": torch.randn(1, 6, 8), "cache_position": torch.arange(6)}
    route.hook(None, (), prefill)
    assert "attention_mask" not in prefill
    pad = torch.ones(1, 1, 1, 7, dtype=torch.bool)
    pad[..., 0] = False
    step = {
        "hidden_states": torch.randn(1, 1, 8),
        "cache_position": torch.tensor([6]),
        "attention_mask": pad,
    }
    route.hook(None, (), step)
    assert torch.isneginf(step["attention_mask"][..., 0]).all()


def test_gt_sync_skips_insertions_and_expires_without_repeating_last_window():
    route, _ = route_fixture()
    route.pointer = "synced"
    route.source = "gt"
    route.set_page("p", [], 6, torch.tensor([[1, 3, 3, 3, 3, 2]]), reference="ab")
    route.char_to_token = [0, 1]
    route.gt_masks = torch.tensor([[[1.0, 0, 0, 0], [0.0, 1, 1, 0]]])
    route.observe_inputs(
        None, (), {"input_ids": torch.tensor([[12]]), "cache_position": torch.tensor([6])}
    )
    assert route.position == 0
    route.observe_inputs(
        None, (), {"input_ids": torch.tensor([[10]]), "cache_position": torch.tensor([7])}
    )
    assert route.position == 1
    assert route._mask_for(1, 8, "cpu", torch.float32)[0, 0, 0, 1:5].tolist() == [0, 1, 1, 0]
    route.observe_inputs(
        None, (), {"input_ids": torch.tensor([[11]]), "cache_position": torch.tensor([8])}
    )
    assert route._mask_for(2, 9, "cpu", torch.float32) is None


def test_gt_binary_window_bias_equals_original_line_hook_for_same_support():
    route, state = route_fixture()
    state.last_mask = torch.tensor([[[0.0, 1, 1, 0]]])
    state.last_patch_positions = torch.tensor(
        [[[0.125, 0.5], [0.375, 0.5], [0.625, 0.5], [0.875, 0.5]]]
    )
    original = AttentionRouting(state, 1.0, 3, Tokenizer(), pointer="step", box_source="line")
    original.set_page(
        "p",
        [{"bbox": [0.25, 0, 0.75, 1]}],
        6,
        torch.tensor([[1, 3, 3, 3, 3, 2]]),
        regions=[{"reading_order": 0, "bbox": [0.25, 0, 0.75, 1]}],
    )
    assert torch.equal(
        route._mask_for(0, 7, "cpu", torch.float32), original._mask_for(0, 7, "cpu", torch.float32)
    )


class Layer(nn.Module):
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        return hidden_states + 0.01


class Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer(), Layer()])

    def forward(self, inputs_embeds, **kwargs):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden_states=hidden, **kwargs)
        return hidden


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 16)
        self.config = SimpleNamespace(image_token_id=3, _attn_implementation="sdpa")
        self.model = nn.Module()
        self.model.language_model = Text()
        self.model.visual = SimpleNamespace(spatial_merge_size=2)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, **kwargs):
        return self.model.language_model(inputs_embeds=self.embedding(input_ids), **kwargs)


def test_first_layer_runtime_generation_then_gt_then_prediction_resets_everything():
    model = Model()
    runtime = FirstLayerWindowRuntime(
        model, Tokenizer(), DecoderMaskConfig(router_dim=8, target_mode="window")
    )
    inputs = {
        "input_ids": torch.tensor([[1, 3, 3, 3, 3, 2]]),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    runtime.set_page(inputs)
    model(input_ids=inputs["input_ids"], cache_position=torch.arange(6))
    assert runtime.runtime.last_mask.shape == (1, 1, 4)
    assert runtime.route._cache_mask is None
    model(input_ids=torch.tensor([[10]]), cache_position=torch.tensor([6]))
    assert runtime.route._cache_mask is not None
    assert model.config._attn_implementation == "sdpa"
    targets = SimpleNamespace(mask=torch.ones(1, 2, 4), char_spans=[(0, 1), (1, 2)])
    runtime.set_page(inputs, gt_targets=targets, reference="ab")
    model(input_ids=inputs["input_ids"], cache_position=torch.arange(6))
    assert runtime.runtime.last_mask is None
    runtime.set_page(inputs)
    assert runtime.route.reference is None and runtime.route.gt_masks is None
    assert runtime.runtime.prev_mask is None
    with pytest.raises(ValueError, match="reference"):
        runtime.set_page(inputs, reference="ab")


def test_acceptance_is_gt_full_protocol_only_and_strictly_below_threshold():
    path = Path(__file__).parents[1] / "tools/evaluation/evaluate_window_mask_routing.py"
    spec = importlib.util.spec_from_file_location("window_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metrics = {"pages": 149, "cer": 0.12999}
    digest = module.PROFILE.validation_sha256
    assert module.acceptance(metrics, digest, "gt")["passed"] is True
    for mode in ("predicted", "legacy-line"):
        assert module.acceptance(metrics, digest, mode)["passed"] is None
    assert module.acceptance(metrics, "wrong-protocol", "gt")["passed"] is None
    assert module.acceptance(metrics, digest, "gt", limited=True)["passed"] is None
    assert module.acceptance(metrics, digest, "gt", legacy_layout=True)["passed"] is None
    metrics["cer"] = 0.13
    assert module.acceptance(metrics, digest, "gt")["passed"] is False


def test_head_checkpoint_roundtrip_preserves_prefill_prediction(tmp_path):
    from layout_ocr.decoder_mask_checkpoint import (
        load_config,
        restore_router,
        save_decoder_mask_checkpoint,
    )

    torch.manual_seed(9)
    first = Model()
    fusion = FirstLayerWindowRuntime(
        first,
        Tokenizer(),
        DecoderMaskConfig(router_dim=8, target_mode="window", input_noise=0, mask_feedback_noise=0),
    )
    fusion.head.eval()
    inputs = {
        "input_ids": torch.tensor([[1, 3, 3, 3, 3, 2]]),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }
    fusion.set_page(inputs)
    first(input_ids=inputs["input_ids"], cache_position=torch.arange(6))
    expected = fusion.runtime.last_mask.detach().clone()
    save_decoder_mask_checkpoint(
        tmp_path, config=fusion.runtime.config, router_state=fusion.head.state_dict()
    )
    second = Model()
    second.embedding.load_state_dict(first.embedding.state_dict())
    restored = FirstLayerWindowRuntime(second, Tokenizer(), load_config(tmp_path))
    restore_router(second, restored.runtime, tmp_path)
    restored.head.eval()
    restored.set_page(inputs)
    second(input_ids=inputs["input_ids"], cache_position=torch.arange(6))
    assert torch.equal(expected, restored.runtime.last_mask)
