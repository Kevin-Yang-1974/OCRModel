"""Tests for the spatial attention-routing bias.

The off-by-one between ``cache_position`` and the emitted character is the part
that cannot be checked by reading the code once: ``generate`` samples the first
token from the prefill, so the decode step at position ``prompt_length + t - 1`` is
the one that emits character ``t``.  These tests pin that down, along with the
guarantees the arms rely on -- the prefill is untouched, the existing mask is
added to rather than replaced, and a missing box leaves the step unbiased.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from layout_ocr.attention_routing import (
    AttentionRouting,
    install_attention_routing,
    pointer_mode,
)

IMAGE_TOKEN_ID = 3
PROMPT_LENGTH = 6
KV_LENGTH = 7
BIAS = 2.0


def _bridge(positions=None):
    if positions is None:
        positions = torch.tensor(
            [[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]], dtype=torch.float32
        )
    return SimpleNamespace(last_patch_positions=positions)


class _FakeTokenizer:
    """Maps ids to single characters, which is all the pointer reads."""

    def __init__(self, mapping):
        self.mapping = mapping

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.mapping.get(int(index), "") for index in ids)


def _runtime(bridge=None, characters=None, pointer="step", reference=None, mapping=None):
    runtime = AttentionRouting(
        bridge or _bridge(),
        BIAS,
        IMAGE_TOKEN_ID,
        _FakeTokenizer(mapping or {}),
        pointer,
    )
    if characters is not None:
        input_ids = torch.tensor([[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID,
                                   IMAGE_TOKEN_ID, 2]])
        runtime.set_page("p0", characters, PROMPT_LENGTH, input_ids, reference=reference)
    return runtime


def _observe(runtime, token_ids):
    """Feed a step's full sequence to the top-level hook, as ``generate`` would."""

    full = torch.tensor([[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, 2]
                         + list(token_ids)])
    runtime.observe_inputs(None, (), {"input_ids": full})


def _call(runtime, cache_position, q_len=1, dtype=torch.float32, existing=None):
    kwargs = {"cache_position": torch.tensor(cache_position)}
    if existing is not None:
        kwargs["attention_mask"] = existing
    hidden = torch.zeros(1, q_len, 8, dtype=dtype)
    runtime.hook(None, (hidden,), kwargs)
    return kwargs


# The four visual tokens sit at the corners of the page, so these two boxes cover
# two tokens each and the biased keys are not adjacent.
LEFT_COLUMN = {"bbox": [0.0, 0.0, 0.5, 1.0]}
BOTTOM_RIGHT = {"bbox": [0.5, 0.5, 1.0, 1.0]}

# The first character is sampled from the prefill's last position and is never
# biased, so every box these tests exercise sits at index 1 or later.
NEVER_BIASED = {"bbox": "unreachable"}


def _characters(*boxes):
    return [NEVER_BIASED] + [{"bbox": box} for box in boxes]


def test_pointer_mode_reads_the_environment(monkeypatch):
    monkeypatch.delenv("GLMOCR_ROUTING_POINTER", raising=False)
    assert pointer_mode() == "synced"
    monkeypatch.setenv("GLMOCR_ROUTING_POINTER", "step")
    assert pointer_mode() == "step"
    monkeypatch.setenv("GLMOCR_ROUTING_POINTER", "sideways")
    with pytest.raises(ValueError, match="must be one of"):
        pointer_mode()


def test_a_zero_bias_is_a_usable_arm():
    """Zero installs the route with an all-zero mask: the wiring-checked baseline."""

    runtime = AttentionRouting(_bridge(), 0.0, IMAGE_TOKEN_ID, _FakeTokenizer({}), "synced")
    assert runtime.bias == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        AttentionRouting(_bridge(), -1.0, IMAGE_TOKEN_ID, _FakeTokenizer({}), "synced")


def test_the_bias_lands_on_the_visual_keys_inside_the_box():
    runtime = _runtime(characters=_characters(LEFT_COLUMN["bbox"]))
    kwargs = _call(runtime, [PROMPT_LENGTH])
    mask = kwargs["attention_mask"]
    assert mask.shape == (1, 1, 1, KV_LENGTH)
    # Visual tokens occupy positions 1..4; the left column holds the first and
    # third of them, and nothing outside the span is touched.
    assert mask[0, 0, 0].tolist() == [0.0, BIAS, 0.0, BIAS, 0.0, 0.0, 0.0]


def test_the_first_decode_step_emits_character_one_not_zero():
    """``generate`` samples character 0 from the prefill, so position L is char 1."""

    runtime = _runtime(characters=_characters(BOTTOM_RIGHT["bbox"], LEFT_COLUMN["bbox"]))
    # The step at position ``prompt_length`` is the first decode step and reaches
    # character 1, whose box is the bottom-right quarter -- so the single biased
    # key is the fourth visual token.
    assert _call(runtime, [PROMPT_LENGTH])["attention_mask"][0, 0, 0].tolist() == [
        0.0, 0.0, 0.0, 0.0, BIAS, 0.0, 0.0,
    ]
    runtime._cache_key = None
    mask = _call(runtime, [PROMPT_LENGTH + 1])["attention_mask"]
    # The mask grows with the key sequence: one more key than the previous step.
    assert mask.shape[-1] == KV_LENGTH + 1
    assert mask[0, 0, 0, 1:5].tolist() == [BIAS, 0.0, BIAS, 0.0]
    assert mask[0, 0, 0, 5:].tolist() == [0.0, 0.0, 0.0]


def test_the_prefill_is_left_alone():
    runtime = _runtime(characters=[LEFT_COLUMN])
    kwargs = _call(runtime, list(range(PROMPT_LENGTH)), q_len=PROMPT_LENGTH)
    assert "attention_mask" not in kwargs
    assert runtime.steps == 0


def test_the_bias_is_added_to_the_existing_mask():
    """Replacing would discard whatever padding structure the mask already carries."""

    runtime = _runtime(characters=_characters(LEFT_COLUMN["bbox"]))
    existing = torch.full((1, 1, 1, KV_LENGTH), 0.5)
    mask = _call(runtime, [PROMPT_LENGTH], existing=existing)["attention_mask"]
    assert mask[0, 0, 0].tolist() == [0.5, 0.5 + BIAS, 0.5, 0.5 + BIAS, 0.5, 0.5, 0.5]


def test_a_boolean_existing_mask_is_converted_before_adding():
    runtime = _runtime(characters=_characters(LEFT_COLUMN["bbox"]))
    existing = torch.ones((1, 1, 1, KV_LENGTH), dtype=torch.bool)
    existing[0, 0, 0, 0] = False
    mask = _call(runtime, [PROMPT_LENGTH], existing=existing)["attention_mask"]
    assert mask[0, 0, 0, 0].item() == float("-inf")
    assert mask[0, 0, 0, 1].item() == BIAS


def test_a_missing_box_leaves_the_step_unbiased_and_counted():
    runtime = _runtime(characters=_characters(None))
    kwargs = _call(runtime, [PROMPT_LENGTH])
    assert "attention_mask" not in kwargs
    assert runtime.missing == 1
    assert runtime.biased == 0


def test_a_step_past_the_annotation_is_not_biased():
    runtime = _runtime(characters=_characters(LEFT_COLUMN["bbox"]))
    kwargs = _call(runtime, [PROMPT_LENGTH + 50])
    assert "attention_mask" not in kwargs
    assert runtime.steps == 1


def test_one_mask_serves_every_layer_of_a_step():
    runtime = _runtime(characters=_characters(LEFT_COLUMN["bbox"]))
    calls = []
    original = runtime._mask_for

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    runtime._mask_for = counted
    for _ in range(3):  # three decoder layers on the same forward
        _call(runtime, [PROMPT_LENGTH])
    assert len(calls) == 1
    assert runtime.biased == 1


def test_the_probe_counts_the_decoding_steps():
    runtime = _runtime(characters=_characters(LEFT_COLUMN["bbox"], BOTTOM_RIGHT["bbox"]))
    _call(runtime, [PROMPT_LENGTH])
    runtime._cache_key = None
    _call(runtime, [PROMPT_LENGTH + 1])
    report = runtime.report()
    assert report["decoding_steps"] == 2
    assert report["biased_steps"] == 2
    assert report["page_id"] == "p0"
    assert report["visual_tokens"] == 4


def test_clearing_a_page_stops_the_route():
    runtime = _runtime(characters=_characters(LEFT_COLUMN["bbox"]))
    runtime.clear_page()
    assert "attention_mask" not in _call(runtime, [PROMPT_LENGTH])


def test_a_missing_patch_grid_is_an_error_not_a_silent_no_op():
    runtime = _runtime(
        bridge=SimpleNamespace(last_patch_positions=None),
        characters=_characters(LEFT_COLUMN["bbox"]),
    )
    with pytest.raises(RuntimeError, match="no patch grid"):
        _call(runtime, [PROMPT_LENGTH])


def test_the_synced_pointer_picks_the_box_not_the_step():
    """Both runs are on the same step; only the pointer differs."""

    runtime = _runtime(
        characters=_characters(BOTTOM_RIGHT["bbox"], LEFT_COLUMN["bbox"]),
        pointer="synced",
        reference="甲乙丙丁戊",
        mapping={7: "甲", 8: "乙"},
    )
    assert runtime.position == 0
    _observe(runtime, [7])
    assert runtime.position == 1
    runtime._cache_key = None
    # Pointer 1 is the bottom-right box, which holds only the fourth token.
    assert _call(runtime, [PROMPT_LENGTH])["attention_mask"][0, 0, 0, 1:5].tolist() == [
        0.0, 0.0, 0.0, BIAS,
    ]
    _observe(runtime, [7, 8])
    assert runtime.position == 2
    runtime._cache_key = None
    # Pointer 2 is the left column, which holds the first and third tokens -- a
    # different box on the very next step, on the same sequence position delta.
    assert _call(runtime, [PROMPT_LENGTH + 1])["attention_mask"][0, 0, 0, 1:5].tolist() == [
        BIAS, 0.0, BIAS, 0.0,
    ]


def test_the_synced_pointer_jumps_over_a_skipped_reference_character():
    """A model that skips a character is still on the page, not one behind it."""

    runtime = _runtime(characters=_characters("a"), pointer="synced",
                       reference="甲乙丙丁戊", mapping={7: "甲", 9: "丁"})
    _observe(runtime, [7, 9])
    assert runtime.position == 4


def test_the_synced_pointer_ignores_an_inserted_character():
    """A character the reference does not have nearby must not move the pointer."""

    runtime = _runtime(characters=_characters("a"), pointer="synced",
                       reference="甲乙丙丁戊", mapping={7: "甲", 1: "Z"})
    _observe(runtime, [7])
    _observe(runtime, [7, 1])
    assert runtime.position == 1


def test_the_synced_pointer_lookahead_is_bounded():
    """Without a bound, a repeated glyph would snap the pointer back to itself."""

    reference = "甲" * 100 + "乙"
    runtime = _runtime(characters=_characters("a"), pointer="synced",
                       reference=reference, mapping={7: "甲"})
    _observe(runtime, [7])
    _observe(runtime, [7, 7])
    assert runtime.position == 2  # advanced, not jumped to a later 甲


def test_a_synced_runtime_needs_a_tokenizer():
    with pytest.raises(ValueError, match="needs a tokenizer"):
        AttentionRouting(_bridge(), BIAS, IMAGE_TOKEN_ID, None, "synced")


def test_the_synced_pointer_only_decodes_the_new_tail():
    decoded = []

    class _Recording(_FakeTokenizer):
        def decode(self, ids, skip_special_tokens=True):
            decoded.append(list(ids.tolist()))
            return super().decode(ids, skip_special_tokens)

    runtime = AttentionRouting(_bridge(), BIAS, IMAGE_TOKEN_ID,
                               _Recording({7: "甲", 8: "乙"}), "synced")
    runtime.set_page("p0", _characters("a"), PROMPT_LENGTH,
                     torch.tensor([[1, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID, IMAGE_TOKEN_ID,
                                    IMAGE_TOKEN_ID, 2]]),
                     reference="甲乙丙")
    _observe(runtime, [7])
    _observe(runtime, [7, 8])
    assert decoded == [[7], [8]]


class _FakeTextModel(nn.Module):
    def __init__(self, hidden_size: int = 8, layers: int = 2):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(layers)])


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(image_token_id=IMAGE_TOKEN_ID)
        self.model = nn.Module()
        self.model.language_model = _FakeTextModel()

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens


def test_installation_hooks_every_decoder_layer():
    model = _FakeModel()
    runtime, handles = install_attention_routing(
        model, _bridge(), bias=BIAS, tokenizer=_FakeTokenizer({7: "甲"}), pointer="synced"
    )
    # One hook per layer, plus the model-level one that advances the pointer.
    assert len(handles) == len(model.model.language_model.layers) + 1
    assert runtime.image_token_id == IMAGE_TOKEN_ID
    for handle in handles:
        handle.remove()


def test_installation_refuses_a_model_without_decoder_layers():
    model = _FakeModel()
    model.model.language_model.layers = nn.ModuleList()
    with pytest.raises(RuntimeError, match="text decoder layers"):
        install_attention_routing(model, _bridge(), bias=BIAS, tokenizer=_FakeTokenizer({}), pointer="synced")
