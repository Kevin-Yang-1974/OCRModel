"""Tests for routing the layout branch into decoder prefix tokens."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch
from torch import Tensor, nn

from layout_ocr.prefix_injection import (
    ENV_VAR,
    PAYLOAD_ENV_VAR,
    PREFIX_TAG,
    PrefixInjector,
    PrefixLogitsMask,
    _find_text_model,
    _select_payload,
    extend_prefix_side_inputs,
    install_prefix_injection,
    payload_mode,
    prefix_token_count,
    publish_prefix_payload,
    reserve_prefix_tokens,
)


class _FakeTokenizer:
    """Minimal stand-in: the real one is only used for its vocab bookkeeping."""

    def __init__(self, base: int = 100) -> None:
        self._vocab = {f"tok{index}": index for index in range(base)}
        self._next = base

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def add_tokens(self, names: list[str], special_tokens: bool = False) -> int:
        added = 0
        for name in names:
            if name in self._vocab:
                continue
            self._vocab[name] = self._next
            self._next += 1
            added += 1
        return added

    def convert_tokens_to_ids(self, name: str) -> int:
        return self._vocab.get(name, -1)

    def __len__(self) -> int:
        # What ``resize_token_embeddings`` is given, so the fake must agree with
        # the real tokenizer's contract here.
        return self._next


class _FakeOutput:
    def __init__(self, queries: Tensor, regions: Tensor | None = None) -> None:
        self.layout_queries = queries
        self.region_output = None if regions is None else _FakeRegionOutput(regions)


class _FakeRegionOutput:
    def __init__(self, features: Tensor) -> None:
        self.region_features = features


class _FakeTextModel(nn.Module):
    """Receives assembled ``inputs_embeds`` exactly like ``GlmOcrTextModel``."""

    def __init__(self, embed: nn.Embedding) -> None:
        super().__init__()
        self.embed_tokens = embed
        self.hidden_size = embed.weight.shape[1]
        self.seen_embeds: Tensor | None = None

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs: Any):
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        self.seen_embeds = inputs_embeds
        return inputs_embeds


class _FakeBridge:
    """Mirrors ``LayoutAwarePatchMerger``: ``adapter.config.hidden_size``."""

    def __init__(self, hidden: int = 8) -> None:
        self.adapter = nn.Module()
        self.adapter.config = type("C", (), {"hidden_size": hidden})()
        self.prefix_splice = None


class _FakeModel(nn.Module):
    def __init__(self, vocab: int = 64, hidden: int = 8) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.model = nn.Module()
        self.model.language_model = _FakeTextModel(self.embed)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed

    def resize_token_embeddings(
        self, new_size: int, mean_resizing: bool = True
    ) -> nn.Embedding:
        """Mirrors the real call: a *new* module, rebound at both access paths.

        Reproducing that matters, because installing the splice before the resize
        would hook the discarded module and the failure would be invisible.
        """

        old = self.embed
        fresh = nn.Embedding(new_size, old.weight.shape[1])
        with torch.no_grad():
            fresh.weight[: old.weight.shape[0]] = old.weight
        self.embed = fresh
        self.model.language_model.embed_tokens = fresh
        return fresh


# --- environment plumbing -------------------------------------------------


def test_prefix_count_defaults_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_VAR, raising=False)
    assert prefix_token_count() == 0
    # Off means off: the unconfigured model is untouched.
    assert prefix_token_count(default=7) == 7


def test_prefix_count_rejects_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_VAR, "many")
    with pytest.raises(ValueError, match="must be an integer"):
        prefix_token_count()
    monkeypatch.setenv(ENV_VAR, "-1")
    with pytest.raises(ValueError, match="non-negative"):
        prefix_token_count()


def test_payload_mode_defaults_and_validates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PAYLOAD_ENV_VAR, raising=False)
    assert payload_mode() == "queries"
    monkeypatch.setenv(PAYLOAD_ENV_VAR, "global")
    assert payload_mode() == "global"
    monkeypatch.setenv(PAYLOAD_ENV_VAR, "boxes")
    with pytest.raises(ValueError, match="must be one of"):
        payload_mode()


# --- reserved tokens ------------------------------------------------------


def test_reserve_prefix_tokens_returns_distinct_ids() -> None:
    tokenizer = _FakeTokenizer()
    ids = reserve_prefix_tokens(tokenizer, 4)
    assert len(ids) == 4
    assert len(set(ids)) == 4
    assert all(token_id >= 100 for token_id in ids)
    assert reserve_prefix_tokens(tokenizer, 0) == []


def test_reserve_prefix_tokens_refuses_a_collision() -> None:
    """A pre-existing name would hand back a real token id and splice over it."""

    tokenizer = _FakeTokenizer()
    tokenizer.add_tokens([f"{PREFIX_TAG[:-1]}_0|>"], special_tokens=True)
    with pytest.raises(ValueError, match="already in the vocabulary"):
        reserve_prefix_tokens(tokenizer, 2)


def test_extend_prefix_side_inputs_keeps_every_side_input_aligned() -> None:
    inputs = {
        "input_ids": torch.tensor([[5, 6, 7]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.tensor([[5, 6, 7]]),
        "mm_token_type_ids": torch.tensor([[1, 1, 0]]),
    }
    out = extend_prefix_side_inputs(inputs, [40, 41])
    assert out["input_ids"].tolist() == [[40, 41, 5, 6, 7]]
    assert out["attention_mask"].shape == (1, 5)
    # The prefix is prompt, so the loss must never ask for it.
    assert out["labels"].tolist() == [[-100, -100, 5, 6, 7]]
    # Text type (0) by the documented convention, even though this sequence
    # leads with an image token -- copying the first token's type would be wrong.
    assert out["mm_token_type_ids"].tolist() == [[0, 0, 1, 1, 0]]


def test_extend_prefix_side_inputs_is_a_no_op_without_reserved_ids() -> None:
    inputs = {"input_ids": torch.tensor([[5, 6]])}
    assert extend_prefix_side_inputs(dict(inputs), [])["input_ids"].tolist() == [[5, 6]]


def test_extend_prefix_side_inputs_rejects_a_batch() -> None:
    inputs = {"input_ids": torch.tensor([[5], [6]])}
    with pytest.raises(ValueError, match="single-page"):
        extend_prefix_side_inputs(inputs, [40])


def test_tail_position_keeps_every_side_input_aligned() -> None:
    """The control that leaves the image tokens' positions untouched."""

    inputs = {
        "input_ids": torch.tensor([[1, 1, 0]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 7]]),
        "mm_token_type_ids": torch.tensor([[1, 1, 0]]),
    }
    # Two leading -100 labels, so the prompt ends after the second token.
    out = extend_prefix_side_inputs(inputs, [40, 41], position="tail")
    assert out["input_ids"].tolist() == [[1, 1, 40, 41, 0]]
    assert out["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]
    assert out["labels"].tolist() == [[-100, -100, -100, -100, 7]]
    assert out["mm_token_type_ids"].tolist() == [[1, 1, 0, 0, 0]]


def test_tail_position_inserts_at_the_prompt_boundary_not_the_sequence_end() -> None:
    """Teacher-forced inputs carry the answer after the prompt.

    Appending would put the reserved rows past the target: they would contribute
    nothing to the loss and the answer would no longer be the last thing the model
    sees, which is the property that made ``tail`` untrainable before.
    """

    inputs = {
        "input_ids": torch.tensor([[1, 1, 0, 7, 8, 9]]),
        "attention_mask": torch.ones(1, 6, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, 7, 8, 9]]),
        "mm_token_type_ids": torch.tensor([[1, 1, 0, 0, 0, 0]]),
    }
    out = extend_prefix_side_inputs(inputs, [40, 41], position="tail")
    # Inserted after the third token, i.e. at the prompt/target boundary.
    assert out["input_ids"].tolist() == [[1, 1, 0, 40, 41, 7, 8, 9]]
    assert out["labels"].tolist() == [[-100, -100, -100, -100, -100, 7, 8, 9]]
    assert out["mm_token_type_ids"].tolist() == [[1, 1, 0, 0, 0, 0, 0, 0]]
    # The answer is still the last thing the model sees.
    assert out["labels"][0, -1].item() == 9


def test_tail_position_on_an_all_prompt_sequence_appends() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[-100, -100, -100]]),
    }
    out = extend_prefix_side_inputs(inputs, [40], position="tail")
    assert out["input_ids"].tolist() == [[1, 2, 3, 40]]
    assert out["labels"].tolist() == [[-100, -100, -100, -100]]


def test_tail_position_on_a_sequence_that_starts_with_a_target() -> None:
    inputs = {
        "input_ids": torch.tensor([[7, 8]]),
        "labels": torch.tensor([[7, 8]]),
    }
    out = extend_prefix_side_inputs(inputs, [40], position="tail")
    assert out["input_ids"].tolist() == [[40, 7, 8]]


def test_position_must_be_front_or_tail() -> None:
    inputs = {"input_ids": torch.tensor([[5, 6]])}
    with pytest.raises(ValueError, match="'front' or 'tail'"):
        extend_prefix_side_inputs(inputs, [40], position="middle")


def test_position_env_defaults_to_front(monkeypatch: pytest.MonkeyPatch) -> None:
    from layout_ocr.prefix_injection import POSITION_ENV_VAR, prefix_position

    monkeypatch.delenv(POSITION_ENV_VAR, raising=False)
    assert prefix_position() == "front"
    monkeypatch.setenv(POSITION_ENV_VAR, "tail")
    assert prefix_position() == "tail"
    monkeypatch.setenv(POSITION_ENV_VAR, "sideways")
    with pytest.raises(ValueError, match="'front' or 'tail'"):
        prefix_position()


def test_disable_env_skips_the_splice_but_still_consumes_the_arming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isolates the cost of the extra positions from the cost of their contents."""

    from layout_ocr.prefix_injection import DISABLE_ENV_VAR

    monkeypatch.setenv(DISABLE_ENV_VAR, "1")
    model, _, injector, splice, _ = _installed()
    _force_identity(injector)
    publish_prefix_payload(splice, _FakeOutput(torch.ones(1, 4, 8)))
    splice.arm()
    embeds = torch.zeros(1, 6, 8)
    seen = model.model.language_model(input_ids=None, inputs_embeds=embeds)
    assert torch.equal(seen, embeds)  # nothing written
    assert splice.applied == 0
    assert splice.skipped == 1
    assert splice._pending == 0  # arming still consumed


def test_explicit_position_wins_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The regression: reading the env var in the hook ignored --prefix-position.

    A run that asked for ``tail`` therefore measured ``front`` and reported
    ``pos=front`` in its own probe -- a wrong number that never raised.
    """

    from layout_ocr.prefix_injection import POSITION_ENV_VAR

    monkeypatch.setenv(POSITION_ENV_VAR, "tail")
    model = _FakeModel(vocab=64, hidden=8)
    bridge = _FakeBridge(hidden=8)
    injector, splice, _ = install_prefix_injection(
        model, bridge, token_count=4, reserved_ids=[50, 51, 52, 53],
        position="front",
    )
    assert splice.position == "front"
    _force_identity(injector)
    publish_prefix_payload(splice, _FakeOutput(torch.ones(1, 4, 8)))
    splice.arm()
    seen = model.model.language_model(input_ids=None, inputs_embeds=torch.zeros(1, 6, 8))
    # Front, despite the environment saying tail.
    assert torch.allclose(seen[0, :4], torch.ones(4, 8))
    assert torch.equal(seen[0, 4:], torch.zeros(2, 8))


def test_install_rejects_an_unknown_position() -> None:
    model = _FakeModel()
    with pytest.raises(ValueError, match="'front' or 'tail'"):
        install_prefix_injection(
            model, _FakeBridge(), token_count=4, reserved_ids=[50, 51, 52, 53],
            position="middle",
        )


def test_disabled_arm_writes_a_probe_record(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deliberately disabled splice must not look like one that never fired."""

    import json

    from layout_ocr.prefix_injection import DISABLE_ENV_VAR, PROBE_ENV_VAR

    probe = tmp_path / "prefix.jsonl"
    monkeypatch.setenv(DISABLE_ENV_VAR, "1")
    monkeypatch.setenv(PROBE_ENV_VAR, str(probe))
    model, _, injector, splice, _ = _installed()
    publish_prefix_payload(splice, _FakeOutput(torch.ones(1, 4, 8)))
    splice.arm()
    model.model.language_model(input_ids=None, inputs_embeds=torch.zeros(1, 6, 8))
    records = [json.loads(line) for line in probe.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["splice_disabled"] is True
    assert records[0]["prefix_norm"] is None
    assert records[0]["prefix_position"] == "front"


def test_tail_position_splices_at_the_end(monkeypatch: pytest.MonkeyPatch) -> None:
    from layout_ocr.prefix_injection import POSITION_ENV_VAR

    monkeypatch.setenv(POSITION_ENV_VAR, "tail")
    model, _, injector, splice, _ = _installed()
    _force_identity(injector)
    publish_prefix_payload(splice, _FakeOutput(torch.ones(1, 4, 8)))
    splice.arm()
    seen = model.model.language_model(input_ids=None, inputs_embeds=torch.zeros(1, 6, 8))
    assert torch.allclose(seen[0, 2:], torch.ones(4, 8))
    assert torch.equal(seen[0, :2], torch.zeros(2, 8))


# --- generation masking ---------------------------------------------------


def test_logits_mask_forces_reserved_ids_out() -> None:
    mask = PrefixLogitsMask([40, 41])
    scores = torch.zeros(1, 4, 64)
    masked = mask(torch.zeros(1, 4, dtype=torch.long), scores)
    assert torch.isinf(masked[..., 40]).all() and masked[..., 40].max() < 0
    assert torch.isinf(masked[..., 41]).all() and masked[..., 41].max() < 0
    assert masked[..., 0].eq(0).all()
    # Untouched entries must be the same object values, not merely finite.
    assert masked[..., 3].eq(0).all()


def test_logits_mask_is_a_no_op_when_nothing_is_reserved() -> None:
    scores = torch.zeros(1, 2, 8)
    assert PrefixLogitsMask([])(torch.zeros(1, 2, dtype=torch.long), scores) is scores


# --- payload selection ----------------------------------------------------


def test_select_payload_queries_is_per_region() -> None:
    queries = torch.randn(1, 4, 8)
    payload = _select_payload(_FakeOutput(queries), "queries", 4)
    assert torch.equal(payload, queries)


def test_select_payload_global_is_page_level() -> None:
    """The ablation arm: one vector repeated, so no region content survives."""

    queries = torch.randn(1, 4, 8)
    payload = _select_payload(_FakeOutput(queries), "global", 4)
    assert payload.shape == (1, 4, 8)
    assert torch.allclose(payload[:, 0], payload[:, 3])
    assert torch.allclose(payload[0, 0], queries[0].mean(dim=0), atol=1e-6)
    # And it genuinely differs from the per-region payload.
    assert not torch.allclose(payload, queries)


def test_select_payload_regions_requires_the_ar_branch() -> None:
    with pytest.raises(ValueError, match="requires the AR region decoder"):
        _select_payload(_FakeOutput(torch.randn(1, 4, 8)), "regions", 4)


def test_select_payload_enforces_the_reserved_width() -> None:
    """A mismatch must fail loudly: silently truncating would drop regions."""

    with pytest.raises(ValueError, match="layout_queries has"):
        _select_payload(_FakeOutput(torch.randn(1, 4, 8)), "queries", 8)


# --- injector -------------------------------------------------------------


def test_injector_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="token_count > 0"):
        PrefixInjector(8, 8, 0, [])
    with pytest.raises(ValueError, match="reserved ids"):
        PrefixInjector(8, 8, 2, [1])


def test_projection_is_zero_initialised_but_the_slots_are_not() -> None:
    """The projection carries no content at step 0; the slots must not be zero.

    An exactly-zero prefix row puts a zero hidden state at the first RMSNorm, whose
    backward is ``0.5 * eps**-1.5``-ish at that point -- measured at 1.0e3 times the
    upstream gradient against 0.12 for a row drawn at 0.02.  A zero-init projection
    with no slot base therefore produced non-finite gradients in layer 0.
    """

    injector = PrefixInjector(8, 8, 3, [50, 51, 52])
    assert torch.equal(injector.projection.weight, torch.zeros(8, 8))
    assert torch.equal(injector.projection.bias, torch.zeros(8))
    payload = torch.randn(1, 3, 8)
    out = injector(payload)
    # No layout content at init: the output is exactly the per-slot base, so the
    # payload has no influence until the projection is trained.
    assert torch.allclose(out[0], injector.slot_bias.detach())
    # Slots differ from each other, like ordinary token embeddings.
    assert not torch.allclose(out[0, 0], out[0, 2])
    # And none of them is zero, which is the point.
    assert out.abs().min() > 0
    assert 0.005 < float(out.detach().std()) < 0.06
    assert injector.slot_bias.requires_grad


def test_injector_gradient_stays_finite_at_initialisation() -> None:
    """The bug this replaced: a zero row made layer-0 RMSNorm backward explode."""

    injector = PrefixInjector(8, 8, 3, [50, 51, 52])
    payload = torch.randn(1, 3, 8, requires_grad=True)
    injector(payload).sum().backward()
    assert torch.isfinite(injector.projection.weight.grad).all()
    assert torch.isfinite(injector.slot_bias.grad).all()


def test_injector_projects_shapes() -> None:
    injector = PrefixInjector(6, 8, 3, [50, 51, 52])
    out = injector(torch.randn(1, 3, 6))
    assert out.shape == (1, 3, 8)


def test_injector_rejects_a_batch() -> None:
    injector = PrefixInjector(6, 8, 3, [50, 51, 52])
    with pytest.raises(ValueError, match="single page"):
        injector(torch.randn(2, 3, 6))


# --- the splice itself ----------------------------------------------------



def _force_identity(injector: PrefixInjector) -> None:
    """Make the injector pass its payload through unchanged, for splice tests.

    The slot base has to be zeroed alongside the projection: it is what keeps the
    prefix rows away from RMSNorm's gradient singularity at zero, so it is nonzero
    by default and would otherwise ride on top of every spliced row.
    """

    with torch.no_grad():
        injector.projection.weight.copy_(torch.eye(injector.projection.weight.shape[0]))
        injector.projection.bias.zero_()
        injector.slot_bias.zero_()


def _installed(hidden: int = 8, tokens: int = 4, payload_mode_: str = "queries"):
    model = _FakeModel(vocab=64, hidden=hidden)
    bridge = _FakeBridge(hidden=hidden)
    injector, splice, handle = install_prefix_injection(
        model, bridge, token_count=tokens, reserved_ids=[50, 51, 52, 53],
        payload_mode_=payload_mode_,
    )
    return model, bridge, injector, splice, handle


def test_install_finds_the_text_model_not_the_embedding() -> None:
    """Splicing at ``embed_tokens`` would miss the scattered image features."""

    model = _FakeModel()
    assert _find_text_model(model) is model.model.language_model
    assert _find_text_model(model) is not model.embed


def test_install_hooks_the_bridge_both_ways() -> None:
    model, bridge, _, splice, handle = _installed()
    assert bridge.prefix_splice is splice
    handle.remove()
    assert bridge.prefix_splice is splice  # removing the hook is the caller's job


def test_injector_lands_in_model_parameters_so_it_can_be_optimized() -> None:
    """A detached injector would run, look correct, and never train.

    ``train_screen`` builds its optimizer from named parameter sets rather than
    from ``model.parameters()``, and DDP only reduces parameters it can see by
    walking the wrapped module.  Registration on the decoder is what puts the
    projection in front of both.
    """

    model, _, injector, _, _ = _installed()
    names = [name for name, _ in model.named_parameters()]
    assert any(name.endswith("projection.weight") for name in names), names
    parameter_ids = {id(parameter) for parameter in model.parameters()}
    assert id(injector.projection.weight) in parameter_ids
    assert injector.projection.weight.requires_grad


def test_install_refuses_a_double_install() -> None:
    """Two injectors would fight over the same reserved rows."""

    model, bridge, _, _, _ = _installed()
    with pytest.raises(RuntimeError, match="already installed"):
        install_prefix_injection(
            model, bridge, token_count=4, reserved_ids=[50, 51, 52, 53]
        )


def test_bridge_publishes_and_hook_splices_in_the_same_forward() -> None:
    """The ordering that makes one vision pass enough: image features first."""

    model, bridge, injector, splice, _ = _installed(hidden=8, tokens=4)
    _force_identity(injector)
    queries = torch.randn(1, 4, 8)
    publish_prefix_payload(splice, _FakeOutput(queries))
    assert torch.allclose(splice.payload, queries)

    input_ids = torch.tensor([[50, 51, 52, 53, 5, 6]])
    splice.arm()
    seen = model.model.language_model(input_ids=input_ids, inputs_embeds=torch.zeros(1, 6, 8))
    assert seen.shape == (1, 6, 8)
    # The reserved rows now carry the payload, and the rest is untouched.
    assert torch.allclose(seen[0, :4], queries[0], atol=1e-6)
    assert torch.allclose(seen[0, 4:], torch.zeros(2, 8))


def test_hook_is_skipped_when_no_payload_was_published() -> None:
    model, _, _, _, _ = _installed()
    embeds = torch.zeros(1, 6, 8)
    seen = model.model.language_model(input_ids=None, inputs_embeds=embeds)
    assert torch.equal(seen, embeds)


def test_hook_does_nothing_unless_armed() -> None:
    """Without arming the splice would overwrite real tokens, not reserved ones.

    The hook cannot see ``input_ids`` -- the text model is reached with
    ``input_ids=None`` -- so it has no way to check the sequence itself.  An
    un-extended forward must therefore splice nothing.
    """

    model, _, injector, splice, _ = _installed()
    _force_identity(injector)
    publish_prefix_payload(splice, _FakeOutput(torch.ones(1, 4, 8)))
    embeds = torch.zeros(1, 6, 8)
    seen = model.model.language_model(input_ids=None, inputs_embeds=embeds)
    assert torch.equal(seen, embeds)
    assert splice.applied == 0


def test_arming_is_consumed_once() -> None:
    """A skipped prefill must not leak the armed flag into the decode steps."""

    model, _, injector, splice, _ = _installed()
    _force_identity(injector)
    publish_prefix_payload(splice, _FakeOutput(torch.ones(1, 4, 8)))
    splice.arm()
    first = model.model.language_model(input_ids=None, inputs_embeds=torch.zeros(1, 6, 8))
    assert torch.allclose(first[0, :4], torch.ones(4, 8))
    # Second call without re-arming: untouched.
    second = model.model.language_model(input_ids=None, inputs_embeds=torch.zeros(1, 6, 8))
    assert torch.equal(second, torch.zeros(1, 6, 8))
    assert splice.applied == 1


def test_runtime_extend_arms_the_splice() -> None:
    """Arming and extending must not be able to drift apart."""

    from layout_ocr.prefix_injection import PrefixRuntime

    model, _, injector, splice, handle = _installed()
    runtime = PrefixRuntime(
        token_count=4, reserved_ids=[50, 51, 52, 53],
        injector=injector, splice=splice, handle=handle,
    )
    assert splice._pending == 0
    inputs = {"input_ids": torch.tensor([[5, 6]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}
    out = runtime.extend(inputs)
    assert out["input_ids"].tolist() == [[50, 51, 52, 53, 5, 6]]
    assert splice._pending == 1


def test_interleaved_forwards_each_consume_their_own_arming() -> None:
    """Validation arms two forwards per page; a flag would starve the second."""

    from layout_ocr.prefix_injection import PrefixRuntime

    model, _, injector, splice, handle = _installed()
    _force_identity(injector)
    runtime = PrefixRuntime(
        token_count=4, reserved_ids=[50, 51, 52, 53],
        injector=injector, splice=splice, handle=handle,
    )
    runtime.extend({"input_ids": torch.tensor([[5, 6]])})
    runtime.extend({"input_ids": torch.tensor([[5, 6]])})
    assert splice._pending == 2
    publish_prefix_payload(splice, _FakeOutput(torch.ones(1, 4, 8)))
    first = model.model.language_model(input_ids=None, inputs_embeds=torch.zeros(1, 6, 8))
    second = model.model.language_model(input_ids=None, inputs_embeds=torch.zeros(1, 6, 8))
    third = model.model.language_model(input_ids=None, inputs_embeds=torch.zeros(1, 6, 8))
    assert torch.allclose(first[0, :4], torch.ones(4, 8))
    assert torch.allclose(second[0, :4], torch.ones(4, 8))
    # Third forward was never armed: untouched rather than corrupted.
    assert torch.equal(third, torch.zeros(1, 6, 8))
    assert splice.applied == 2
    assert splice._pending == 0


def test_runtime_logits_mask_covers_the_reserved_ids() -> None:
    from layout_ocr.prefix_injection import PrefixRuntime

    model, _, injector, splice, handle = _installed()
    runtime = PrefixRuntime(
        token_count=4, reserved_ids=[50, 51, 52, 53],
        injector=injector, splice=splice, handle=handle,
    )
    scores = torch.zeros(1, 2, 64)
    masked = runtime.logits_mask(torch.zeros(1, 2, dtype=torch.long), scores)
    assert torch.isinf(masked[..., 53]).all()
    assert masked[..., 0].eq(0).all()


def test_hook_does_not_resplice_during_decoding() -> None:
    """At decode time the reserved rows live in the KV cache, not in the input."""

    class _Past:
        def get_seq_length(self) -> int:
            return 12

    model, _, injector, splice, _ = _installed()
    _force_identity(injector)
    publish_prefix_payload(splice, _FakeOutput(torch.randn(1, 4, 8)))
    embeds = torch.zeros(1, 1, 8)
    splice.arm()
    seen = model.model.language_model(
        input_ids=None, inputs_embeds=embeds, past_key_values=_Past()
    )
    assert torch.equal(seen, embeds)
    assert splice.applied == 0


def test_hook_refuses_to_drop_the_prefix_silently() -> None:
    """If the caller passes only input_ids the prefix would vanish; say so."""

    model, _, injector, splice, _ = _installed()
    _force_identity(injector)
    publish_prefix_payload(splice, _FakeOutput(torch.randn(1, 4, 8)))
    splice.arm()
    with pytest.raises(RuntimeError, match="expected inputs_embeds"):
        model.model.language_model(input_ids=torch.tensor([[50, 51, 52, 53, 5]]))


def test_hook_ignores_a_sequence_shorter_than_the_prefix() -> None:
    model, _, injector, splice, _ = _installed()
    _force_identity(injector)
    publish_prefix_payload(splice, _FakeOutput(torch.randn(1, 4, 8)))
    splice.arm()
    embeds = torch.zeros(1, 2, 8)
    seen = model.model.language_model(input_ids=None, inputs_embeds=embeds)
    assert torch.equal(seen, embeds)


def test_splice_respects_the_intervention_arms(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prefix route is attributable with the same arms as the residual route."""

    monkeypatch.setenv("GLMOCR_LAYOUT_INTERVENE", "zero")
    model, _, injector, splice, _ = _installed()
    _force_identity(injector)
    publish_prefix_payload(splice, _FakeOutput(torch.ones(1, 4, 8)))
    assert torch.equal(splice.payload, torch.zeros(1, 4, 8))
    splice.arm()
    seen = model.model.language_model(
        input_ids=None, inputs_embeds=torch.ones(1, 6, 8)
    )
    assert torch.equal(seen[0, :4], torch.zeros(4, 8))
    assert torch.equal(seen[0, 4:], torch.ones(2, 8))


def test_splice_trains_the_projection() -> None:
    """The whole point: the projection must receive gradient from the LM loss."""

    model, _, injector, splice, _ = _installed()
    payload = torch.randn(1, 4, 8, requires_grad=True)
    publish_prefix_payload(splice, _FakeOutput(payload))
    splice.arm()
    seen = model.model.language_model(
        input_ids=None, inputs_embeds=torch.zeros(1, 6, 8)
    )
    seen[0, :4].sum().backward()
    assert injector.projection.weight.grad is not None
    assert injector.projection.weight.grad.abs().sum() > 0


def test_global_payload_arm_reaches_the_hook_as_a_page_vector() -> None:
    model, _, injector, splice, _ = _installed(payload_mode_="global")
    assert injector.payload_mode == "global"
    _force_identity(injector)
    queries = torch.randn(1, 4, 8)
    publish_prefix_payload(splice, _FakeOutput(queries))
    # Every slot gets the same page-level vector, which is what makes this the
    # control for "is it region information or page conditioning?".
    assert torch.allclose(splice.payload[:, 0], splice.payload[:, 3])
    assert torch.allclose(
        splice.payload[0, 0], queries[0].mean(dim=0), atol=1e-6
    )


def test_enable_reserves_resizes_freezes_and_installs_in_order() -> None:
    """The ordering is the whole point, and every step of it is silent if wrong.

    ``resize_token_embeddings`` builds a *new* embedding module, so installing the
    splice before it would hook a discarded module; and it leaves the fresh rows
    with ``requires_grad=True`` on a model that had been frozen wholesale, so
    without the re-freeze the entire embedding matrix quietly becomes trainable.
    """

    from layout_ocr.prefix_injection import enable_prefix_injection

    hidden = 8
    model = _FakeModel(vocab=64, hidden=hidden)
    bridge = _FakeBridge(hidden=hidden)
    tokenizer = _FakeTokenizer(base=64)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    runtime = enable_prefix_injection(
        model, bridge, tokenizer, token_count=3, payload_mode_="queries"
    )

    assert runtime.token_count == 3
    assert runtime.reserved_ids == [64, 65, 66]
    assert model.embed.weight.shape[0] == 67
    # The resize must not have re-opened the frozen embedding.
    assert not model.embed.weight.requires_grad
    injector_ids = {id(p) for p in runtime.injector.parameters()}
    assert all(
        not p.requires_grad
        for p in model.parameters()
        if id(p) not in injector_ids
    )
    # Only the injector trains.
    assert runtime.injector.projection.weight.requires_grad
    assert runtime.injector.projection.bias.requires_grad
    assert {id(p) for p in runtime.parameters()} == {
        id(runtime.injector.projection.weight),
        id(runtime.injector.projection.bias),
        id(runtime.injector.slot_bias),
    }
    assert bridge.prefix_splice is runtime.splice


def test_enable_preserves_pre_existing_trainable_parameters() -> None:
    """The stage-two semantic path must survive the resize.

    ``freeze_layout_branch`` deliberately leaves ``content_gate`` and
    ``sem_adapter.*`` trainable.  A blanket ``requires_grad_(False)`` after the
    resize removed them from the optimizer without reporting anything, so the
    prefix arm trained 0 adapter parameters against a baseline's 4.7M -- a
    comparison of two different recipes presented as one feature apart.
    """

    from layout_ocr.prefix_injection import enable_prefix_injection

    hidden = 8
    model = _FakeModel(vocab=64, hidden=hidden)
    bridge = _FakeBridge(hidden=hidden)
    tokenizer = _FakeTokenizer(base=64)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Stand in for the semantic path the caller deliberately left trainable.
    keep = nn.Parameter(torch.randn(4, 4))
    model.sem_path = nn.Module()
    model.sem_path.weight = nn.Parameter(keep.detach().clone())
    model.sem_path.weight.requires_grad_(True)

    runtime = enable_prefix_injection(
        model, bridge, tokenizer, token_count=3, payload_mode_="queries"
    )

    assert model.sem_path.weight.requires_grad, "the semantic path was silently frozen"
    # The fresh embedding rows are still frozen.
    assert not model.embed.weight.requires_grad
    # And the injector trains.
    assert runtime.injector.projection.weight.requires_grad


def test_enable_rejects_a_zero_count() -> None:
    from layout_ocr.prefix_injection import enable_prefix_injection

    model = _FakeModel()
    with pytest.raises(ValueError, match="token_count > 0"):
        enable_prefix_injection(
            model, _FakeBridge(), _FakeTokenizer(), token_count=0
        )


def test_payload_width_uses_the_region_decoder_width_when_asked() -> None:
    """``regions`` payloads are narrower than the adapter width."""

    model = _FakeModel(vocab=64, hidden=8)
    bridge = _FakeBridge(hidden=8)
    decoder_config = type("C", (), {"decoder_hidden_size": 5})()
    bridge.adapter.region_decoder = type("D", (), {"config": decoder_config})()
    injector, _, _ = install_prefix_injection(
        model, bridge, token_count=4, reserved_ids=[50, 51, 52, 53],
        payload_mode_="regions",
    )
    assert injector.projection.in_features == 5
    assert math.isclose(
        float(injector.projection.weight.detach().sum()), 0.0, abs_tol=1e-9
    )


# --- oracle encoder -------------------------------------------------------


def test_oracle_features_carry_box_order_direction_and_validity() -> None:
    from layout_ocr.prefix_injection import ORACLE_FEATURE_DIM, region_feature_tensor

    regions = [
        {"bbox": [0.5, 0.6, 0.7, 0.8], "reading_order": 1, "writing_direction": "horizontal_ltr"},
        {"bbox": [0.1, 0.2, 0.3, 0.4], "reading_order": 0, "writing_direction": "vertical_rtl"},
    ]
    f = region_feature_tensor(regions, 4, torch.device("cpu"))
    assert f.shape == (1, 4, ORACLE_FEATURE_DIM)
    # Sorted into reading order, so the vertical_rtl region lands in slot 0.
    assert f[0, 0, 0:4].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4], abs=1e-6)
    assert f[0, 0, 5].item() == 1.0  # vertical_rtl one-hot
    # One-hot occupies 5:8 in the fixed order vertical_rtl, horizontal_ltr, unknown.
    assert f[0, 1, 6].item() == 1.0  # horizontal_ltr
    # Reading order normalized over the real regions and increasing.
    assert 0 < f[0, 0, 4].item() < f[0, 1, 4].item() <= 1.0
    # Validity separates real regions from padding.
    assert f[0, :, 8].tolist() == [1.0, 1.0, 0.0, 0.0]


def test_oracle_features_reject_more_regions_than_slots() -> None:
    from layout_ocr.prefix_injection import region_feature_tensor

    regions = [
        {"bbox": [0, 0, 1, 1], "reading_order": i, "writing_direction": "unknown"}
        for i in range(3)
    ]
    with pytest.raises(ValueError, match="only 2 prefix slots"):
        region_feature_tensor(regions, 2, torch.device("cpu"))


def test_oracle_encoder_is_trainable_and_finite() -> None:
    from layout_ocr.prefix_injection import (
        LayoutOracleEncoder,
        ORACLE_FEATURE_DIM,
    )

    encoder = LayoutOracleEncoder(16)
    features = torch.randn(1, 4, ORACLE_FEATURE_DIM, requires_grad=True)
    out = encoder(features)
    assert out.shape == (1, 4, 16)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_oracle_mode_requires_the_encoder_and_the_features() -> None:
    injector = PrefixInjector(8, 8, 3, [50, 51, 52], payload_mode_="oracle")
    assert injector.oracle_encoder is not None
    # Wrong slot count is a loud error, not a silent truncation.
    with pytest.raises(ValueError, match="regions but"):
        injector.from_oracle(torch.randn(1, 5, 9))
    # A non-oracle mode has no encoder to call.
    plain = PrefixInjector(8, 8, 3, [50, 51, 52], payload_mode_="queries")
    assert plain.oracle_encoder is None
    with pytest.raises(ValueError, match="not 'oracle'"):
        plain.from_oracle(torch.randn(1, 3, 9))


def test_extend_requires_regions_in_oracle_mode() -> None:
    from layout_ocr.prefix_injection import PrefixRuntime

    model, _, injector, splice, handle = _installed(payload_mode_="oracle")
    runtime = PrefixRuntime(
        token_count=4, reserved_ids=[50, 51, 52, 53],
        injector=injector, splice=splice, handle=handle,
    )
    with pytest.raises(RuntimeError, match="ground-truth regions"):
        runtime.extend({"input_ids": torch.tensor([[5, 6]])})
