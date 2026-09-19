"""Inject the layout branch's output as prefix tokens at the front of the LM input.

The residual write-back adds ``alpha * layout_context`` to the *pre-merger* visual
tokens, which makes it subject to three things the decoder cannot be argued out of:
the residual cap (``0.03`` in every current entry point), the merger's transfer
function, and the averaging that puts the context into the same slots as the
pixels.  The 2026-09-19 intervention matrix measured what that seam actually
delivers -- the whole effect lives in the patch-mean component, and the per-patch
component was inert at the amplitude it was measured at.  This module takes the
branch's output out of that seam entirely and hands it to the decoder as ``K``
reserved tokens at the very front of the sequence, where it gets dedicated
positions, full attention from every text token, and no cap.

## Why the splice point is ``GlmOcrTextModel.forward``

Three constraints force it, and all three were read off the installed modeling
source rather than assumed:

1. ``GlmOcrTextModel.forward`` does ``inputs_embeds = self.embed_tokens(input_ids)``
   and the text model is reached with ``input_ids=None, inputs_embeds=<assembled>``
   from ``GlmOcrModel.forward``.  So ``embed_tokens`` is *not* where the decoder's
   input is finalised in the training path, and hooking it would also miss the
   image features that are scattered in afterwards.
2. ``compute_3d_position_ids`` only takes the mrope branch when
   ``can_compute_mrope = input_ids is not None and mm_token_type_ids is not None
   and image_grid_thw is not None``.  Passing ``inputs_embeds`` *up* from the top
   level would therefore silently degrade every visual token to 1D positions.  The
   splice must not touch ``input_ids``, and it does not: ``GlmOcrModel`` computes
   the 3D ``position_ids`` from the real ``input_ids`` before calling the text
   model, so by the splice point the positions are already correct for the longer
   sequence.
3. The payload does not exist until the visual tower has run.
   ``GlmOcrModel.forward`` calls ``get_image_features`` (the bridge, which is where
   the layout branch lives) *before* ``language_model(...)``, but *after*
   ``get_input_embeddings``.  Hooking the embedding would read the payload one
   forward stale on the first step and every step after; hooking the text model
   reads it in the same forward, so the vision tower runs exactly once.

The splice is applied by a forward pre-hook so the text model's own ``forward``
stays untouched, and it is guarded on the prefill: during decoding the text model
is called with a single new position and the reserved rows are already in the KV
cache, so re-splicing there would be wrong (and out of range).

## Reserved tokens

The reserved ids are added to the tokenizer and the embedding matrix is resized,
which makes a collision with real prompt text impossible by construction -- the
alternative of borrowing K existing ids relies on those ids never appearing in the
corpus, which is a property of the data and not of the code.  Their embedding rows
are overwritten on every forward, so their initial values never reach the decoder
input; on the output side they are masked out of generation by
:class:`PrefixLogitsMask`, because the tied ``lm_head`` rows would otherwise stay
at their initialiser and remain samplable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Literal, get_args

import torch
from torch import Tensor, nn

ENV_VAR = "GLMOCR_PREFIX_TOKENS"
PAYLOAD_ENV_VAR = "GLMOCR_PREFIX_PAYLOAD"
POSITION_ENV_VAR = "GLMOCR_PREFIX_POSITION"
DISABLE_ENV_VAR = "GLMOCR_PREFIX_DISABLE"
PROBE_ENV_VAR = "GLMOCR_PREFIX_PROBE"


def prefix_position() -> str:
    """Where the reserved span sits; ``front`` is the design, ``tail`` a control."""

    raw = os.environ.get(POSITION_ENV_VAR, "").strip().lower()
    if not raw:
        return "front"
    if raw not in {"front", "tail"}:
        raise ValueError(f"{POSITION_ENV_VAR} must be 'front' or 'tail', got {raw!r}")
    return raw

PayloadMode = Literal["queries", "regions", "global"]
PAYLOAD_MODES: tuple[str, ...] = get_args(PayloadMode)

PREFIX_TAG = "<|layout_prefix|>"

# Name the injector is registered under on the decoder, so the training entry
# point can find its parameters for the optimizer.
PREFIX_MODULE_NAME = "layout_prefix_injector"


def prefix_token_count(default: int = 0) -> int:
    """Number of reserved prefix tokens, from the environment.

    Zero keeps the model exactly as it was, so every existing run is unaffected
    until the feature is asked for explicitly.
    """

    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return default
    try:
        count = int(raw)
    except ValueError as error:
        raise ValueError(f"{ENV_VAR} must be an integer, got {raw!r}") from error
    if count < 0:
        raise ValueError(f"{ENV_VAR} must be non-negative, got {count}")
    return count


def payload_mode() -> PayloadMode:
    raw = os.environ.get(PAYLOAD_ENV_VAR, "").strip().lower()
    if not raw:
        return "queries"
    if raw not in PAYLOAD_MODES:
        raise ValueError(f"{PAYLOAD_ENV_VAR} must be one of {PAYLOAD_MODES}, got {raw!r}")
    return raw  # type: ignore[return-value]


def reserve_prefix_tokens(tokenizer: Any, count: int) -> list[int]:
    """Add ``count`` distinct prefix tokens and return their ids in order."""

    if count <= 0:
        return []
    names = [f"{PREFIX_TAG[:-1]}_{index}|>" for index in range(count)]
    existing = tokenizer.get_vocab()
    # Adding a token that is already present would hand back the pre-existing id
    # and silently splice the payload over a real token.
    clash = [name for name in names if name in existing]
    if clash:
        raise ValueError(f"prefix tokens already in the vocabulary: {clash[:4]}")
    added = tokenizer.add_tokens(names, special_tokens=True)
    if added != count:
        raise ValueError(f"expected to add {count} prefix tokens, tokenizer added {added}")
    ids = [int(tokenizer.convert_tokens_to_ids(name)) for name in names]
    if len(set(ids)) != count:
        raise ValueError("prefix token ids are not distinct")
    if min(ids) < 0:
        raise ValueError("prefix token id could not be resolved")
    return ids


def build_prefix_input_ids(
    input_ids: Tensor, reserved_ids: list[int]
) -> tuple[Tensor, Tensor | None]:
    """Prepend the reserved ids to a ``[1, L]`` sequence.

    Returns the extended ``input_ids`` and, when the caller supplied one, the
    matching extension of ``attention_mask``.  ``mm_token_type_ids`` is *not*
    extended here: the reserved tokens are text, and every real entry point
    consumes ``mm_token_type_ids`` only for its image/video spans, but callers
    that pass it must extend it with the text-stream type themselves -- see
    :func:`extend_prefix_side_inputs`.
    """

    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("prefix injection requires a single-page [1, L] input_ids")
    if not reserved_ids:
        return input_ids, None
    prefix = torch.tensor(
        [reserved_ids], dtype=input_ids.dtype, device=input_ids.device
    )
    return torch.cat((prefix, input_ids), dim=1), prefix


def extend_prefix_side_inputs(
    inputs: dict[str, Any],
    reserved_ids: list[int],
    *,
    position: str = "front",
    splice_disabled: bool = False,
) -> dict[str, Any]:
    """Add the reserved ids to ``input_ids`` and keep every side input aligned.

    ``labels`` gets ``-100`` over the reserved span, which is the same treatment
    the prompt already receives, so the loss never asks the model to predict a
    reserved id.

    ``position`` places the span at the front of the sequence or at its end, which
    is a diagnostic rather than a design choice.  Leading text shifts the mrope
    position of every following token, image tokens included; trailing the prompt
    leaves the image positions untouched and shifts only the text tail.  The two
    therefore separate "the vision positions moved" from "there is more attention
    mass in the sequence", and a 2026-09-19 eval measured the two placements
    differently, so the distinction is load-bearing rather than cosmetic.

    ``splice_disabled`` inserts the span without letting the hook overwrite it,
    which isolates the cost of *having* K extra positions from the cost of the
    vectors written into them.
    """

    if not reserved_ids:
        return inputs
    if position not in {"front", "tail"}:
        raise ValueError(f"prefix position must be 'front' or 'tail', got {position!r}")
    input_ids = inputs["input_ids"]
    if position == "front":
        extended, prefix = build_prefix_input_ids(input_ids, reserved_ids)
    else:
        prefix = torch.tensor(
            [reserved_ids], dtype=input_ids.dtype, device=input_ids.device
        )
        extended = torch.cat((input_ids, prefix), dim=1)
    if prefix is None:
        return inputs
    width = prefix.shape[1]
    inputs["input_ids"] = extended
    attention_mask = inputs.get("attention_mask")
    if isinstance(attention_mask, Tensor):
        pad = torch.ones(
            (1, width), dtype=attention_mask.dtype, device=attention_mask.device
        )
        inputs["attention_mask"] = (
            torch.cat((pad, attention_mask), dim=1)
            if position == "front"
            else torch.cat((attention_mask, pad), dim=1)
        )
    labels = inputs.get("labels")
    if isinstance(labels, Tensor):
        ignore = torch.full(
            (1, width), -100, dtype=labels.dtype, device=labels.device
        )
        inputs["labels"] = (
            torch.cat((ignore, labels), dim=1)
            if position == "front"
            else torch.cat((labels, ignore), dim=1)
        )
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if isinstance(mm_token_type_ids, Tensor):
        # ``get_rope_index`` groups this stream to place vision tokens, with the
        # documented convention text=0, image=1, video=2.  The prefix is text, so
        # it is filled with text rather than copied from the first real token --
        # copying would be wrong the moment a template leads with an image.
        text_type = torch.zeros(
            (1, width), dtype=mm_token_type_ids.dtype, device=mm_token_type_ids.device
        )
        inputs["mm_token_type_ids"] = (
            torch.cat((text_type, mm_token_type_ids), dim=1)
            if position == "front"
            else torch.cat((mm_token_type_ids, text_type), dim=1)
        )
    return inputs


class PrefixLogitsMask:
    """Force the reserved ids out of the sampling distribution.

    ``lm_head.weight`` is tied to the embedding matrix, so a reserved row that is
    only ever overwritten on the *input* side keeps its initialiser on the output
    side and stays samplable.  A reserved id in the generated text would be both
    meaningless and unparseable, so it is masked rather than left to chance.
    """

    def __init__(self, reserved_ids: list[int]) -> None:
        self.reserved_ids = [int(token_id) for token_id in reserved_ids]
        self._tensor: Tensor | None = None

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        if not self.reserved_ids:
            return scores
        if self._tensor is None or self._tensor.device != scores.device:
            self._tensor = torch.tensor(
                self.reserved_ids, dtype=torch.long, device=scores.device
            )
        return scores.index_fill(-1, self._tensor, float("-inf"))


class PrefixInjector(nn.Module):
    """Project the layout branch's output into ``K`` decoder prefix embeddings."""

    def __init__(
        self,
        source_hidden_size: int,
        model_hidden_size: int,
        token_count: int,
        reserved_ids: list[int],
        *,
        payload_mode_: PayloadMode | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if token_count <= 0:
            raise ValueError("PrefixInjector requires token_count > 0")
        if len(reserved_ids) != token_count:
            raise ValueError(
                f"expected {token_count} reserved ids, got {len(reserved_ids)}"
            )
        self.token_count = token_count
        self.reserved_ids = [int(token_id) for token_id in reserved_ids]
        self.payload_mode: PayloadMode = payload_mode_ or "queries"
        self.projection = nn.Linear(source_hidden_size, model_hidden_size)
        # Start as an exact no-op contribution: the prefix rows are the projection
        # of a zero vector, so the feature can be switched on without perturbing
        # the checkpoint it is attached to.  This mirrors the zero-initialised
        # residual gate, for the same reason.
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
        if dtype is not None:
            self.to(dtype=dtype)

    def forward(self, payload: Tensor) -> Tensor:
        """``[batch, K, source]`` -> ``[batch, K, model]``, replacing the rows."""

        if payload.shape[0] != 1:
            raise ValueError("prefix injection supports a single page at a time")
        if payload.shape[1] != self.token_count:
            raise ValueError(
                f"payload has {payload.shape[1]} entries but {self.token_count} "
                "reserved tokens are allocated"
            )
        return self.projection(payload.to(self.projection.weight.dtype))


def _select_payload(output: Any, mode: PayloadMode, token_count: int) -> Tensor:
    """Pick the branch output that becomes the prefix, at the reserved width.

    ``queries``  the layout queries themselves, one prefix token per query.  Each
                 query is a region hypothesis carrying a box and an order score,
                 so this is the form that can actually tell the decoder about
                 reading order -- the thing the residual path was meant to carry
                 and measurably does not.
    ``regions``  the AR region decoder's ordered region features.  Available only
                 for the AR branch.
    ``global``   the *mean of the queries*, broadcast to every prefix slot.  A
                 page-level payload with no per-region content, kept as the arm
                 that separates "the decoder uses region information" from "the
                 decoder uses a page-level conditioning vector" -- the same
                 question the write-back matrix answered for the residual seam.
    """

    if mode == "queries":
        queries = output.layout_queries
        if queries.shape[1] != token_count:
            raise ValueError(
                f"prefix needs {token_count} payload entries, layout_queries has "
                f"{queries.shape[1]}; set GLMOCR_PREFIX_TOKENS to num_queries"
            )
        return queries
    if mode == "regions":
        region_output = getattr(output, "region_output", None)
        if region_output is None:
            raise ValueError("payload mode 'regions' requires the AR region decoder")
        features = region_output.region_features
        if features.shape[1] != token_count:
            raise ValueError(
                f"prefix needs {token_count} payload entries, region_features has "
                f"{features.shape[1]}"
            )
        return features
    if mode == "global":
        queries = output.layout_queries
        return queries.mean(dim=1, keepdim=True).expand(-1, token_count, -1)
    raise ValueError(f"unsupported payload mode: {mode}")


def publish_prefix_payload(splice: Any, output: Any) -> None:
    """Hand the branch output to the text-model hook, subject to the active arm.

    Called by the bridge once per vision forward.  The intervention arms apply
    here unchanged, so the prefix route is attributable with exactly the controls
    the residual route was judged with -- including ``global`` (page-level only)
    and ``shuffle`` (per-slot content destroyed), which are the two comparisons
    that decided the residual seam.
    """

    splice.set_payload(
        _select_payload(
            output, splice.injector.payload_mode, splice.injector.token_count
        )
    )


def _probe(path: str, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


class _Splice:
    """State shared between the bridge and the text-model pre-hook."""

    def __init__(self, injector: PrefixInjector, position: str = "front") -> None:
        self.injector = injector
        # Resolved once at install time from the caller, not read from the
        # environment on every forward.  An earlier revision read the env var here
        # and ignored the ``--prefix-position`` argument entirely, so a run that
        # asked for ``tail`` reported ``pos=front`` in its own probe and measured
        # the wrong arm without failing.
        if position not in {"front", "tail"}:
            raise ValueError(f"prefix position must be 'front' or 'tail', got {position!r}")
        self.position = position
        self.payload: Tensor | None = None
        self.applied = 0
        self.skipped = 0
        self._pending = 0

    def arm(self) -> None:
        """Declare that one coming forward's ``input_ids`` carry the reserved ids.

        The hook cannot see ``input_ids`` -- ``GlmOcrModel`` calls the text model
        with ``input_ids=None`` and assembled embeddings -- so it cannot confirm
        from the sequence that positions ``0..K-1`` are the reserved rows.  Without
        this flag the hook would happily overwrite the first K *real* tokens on any
        forward whose inputs were never extended, which is a silent corruption
        rather than a visible failure.  Arming is therefore explicit, so an
        un-extended forward splices nothing.

        A counter rather than a flag because validation interleaves forwards: the
        teacher-forcing pass and the generation prefill both consume extended
        inputs, in that order, within one page.  A flag would be spent by the first
        and the second would silently lose its prefix.
        """

        self._pending += 1

    def set_payload(self, payload: Tensor) -> None:
        from .writeback_intervention import apply_intervention, intervention_mode

        mode = intervention_mode()
        if mode != "full":
            payload = apply_intervention(payload, mode)
        self.payload = payload

    def hook(self, module: nn.Module, args: Any, kwargs: dict) -> Any:
        if self._pending <= 0:
            return None
        # Consume on this forward whatever the outcome, so a skipped splice cannot
        # leak into the next one.
        self._pending -= 1
        if os.environ.get(DISABLE_ENV_VAR, "").strip() not in {"", "0"}:
            # Diagnostic: the reserved span is in the sequence but nothing is
            # written into it, which separates the cost of the extra positions
            # from the cost of the vectors that go in them.  Note this is not the
            # same as "no vectors in the slots": the slots then hold whatever the
            # reserved tokens' own embedding rows contain, which are untrained.
            self.skipped += 1
            probe_path = os.environ.get(PROBE_ENV_VAR)
            if probe_path:
                # Recorded so the artifact distinguishes "the splice was
                # deliberately disabled" from "the splice never fired", which
                # otherwise look identical downstream.
                _probe(
                    probe_path,
                    {
                        "prefix_tokens": 0,
                        "prefix_offset": None,
                        "prefix_position": self.position,
                        "payload_mode": self.injector.payload_mode,
                        "splice_disabled": True,
                        "payload_norm": None,
                        "prefix_norm": None,
                        "embeds_norm": None,
                    },
                )
            return None
        payload = self.payload
        if payload is None:
            return None
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None:
            # The training path always reaches the text model with assembled
            # embeddings; if that ever changes, failing loudly is better than
            # quietly dropping the prefix.
            raise RuntimeError(
                "prefix injection expected inputs_embeds at the text-model seam"
            )
        past = kwargs.get("past_key_values")
        if past is not None and past.get_seq_length() > 0:
            # Decoding step: the reserved rows are already in the KV cache.
            return None
        width = payload.shape[1]
        if inputs_embeds.shape[1] < width:
            return None
        projected = self.injector(payload).to(inputs_embeds.dtype)
        start = 0 if self.position == "front" else inputs_embeds.shape[1] - width
        # Non-mutating splice: the caller's tensor may be a graph leaf.
        spliced = torch.cat(
            (
                inputs_embeds[:, :start, :],
                projected,
                inputs_embeds[:, start + width :, :],
            ),
            dim=1,
        )
        kwargs["inputs_embeds"] = spliced
        self.applied += 1
        probe_path = os.environ.get(PROBE_ENV_VAR)
        if probe_path:
            with torch.no_grad():
                _probe(
                    probe_path,
                    {
                        "prefix_tokens": int(width),
                        "prefix_offset": int(start),
                        "prefix_position": self.position,
                        "payload_mode": self.injector.payload_mode,
                        "payload_norm": float(payload.float().norm(dim=-1).mean()),
                        "prefix_norm": float(projected.float().norm(dim=-1).mean()),
                        "embeds_norm": float(inputs_embeds.float().norm(dim=-1).mean()),
                    },
                )
        return None


def install_prefix_injection(
    model: nn.Module,
    bridge: Any,
    *,
    token_count: int,
    reserved_ids: list[int],
    payload_mode_: PayloadMode | None = None,
    position: str | None = None,
) -> tuple[PrefixInjector, _Splice, Any]:
    """Route the bridge's branch output into ``K`` reserved prefix positions.

    Returns the injector, the splice state, and the removable hook handle.  The
    caller owns the optimizer: only ``injector.projection`` is new here, and it is
    zero-initialised, so a model with the prefix installed and no training step
    reproduces the un-prefixed model exactly.
    """

    text_model = _find_text_model(model)
    source_hidden = int(_payload_width(bridge, payload_mode_ or payload_mode()))
    model_hidden = int(getattr(text_model, "hidden_size", 0)) or int(
        model.get_input_embeddings().weight.shape[1]
    )
    injector = PrefixInjector(
        source_hidden,
        model_hidden,
        token_count,
        reserved_ids,
        payload_mode_=payload_mode_,
        dtype=next(model.parameters()).dtype,
    ).to(next(model.parameters()).device)
    # Attached to the decoder rather than left standalone.  DDP discovers
    # parameters by walking the wrapped module, and the optimizer in
    # ``train_screen`` collects ``adapter.parameters()`` plus LoRA explicitly, so a
    # detached injector would build fine, run fine, and never train -- the
    # projection would stay at its zero initialiser and the prefix would be a
    # constant.  Registering it puts it in ``model.parameters()`` where both DDP
    # and ``trainable_parameters`` (grad clipping, reporting) already look.
    if getattr(text_model, PREFIX_MODULE_NAME, None) is not None:
        raise RuntimeError(f"{PREFIX_MODULE_NAME} is already installed on the decoder")
    text_model.add_module(PREFIX_MODULE_NAME, injector)
    splice = _Splice(injector, position or prefix_position())
    bridge.prefix_splice = splice
    handle = text_model.register_forward_pre_hook(splice.hook, with_kwargs=True)
    return injector, splice, handle


@dataclass
class PrefixRuntime:
    """Everything the data path and the optimizer need to know about the prefix."""

    token_count: int
    reserved_ids: list[int]
    injector: PrefixInjector
    splice: _Splice
    handle: Any

    @property
    def position(self) -> str:
        # Single source of truth: the splice resolved it at install time from the
        # caller's argument.  Reading the environment here instead would let the
        # CLI flag and the actual placement disagree.
        return self.splice.position

    @property
    def logits_mask(self) -> PrefixLogitsMask:
        return PrefixLogitsMask(self.reserved_ids)

    def extend(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Add the reserved rows and arm the splice for the coming forward.

        Arming is part of extending on purpose: the two must not be able to drift
        apart, because arming without extending corrupts real tokens.
        """

        extended = extend_prefix_side_inputs(
            inputs, self.reserved_ids, position=self.position
        )
        self.splice.arm()
        return extended

    def parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.injector.parameters())


def enable_prefix_injection(
    model: nn.Module,
    bridge: Any,
    tokenizer: Any,
    *,
    token_count: int,
    payload_mode_: PayloadMode | None = None,
    position: str | None = None,
) -> PrefixRuntime:
    """Reserve the tokens, resize the embedding, install, and return the runtime.

    Order matters.  ``resize_token_embeddings`` builds a *new* embedding module, so
    the splice must be installed after it or it would hook a discarded module; and
    it leaves the fresh rows with ``requires_grad=True`` on a model whose
    parameters were all frozen before the adapter was installed, so the freeze is
    reapplied before the injector is added.  Without that second step the whole
    embedding matrix would silently become trainable.
    """

    if token_count <= 0:
        raise ValueError("enable_prefix_injection requires token_count > 0")
    reserved_ids = reserve_prefix_tokens(tokenizer, token_count)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    # The resize un-freezes the embedding it just built; restore the contract that
    # only the adapter, LoRA and the injector train.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    injector, splice, handle = install_prefix_injection(
        model,
        bridge,
        token_count=token_count,
        reserved_ids=reserved_ids,
        payload_mode_=payload_mode_,
        position=position,
    )
    return PrefixRuntime(
        token_count=token_count,
        reserved_ids=reserved_ids,
        injector=injector,
        splice=splice,
        handle=handle,
    )


def _find_text_model(model: nn.Module) -> nn.Module:
    """Locate the decoder that receives assembled ``inputs_embeds``.

    Found by identity rather than by a hard-coded attribute path, so a
    transformers refactor that renames ``language_model`` fails with a clear
    message instead of splicing into the wrong module.
    """

    embed_owner = None
    target = model.get_input_embeddings()
    for _, module in model.named_modules():
        for child in module.modules():
            if child is target:
                embed_owner = module
                break
        if embed_owner is not None:
            break
    for name in ("language_model", "model"):
        candidate = getattr(getattr(model, "model", None), name, None)
        if isinstance(candidate, nn.Module) and candidate is not embed_owner:
            return candidate
    for _, module in model.named_modules():
        if module is embed_owner:
            continue
        if getattr(module, "embed_tokens", None) is target:
            return module
    raise RuntimeError("could not locate the GLM-OCR text decoder seam")


def _payload_width(bridge: Any, mode: PayloadMode) -> int:
    """Width of the branch output the prefix carries before projection."""

    adapter = getattr(bridge, "adapter", None)
    if adapter is not None:
        hidden = getattr(getattr(adapter, "config", None), "hidden_size", None)
        if hidden:
            if mode == "regions":
                decoder = getattr(adapter, "region_decoder", None)
                config = getattr(decoder, "config", None)
                region_hidden = getattr(config, "decoder_hidden_size", None)
                if region_hidden:
                    return int(region_hidden)
            return int(hidden)
    raise RuntimeError("could not determine the layout branch hidden size")
