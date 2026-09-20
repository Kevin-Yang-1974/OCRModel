"""Route the decoder's attention with a per-step spatial bias on the visual tokens.

Every layout injection tried so far put *information* into the sequence -- residuals
at the pre-merger seam, reserved prefix slots -- and the write-back matrix closed
that route: the whole usable contribution was a page-level vector, and the
per-patch component was adversarial.  This module changes something else.  It
leaves the sequence exactly as it was and biases *where the decoder looks*: at each
decoding step, an additive term on the attention logits of the visual keys that lie
inside a given box.

``bias`` is the strength in logits.  Zero is a *usable* arm, not an off switch: the
hooks are installed and fire on every decoding step while the mask they add stays
all-zero, so the arm is bit-identical to a run without this module *and* proves the
wiring.  A run that does not ask for the route at all -- no ``--layout-routing-bias``
-- is the one this module leaves untouched.

## Why only the decoding steps

``sdpa_attention_forward`` computes

    is_causal = query.shape[2] > 1 and attention_mask is None and is_causal

so a decode step (``q_len == 1``) already runs with ``is_causal=False``: a single
query at the end of the sequence attends to everything, and there is no causal
structure left to encode.  A bias added on those steps therefore needs no
hand-written causal mask, and the prefill -- the expensive pass, and the one that
would have to give up the flash kernel -- is left untouched.  The cost is that the
very first generated character comes from the prefill's last position and is not
biased; it is one position out of several hundred.

The bias is *added* to whatever mask ``create_causal_mask`` built rather than
replacing it.  On a decode step that mask carries no causal entries, but it may
still carry a padding structure, and adding preserves it for free.

## What picks the box

``characters[i]`` is the box of the character the decoder emits at step ``i``, but
which ``i`` to use is a real choice and the two answers measure different things.

``step``   index by generation step.  This is the only form a detector could
           supply -- it needs no text -- and it is what a deployed policy would be
           worth.  It is *not* a usable upper bound on this checkpoint: measured
           against the truth text, the step index drifts from the true reading
           position by a median of 6 characters and up to 1271, because the model
           over-generates by up to 4x on the pages that hit the token limit.
``synced`` walk the truth text alongside the generated one and point at the
           character the model has actually reached.  This is the upper bound.

The pointer is advanced from a hook on the model ``generate`` calls, not from the
decoder layers: this is the only place the *token ids* are visible (the text model
is reached with assembled embeddings), and it runs before any layer, so those
layers see the pointer this same forward produced.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch
from torch import Tensor, nn

from .prefix_injection import _find_text_model

POINTER_ENV_VAR = "GLMOCR_ROUTING_POINTER"
PROBE_ENV_VAR = "GLMOCR_ROUTING_PROBE"
ROUTING_ATTR = "layout_attention_routing"

# How the box for the current step is chosen.
#
# ``step``   the t-th box for the t-th generated character.  This is the only
#            form a detector could supply -- it needs no text -- and it is the
#            arm that says what a deployed policy would be worth.
# ``synced`` walk the truth text alongside the generated text and point at the
#            character the model has actually reached.  Measured on this
#            checkpoint, the step index drifts from the true reading position by
#            a median of 6 characters and up to 1271 (the model over-generates on
#            the pages that hit the token limit), so ``step`` is not a usable
#            upper bound and ``synced`` is the default for the truth arm.
POINTER_MODES = ("step", "synced")


def pointer_mode(default: str = "synced") -> str:
    raw = os.environ.get(POINTER_ENV_VAR, "").strip().lower()
    if not raw:
        return default
    if raw not in POINTER_MODES:
        raise ValueError(f"{POINTER_ENV_VAR} must be one of {POINTER_MODES}, got {raw!r}")
    return raw


def _probe(path: str, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def probe_path() -> str | None:
    """Where the per-page routing report is appended, if anywhere."""

    raw = os.environ.get(PROBE_ENV_VAR, "").strip()
    return raw or None


class AttentionRouting:
    """Per-step spatial bias on the visual keys, applied to every decoder layer."""

    # How far ahead in the truth text a generated character may match before the
    # pointer is allowed to jump.  Bounds the cost of a wrong character: without
    # it, a common glyph early in the page would snap the pointer back to it.
    LOOKAHEAD = 16

    def __init__(
        self,
        bridge: Any,
        bias: float,
        image_token_id: int | None,
        tokenizer: Any | None = None,
        pointer: str = "synced",
    ) -> None:
        if bias < 0:
            raise ValueError("AttentionRouting needs a non-negative bias")
        if pointer not in POINTER_MODES:
            raise ValueError(f"pointer must be one of {POINTER_MODES}, got {pointer!r}")
        if pointer == "synced" and tokenizer is None:
            raise ValueError("the synced pointer needs a tokenizer to read the generated ids")
        self.bridge = bridge
        self.bias = float(bias)
        self.image_token_id = image_token_id
        self.tokenizer = tokenizer
        self.pointer = pointer
        self.characters: list[dict[str, Any]] | None = None
        self.prompt_length: int | None = None
        self.visual_start: int | None = None
        self.visual_count: int | None = None
        self.page_id: str | None = None
        # Synced-pointer state: the truth text, the generated text so far, how many
        # generated ids have been decoded, and the truth index reached.
        self.reference: str | None = None
        self.position = 0
        self._decoded_ids = 0
        # One mask serves every layer of a forward: they all see the same
        # ``cache_position`` and the same page, so rebuilding it per layer would
        # repeat identical work 32 times per step.
        self._cache_key: int | None = None
        self._cache_mask: Tensor | None = None
        self.steps = 0
        self.biased = 0
        self.missing = 0
        self.boxes_hit = 0.0

    def set_page(
        self,
        page_id: str,
        characters: list[dict[str, Any]] | None,
        prompt_length: int,
        input_ids: Tensor,
        reference: str | None = None,
    ) -> None:
        """Point the route at one page's character boxes.

        The visual span is read off the prompt's own token ids rather than assumed
        to sit at a fixed offset, so a prefix span or a template change moves it
        with the sequence instead of silently biasing the wrong keys.
        """

        self.characters = characters
        self.prompt_length = int(prompt_length)
        self.page_id = page_id
        self.reference = reference
        self.position = 0
        self._decoded_ids = 0
        self._cache_key = None
        self._cache_mask = None
        self.steps = 0
        self.biased = 0
        self.missing = 0
        self.boxes_hit = 0.0
        if self.image_token_id is None:
            raise RuntimeError(
                "attention routing needs the model's image token id; "
                "install_attention_routing could not resolve it"
            )
        positions = (input_ids[0] == int(self.image_token_id)).nonzero().flatten()
        self.visual_count = int(positions.numel())
        self.visual_start = int(positions[0].item()) if self.visual_count else None

    def clear_page(self) -> None:
        self.characters = None
        self.prompt_length = None
        self.visual_start = None
        self.visual_count = None
        self._cache_key = None
        self._cache_mask = None

    def _advance(self, text: str) -> None:
        """Walk the pointer along the truth text by one run of generated characters.

        Greedy and monotone, with a bounded lookahead.  A character that matches
        where the pointer already is advances it; one that matches a little further
        on means the model skipped or the annotation dropped something, so the
        pointer moves there; anything else is an insertion and leaves the pointer
        alone.  The bound is what stops a common glyph from snapping the pointer
        back to an earlier occurrence of itself.
        """

        if self.reference is None:
            return
        for char in text:
            if self.position < len(self.reference) and self.reference[self.position] == char:
                self.position += 1
                continue
            limit = min(len(self.reference), self.position + self.LOOKAHEAD)
            found = self.reference.find(char, self.position, limit)
            if found != -1:
                self.position = found + 1

    def observe_inputs(self, module: nn.Module, args: Any, kwargs: dict) -> None:
        """Read the current step's sequence off the top-level model call.

        Hooked on the model ``generate`` calls rather than on the decoder layers
        because this is the only place the *token ids* are visible -- the text model
        is reached with assembled embeddings.  It runs before any layer, so the
        layers of this same forward see the pointer this sets.
        """

        if self.pointer != "synced" or self.characters is None or self.prompt_length is None:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            return None
        generated = input_ids.shape[1] - self.prompt_length
        if generated <= self._decoded_ids:
            return None
        tail = input_ids[0, self.prompt_length + self._decoded_ids :]
        self._decoded_ids = generated
        self._advance(self.tokenizer.decode(tail, skip_special_tokens=True))
        return None

    def _mask_for(self, step: int, kv_length: int, device: Any, dtype: torch.dtype) -> Tensor | None:
        if self.characters is None or self.visual_count in (None, 0) or self.visual_start is None:
            return None
        if not 0 <= step < len(self.characters):
            return None
        box = (self.characters[step] or {}).get("bbox")
        if box is None:
            # The annotation has no box for this character.  Fabricating one would
            # put an invented location under the one arm whose point is spatial
            # truth, so the step is simply left unbiased and counted.
            self.missing += 1
            return None
        positions = getattr(self.bridge, "last_patch_positions", None)
        if positions is None or positions.shape[1] != self.visual_count:
            raise RuntimeError(
                "attention routing has no patch grid for this page: the visual "
                "tower must run before the first biased decoding step"
            )
        grid = positions[0].to(device=device, dtype=torch.float32)
        inside = (
            (grid[:, 0] >= float(box[0]))
            & (grid[:, 0] <= float(box[2]))
            & (grid[:, 1] >= float(box[1]))
            & (grid[:, 1] <= float(box[3]))
        )
        count = int(inside.sum().item())
        self.boxes_hit += count
        self.biased += 1
        mask = torch.zeros(1, 1, 1, kv_length, device=device, dtype=dtype)
        mask[0, 0, 0, self.visual_start : self.visual_start + self.visual_count] = (
            inside.to(dtype) * self.bias
        )
        return mask

    def hook(self, module: nn.Module, args: Any, kwargs: dict) -> None:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if hidden_states is None or hidden_states.shape[1] != 1:
            return None
        cache_position = kwargs.get("cache_position")
        if cache_position is None or self.prompt_length is None:
            return None
        current = int(cache_position[-1].item())
        self.steps += 1
        if self.pointer == "synced":
            step = self.position
        else:
            # ``generate`` samples the first token from the prefill's last position,
            # so the decode step at position ``prompt_length + t - 1`` is the one
            # that emits character ``t``.
            step = current - self.prompt_length + 1
        if self._cache_key != current:
            self._cache_key = current
            self._cache_mask = self._mask_for(
                step, current + 1, hidden_states.device, hidden_states.dtype
            )
        mask = self._cache_mask
        if mask is None:
            return None
        existing = kwargs.get("attention_mask")
        if existing is None:
            kwargs["attention_mask"] = mask
        elif existing.dtype == torch.bool:
            additive = torch.zeros_like(existing, dtype=mask.dtype).masked_fill(
                ~existing, float("-inf")
            )
            kwargs["attention_mask"] = additive + mask
        else:
            kwargs["attention_mask"] = existing + mask.to(existing.dtype)
        return None

    def write_probe(self) -> None:
        """Append this page's report, so a run's artifacts show what it actually did.

        Without it a run whose ``set_page`` never fired -- a manifest missing the
        character channel, say -- would score the same as an unbiased baseline and
        read as "the bias does nothing".
        """

        path = probe_path()
        if path:
            _probe(path, self.report())

    def report(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "bias": self.bias,
            "pointer": self.pointer,
            "pointer_position": self.position if self.pointer == "synced" else None,
            "reference_characters": len(self.reference) if self.reference else None,
            "decoding_steps": self.steps,
            "biased_steps": self.biased,
            "missing_box_steps": self.missing,
            "mean_boxes_hit": (self.boxes_hit / self.biased) if self.biased else None,
            "visual_tokens": self.visual_count,
            "visual_start": self.visual_start,
        }


def install_attention_routing(
    model: nn.Module,
    bridge: Any,
    *,
    bias: float,
    tokenizer: Any | None = None,
    pointer: str = "synced",
) -> tuple[AttentionRouting, list[Any]]:
    """Register the bias hook on every text decoder layer.

    Returns the runtime and the removable hook handles.  The layers are found
    through the text model rather than by importing the class, so a transformers
    refactor that renames the layer fails with a clear message instead of silently
    biasing nothing.
    """

    text_model = _find_text_model(model)
    layers = list(getattr(text_model, "layers", []))
    if not layers:
        raise RuntimeError("could not find the GLM-OCR text decoder layers to route")
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(getattr(model.config, "text_config", None), "image_token_id", None)
    runtime = AttentionRouting(bridge, bias, image_token_id, tokenizer, pointer)
    handles = [
        # The model's own pre-hook first: it runs outside every layer, so the
        # pointer it advances is the one those layers use for this same forward.
        model.register_forward_pre_hook(runtime.observe_inputs, with_kwargs=True),
        *[
            layer.register_forward_pre_hook(runtime.hook, with_kwargs=True)
            for layer in layers
        ],
    ]
    return runtime, handles
