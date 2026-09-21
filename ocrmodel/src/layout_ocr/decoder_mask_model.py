"""Couple the learned mask head to the GLM-OCR decoder (plan section 3 and 6).

The head lives in :mod:`decoder_mask_router` and is deliberately tensor-in /
tensor-out.  This module owns everything that has to reach *inside* the decoder:
reading the visual tokens, reading the layer-``split_layer`` hidden state at the
query positions, running the head's recurrent scan, and adding the resulting
mask as an additive bias to the attention logits of layers ``split_layer..``.

The one non-negotiable constraint here is the attention backend.  A
differentiable additive float mask only propagates gradient through the eager
(math) attention path; SDPA and FlashAttention treat a float mask as a constant
and either drop the gradient or reject the tensor.  The installer therefore
forces ``text_config._attn_implementation = 'eager'`` before the first forward.

## Query indexing (plan 3)

The head predicts ``M_t`` for the token ``y_t`` from the hidden state at position
``q = P-1+t``, where ``P`` is the prompt length.  So the query positions are
``[P-1, P, ..., P+T-2]`` for ``T`` target tokens, and ``M_t`` supervises the box
of ``labels[:, q+1]`` -- never ``input_ids[:, q]``.  The last prompt position
(``P-1``) is what predicts the *first* target token, which is why the first
character's bias must fire inside the prefill and cannot inherit the old oracle
routing's "skip the prefill" behaviour.

## State discipline

Per-page state (projected keys, grid, previous mask, absolute positions) is a
plain object held by the runtime and cleared on every new page.  None of it is a
persistent buffer and none of it is saved to a checkpoint; the mask recurrence
is a page-local quantity, not a model weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .decoder_mask_router import DecoderMaskConfig, DecoderMaskRouter, _normalized_grid_xywh
from .prefix_injection import _find_text_model

ROUTER_MODULE_NAME = "decoder_mask_router"


@dataclass
class DecoderMaskRuntime:
    """The hook state and per-page bookkeeping for one installed mask head."""

    router: DecoderMaskRouter
    config: DecoderMaskConfig
    image_token_id: int
    spatial_merge_size: int

    # --- per-page state (cleared on ``clear_page``) ---
    xywh: Tensor | None = None
    spatial_shape: tuple[int, int] | None = None
    grid_thw: Tensor | None = None
    keys: Tensor | None = None
    image_positions: Tensor | None = None  # [N] absolute sequence positions
    query_positions: Tensor | None = None  # [T] absolute sequence positions
    mask_targets: Any | None = None
    prompt_length: int | None = None
    bias_strength: float = 0.0  # beta, ramped by the trainer from 0 to bias_max
    last_mask: Tensor | None = None  # [B, T, N] clean predicted mask
    last_stop: Tensor | None = None  # [B, T, 1] stop probability
    last_logits: Tensor | None = None  # [B, T, N] pre-sigmoid z

    # --- generation recurrence state ---
    prev_mask: Tensor | None = None  # [B, N] the previous step's mask
    _first_bias_layer: nn.Module | None = None
    _bias: Tensor | None = None

    def set_bias_strength(self, beta: float) -> None:
        if not 0.0 <= beta:
            raise ValueError("bias strength must be non-negative")
        self.bias_strength = float(beta)

    def set_noise(self, feedback_noise: float, input_noise: float) -> None:
        """Update the annealed noise levels for the coming forward."""
        self.router.feedback_noise = float(feedback_noise)
        self.router.input_noise = float(input_noise)

    def set_page(
        self,
        grid_thw: Tensor,
        input_ids: Tensor,
        prompt_length: int,
        mask_targets: Any | None,
    ) -> None:
        """Point the runtime at one page and compute its fixed geometry.

        ``mask_targets`` is the per-target-token supervision for a training
        forward, or ``None`` for a prompt-only generation forward (where the
        only query is the last prompt position).
        """

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("the mask runtime expects a single-page [1, L] input_ids")
        self.xywh, self.spatial_shape = _normalized_grid_xywh(grid_thw, self.spatial_merge_size)
        self.grid_thw = grid_thw
        positions = (input_ids[0] == int(self.image_token_id)).nonzero().flatten()
        if positions.numel() != self.xywh.shape[1]:
            raise RuntimeError(
                "image token count does not match the merged grid: "
                f"{positions.numel()} image tokens for {self.xywh.shape[1]} grid cells"
            )
        self.image_positions = positions.to(device=input_ids.device)
        total = int(input_ids.shape[1])
        self.prompt_length = int(prompt_length)
        if mask_targets is None:
            # Generation prefill: the last prompt position predicts the first token.
            if prompt_length < 1:
                raise ValueError("generation forward has an empty prompt")
            self.query_positions = torch.tensor(
                [prompt_length - 1], device=input_ids.device, dtype=torch.long
            )
        else:
            target_tokens = total - prompt_length
            if target_tokens < 1:
                raise ValueError("training forward has no target tokens")
            self.query_positions = torch.arange(
                prompt_length - 1, total - 1, device=input_ids.device, dtype=torch.long
            )
            if self.query_positions.numel() != target_tokens:
                raise RuntimeError("query-position count does not match the target tokens")
        self.mask_targets = mask_targets
        self.keys = None
        self.last_mask = None
        self.last_stop = None
        self.last_logits = None
        self.prev_mask = None

    def clear_page(self) -> None:
        self.xywh = None
        self.spatial_shape = None
        self.grid_thw = None
        self.keys = None
        self.image_positions = None
        self.query_positions = None
        self.mask_targets = None
        self.prompt_length = None
        self.last_mask = None
        self.last_stop = None
        self.last_logits = None
        self.prev_mask = None
        self._bias = None

    # --- hooks ---

    def capture_visual(self, module: nn.Module, args: Any, kwargs: dict) -> None:
        """Text-model pre-hook: project the image embeddings once per page.

        The text model is reached with ``input_ids=None`` and assembled
        ``inputs_embeds`` (see :mod:`prefix_injection`), so the visual tokens are
        read off the embeddings at the image positions resolved in ``set_page``.
        """

        if self.image_positions is None:
            return None
        if self.keys is not None:
            # A decode step re-embeds only the new token; the image tokens live in
            # the KV cache and were already projected on the prefill.
            return None
        inputs_embeds = kwargs.get("inputs_embeds")
        if inputs_embeds is None and args:
            inputs_embeds = args[0]
        if inputs_embeds is None:
            raise RuntimeError("mask routing expected inputs_embeds at the text-model seam")
        if self.xywh is None or self.grid_thw is None:
            return None
        visual = inputs_embeds[:, self.image_positions, :]  # [B, N, hidden]
        self.keys, self.xywh, self.spatial_shape = self.router.project_visual(
            visual, self.grid_thw, self.spatial_merge_size
        )
        return None

    def _run_head(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run the head for this forward and return ``(M, e, z)`` in ``[B, T, N]``.

        A prefill scans every query position; a decode step (``q_len == 1``) runs
        one step and carries the recurrence in ``prev_mask``.
        """

        router = self.router
        if hidden_states.shape[1] == 1:
            h = hidden_states[:, 0]  # [B, hidden]
            prev = self.prev_mask
            if prev is None:
                prev = torch.zeros(
                    (hidden_states.shape[0], self.keys.shape[1]),
                    device=hidden_states.device,
                    dtype=self.keys.dtype,
                )
            feedback = router._feedback(prev)
            mask, stop, z = router._step(h, feedback, self.keys, self.xywh, self.spatial_shape)
            self.prev_mask = mask
            return mask.unsqueeze(1), stop.unsqueeze(1), z.unsqueeze(1)
        h = hidden_states[:, self.query_positions, :]  # [B, T, hidden]
        mask, stop, z = router.scan(
            h, self.keys, self.xywh, self.spatial_shape, detach_every=router.config.detach_every
        )
        self.prev_mask = mask[:, -1]
        return mask, stop, z

    def _kv_length(self, hidden_states: Tensor, kwargs: dict) -> int:
        """Number of keys this forward attends over.

        Derived from ``cache_position`` rather than ``past_key_value`` so it does
        not depend on the ``Cache`` object's API (the proven pattern in
        :mod:`attention_routing` uses ``cache_position[-1] + 1``).  On a prefill
        the cache positions cover ``0..L-1`` and ``cache_position[-1] + 1 == L``;
        on a decode step they are exactly the one new position, so the same rule
        gives the full cached length.  When the model passes no ``cache_position``
        (a training forward without cache), the key length is simply ``q_len``.
        """

        cache_position = kwargs.get("cache_position")
        if isinstance(cache_position, Tensor) and cache_position.numel() > 0:
            return int(cache_position[-1].item()) + 1
        return int(hidden_states.shape[1])

    def _build_bias(self, mask: Tensor, hidden_states: Tensor, kwargs: dict) -> Tensor:
        """Scatter ``beta * M`` into a ``[1, 1, q_len, kv_len]`` bias on image keys.

        The bias is cast to the decoder's own dtype (BF16) so it adds cleanly to
        the eager attention logits; the gradient still reaches the FP32 mask
        through the cast.
        """

        q_len = hidden_states.shape[1]
        kv_len = self._kv_length(hidden_states, kwargs)
        bias = torch.zeros(1, 1, q_len, kv_len, device=mask.device, dtype=hidden_states.dtype)
        beta = self.bias_strength
        if beta == 0.0:
            return bias
        ip = self.image_positions  # [N] absolute, all present in the key range
        if q_len == 1:
            bias[0, 0, 0, ip] = (beta * mask[0, 0]).to(hidden_states.dtype)
        else:
            qp = self.query_positions  # [T] absolute == local rows on the prefill
            bias[0, 0, qp.unsqueeze(1), ip.unsqueeze(0)] = (beta * mask[0]).to(hidden_states.dtype)
        return bias

    def _causal_mask(self, hidden_states: Tensor, kwargs: dict) -> Tensor:
        """A float ``[1, 1, q_len, kv_len]`` causal mask when the model passes none.

        On the prefill the query rows are ``0..q_len-1``; on a decode step the one
        query sits at the last key position, so it attends to everything and the
        mask is all-zeros.
        """

        device = hidden_states.device
        q_len = hidden_states.shape[1]
        kv_len = self._kv_length(hidden_states, kwargs)
        key_pos = torch.arange(kv_len, device=device)
        if q_len == 1:
            query_pos = key_pos[-1:]
        else:
            query_pos = torch.arange(q_len, device=device)
        allowed = key_pos.unsqueeze(0) <= query_pos.reshape(-1, 1)  # [q_len, kv_len]
        mask = torch.zeros(1, 1, q_len, kv_len, device=device, dtype=hidden_states.dtype)
        return mask.masked_fill(~allowed.unsqueeze(0).unsqueeze(0), float("-inf"))

    def layer_hook(self, module: nn.Module, args: Any, kwargs: dict) -> None:
        """Layer pre-hook: compute the mask once, then add the bias to the mask.

        Registered on layers ``split_layer..``.  The first such layer owns the
        head scan (so the bias exists before any biased layer runs); every biased
        layer adds the same ``[1, 1, q_len, kv_len]`` bias to its attention mask.
        """

        hidden_states = kwargs.get("hidden_states")
        if hidden_states is None and args:
            hidden_states = args[0]
        if hidden_states is None:
            return None
        if self.keys is None:
            raise RuntimeError("mask routing keys were not projected before the biased layers")
        if module is self._first_bias_layer:
            mask, stop, z = self._run_head(hidden_states)
            self.last_mask = mask
            self.last_stop = stop
            self.last_logits = z
            self._bias = self._build_bias(mask, hidden_states, kwargs)
        bias = self._bias
        if bias is None:
            return None
        existing = kwargs.get("attention_mask")
        if existing is None:
            kwargs["attention_mask"] = self._causal_mask(hidden_states, kwargs) + bias
            return None
        if existing.ndim == 2:
            causal = self._causal_mask(hidden_states, kwargs)
            pad = existing[0].to(torch.bool)
            causal = causal.masked_fill(~pad.view(1, 1, 1, -1), float("-inf"))
            kwargs["attention_mask"] = causal + bias
            return None
        if existing.dtype == torch.bool:
            additive = torch.zeros_like(existing, dtype=bias.dtype).masked_fill(
                ~existing, float("-inf")
            )
            kwargs["attention_mask"] = additive + bias
        else:
            kwargs["attention_mask"] = existing + bias.to(existing.dtype)
        return None

    def report(self) -> dict[str, Any]:
        return {
            "bias_strength": self.bias_strength,
            "feedback_noise": self.router.feedback_noise,
            "input_noise": self.router.input_noise,
            "image_tokens": int(self.image_positions.numel()) if self.image_positions is not None else None,
            "query_positions": int(self.query_positions.numel()) if self.query_positions is not None else None,
            "grid_height": self.spatial_shape[0] if self.spatial_shape else None,
            "grid_width": self.spatial_shape[1] if self.spatial_shape else None,
        }


def install_decoder_mask_router(
    model: nn.Module,
    config: DecoderMaskConfig,
    image_token_id: int,
    spatial_merge_size: int,
) -> DecoderMaskRuntime:
    """Register the mask head on the decoder and wire its hooks.

    Returns the runtime.  The router is added as a submodule of the text model so
    DDP discovers its parameters and the optimizer collects them; the visual
    tower and the decoder stay frozen, exactly as in the LoRA recipe.
    """

    text_model = _find_text_model(model)
    if getattr(text_model, ROUTER_MODULE_NAME, None) is not None:
        raise RuntimeError(f"{ROUTER_MODULE_NAME} is already installed on the decoder")
    hidden = int(getattr(text_model, "hidden_size", 0)) or int(
        model.get_input_embeddings().weight.shape[1]
    )
    config = DecoderMaskConfig(**{**config.__dict__, "hidden_size": hidden})
    router = DecoderMaskRouter(config).to(
        device=next(model.parameters()).device, dtype=torch.float32
    )
    text_model.add_module(ROUTER_MODULE_NAME, router)
    runtime = DecoderMaskRuntime(router, config, int(image_token_id), int(spatial_merge_size))

    layers = list(getattr(text_model, "layers", []))
    if not layers:
        raise RuntimeError("could not find the GLM-OCR text decoder layers")
    if config.split_layer < 0 or config.split_layer >= len(layers):
        raise RuntimeError(
            f"split_layer {config.split_layer} is out of range for {len(layers)} layers"
        )
    runtime._first_bias_layer = layers[config.split_layer]
    # The visual capture must run before the first biased layer; a pre-hook on the
    # text model runs before every layer, so the ordering is guaranteed.
    handles = [
        text_model.register_forward_pre_hook(runtime.capture_visual, with_kwargs=True),
        *[
            layer.register_forward_pre_hook(runtime.layer_hook, with_kwargs=True)
            for layer in layers[config.split_layer :]
        ],
    ]
    runtime.handles = handles  # type: ignore[attr-defined]
    return runtime


def enable_eager_backend(model: nn.Module) -> None:
    """Force the math attention path so the additive mask keeps its gradient."""

    text_config = getattr(model.config, "text_config", None)
    if text_config is not None:
        text_config._attn_implementation = "eager"
    model.config._attn_implementation = "eager"
