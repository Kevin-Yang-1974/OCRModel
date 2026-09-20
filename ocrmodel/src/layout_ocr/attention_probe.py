"""Observe, without intervening, where the decoder's attention falls each step.

This is the Stage 0/1 deliverable of ``plans/LAYOUT_ATTENTION_TRACKING.md``: a
*probe-only* instrument that records, per decoding step and per selected
layer/head, where the decoder looks, and then stops.  It adds no bias, does not
touch ``attention_mask``, and changes no weight and no cache -- the generation it
runs alongside must be bit-identical to a run without the probe, which is exactly
what the Stage 0 ``probe_only`` vs ``noroute`` comparison checks.

## Why this exists

The write-back diagnosis (``LAYOUT_BRANCH_WRITEBACK_DIAGNOSIS.md``) showed the
trained branch writes real spatial structure into the features (``lc_flat=0.39``)
while CER is unchanged -- the decoder does not *use* spatial information encoded in
the features.  The routing result (``LAYOUT_ATTENTION_ROUTING_RESULT.md``) showed
that a truth-boxed *bias on attention* does help.  Together the two point at
"layout information must reach the decoder's attention, not its features".  But the
routing arm is an oracle: it needs the true character box at every step.  A
deployable version has to *predict* the box, and it can only do that if the model's
own attention carries a readable "where am I now" signal.

The probe asks whether that signal exists, before any tracking machinery is built.
Its failure mode is not "localization is inaccurate" but "there is no signal": if
the visual total mass :math:`m_t` stays near zero, the decoder barely attends to the
visual tokens and no amount of layer/head tuning recovers a spatial readout.  That
is worth finding out early and cheaply.

## The two numbers

Per decode step :math:`t`, per selected query head, with visual keys :math:`V`, all
keys :math:`K_t`, and the logits the model actually used :math:`s_{t,k}`:

.. math::
    a^{\\mathrm{vis}}_{t,k} = \\frac{\\exp(s_{t,k})}{\\sum_{j \\in V} \\exp(s_{t,j})}
    \\qquad
    m_t = \\frac{\\sum_{k \\in V} \\exp(s_{t,k})}{\\sum_{j \\in K_t} \\exp(s_{t,j})}

:math:`a^{\\mathrm{vis}}` is the *visual-conditional* position distribution,
normalized over the visual keys alone, so it sums to 1 even when the model barely
looks at the image.  :math:`m_t` is the visual total mass, and it is the number that
separates "looks at the image" from "does not".

:math:`m_t` is a function of generation length, not a pure measure of looking: the
visual keys are fixed while the text keys grow, so :math:`m_t` falls as generation
proceeds.  The probe therefore records the log-sum-exp of the visual and the text
logits *separately* (``lse_vis``/``lse_text``), so the length effect can be removed
offline -- a fixed window, or the ``lse_vis - lse_text`` logit gap -- before "signal
exists" is judged.

## How the weights are obtained

The probe recomputes the attention logits from the model's own projections rather
than asking SDPA to materialize its weights, which would change the kernel path and
break the bit-identical guarantee.  For each selected attention module it re-runs
``q_proj``/``k_proj`` on the captured ``hidden_states`` in ``no_grad`` and applies
the *same* rotary embedding the forward used, which it reads from the
``position_embeddings`` argument.

That last point is the one to get wrong quietly.  Checked against transformers 5.3
(``models/glm_ocr/modeling_glm_ocr.py``):

* ``GlmOcrTextAttention`` holds **no** ``rotary_emb`` and **no** ``q_norm``/``k_norm``
  -- the per-head normalization the architecture suggests belongs to
  ``GlmOcrVisionAttention``, not the text tower.  Rotary reaches the text attention
  only as ``position_embeddings=(cos, sin)``, computed once by the text model.
* ``position_ids`` is **not** an argument of the attention forward, so a transform
  written against it silently applies no rotary at all and reports logits from
  un-rotated queries and keys -- plausible-looking and completely wrong.
  ``_project`` therefore treats a missing ``position_embeddings`` as a hard failure
  of that layer rather than quietly skipping the rotation; see ``transform_failed``
  in the report.

What remains is the ordering rather than the math, and it is what the Stage 0
single-step eager check compares: the recomputed weights against the weights an
eager attention kernel returns for the same forward.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

import torch
from torch import Tensor, nn

from .prefix_injection import _find_text_model

PROBE_ENV_VAR = "GLMOCR_ATTENTION_PROBE"
LAYERS_ENV_VAR = "GLMOCR_ATTENTION_PROBE_LAYERS"
HEADS_ENV_VAR = "GLMOCR_ATTENTION_PROBE_HEADS"
PROBE_ATTR = "layout_attention_probe"

# The plan asks to probe layers 0/4/8/12 first and keep per-head statistics.  This
# is a development default, not a property of the model: a run reads the loaded
# config and records what it actually found rather than trusting these numbers.
DEFAULT_LAYERS = (0, 4, 8, 12)

# Which box list the line readout is expressed in.
#
# ``regions``    the page's annotated lines.  Every localization number reported so far is in this
#                space, and it is an oracle: the labels exist only in the annotation.
# ``predicted``  the detector's boxes.  Nothing in the path then needs the annotation, which is what
#                makes a tracked arm deployable -- the line it names has to be a line a detector
#                could have produced, or the tracker is aiming at a map that will not exist.
BOX_MAPS = ("regions", "predicted")


def probe_path() -> str | None:
    """Where the per-page attention report is appended, if anywhere."""

    raw = os.environ.get(PROBE_ENV_VAR, "").strip()
    return raw or None


def _env_ints(name: str) -> tuple[int, ...] | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        values = tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated list of ints") from error
    if not values or min(values) < 0:
        raise ValueError(f"{name} must be a non-empty list of non-negative ints")
    return values


def probe_layers(default: tuple[int, ...] = DEFAULT_LAYERS) -> tuple[int, ...]:
    """Layer indices to observe, from the environment as a comma-separated list."""

    return _env_ints(LAYERS_ENV_VAR) or default


def probe_heads(default: tuple[int, ...] | None = None) -> tuple[int, ...] | None:
    """Query-head indices to observe; ``None`` means every head."""

    return _env_ints(HEADS_ENV_VAR) or default


def _incremental_text(decoder: Any, tokens: list[int]) -> list[str]:
    """Per-token text, such that concatenating it reproduces the full decode.

    One token can carry several characters and one character can span several tokens,
    so the attribution has to come from the cumulative decode rather than from
    ``decode([token])`` -- which on a byte-level tokenizer returns a replacement
    character for any token holding half of a multi-byte character.

    The subtlety is what to do when a partial character completes.  Decoding the prefix
    up to token ``s`` can end in U+FFFD, and decoding up to ``s + 1`` replaces it with
    the finished character, so the longer decode is *not* an extension of the shorter
    one.  Diffing on the longest common prefix handles that, but it leaves the fragment
    attributed to token ``s`` -- which would duplicate it, because the finished
    character also contains it.  The fragment is therefore retracted from the step that
    emitted it before the completion is appended to this one.

    An earlier version fell back to appending the whole cumulative string whenever the
    prefix property failed.  That is not a rare fallback: on a real page roughly one
    token in ten ends on a partial character, and each one re-appended the entire text
    so far.  One 245-step page came out as 3485 characters instead of 284, which the
    alignment then read as seventeen thousand insertions.
    """

    emitted: list[str] = []
    previous = ""
    for index in range(len(tokens)):
        current = decoder.decode(tokens[: index + 1], skip_special_tokens=True)
        common = 0
        limit = min(len(previous), len(current))
        while common < limit and previous[common] == current[common]:
            common += 1
        retract = len(previous) - common
        while retract > 0 and emitted:
            # Walk back over the slots that actually hold text; an emptied slot keeps
            # its position so step indices stay aligned with token indices.
            for position in range(len(emitted) - 1, -1, -1):
                if retract <= 0:
                    break
                text = emitted[position]
                take = min(retract, len(text))
                emitted[position] = text[: len(text) - take]
                retract -= take
        emitted.append(current[common:])
        previous = current
    return emitted


def _region_owners(regions: list[dict[str, Any]], positions: Tensor) -> Tensor:
    """Return the region index owning each visual token, or ``-1`` for background.

    Mirrors ``layout_targets``: a token belongs to the first region (by reading
    order) whose normalized box contains its normalized patch centre.  Overlap is
    resolved by ``argmax`` over the ``inside`` flags, so the tie-break is the
    reading-order index, exactly as the target builder does it.
    """

    owners = torch.full((positions.shape[1],), -1, dtype=torch.long, device=positions.device)
    if not regions:
        return owners
    boxes = torch.tensor(
        [region["bbox"] for region in regions], dtype=torch.float32, device=positions.device
    )
    centers = positions[0]
    inside = (
        (centers[:, None, 0] >= boxes[None, :, 0])
        & (centers[:, None, 0] <= boxes[None, :, 2])
        & (centers[:, None, 1] >= boxes[None, :, 1])
        & (centers[:, None, 1] <= boxes[None, :, 3])
    )
    has_owner = inside.any(dim=-1)
    owners[has_owner] = inside[has_owner].float().argmax(dim=-1)
    return owners


def _logsumexp(values: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable per-slice log-sum-exp, ``-inf`` on an empty or fully-masked slice.

    A row can be entirely ``-inf`` once a padding or routing mask is added, and the
    naive ``(v - max).exp().log() + max`` turns that into ``nan``, which would then
    spread through every downstream mean.  ``nan`` in a measurement is worse than
    ``-inf``: ``-inf`` reads as "this head could not see anything".
    """

    dim = dim % values.ndim  # so the ``-1`` default indexes the same axis in both slices
    shape = values.shape[:dim] + values.shape[dim + 1 :]
    if values.shape[dim] == 0:
        return torch.full(shape, float("-inf"), dtype=values.dtype, device=values.device)
    maximum = values.amax(dim=dim, keepdim=True)
    finite = torch.isfinite(maximum)
    safe_max = torch.where(finite, maximum, torch.zeros_like(maximum))
    total = (values - safe_max).exp().sum(dim=dim).log() + safe_max.squeeze(dim)
    return torch.where(finite.squeeze(dim), total, torch.full_like(total, float("-inf")))


class AttentionProbe:
    """Hook the selected attention layers and reduce their per-step visual focus.

    Split cleanly into the parts a fake model can exercise (state, the reduction
    math, the report) and the parts specific to the running ``modeling_glm_ocr.py``
    (``_project`` and the hook wiring).  The projection/rotary transform is the one
    piece that cannot be argued correct by reading this file alone -- see the module
    docstring.
    """

    def __init__(
        self,
        bridge: Any,
        image_token_id: int | None,
        *,
        layers: tuple[int, ...] = DEFAULT_LAYERS,
        heads: tuple[int, ...] | None = None,
        tracked: Any | None = None,
        box_map: str = "regions",
        routing_bias: float = 0.0,
        correct_confidence: bool = False,
        next_line_scale: float = 0.0,
    ) -> None:
        if box_map not in BOX_MAPS:
            raise ValueError(f"box_map must be one of {BOX_MAPS}, got {box_map!r}")
        if correct_confidence and routing_bias <= 0.0:
            # With no bias there is nothing to divide out, and a run that asked for the
            # correction would silently report uncorrected numbers.
            raise ValueError("bias-corrected confidence needs the routing bias it inverts")
        self.box_map = box_map
        # The bias the routing applies while this step's attention is being read, and whether to
        # divide it back out before estimating the line.  See ``_uncorrected_dist``.
        self.routing_bias = float(routing_bias)
        self.correct_confidence = bool(correct_confidence)
        # The share of the bias the routing put on the next line, so the correction divides out
        # both.  Zero is the one-line arm every earlier result used.
        self.next_line_scale = float(next_line_scale)
        self.bias_corrected_steps = 0
        self.bridge = bridge
        self.image_token_id = image_token_id
        self.layers = tuple(layers)
        self.heads = None if heads is None else tuple(heads)
        # Where to publish this step's line estimate, for an arm that aims the bias by it.  The
        # probe does not know or care what reads it; see attention_tracking.
        self.tracked = tracked
        # Per-page state, resolved by ``set_page``.
        self.page_id: str | None = None
        self.visual_start: int | None = None
        self.visual_count: int | None = None
        self.num_regions: int | None = None
        self.owners: Tensor | None = None
        self._regions: list[dict[str, Any]] = []
        self._line_direction: list[str] = []
        self.grid_missing = 0
        # Captured keys, per KV head, as ``[num_kv_heads, seq, head_dim]``.  The
        # prefill carries the whole prompt, so its key span holds every visual token
        # and the prompt text; each decode step contributes exactly one new text key.
        self._visual_keys: dict[int, Tensor] = {}
        self._prompt_text_keys: dict[int, Tensor] = {}
        self._gen_keys: dict[int, list[Tensor]] = {}
        self._geometry: dict[int, tuple[int, int, int, float]] = {}
        self._transform_failed: dict[int, str] = {}
        self.steps = 0
        self._step_key: int | None = None
        self.emitted_missing = 0
        self.emitted_join_mismatch = 0
        self._records: list[dict[str, Any]] = []

    # -- page lifecycle -----------------------------------------------------

    def set_page(
        self,
        page_id: str,
        regions: list[dict[str, Any]],
        prompt_length: int,
        input_ids: Tensor,
        predicted_lines: list[list[float]] | None = None,
    ) -> None:
        """Point the probe at one page: resolve the visual span and the line map."""

        self.page_id = page_id
        self.owners = None
        self._regions = []
        self._visual_keys = {}
        self._prompt_text_keys = {}
        self._gen_keys = {}
        self._transform_failed = {}
        self.steps = 0
        self._step_key = None
        self.grid_missing = 0
        self.emitted_missing = 0
        self.emitted_join_mismatch = 0
        self.bias_corrected_steps = 0
        self._records = []
        if self.image_token_id is None:
            raise RuntimeError(
                "attention probe needs the model's image token id; "
                "install_attention_probe could not resolve it"
            )
        positions = (input_ids[0] == int(self.image_token_id)).nonzero().flatten()
        self.visual_count = int(positions.numel())
        self.visual_start = int(positions[0].item()) if self.visual_count else None
        # Sorted by reading order, exactly as ``layout_targets`` does it: the region
        # index is the line label the offline evaluation compares against, and the
        # manifest's own order is not guaranteed to be reading order.  Sorting here
        # rather than at the call site keeps the probe's labels and the truth's
        # labels the same kind of thing.
        if self.box_map == "predicted":
            # The same ordered list the bias reads, so an index means one box on both sides.  A
            # predicted box has no direction -- deriving it is a separate problem, and the in-line
            # position only wants it as an axis choice.
            self._regions = [
                {"bbox": list(box), "reading_order": index}
                for index, box in enumerate(predicted_lines or [])
            ]
        else:
            self._regions = sorted(regions, key=lambda item: int(item["reading_order"]))
        self.num_regions = len(self._regions)
        self._line_direction = [
            region.get("writing_direction", "unknown") for region in self._regions
        ]
        # The token -> line map is resolved lazily by ``_ensure_owners``: it needs the
        # patch grid, and that grid is written by the visual tower *during* the
        # generation prefill, which has not happened yet at ``set_page`` time.

    def _ensure_owners(self) -> None:
        """Resolve the token -> line map, once the visual tower has produced a grid."""

        if self.owners is not None or not self.visual_count:
            return
        grid = getattr(self.bridge, "last_patch_positions", None)
        if grid is None:
            return
        if grid.shape[1] != self.visual_count:
            raise RuntimeError(
                "attention probe has no matching patch grid for this page: the visual "
                f"tower produced {grid.shape[1]} patches for {self.visual_count} image tokens"
            )
        self.owners = _region_owners(self._regions, grid)

    def clear_page(self) -> None:
        self.page_id = None
        self.visual_start = None
        self.visual_count = None
        self.num_regions = None
        self.owners = None
        self._regions = []
        self._line_direction = []
        self._visual_keys = {}
        self._prompt_text_keys = {}
        self._gen_keys = {}

    # -- capture ------------------------------------------------------------

    def _store_prompt(self, layer: int, key: Tensor) -> None:
        """Split one prefill key span into the visual prefix and the prompt text.

        The visual span is contiguous in the LLM sequence (``attention_routing``
        relies on the same fact), so the two slices cover it exactly, and the text is
        stored in sequence order -- before the span, then after -- which is the order
        the mask is split in as well.
        """

        if self.visual_start is None or not self.visual_count:
            return
        end = self.visual_start + self.visual_count
        # Dim 2 is the sequence.  Keys are ``[batch, kv_heads, seq, head_dim]``, so a
        # slice taken on dim 1 selects *KV heads* -- which on a GQA model silently
        # returns one head's keys and calls them a token span.
        self._visual_keys[layer] = key[:, :, self.visual_start : end].detach()
        text = torch.cat((key[:, :, : self.visual_start], key[:, :, end:]), dim=2)
        if text.shape[2] > 0:
            self._prompt_text_keys[layer] = text.detach()

    def _text_keys(self, layer: int) -> Tensor | None:
        """Every non-visual key, in sequence order, as ``[1, kv_heads, T, head_dim]``."""

        parts = [
            part
            for part in (
                self._prompt_text_keys.get(layer),
                torch.cat(self._gen_keys[layer], dim=2) if self._gen_keys.get(layer) else None,
            )
            if part is not None
        ]
        return torch.cat(parts, dim=2) if parts else None

    def _text_key_count(self, layer: int) -> int:
        prompt = self._prompt_text_keys[layer].shape[2] if layer in self._prompt_text_keys else 0
        generated = sum(keys.shape[2] for keys in self._gen_keys.get(layer, []))
        return int(prompt + generated)

    def _split_mask(self, mask: Tensor, layer: int) -> tuple[Tensor | None, Tensor | None]:
        """Split a full-sequence attention mask the way the keys were split.

        Only needed when something has already put a mask on this forward -- the
        routing bias, or a padding structure.  Without one the decode step runs
        unmasked and there is nothing to split.
        """

        if mask.dtype == torch.bool:
            # A boolean mask says "attend / do not attend", not "add this".  Adding it
            # would raise the allowed keys' logits by one instead of excluding the
            # others, so it is converted to the additive form the model's own attention
            # then consumes -- the same conversion the routing module applies.
            mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                ~mask, float("-inf")
            )
        # Flatten everything but the key axis.  A decode step arrives as a single row, but
        # a prefill arrives with one row *per query position* -- the causal mask is not
        # one row -- and taking row 0 there would silently compare every position against
        # the first position's visible set.
        rows = mask.reshape(-1, mask.shape[-1]).float()
        expected = self._text_key_count(layer) + int(self.visual_count or 0)
        if rows.shape[-1] != expected:
            raise RuntimeError(
                "attention probe cannot align the mask with its captured keys on layer "
                f"{layer}: the mask covers {rows.shape[-1]} keys, the probe holds {expected}"
            )
        if self.visual_start is None or not self.visual_count:
            return None, rows.unsqueeze(0).unsqueeze(0)
        end = self.visual_start + self.visual_count
        visual = rows[:, self.visual_start : end]
        text = torch.cat((rows[:, : self.visual_start], rows[:, end:]), dim=-1)
        # Two leading axes to broadcast against ``[kv_heads, groups, q_len, keys]``.
        return visual.unsqueeze(0).unsqueeze(0), text.unsqueeze(0).unsqueeze(0)

    # -- reduction ----------------------------------------------------------

    def _reduce(
        self,
        layer: int,
        query: Tensor,
        visual: Tensor,
        text: Tensor | None,
        mask_vis: Tensor | None = None,
        mask_text: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Per-head visual total mass and visual-conditional distribution.

        Shapes: ``query`` is ``[1, num_heads, q_len, head_dim]``, ``visual`` is
        ``[1, kv_heads, V, head_dim]``, ``text`` is ``[1, kv_heads, T, head_dim]``.
        Returns ``mass``, ``dist``, ``lse_vis`` and ``lse_text`` as
        ``[num_heads, q_len]``, ``[num_heads, q_len, V]``, ``[num_heads, q_len]`` and
        ``[num_heads, q_len]``.  The two log-sum-exps come back because the report keeps
        them: ``m_t`` alone cannot separate "looks at the image" from "generated a lot",
        and the difference between them is what removes the length effect.

        Kept as its own function, taking the keys rather than reading the captured state,
        for one reason: it is the only place the probe's recomputed logits become a
        number, and the single-step eager check has to run this same math over key
        tensors it projected itself.  Two copies of it would let the validated version
        and the run version drift apart.
        """

        num_heads, kv_heads, head_dim, scale = self._geometry[layer]
        groups = num_heads // kv_heads
        q_len = int(query.shape[2])
        # Map each query head onto its KV head (GQA).  The flatten is kv-major so that
        # query head ``kv * groups + g`` reads KV head ``kv``, which is the layout
        # ``repeat_kv`` produces; the other order (groups-major) is a silent permutation
        # of the heads and would mislabel every per-head number.
        probe = query[0].reshape(kv_heads, groups, q_len, head_dim).float()
        vis = visual.reshape(kv_heads, -1, head_dim).float()
        # sdpa_attention_forward scales first and adds the mask to the scaled logits, so
        # the mask is added unscaled: multiplying it by ``scale`` would divide the routing
        # arm's B by sqrt(head_dim) and report a different arm than the one that ran.
        logits_vis = torch.einsum("khqd,kvd->khqv", probe, vis) * scale
        if text is not None:
            logits_text = torch.einsum(
                "khqd,ktd->khqt", probe, text.reshape(kv_heads, -1, head_dim).float()
            ) * scale
        else:
            logits_text = torch.empty(*logits_vis.shape[:3], 0, dtype=torch.float32)
        # Both mask halves are one entry per key, so they broadcast along the trailing
        # axis of the ``[kv_heads, groups, q_len, keys]`` logits; a reshape would need
        # ``kv_heads * groups`` copies of the row and is the wrong shape for any GQA model.
        if mask_vis is not None:
            logits_vis = logits_vis + mask_vis
        if mask_text is not None and logits_text.shape[-1]:
            logits_text = logits_text + mask_text
        logits_vis = logits_vis.reshape(num_heads, q_len, -1)
        logits_text = logits_text.reshape(num_heads, q_len, -1)

        lse_vis = _logsumexp(logits_vis)
        lse_text = _logsumexp(logits_text)
        lse_all = torch.logaddexp(lse_vis, lse_text)
        # A row a mask emptied entirely leaves ``lse_all`` at ``-inf``, and ``-inf - -inf``
        # is ``nan``.  Zero is the honest reading of "this head had no mass anywhere";
        # ``nan`` would spread through every downstream mean.
        mass = torch.where(
            torch.isfinite(lse_vis) & torch.isfinite(lse_all),
            (lse_vis - lse_all).exp(),
            torch.zeros_like(lse_vis),
        )
        # A fully-masked visual row gives every logit ``-inf``; subtracting an ``-inf`` lse
        # would make the whole row ``nan``.  An all-zero row is the honest answer: this
        # head saw no visual token at all.
        visible = torch.isfinite(lse_vis).unsqueeze(-1)
        dist = torch.where(
            visible,
            (logits_vis - lse_vis.unsqueeze(-1)).softmax(dim=-1),
            torch.zeros_like(logits_vis),
        )
        return mass, dist, lse_vis, lse_text

    def reduce_captured(
        self, layer: int, query: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | None:
        """Reduce a query against the keys captured for this page.

        The mask is split here rather than inside ``_reduce`` because it arrives as one
        whole-sequence tensor that has to be cut against the captured key counts.
        """

        visual = self._visual_keys.get(layer)
        if visual is None or not self.visual_count:
            return None
        mask_vis = mask_text = None
        if mask is not None:
            mask_vis, mask_text = self._split_mask(mask, layer)
        return self._reduce(layer, query, visual, self._text_keys(layer), mask_vis, mask_text)

    def _bias_added_per_token(self, line: int) -> Tensor | None:
        """The logit the routing added to each visual token, exactly as the routing decided it.

        The box test ``attention_routing`` applies to the patch grid, rebuilt here rather than
        reusing the region owners: an owner is resolved by first-in-reading-order among the boxes
        that contain a token, so two overlapping regions would put a token in one line here and
        bias it under the other there.  The correction has to subtract the bias that was actually
        added, so it follows the routing's rule -- including the share that lands on the next line
        when one is configured.
        """

        outside = self._blank_bias()
        if outside is None or line < 0:
            return None
        added = torch.zeros_like(outside)
        covered = torch.zeros_like(outside, dtype=torch.bool)
        for offset, share in ((0, 1.0), (1, self.next_line_scale)):
            index = line + offset
            if share <= 0.0 or index >= len(self._regions):
                continue
            inside = self._tokens_inside_box(self._regions[index].get("bbox"))
            if inside is None:
                continue
            # The routing adds the share only where the current line does not already cover the
            # token, so a token in both boxes must not be divided by more than was added.
            added = added + (inside & ~covered).to(added.dtype) * (self.routing_bias * share)
            covered = covered | inside
        return added if bool(covered.any()) else None

    def _blank_bias(self) -> Tensor | None:
        """A zero tensor over the visual tokens, or ``None`` when the grid is not available.

        The grid is only read for its device and length: the accumulator's dtype has to be a
        float, and the patch positions are integers.
        """

        grid = getattr(self.bridge, "last_patch_positions", None)
        if not self.visual_count or grid is None or grid.shape[1] != self.visual_count:
            return None
        return torch.zeros(self.visual_count, dtype=torch.float32, device=grid.device)

    def _tokens_inside_box(self, box: list[float] | None) -> Tensor | None:
        if not box or not self.visual_count:
            return None
        grid = getattr(self.bridge, "last_patch_positions", None)
        if grid is None or grid.shape[1] != self.visual_count:
            return None
        points = grid[0].to(device=grid.device, dtype=torch.float32)
        return (
            (points[:, 0] >= float(box[0]))
            & (points[:, 0] <= float(box[2]))
            & (points[:, 1] >= float(box[1]))
            & (points[:, 1] <= float(box[3]))
        )

    def _tokens_inside_line(self, line: int) -> Tensor | None:
        """The routing's box mask for one line, ignoring any next-line share."""

        if line < 0 or line >= len(self._regions):
            return None
        return self._tokens_inside_box(self._regions[line].get("bbox"))

    def _observe_step(self, layer: int, step: int, query: Tensor, mask: Tensor | None) -> None:
        """Reduce one decode step's attention over the visual keys and record it."""

        if query.shape[2] != 1:
            raise RuntimeError(
                f"the probe reduces one decode step at a time, but layer {layer} got a "
                f"query of length {query.shape[2]}; a prefill should have gone to "
                "_store_prompt instead"
            )
        reduced = self.reduce_captured(layer, query, mask)
        if reduced is None:
            # The prefill stored no visual keys for this layer (the transform failed), so
            # there is nothing to score and a fabricated row would be worse than a missing
            # one.
            return
        mass, dist, lse_vis, lse_text = reduced
        # One query position per observed step, so the step's numbers are position 0.
        mass = mass[:, 0]
        dist = dist[:, 0]
        lse_vis = lse_vis[:, 0]
        lse_text = lse_text[:, 0]
        num_heads = dist.shape[0]
        # Divide the routing bias back out before anything reads this distribution.  Observation
        # and intervention act on one mechanism, so the estimate is being read off a quantity the
        # previous step's bias has moved: adding B to one line's keys multiplies that line's mass
        # by e^B, which inflates exactly the confidence the gate then tests.  The biased line is
        # ``tracked.line`` as it stands right now -- written by the previous step, and the value
        # the routing hook carried into this same forward.
        raw_dist = dist
        corrected_from = -1
        if self.correct_confidence and self.tracked is not None:
            biased_line = int(getattr(self.tracked, "line", -1))
            added = self._bias_added_per_token(biased_line)
            if added is not None and bool(added.any()):
                # Every visual token's weight was multiplied by e^added, so dividing by exactly
                # that puts the distribution back where no bias was applied.
                dist = dist * torch.exp(-added).unsqueeze(0)
                total = dist.sum(dim=-1, keepdim=True)
                # A head that put no mass inside the biased line is unchanged by the bias, and
                # dividing by its own total leaves it that way; clamp only guards a zero row.
                dist = dist / total.clamp_min(torch.finfo(dist.dtype).tiny)
                corrected_from = biased_line
                self.bias_corrected_steps += 1
        entropy = -(dist * dist.clamp_min(1e-12).log()).sum(dim=-1)
        # Normalized by log|V| so the number is comparable across resolutions: the
        # 1M and 4M arms see 630 and 2496 visual tokens.
        entropy_norm = entropy / math.log(self.visual_count) if self.visual_count > 1 else entropy

        # ``_observe_step`` only runs after the prefill, so the grid exists by now; if it
        # still does not, the per-line fields are absent -- counted, because a report of
        # "no localization signal" and a report of "never measured localization" must
        # not look the same.
        self._ensure_owners()
        if self.owners is None and self._regions:
            self.grid_missing += 1
        owners = self.owners.to(dist.device) if self.owners is not None else None
        heads = self.heads if self.heads is not None else range(num_heads)
        records: list[dict[str, Any]] = []
        # Every head at once.  The per-head loop this replaces did a scatter and a masked
        # weighted sum per head per step, which at four layers, sixteen heads and a few hundred
        # steps a page is millions of small torch calls -- the probe's own overhead on top of the
        # generation it is watching, and the one part of it that is mine to remove.
        per_line_by_head: list[list[float]] | None = None
        raw_top_mass_by_head: list[float] | None = None
        raw_argmax_by_head: list[int] | None = None
        in_line_by_head: list[float] | None = None
        argmax_by_head: list[int] | None = None
        if owners is not None and self.num_regions:
            # The last slot holds everything off every line -- background, or a token no box
            # covers.  Keeping it visible matters: a high background share means the "line"
            # readout is reading noise, not a line.
            slots = owners.clamp_min(0) + (owners < 0).long() * self.num_regions
            per_line = torch.zeros(
                dist.shape[0], self.num_regions + 1, dtype=dist.dtype, device=dist.device
            )
            per_line.index_add_(1, slots, dist)
            lines_out = per_line[:, : self.num_regions]
            in_line_by_head = self._in_line_position(dist, owners, int(lines_out.shape[1]))
            per_line_by_head = per_line.tolist()
            argmax_by_head = lines_out.argmax(dim=1).tolist()
            if corrected_from >= 0:
                # What the same reduction would have said with the bias still in it.  Carried on
                # every corrected step so the offline analysis can price the correction instead of
                # comparing two runs that differ in more than one way.
                raw_per_line = torch.zeros_like(per_line)
                raw_per_line.index_add_(1, slots, raw_dist)
                raw_lines = raw_per_line[:, : self.num_regions]
                raw_argmax_by_head = raw_lines.argmax(dim=1).tolist()
                raw_top_mass_by_head = [
                    float(raw_per_line[head, int(raw_argmax_by_head[head])])
                    for head in range(raw_per_line.shape[0])
                ]

        for head in heads:
            head = int(head)
            row: dict[str, Any] = {
                "layer": layer,
                "head": head,
                "m_t": float(mass[head]),
                "lse_vis": float(lse_vis[head]),
                "lse_text": float(lse_text[head]),
                "entropy_norm": float(entropy_norm[head]),
            }
            if per_line_by_head is not None:
                line_probs = per_line_by_head[head]
                argmax = int(argmax_by_head[head])
                row["line_probs"] = line_probs
                row["argmax_line"] = argmax
                row["top_line_mass"] = line_probs[argmax]
                row["background_mass"] = line_probs[-1]
                row["in_line_pos"] = in_line_by_head[head]
                if raw_top_mass_by_head is not None and head < len(raw_top_mass_by_head):
                    row["top_line_mass_raw"] = raw_top_mass_by_head[head]
                    row["argmax_line_raw"] = int(raw_argmax_by_head[head])
            records.append(row)
        # One entry per layer per step: ``heads`` carries the layer, so the offline
        # tool can group by step without the probe having to accumulate a nested
        # structure it would then have to keep in memory for the whole page.
        self._records.append(
            {
                "step": step,
                "text_keys": self._text_key_count(layer),
                "heads": records,
                "bias_corrected_from": corrected_from,
            }
        )
        if self.tracked is not None:
            # Publish this step's estimate for an arm that aims its bias by it.  Written here, on a
            # post-hook, so a reader hooked earlier in the next forward sees the previous step's
            # value -- which is the one-step lag the plan asks for, and it needs no extra state.
            from .attention_tracking import (
                aggregate_line_distribution,
                aggregate_line_estimate,
            )

            num_regions = int(self.num_regions or 0)
            line, confidence = aggregate_line_estimate(records, num_regions)
            # What the line already in use still holds, so a switch margin can compare like with
            # like. Absent when no line is in use, in which case there is nothing to hold.
            probs = aggregate_line_distribution(records, num_regions)
            held = None
            if probs is not None and 0 <= self.tracked.line < len(probs):
                held = float(probs[self.tracked.line])
            self.tracked.observe(line, confidence, held_mass=held, num_regions=num_regions)

    def _in_line_position(self, dist: Tensor, owners: Tensor, num_lines: int) -> list[float]:
        """Attention-weighted position along each line's reading direction, one per head.

        A vertical column is read top-to-bottom, so the position axis is ``y``; a horizontal line
        uses ``x``.  A line that holds no token, or a head that put no weight on it, comes back as
        ``-1.0`` rather than a fabricated coordinate.

        Every token belongs to exactly one line, so its coordinate along *its own* line's axis is a
        single per-token number, and both the numerator and the denominator are then one grouped sum
        per head and line.  The per-head version this replaces did all of that once per head per
        step, which at four layers and sixteen heads over a few hundred steps is the probe's own
        cost sitting on top of the generation it is watching -- and the only part of that cost that
        is mine rather than the model's.
        """

        heads = int(dist.shape[0])
        out = [-1.0] * heads
        grid = getattr(self.bridge, "last_patch_positions", None)
        # A line whose direction was never recorded gets no position: choosing an axis for it would
        # be inventing the reading direction the number is measured along.
        known = [line < len(self._line_direction) for line in range(num_lines)]
        if grid is None or num_lines <= 0 or not any(known):
            return out
        inside = owners >= 0
        if not bool(inside.any()):
            return out
        token_line = owners[inside]
        weights = dist[:, inside].float()
        # Which axis each token's own line is read along, so one gather serves every line.
        vertical = torch.tensor(
            [
                known[line] and self._line_direction[line] == "vertical_rtl"
                for line in range(num_lines)
            ],
            device=dist.device,
        )
        points = grid[0].to(dist.device).float()[inside]
        coords = torch.where(vertical[token_line], points[:, 1], points[:, 0])

        numerator = torch.zeros(heads, num_lines, dtype=weights.dtype, device=dist.device)
        denominator = torch.zeros_like(numerator)
        numerator.index_add_(1, token_line, weights * coords)
        denominator.index_add_(1, token_line, weights)

        positions = (numerator / denominator.clamp_min(1e-12)).tolist()
        present = (denominator > 0).tolist()
        for head in range(heads):
            for line in range(num_lines):
                if known[line] and present[head][line]:
                    out[head] = positions[head][line]
                    break
        return out

    # -- the transform (validate on A100) -----------------------------------

    def _read_shape(self, module: nn.Module, layer: int) -> None:
        """Resolve the attention geometry from the module, once per layer.

        Read off the module rather than recomputed: ``scaling`` in particular is the
        model's own attribute, so a config that changes it changes the probe with it.
        """

        num_heads = int(getattr(module, "num_heads", 0) or 0)
        kv_heads = int(getattr(module, "num_key_value_heads", 0) or num_heads)
        head_dim = int(getattr(module, "head_dim", 0) or 0)
        q_proj = getattr(module, "q_proj", None)
        if not head_dim and q_proj is not None and num_heads:
            head_dim = int(q_proj.out_features) // num_heads
        if num_heads <= 0 or head_dim <= 0:
            raise RuntimeError(
                f"attention probe could not resolve the head geometry of layer {layer}"
            )
        scale = float(getattr(module, "scaling", 0.0) or head_dim**-0.5)
        self._geometry[layer] = (num_heads, kv_heads, head_dim, scale)
        if self.heads is not None and max(self.heads) >= num_heads:
            raise RuntimeError(
                f"attention probe was asked for head {max(self.heads)} but layer {layer} has "
                f"{num_heads} query heads"
            )

    def _project(
        self, module: nn.Module, layer: int, hidden_states: Tensor, position_embeddings: Any
    ) -> tuple[Tensor, Tensor] | None:
        """Reproduce one forward's rotated query and key states.

        Returns ``[1, heads, seq, head_dim]`` for both, or ``None`` if the transform
        could not be established, in which case the reason is kept for the report.  A
        failed layer is dropped, not approximated: numbers from a transform the probe
        does not understand are worse than no numbers.
        """

        if layer in self._transform_failed:
            return None
        num_heads, kv_heads, head_dim, _ = self._geometry[layer]
        try:
            query = module.q_proj(hidden_states).view(1, -1, num_heads, head_dim).transpose(1, 2)
            key = module.k_proj(hidden_states).view(1, -1, kv_heads, head_dim).transpose(1, 2)
            # The text attention carries no per-head norm (that is the vision tower),
            # but honour one if a future config adds it.
            q_norm = getattr(module, "q_norm", None)
            k_norm = getattr(module, "k_norm", None)
            if q_norm is not None:
                query = q_norm(query)
            if k_norm is not None:
                key = k_norm(key)
            cos, sin = self._position_embeddings(position_embeddings, hidden_states)
            query, key = _apply_rotary_pos_emb(query, key, cos, sin)
            return query, key
        except Exception as error:  # noqa: BLE001 - one bad layer must not kill the run
            self._transform_failed[layer] = f"{type(error).__name__}: {error}"
            return None

    @staticmethod
    def _position_embeddings(
        position_embeddings: Any, hidden_states: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Pull ``(cos, sin)`` out of the attention forward's arguments.

        This is the trap the module docstring warns about: the text attention has no
        ``rotary_emb`` of its own and never sees ``position_ids``, so the rotary has to
        arrive as this argument.  If it is absent the logits would be computed from
        un-rotated states, so the absence raises instead of being absorbed.
        """

        if (
            isinstance(position_embeddings, (tuple, list))
            and len(position_embeddings) == 2
            and all(isinstance(part, Tensor) for part in position_embeddings)
        ):
            cos, sin = position_embeddings
            return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)
        raise RuntimeError(
            "the attention forward carried no position_embeddings=(cos, sin); the probe "
            "cannot reproduce the rotary transform and will not report un-rotated logits"
        )

    # -- hooking ------------------------------------------------------------

    @staticmethod
    def _layer_of(module: nn.Module) -> int:
        return int(getattr(module, "_probe_layer_idx", -1))

    def _pre(self, module: nn.Module, args: Any, kwargs: dict) -> None:
        # Stash this forward's inputs on the module so the post-hook can reduce them
        # once the projections have run.  The mask is read here rather than in the
        # post-hook because this dict carries whatever the routing pre-hook on the
        # enclosing decoder layer already put there -- so a run with both installed
        # reports the logits generation actually used.
        hidden = kwargs.get("hidden_states")
        if hidden is None and args:
            hidden = args[0]
        if hidden is None or not isinstance(hidden, Tensor):
            return
        setattr(module, "_probe_hidden", hidden)
        position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is None:
            position_embeddings = next(
                (arg for arg in args if isinstance(arg, (tuple, list)) and len(arg) == 2), None
            )
        setattr(module, "_probe_position_embeddings", position_embeddings)
        setattr(module, "_probe_mask", kwargs.get("attention_mask"))

    def _step_index(self, layer: int, cache_position: Any) -> int:
        """The step index for this forward, advancing the counter once per step.

        Every selected layer runs this hook on the same forward, so a naive count would
        report the decoding steps multiplied by the depth of the model -- and that number
        is the arm's coverage.  ``cache_position`` identifies the forward; when it is
        unavailable the first selected layer stands in for it.

        This returns the step for *every* layer, including the ones that do not advance
        the counter.  An earlier version returned ``None`` for those, which counted
        correctly and skipped their observation entirely: four probed layers produced one
        layer's worth of rows, and the per-layer statistics the plan asks for were three
        quarters missing while looking complete.
        """

        current = None
        if isinstance(cache_position, Tensor) and cache_position.numel():
            current = int(cache_position.reshape(-1)[-1].item())
        if current is not None:
            if current != self._step_key:
                self._step_key = current
                self.steps += 1
            return self.steps
        # Without a cache_position the layers are indistinguishable, so the counter rides
        # on the first selected layer, which runs before the others in layer order.
        if layer == self.layers[0]:
            self.steps += 1
        return self.steps

    def _post(self, module: nn.Module, args: Any, kwargs: dict, output: Any) -> None:
        # Read the stashed inputs and clear them in the same breath: a forward whose
        # ``_pre`` did not run must not be reduced against the previous forward's
        # states, which would silently report one step's attention under another's.
        hidden = getattr(module, "_probe_hidden", None)
        position_embeddings = getattr(module, "_probe_position_embeddings", None)
        mask = getattr(module, "_probe_mask", None)
        cache_position = kwargs.get("cache_position")
        for name in ("_probe_hidden", "_probe_position_embeddings", "_probe_mask"):
            if hasattr(module, name):
                delattr(module, name)
        if hidden is None:
            return
        layer = self._layer_of(module)
        with torch.no_grad():
            if layer not in self._geometry:
                self._read_shape(module, layer)
            projected = self._project(module, layer, hidden, position_embeddings)
            if projected is None:
                # A layer whose transform failed records nothing at all, so the gap is
                # visible as a missing layer rather than as a plausible reading.
                return
            query, key = projected
            if hidden.shape[1] > 1:
                # Prefill: the one forward carrying the whole prompt.  Its visual keys
                # are stored and never recomputed, which is how the probe avoids
                # reconstructing the KV cache.
                self._store_prompt(layer, key)
                return
            step = self._step_index(layer, cache_position)
            self._gen_keys.setdefault(layer, []).append(key.detach())
            self._observe_step(layer, step, query, mask)

    def attach_emitted(self, ids: Tensor, decoder: Any) -> int:
        """Stamp each observed step with the text generation emitted from it.

        The probe cannot read the sampled token: sampling happens after the forward it
        hooks.  The eval loop can, so it hands the generated span over here, and this
        keeps the whole sampling concern out of the observation path.

        Step ``s`` saw the query at position ``prompt_length + s - 1``, which predicts
        the ``s``-th generated token -- token 0 is sampled from the prefill and is
        never observed at all.  That offset is the one the offline alignment has to
        respect, so the mapping is applied here rather than guessed downstream.

        The text is built by cumulative decode, because one token can carry several
        characters and one character can span several tokens.  Returns the number of
        steps left without text, so a misalignment is reported instead of absorbed.
        """

        tokens = [int(token) for token in ids.reshape(-1).tolist()]
        emitted = _incremental_text(decoder, tokens)
        # The decomposition reproduces the full decode by construction -- the retraction
        # above removes exactly the diverged tail before the extension is appended --
        # so this cannot fire for an honest tokenizer.  What it does catch is a decoder
        # that returns different text for the same ids across calls, which would make
        # every character-to-step mapping downstream a fiction.  Cheap, and the failure
        # it guards against is silent otherwise.
        self.emitted_join_mismatch = int(
            "".join(emitted) != decoder.decode(tokens, skip_special_tokens=True)
        )
        self.emitted_missing = 0
        for record in self._records:
            step = record["step"]
            if 0 < step < len(emitted):
                record["emitted"] = emitted[step]
            else:
                self.emitted_missing += 1
        return self.emitted_missing

    def write_probe(self) -> None:
        """Append this page's report, so a run's artifacts show what was observed."""

        path = probe_path()
        if path:
            _probe(path, self.report())

    def report(self) -> dict[str, Any]:
        return {
            "page_id": self.page_id,
            "layers": list(self.layers),
            "heads": None if self.heads is None else list(self.heads),
            "visual_tokens": self.visual_count,
            "visual_start": self.visual_start,
            "num_regions": self.num_regions,
            "box_map": self.box_map,
            "decoding_steps": self.steps,
            "grid_missing_steps": self.grid_missing,
            "emitted_missing_steps": self.emitted_missing,
            "emitted_join_mismatch": self.emitted_join_mismatch,
            "bias_corrected_steps": self.bias_corrected_steps,
            "routing_bias": self.routing_bias if self.correct_confidence else None,
            "layer_geometry": {
                str(layer): {
                    "num_heads": geometry[0],
                    "num_kv_heads": geometry[1],
                    "head_dim": geometry[2],
                    "scaling": geometry[3],
                }
                for layer, geometry in sorted(self._geometry.items())
            },
            "transform_failed": {
                str(layer): reason for layer, reason in self._transform_failed.items()
            },
            "steps": self._records,
        }


def _rotary_helper() -> Any:
    """The model's own rotary helper, imported where it lives.

    Kept as its own function so a test can substitute one without a transformers
    install, and so the import failure is a single visible fallback rather than a
    silent one inside the transform.
    """

    try:
        from transformers.models.glm_ocr.modeling_glm_ocr import apply_rotary_pos_emb
    except ImportError:  # pragma: no cover - depends on the installed transformers
        from transformers.modeling_rope_utils import apply_rotary_pos_emb  # type: ignore

    return apply_rotary_pos_emb


def _apply_rotary_pos_emb(
    query: Tensor, key: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    """Rotate query and key with the model's own helper.

    ``apply_rotary_pos_emb`` uses ``rotate_half_llm`` (even/odd interleave) rather
    than the classic half split, which is exactly why this calls it instead of
    reimplementing the rotation: the interleave convention is not visible from the
    call site, and getting it wrong produces a wrong-but-plausible distribution.
    """

    rotated = _rotary_helper()(query, key, cos, sin)
    if not isinstance(rotated, (tuple, list)) or len(rotated) != 2:
        raise RuntimeError(
            "apply_rotary_pos_emb returned a single tensor, so the key states would be left "
            "un-rotated; the probe needs the two-tensor (query, key) signature"
        )
    return rotated[0], rotated[1]


def _probe(path: str, record: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _find_attention_modules(model: nn.Module) -> list[tuple[int, nn.Module]]:
    """Locate the text decoder's attention modules by their projections.

    Found by structure (a module carrying ``q_proj`` and ``k_proj``) rather than by
    class name, so a transformers refactor that renames the attention class fails
    with a clear message instead of observing nothing.
    """

    text_model = _find_text_model(model)
    layers = list(getattr(text_model, "layers", []))
    if not layers:
        raise RuntimeError("could not find the GLM-OCR text decoder layers to probe")
    found: list[tuple[int, nn.Module]] = []
    for index, layer in enumerate(layers):
        for _, module in layer.named_modules():
            if (
                getattr(module, "q_proj", None) is not None
                and getattr(module, "k_proj", None) is not None
            ):
                # A probe-private name, not ``layer_idx``: transformers attention
                # modules already carry ``layer_idx`` for the KV cache, and
                # overwriting it would corrupt the cache update.
                module._probe_layer_idx = index
                found.append((index, module))
                break
    if not found:
        raise RuntimeError("could not locate any attention modules to probe")
    return found


def install_attention_probe(
    model: nn.Module,
    bridge: Any,
    *,
    layers: tuple[int, ...] = DEFAULT_LAYERS,
    heads: tuple[int, ...] | None = None,
    tracked: Any | None = None,
    box_map: str = "regions",
    routing_bias: float = 0.0,
    correct_confidence: bool = False,
    next_line_scale: float = 0.0,
) -> tuple[AttentionProbe, list[Any]]:
    """Register observation hooks on the selected decoder attention modules.

    Returns the runtime and the removable hook handles.  The hooks read the forward's
    inputs and re-run the projections in ``no_grad``; they never write into
    ``kwargs``, so generation is unchanged.
    """

    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(getattr(model.config, "text_config", None), "image_token_id", None)
    runtime = AttentionProbe(
        bridge,
        image_token_id,
        layers=layers,
        heads=heads,
        tracked=tracked,
        box_map=box_map,
        routing_bias=routing_bias,
        correct_confidence=correct_confidence,
        next_line_scale=next_line_scale,
    )
    candidates = _find_attention_modules(model)
    available = {index for index, _ in candidates}
    missing = [index for index in layers if index not in available]
    if missing:
        raise RuntimeError(
            f"attention probe requested layers {missing} that the model does not have "
            f"(available: {sorted(available)})"
        )
    selected = set(layers)
    handles: list[Any] = []
    for index, module in candidates:
        if index not in selected:
            continue
        handles.append(module.register_forward_pre_hook(runtime._pre, with_kwargs=True))
        handles.append(module.register_forward_hook(runtime._post, with_kwargs=True))
    return runtime, handles
