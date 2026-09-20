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
all-zero, so the arm carries no bias and proves the wiring.

A zero mask is not *obviously* the same run as no route at all: a non-``None`` mask
can take SDPA off the flash path and make ``use_gqa_in_sdpa`` return false, so
``repeat_kv`` would materialise the heads instead of using ``enable_gqa``.  That
worry was measured rather than argued, with an otherwise identical run that installs
nothing: the two came out bit-identical (identical edit counts on every page, paired
bootstrap CI ``[+0.000000, +0.000000]``), so on this model the kernel difference has
no effect and a gain over the zero arm is a gain over no route.  The arm that
established that is kept in the results, because the next model or shape may not be
so forgiving.

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

# Which box the bias is aimed at.
#
# ``char``  the character being generated, from the character box channel.  This is the arm
#           the recorded result used, and it is an oracle in the strong sense: it needs a box
#           per character, which only the annotation supplies.
# ``line``  the line (region) that character sits on.  Coarser, and the coarseness is the
#           whole question.  The recorded gain was concentrated in deletions falling by half,
#           which is what "do not skip" looks like, and a line is exactly the scale at which a
#           deployable page map could plausibly exist.  If a line box carries none of that
#           gain, then the character-level precision -- which nothing deployable supplies --
#           was where the gain lived, and the route has no deployable form.
# ``pred_static``  the union of the page's *predicted* line boxes, the same mask on every step.
#               No pointer and no annotation: this is the first arm whose boxes could exist in a
#               real run at all.  It says "look at text rather than background" without saying
#               *which* text, which is exactly the weaker claim the plan wants measured before
#               anything is built on top of it.
#
#               The dose has to be matched to the dynamic arms, and that is not a small
#               correction: the union of ~24 predicted columns covers around 74% of the visual
#               tokens where the true line box holds about 3%, so the same bias value would add
#               roughly 24x the logit mass.  The plan asks for the static and dynamic arms to be
#               matched in total bias weight for exactly this reason -- without it, a static arm
#               that looks better may only be looking at more of the page.
BOX_SOURCES = ("char", "line", "pred_static")

# What decides *which* box gets biased, for the sources where that is a separate question.
#
# ``pointer``  the recorded design: walk the reference text alongside the generated one and take the
#              line of the character the model has reached.  Needs the reference, so it is an
#              oracle, and it is what the 26.6% arm uses.
# ``tracked``  the model's own attention estimate of the current line, taken from
#              ``attention_tracking`` one step behind.  No reference text anywhere in the path:
#              this is the arm that says whether the deployable form can exist at all.
LINE_SOURCES = ("pointer", "tracked")

# Which box list a line index refers to.  ``regions`` is the annotation, which every result so
# far is expressed in; ``predicted`` is the detector's, and it is what makes the tracked arm
# deployable -- with this set, nothing in the tracker's path reads the annotation.
LINE_MAPS = ("regions", "predicted")


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
        box_source: str = "char",
        line_source: str = "pointer",
        tracked: Any | None = None,
        line_map: str = "regions",
        next_line_scale: float = 0.0,
    ) -> None:
        if bias < 0:
            raise ValueError("AttentionRouting needs a non-negative bias")
        if next_line_scale < 0:
            raise ValueError("the next line's share of the bias cannot be negative")
        if pointer not in POINTER_MODES:
            raise ValueError(f"pointer must be one of {POINTER_MODES}, got {pointer!r}")
        if box_source not in BOX_SOURCES:
            raise ValueError(f"box_source must be one of {BOX_SOURCES}, got {box_source!r}")
        if line_source not in LINE_SOURCES:
            raise ValueError(f"line_source must be one of {LINE_SOURCES}, got {line_source!r}")
        if line_map not in LINE_MAPS:
            raise ValueError(f"line_map must be one of {LINE_MAPS}, got {line_map!r}")
        if line_source == "tracked" and (box_source != "line" or tracked is None):
            # The tracked source replaces the pointer for choosing *which* line, so it only means
            # anything where a line is what gets biased, and it needs the state to read from.
            raise ValueError(
                "line_source='tracked' needs box_source='line' and a tracked state to read"
            )
        # The tokenizer is only needed when a pointer will actually decide something. The static
        # source biases every line and the tracked source reads its line from the attention, so in
        # both the pointer is inert and demanding a tokenizer for it would be a requirement on
        # nothing -- and would push a caller into passing the reference text to an arm whose point
        # is that it has none.
        pointer_in_use = box_source != "pred_static" and line_source != "tracked"
        if pointer == "synced" and pointer_in_use and tokenizer is None:
            raise ValueError("the synced pointer needs a tokenizer to read the generated ids")
        self.bridge = bridge
        self.bias = float(bias)
        self.image_token_id = image_token_id
        self.tokenizer = tokenizer
        self.pointer = pointer
        self.box_source = box_source
        self.line_source = line_source
        self.line_map = line_map
        # How much of the bias the *next* line in reading order carries.  The applied line is one
        # decoding step stale by construction, and the offline replay puts the cost at 8.5% of
        # characters -- and 88-95% of those are among the first three of their own line, which is
        # exactly the step where the tracker had not moved yet.  Covering the next line as well
        # aims at the right column on those characters; on the rest it adds a wrong column at this
        # share.  The plan asks for it; the dilution it costs is not measurable offline.
        self.next_line_scale = float(next_line_scale)
        self.tracked = tracked
        self.gated = 0
        self.past_annotation = 0
        self.next_line_hits = 0.0
        # The line each character sits on, resolved once per page from the character boxes and
        # the region boxes.  Per character index, not per step, so a step only looks it up.
        self.regions: list[dict[str, Any]] = []
        self._char_lines: list[int] = []
        # Predicted line boxes, normalized, for the ``pred_static`` source.  The union's inside
        # flags are computed once per page and reused every step: the mask does not depend on the
        # step at all, which is what makes this arm deployable.
        self.predicted_lines: list[list[float]] = []
        self._static_inside: Tensor | None = None
        self.characters: list[dict[str, Any]] | None = None
        self.prompt_length: int | None = None
        self.visual_start: int | None = None
        self.visual_count: int | None = None
        self.page_id: str | None = None
        # Synced-pointer state: the truth text, the truth index reached, and the
        # highest sequence position already folded into it.
        self.reference: str | None = None
        self.position = 0
        self._last_observed = -1
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
        regions: list[dict[str, Any]] | None = None,
        predicted_lines: list[list[float]] | None = None,
    ) -> None:
        """Point the route at one page's character boxes.

        The visual span is read off the prompt's own token ids rather than assumed
        to sit at a fixed offset, so a prefix span or a template change moves it
        with the sequence instead of silently biasing the wrong keys.

        ``regions`` is only needed by the ``line`` box source.  It is sorted by reading order
        here, matching ``layout_targets`` and the probe, so a line index means the same thing
        in all three.
        """

        self.gated = 0
        self.past_annotation = 0
        self.next_line_hits = 0.0
        if self.tracked is not None:
            # A new page invalidates the estimate: the previous page's line says nothing about
            # this one, and leaving it would aim the first steps at a line that may not exist.
            self.tracked.set_page()
        self.predicted_lines = [list(box) for box in (predicted_lines or [])]
        self._static_inside = None
        self.regions = sorted(regions or [], key=lambda item: int(item["reading_order"]))
        # Only the line source needs the character-to-line map; computing it for the character
        # source would put a number in the report that the arm never used.
        self._char_lines = (
            self._resolve_char_lines(characters) if self.box_source == "line" else []
        )
        self.characters = characters
        self.prompt_length = int(prompt_length)
        self.page_id = page_id
        self.reference = reference
        self.position = 0
        self._last_observed = -1
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

    def _resolve_char_lines(self, characters: list[dict[str, Any]] | None) -> list[int]:
        """The region each character sits on, or ``-1`` when no box holds it.

        Resolved geometrically -- the first region in reading order whose box contains the
        character's centre -- rather than by trusting the manifest's ``line_index`` to be in the
        same order as ``regions``.  That rule is identical to ``layout_targets`` and to the
        probe's token map, so the three agree on what "line 4" means.
        """

        if not characters or not self.regions:
            return [-1] * len(characters or [])
        boxes = [(region["bbox"], index) for index, region in enumerate(self.regions)]
        resolved: list[int] = []
        for entry in characters:
            box = (entry or {}).get("bbox")
            if not box:
                # The manifest could not place this character, so there is no line to bias.
                # Fabricating one would put an invented location under the arm whose whole
                # point is spatial truth.
                resolved.append(-1)
                continue
            x = (float(box[0]) + float(box[2])) / 2.0
            y = (float(box[1]) + float(box[3])) / 2.0
            found = -1
            for region_box, index in boxes:
                if (
                    float(region_box[0]) <= x <= float(region_box[2])
                    and float(region_box[1]) <= y <= float(region_box[3])
                ):
                    found = index
                    break
            resolved.append(found)
        return resolved

    def clear_page(self) -> None:
        self.characters = None
        self.regions = []
        self._char_lines = []
        self.predicted_lines = []
        self._static_inside = None
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
        """Read this step's new tokens off the top-level model call.

        Hooked on the model ``generate`` calls rather than on the decoder layers
        because this is the only place the *token ids* are visible -- the text model
        is reached with assembled embeddings.  It runs before any layer, so the
        layers of this same forward see the pointer this sets.

        The ids are aligned by ``cache_position`` and not by length.  ``generate``
        slices the model's ``input_ids`` to the positions it has not yet seen -- the
        prefill carries the whole prompt, a decode step carries exactly the one new
        token -- so a length-based rule reads a negative "generated so far" on every
        decode step and silently leaves the pointer at zero.  That is exactly what
        the first run of this arm did: every step biased the first character's box.
        """

        if (
            self.pointer != "synced"
            or self.characters is None
            or self.prompt_length is None
            or self.box_source == "pred_static"
        ):
            # The static arm has no pointer to advance: it biases every predicted line, so
            # tracking which character is being read would be work whose result is discarded.
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
            return None
        cache_position = kwargs.get("cache_position")
        if not isinstance(cache_position, Tensor):
            return None
        positions = cache_position.flatten()
        if positions.numel() != input_ids.shape[1]:
            raise RuntimeError(
                "attention routing cannot align input_ids with cache_position: "
                f"{input_ids.shape[1]} ids for {positions.numel()} positions"
            )
        generated = positions >= self.prompt_length
        if not bool(generated.any()):
            return None  # the prefill: nothing has been generated yet
        first = int(positions[generated][0].item())
        if first <= self._last_observed:
            return None  # this step has already been folded in
        self._last_observed = int(positions[generated][-1].item())
        self._advance(self.tokenizer.decode(input_ids[0][generated], skip_special_tokens=True))
        return None

    def _static_mask(self, kv_length: int, device: Any, dtype: torch.dtype) -> Tensor | None:
        """One mask for the whole page: every predicted line biased, no pointer involved.

        The union of the boxes is computed once and kept, because it does not change from step to
        step -- which is the whole point of this arm.  Nothing here reads the annotation.
        """

        if not self.predicted_lines or self.visual_count in (None, 0) or self.visual_start is None:
            self.missing += 1
            return None
        positions = getattr(self.bridge, "last_patch_positions", None)
        if positions is None or positions.shape[1] != self.visual_count:
            raise RuntimeError(
                "attention routing has no patch grid for this page: the visual tower must run "
                "before the first biased decoding step"
            )
        if self._static_inside is None:
            grid = positions[0].to(device=device, dtype=torch.float32)
            inside = torch.zeros(grid.shape[0], dtype=torch.bool, device=device)
            for box in self.predicted_lines:
                inside |= (
                    (grid[:, 0] >= float(box[0]))
                    & (grid[:, 0] <= float(box[2]))
                    & (grid[:, 1] >= float(box[1]))
                    & (grid[:, 1] <= float(box[3]))
                )
            self._static_inside = inside
        count = int(self._static_inside.sum().item())
        self.boxes_hit += count
        self.biased += 1
        mask = torch.zeros(1, 1, 1, kv_length, device=device, dtype=dtype)
        mask[0, 0, 0, self.visual_start : self.visual_start + self.visual_count] = (
            self._static_inside.to(dtype) * self.bias
        )
        return mask

    def _mask_for(self, step: int, kv_length: int, device: Any, dtype: torch.dtype) -> Tensor | None:
        if self.box_source == "pred_static":
            # No pointer and no annotation: the mask is the page's, not the step's.
            return self._static_mask(kv_length, device, dtype)
        if self.visual_count in (None, 0) or self.visual_start is None:
            return None
        if self.box_source == "line" and self.line_source == "tracked":
            # Reads the attention's line, so it needs neither the character channel nor the
            # reference.  A gated or not-yet-available estimate is counted apart from a missing
            # box: "the gate withheld the bias" and "there was no box to bias" are different
            # failures and the report has to tell them apart.
            line = int(getattr(self.tracked, "line", -1))
            if line < 0:
                self.gated += 1
                return None
            if self.line_map == "predicted":
                boxes = self.predicted_lines
                if line >= len(boxes):
                    self.missing += 1
                    return None
                box = list(boxes[line])
            else:
                if line >= len(self.regions):
                    self.missing += 1
                    return None
                box = self.regions[line]["bbox"]
        else:
            # Everything else is driven by the character being read, so it needs the character
            # channel.  Guarding that here rather than above keeps the tracked source from being
            # rejected for lacking something it never uses.
            if self.characters is None or not 0 <= step < len(self.characters):
                # The pointer has walked past the annotation -- the model generated more
                # characters than the reference has boxes for.  Counted, because an
                # uncounted return here is a step that silently disappears from the
                # wiring accounting.
                self.past_annotation += 1
                return None
            if self.box_source == "line":
                # The line that character sits on.  A missing character box means no line either --
                # see _resolve_char_lines.
                line = self._char_lines[step] if step < len(self._char_lines) else -1
                if line < 0 or line >= len(self.regions):
                    self.missing += 1
                    return None
                box = self.regions[line]["bbox"]
            else:
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
        added = inside.to(dtype) * self.bias
        if self.next_line_scale > 0.0:
            next_box = self._next_line_box(line, box)
            if next_box is not None:
                next_inside = (
                    (grid[:, 0] >= float(next_box[0]))
                    & (grid[:, 0] <= float(next_box[2]))
                    & (grid[:, 1] >= float(next_box[1]))
                    & (grid[:, 1] <= float(next_box[3]))
                )
                # The share lands only where the current line does not already cover it, so a token
                # in both boxes is not biased twice.
                share = next_inside & ~inside
                self.next_line_hits += float(share.sum().item())
                added = added + share.to(dtype) * (self.bias * self.next_line_scale)
        mask = torch.zeros(1, 1, 1, kv_length, device=device, dtype=dtype)
        mask[0, 0, 0, self.visual_start : self.visual_start + self.visual_count] = added
        return mask

    def _next_line_box(self, line: int, current_box: list[float]) -> list[float] | None:
        """The next line's box in reading order, or ``None`` when this is the last one.

        Only the ``line`` source has a next line: the character source aims at one character's
        box, and the static source already covers every line it knows about.
        """

        if self.box_source != "line" or self.next_line_scale <= 0.0:
            return None
        boxes = (
            self.predicted_lines
            if self.line_map == "predicted"
            else [region["bbox"] for region in self.regions]
        )
        index = line + 1
        if index < 0 or index >= len(boxes):
            return None
        return list(boxes[index])

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
        if self.pointer == "synced":
            step = self.position
        else:
            # ``generate`` samples the first token from the prefill's last position,
            # so the decode step at position ``prompt_length + t - 1`` is the one
            # that emits character ``t``.
            step = current - self.prompt_length + 1
        if self._cache_key != current:
            # Every decoder layer runs this hook, so a per-layer count would report
            # the decoding steps multiplied by the depth of the model.
            self._cache_key = current
            self.steps += 1
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
        # Reported whenever the arm actually reads them: the count is part of the dose,
        # since the confidence unit multiplies by it.
        reads_predicted = self.box_source == "pred_static" or self.line_map == "predicted"
        return {
            "page_id": self.page_id,
            "bias": self.bias,
            "pointer": self.pointer,
            "box_source": self.box_source,
            "line_source": self.line_source,
            "line_map": self.line_map,
            "gated_steps": self.gated,
            "past_annotation_steps": self.past_annotation,
            "tracked": (self.tracked.report() if self.line_source == "tracked" else None),
            "characters_on_a_line": (
                sum(1 for line in self._char_lines if line >= 0) if self._char_lines else None
            ),
            "predicted_lines": len(self.predicted_lines) if reads_predicted else None,
            "pointer_position": self.position if self.pointer == "synced" else None,
            "reference_characters": len(self.reference) if self.reference else None,
            "decoding_steps": self.steps,
            "biased_steps": self.biased,
            "missing_box_steps": self.missing,
            "mean_boxes_hit": (self.boxes_hit / self.biased) if self.biased else None,
            "next_line_scale": self.next_line_scale or None,
            "mean_next_line_hits": (
                (self.next_line_hits / self.biased) if self.biased and self.next_line_scale else None
            ),
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
    box_source: str = "char",
    line_source: str = "pointer",
    tracked: Any | None = None,
    line_map: str = "regions",
    next_line_scale: float = 0.0,
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
    runtime = AttentionRouting(
        bridge,
        bias,
        image_token_id,
        tokenizer,
        pointer,
        box_source,
        line_source,
        tracked,
        line_map,
        next_line_scale,
    )
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
