"""Ground-truth masks for the decoder mask head (plan section 4).

This module turns a page's character boxes into a per-token soft mask over the
post-merge visual grid.  It never produces a mask the model could not in
principle learn: the only input that matters at train time is the character box
annotation, aligned to the *actual* target token ids of the chat template.

Two alignments happen here, and both are explicit and reported:

* token -> character: each assistant target token is mapped to the span of
  ``page_text`` characters it reproduces, by monotone alignment of the decoded
  token sequence against ``page_text``.  A token covering several characters
  takes the union of their boxes; several tokens covering one character share
  that box.  This is the fallback path required by plan 4.2 when the tokenizer
  has no reliable offset mapping.
* character box -> grid: a box is rasterised onto the ``N`` merged visual cells
  by overlap-area share, normalised per box (plan 4.3), so a box smaller than a
  cell still gets a non-zero, well-shaped target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor

MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_SCORE = -1


@dataclass(frozen=True)
class MaskTargets:
    """Per-target-token supervision for one page (batch size 1)."""

    mask: Tensor  # [1, T, N] rasterised box target in [0, 1]
    spatial_valid: Tensor  # [1, T] bool: does this token carry a mask target
    stop_target: Tensor  # [1, T] float: 1 for EOS, 0 otherwise
    char_spans: list[tuple[int, int] | None]  # per token, span into page_text
    alignment_status: list[str]  # per token: exact/placeholder/missing/blank/unmapped
    alignment_report: dict[str, Any]  # token-char coverage counters for Gate A


def char_boxes(record: dict[str, Any]) -> tuple[list[list[float] | None], list[str]]:
    """Return per-character normalised boxes and an alignment status.

    ``characters`` is index-aligned with ``page_text`` (produced by
    ``prepare_mthv2_char_manifest.py``).  A missing box is ``None`` and is *not*
    treated as an empty-mask ground truth.
    """

    characters = record.get("characters") or []
    boxes: list[list[float] | None] = []
    statuses: list[str] = []
    for entry in characters:
        if not isinstance(entry, dict):
            boxes.append(None)
            statuses.append("missing")
            continue
        box = entry.get("bbox")
        if box is None:
            boxes.append(None)
            statuses.append("missing")
            continue
        boxes.append([float(value) for value in box])
        statuses.append(entry.get("alignment_status", "exact"))
    return boxes, statuses


def _align(target: str, source: str) -> list[int | None]:
    """Monotone alignment of every ``target`` char to a ``source`` char (or None).

    Same scoring as ``prepare_mthv2_char_manifest.align``: mismatch is cheaper
    than a gap, so a placeholder or variant glyph is paired rather than skipped.
    """

    n, m = len(target), len(source)
    move = [bytearray(m + 1) for _ in range(n + 1)]
    previous = [j * GAP_SCORE for j in range(m + 1)]
    for j in range(1, m + 1):
        move[0][j] = 2
    for i in range(1, n + 1):
        current = [0] * (m + 1)
        current[0] = i * GAP_SCORE
        move[i][0] = 1
        for j in range(1, m + 1):
            best = previous[j - 1] + (MATCH_SCORE if target[i - 1] == source[j - 1] else MISMATCH_SCORE)
            step = 0
            if previous[j] + GAP_SCORE > best:
                best, step = previous[j] + GAP_SCORE, 1
            if current[j - 1] + GAP_SCORE > best:
                best, step = current[j - 1] + GAP_SCORE, 2
            current[j] = best
            move[i][j] = step
        previous = current
    pairs: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        step = move[i][j]
        if step == 0:
            pairs[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return pairs


def token_char_spans(
    tokenizer: Any,
    page_text: str,
    target_ids: Sequence[int],
    char_statuses: Sequence[str],
) -> tuple[list[tuple[int, int] | None], list[str], dict[str, Any]]:
    """Map each assistant target token to a ``page_text`` char span.

    Returns ``(spans, statuses, report)``.  ``spans[i]`` is the half-open char
    span covered by token ``i`` (``None`` when unmapped), and ``statuses[i]``
    classifies the token for the mask builder.
    """

    n_tokens = len(target_ids)
    decoded: list[str] = []
    for token_id in target_ids:
        text = tokenizer.decode([int(token_id)], skip_special_tokens=True)
        decoded.append(text or "")
    concatenated = "".join(decoded)
    # Map every position of the concatenated decode back to the token that
    # produced it, then align the concatenated string to page_text.
    token_of_pos: list[int] = []
    for token_index, text in enumerate(decoded):
        token_of_pos.extend([token_index] * len(text))
    pairs = _align(concatenated, page_text)

    # Group aligned positions by token: the span of page_text characters a token
    # covers is the contiguous run it owns.
    per_token_positions: list[list[int]] = [[] for _ in range(n_tokens)]
    for concat_pos, page_pos in enumerate(pairs):
        if page_pos is not None and concat_pos < len(token_of_pos):
            per_token_positions[token_of_pos[concat_pos]].append(page_pos)

    spans: list[tuple[int, int] | None] = []
    statuses: list[str] = []
    for token_index in range(n_tokens):
        positions = sorted(set(per_token_positions[token_index]))
        if not positions:
            spans.append(None)
            statuses.append("blank" if decoded[token_index].strip() == "" else "unmapped")
            continue
        start, end = positions[0], positions[-1] + 1
        spans.append((start, end))
        if any(char_statuses[i] != "exact" for i in range(start, end) if i < len(char_statuses)):
            statuses.append("placeholder")
        else:
            statuses.append("exact")

    covered = sum(1 for status in statuses if status in ("exact", "placeholder"))
    report = {
        "target_tokens": n_tokens,
        "decoded_characters": len(concatenated),
        "page_characters": len(page_text),
        "mapped_tokens": covered,
        "blank_tokens": sum(1 for status in statuses if status == "blank"),
        "unmapped_tokens": sum(1 for status in statuses if status == "unmapped"),
    }
    return spans, statuses, report


def _cell_bounds(xywh: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Reconstruct cell corners from ``[x, y, w, h]`` centres."""

    x0 = xywh[..., 0] - xywh[..., 2] / 2.0
    x1 = xywh[..., 0] + xywh[..., 2] / 2.0
    y0 = xywh[..., 1] - xywh[..., 3] / 2.0
    y1 = xywh[..., 1] + xywh[..., 3] / 2.0
    return x0, x1, y0, y1


def rasterize_box(box: Sequence[float], xywh: Tensor) -> Tensor:
    """Overlap-area share of each cell against ``box``, normalised by its max.

    ``box`` is ``[x0, y0, x1, y1]`` in normalised page coordinates.  Returns a
    ``[N]`` tensor in ``[0, 1]`` where the best-covered cell is 1.
    """

    x0, x1, y0, y1 = _cell_bounds(xywh)
    ix0 = torch.maximum(x0, torch.tensor(float(box[0]), device=xywh.device))
    ix1 = torch.minimum(x1, torch.tensor(float(box[2]), device=xywh.device))
    iy0 = torch.maximum(y0, torch.tensor(float(box[1]), device=xywh.device))
    iy1 = torch.minimum(y1, torch.tensor(float(box[3]), device=xywh.device))
    inter = (ix1 - ix0).clamp_min(0.0) * (iy1 - iy0).clamp_min(0.0)
    cell_area = (x1 - x0) * (y1 - y0)
    share = inter / cell_area.clamp_min(1e-12)
    peak = share.max()
    if peak.item() <= 0.0:
        return torch.zeros_like(share)
    return share / peak


def build_mask_targets(
    tokenizer: Any,
    record: dict[str, Any],
    target_ids: Sequence[int],
    eos_ids: Iterable[int],
    xywh: Tensor,
) -> MaskTargets:
    """Assemble the per-token mask targets for one page.

    ``target_ids`` is the assistant target token id list (including the trailing
    EOS) taken from the actual ``input_ids``, and ``xywh`` is the grid returned
    by ``DecoderMaskRouter.project_visual`` (shape ``[1, N, 4]``).
    """

    page_text = record["page_text"]
    boxes, char_statuses = char_boxes(record)
    spans, statuses, report = token_char_spans(tokenizer, page_text, target_ids, char_statuses)

    eos = set(int(t) for t in eos_ids)
    n_tokens = len(target_ids)
    n_cells = xywh.shape[1]
    mask = torch.zeros(1, n_tokens, n_cells, dtype=torch.float32, device=xywh.device)
    spatial_valid = torch.zeros(1, n_tokens, dtype=torch.bool, device=xywh.device)
    stop_target = torch.zeros(1, n_tokens, dtype=torch.float32, device=xywh.device)

    for index, (span, status, token_id) in enumerate(zip(spans, statuses, target_ids)):
        if int(token_id) in eos:
            stop_target[0, index] = 1.0
            spatial_valid[0, index] = True  # empty mask, stop=1 (plan 3.2)
            continue
        if status == "blank":
            spatial_valid[0, index] = True  # empty mask, stop=0 (plan 3.2)
            continue
        if span is None or status == "unmapped":
            spatial_valid[0, index] = False
            continue
        # Union of the boxes covered by this token.
        covered = [boxes[i] for i in range(span[0], span[1]) if i < len(boxes)]
        boxes_present = [box for box in covered if box is not None]
        if not boxes_present:
            spatial_valid[0, index] = False  # missing boxes: ignore, never train as background
            continue
        for box in boxes_present:
            mask[0, index] = torch.maximum(mask[0, index], rasterize_box(box, xywh[0]))
        spatial_valid[0, index] = True

    return MaskTargets(
        mask=mask,
        spatial_valid=spatial_valid,
        stop_target=stop_target,
        char_spans=spans,
        alignment_status=statuses,
        alignment_report=report,
    )
