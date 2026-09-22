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

from dataclasses import dataclass, field
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
    # Present only in window mode; carried so line grouping (which poisons every
    # target on a page when it is wrong) is logged rather than inferred.
    window_report: dict[str, Any] = field(default_factory=dict)
    xywh: Tensor | None = None  # [1, N, 4]; optional for older tensor-only callers


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


def char_lines(record: dict[str, Any]) -> tuple[list[int | None], bool]:
    """Per-page_text-position line id, index-aligned with ``page_text``.

    ``prepare_mthv2_char_manifest`` already emits an authoritative ``line_index``
    per character, derived from the page's own ``textlines`` annotation and
    verified by concatenating the lines in reading order back to ``page_text``.
    That is better than any geometric rule: it is exact, it needs no thresholds,
    and it works for vertical as well as horizontal writing (a rule based on y
    overlap would merge two columns on a vertical page).

    Returns ``(line_ids, annotated)`` where ``annotated`` reports whether the
    manifest actually carried the field -- a stale cache built before it existed
    must not silently degrade into a one-line-per-page grouping.
    """

    characters = record.get("characters") or []
    ids: list[int | None] = []
    annotated = False
    for entry in characters:
        if isinstance(entry, dict) and "line_index" in entry:
            value = entry.get("line_index")
            ids.append(None if value is None else int(value))
            annotated = True
        else:
            ids.append(None)
    width = len(record.get("page_text", ""))
    if len(ids) < width:
        ids.extend([None] * (width - len(ids)))
    return ids[:width], annotated


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


def rasterize_box_hard(box: Sequence[float], xywh: Tensor) -> Tensor:
    """Return the binary patch-centre mask used by attention routing.

    A cell is selected exactly when its normalised centre lies inside the box.
    This deliberately does not use overlap area or peak normalisation: one
    selected visual key receives the full decoder bias, matching
    ``attention_routing.AttentionRouting._mask_for``.
    """

    centres = xywh[..., :2]
    inside = (
        (centres[:, 0] >= float(box[0]))
        & (centres[:, 0] <= float(box[2]))
        & (centres[:, 1] >= float(box[1]))
        & (centres[:, 1] <= float(box[3]))
    )
    return inside.to(dtype=xywh.dtype)


def convex_hull(points: Tensor) -> Tensor:
    """Monotone-chain hull of ``[P, 2]`` points, returned counter-clockwise.

    Degenerate inputs (fewer than three distinct points, or all collinear) return
    the input points unchanged, so the caller gets a zero-area polygon rather
    than an exception.
    """

    unique = sorted({(float(x), float(y)) for x, y in points.tolist()})
    if len(unique) < 3:
        return torch.tensor(unique, dtype=points.dtype, device=points.device) if unique else points.new_zeros((0, 2))

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return torch.tensor(hull, dtype=points.dtype, device=points.device)


def rasterize_polygon(points: Tensor, xywh: Tensor, samples: int = 4) -> Tensor:
    """Supersampled cell-coverage share of a convex polygon, peak-normalised to 1.

    ``points`` is ``[P, 2]`` in the same normalised page coordinates as ``xywh``
    (``[N, 4]`` cell centres and sizes).  Each cell is probed on an ``S x S``
    sub-grid and its share is the fraction of probes inside the hull, so a cell
    the polygon fully covers scores 1 and one it misses scores 0.

    Supersampling rather than exact polygon clipping is deliberate: the target is
    soft supervision normalised to a peak of 1, so what matters is where the peak
    lands and how the blob is shaped, not the sub-cell area to five decimals --
    and it avoids the degenerate-polygon failure modes of a clipper.
    """

    if points.numel() == 0:
        return torch.zeros(xywh.shape[0], device=xywh.device, dtype=xywh.dtype)
    hull = convex_hull(points.to(torch.float64))
    if hull.shape[0] < 3:
        return torch.zeros(xywh.shape[0], device=xywh.device, dtype=xywh.dtype)
    offset = (torch.arange(samples, device=xywh.device, dtype=xywh.dtype) + 0.5) / samples - 0.5
    dy, dx = torch.meshgrid(offset, offset, indexing="ij")
    probe_x = xywh[:, 0].unsqueeze(1) + dx.reshape(1, -1) * xywh[:, 2].unsqueeze(1)  # [N, S*S]
    probe_y = xywh[:, 1].unsqueeze(1) + dy.reshape(1, -1) * xywh[:, 3].unsqueeze(1)
    # Cross product of every probe against every hull edge; inside iff all agree.
    edge_a = hull
    edge_b = torch.roll(hull, shifts=-1, dims=0)
    ex = (edge_b[:, 0] - edge_a[:, 0]).to(xywh.dtype)  # [K]
    ey = (edge_b[:, 1] - edge_a[:, 1]).to(xywh.dtype)
    ax = edge_a[:, 0].to(xywh.dtype)
    ay = edge_a[:, 1].to(xywh.dtype)
    # [N, S*S, K]
    cross = ex.view(1, 1, -1) * (probe_y.unsqueeze(-1) - ay.view(1, 1, -1)) - ey.view(
        1, 1, -1
    ) * (probe_x.unsqueeze(-1) - ax.view(1, 1, -1))
    sign = torch.sign(cross)
    # Inside a convex polygon a point lies on the SAME side of every edge, so the
    # signed crossings must all agree in sign.  Testing ``|sign|.sum() == K``
    # instead only asks "did no probe land exactly on an edge" -- true almost
    # everywhere -- which marks the entire grid as inside and yields an all-ones
    # target.
    inside = (sign.sum(dim=-1).abs() == cross.shape[-1]).to(xywh.dtype)
    share = inside.mean(dim=-1)
    peak = share.max()
    if float(peak) <= 0.0:
        return torch.zeros_like(share)
    return share / peak


def rasterize_polygon_hard(points: Tensor, xywh: Tensor) -> Tensor:
    """Return a binary patch-centre mask for a convex polygon."""

    if points.numel() == 0:
        return torch.zeros(xywh.shape[0], device=xywh.device, dtype=xywh.dtype)
    hull = convex_hull(points.to(torch.float64))
    if hull.shape[0] < 3:
        return torch.zeros(xywh.shape[0], device=xywh.device, dtype=xywh.dtype)
    edge_a = hull
    edge_b = torch.roll(hull, shifts=-1, dims=0)
    ex = (edge_b[:, 0] - edge_a[:, 0]).to(xywh.dtype)
    ey = (edge_b[:, 1] - edge_a[:, 1]).to(xywh.dtype)
    ax = edge_a[:, 0].to(xywh.dtype)
    ay = edge_a[:, 1].to(xywh.dtype)
    centres = xywh[:, :2]
    cross = ex.view(1, -1) * (centres[:, 1].unsqueeze(-1) - ay.view(1, -1)) - ey.view(
        1, -1
    ) * (centres[:, 0].unsqueeze(-1) - ax.view(1, -1))
    eps = torch.finfo(xywh.dtype).eps * 8
    inside = (cross >= -eps).all(dim=-1) | (cross <= eps).all(dim=-1)
    return inside.to(dtype=xywh.dtype)


def build_mask_targets(
    tokenizer: Any,
    record: dict[str, Any],
    target_ids: Sequence[int],
    eos_ids: Iterable[int],
    xywh: Tensor,
    *,
    target_mode: str = "token",
    window_min: int = 3,
    window_max: int = 5,
    line_source: str = "auto",
    raster_mode: str = "soft",
) -> MaskTargets:
    """Assemble the per-token mask targets for one page.

    ``target_ids`` is the assistant target token id list (including the trailing
    EOS) taken from the actual ``input_ids``, and ``xywh`` is the grid returned
    by ``DecoderMaskRouter.project_visual`` (shape ``[1, N, 4]``).
    """

    if raster_mode not in ("soft", "hard"):
        raise ValueError(f"unknown raster_mode {raster_mode!r}")

    page_text = record["page_text"]
    boxes, char_statuses = char_boxes(record)
    spans, statuses, report = token_char_spans(tokenizer, page_text, target_ids, char_statuses)

    # --- window mode: the target is the convex hull of an in-line run of chars.
    # Line membership comes from the manifest's authoritative ``line_index``; a
    # geometric rule would merge columns on a vertical page, so it is only a
    # reported fallback, never a silent substitute.
    line_ids, annotated = char_lines(record)
    line_modes = ("window", "line", "anchored")
    line_source = line_source if target_mode in line_modes else "token"
    if target_mode in line_modes and line_source == "annotation" and not annotated:
        raise ValueError(
            f"target_mode='{target_mode}' with line_source='annotation' needs a 'line_index' in the "
            "manifest; regenerate the char manifest or use line_source='auto'"
        )
    use_lines = target_mode in line_modes and annotated
    line_members: dict[int, list[int]] = {}
    if use_lines:
        for char_index, line_id in enumerate(line_ids):
            if line_id is None or char_index >= len(boxes) or boxes[char_index] is None:
                continue
            line_members.setdefault(int(line_id), []).append(char_index)
    window_report: dict[str, Any] = {
        "mode": target_mode,
        "line_source": "annotation" if use_lines else ("geometry_unavailable" if target_mode in line_modes else "token"),
        "lines": len(line_members),
        "singleton_lines": sum(1 for members in line_members.values() if len(members) == 1),
        "max_line_length": max((len(m) for m in line_members.values()), default=0),
        "window_fallbacks": 0,
        "span_over_window": 0,
        "short_windows_at_line_end": 0,
        "raster_mode": raster_mode,
    }

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

        window_boxes: list[list[float]] | None = None
        if use_lines:
            line_id = line_ids[span[0]] if span[0] < len(line_ids) else None
            members = line_members.get(int(line_id)) if line_id is not None else None
            if members:
                if target_mode == "line":
                    # Line-level target: the full hull of the token's own text line,
                    # so every token on a line attends to the whole line's region.
                    window_boxes = [boxes[k] for k in members]
                elif target_mode == "anchored":
                    # The in-line window UNION the rest of the current line.  The window
                    # gives a sharp "which characters" cue; the line remainder restores
                    # the sustained line-level context that the plain window loses --
                    # measured at 8.4 hit tokens per step against the whole line's 77.6,
                    # at the same B.  Half of the line is unaffected by ``window_max``,
                    # so this sits on the coverage axis rather than the window-size one.
                    start = next((k for k, char_index in enumerate(members) if char_index >= span[0]), None)
                    if start is not None:
                        want = max(window_min, min(window_max, span[1] - span[0]))
                        if span[1] - span[0] > window_max:
                            window_report["span_over_window"] += 1
                        end = min(len(members), start + want)
                        if end - start < window_min:  # line tail: top up to the left
                            start = max(0, end - window_min)
                            window_report["short_windows_at_line_end"] += 1
                        if end > start:
                            window_boxes = [boxes[members[k]] for k in range(start, end)]
                            # Line remainder: from the window's end to the line's end.
                            window_boxes += [boxes[members[k]] for k in range(end, len(members))]
                else:
                    start = next((k for k, char_index in enumerate(members) if char_index >= span[0]), None)
                    if start is not None:
                        want = max(window_min, min(window_max, span[1] - span[0]))
                        if span[1] - span[0] > window_max:
                            window_report["span_over_window"] += 1
                        end = min(len(members), start + want)
                        if end - start < window_min:  # line tail: top up to the left
                            start = max(0, end - window_min)
                            window_report["short_windows_at_line_end"] += 1
                        if end > start:
                            window_boxes = [boxes[members[k]] for k in range(start, end)]

        if window_boxes:
            # Convex hull of the window's box corners: adjacent characters on one
            # line are near-collinear, so the hull is close to their union's
            # rectangle while still filling the gaps between glyphs -- one
            # connected blob for the bias instead of a row of separate dots.
            corners = torch.tensor(
                [[box[0], box[1]] for box in window_boxes] + [[box[2], box[3]] for box in window_boxes],
                device=xywh.device,
                dtype=xywh.dtype,
            )
            polygon_mask = (
                rasterize_polygon_hard(corners, xywh[0])
                if raster_mode == "hard"
                else rasterize_polygon(corners, xywh[0])
            )
            mask[0, index] = torch.maximum(mask[0, index], polygon_mask)
        else:
            if target_mode in ("window", "anchored"):
                window_report["window_fallbacks"] += 1
            for box in boxes_present:
                box_mask = (
                    rasterize_box_hard(box, xywh[0])
                    if raster_mode == "hard"
                    else rasterize_box(box, xywh[0])
                )
                mask[0, index] = torch.maximum(mask[0, index], box_mask)
        spatial_valid[0, index] = True

    return MaskTargets(
        mask=mask,
        xywh=xywh,
        spatial_valid=spatial_valid,
        stop_target=stop_target,
        char_spans=spans,
        alignment_status=statuses,
        alignment_report=report,
        window_report=window_report,
    )
