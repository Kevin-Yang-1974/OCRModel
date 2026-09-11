"""Small, opt-in stabilization utilities for GLM-OCR generation.

The text path is deliberately separate from layout generation.  The natural
loop training path first observes a real, no-grad autoregressive trajectory
and then computes a differentiable loss from a detached loop prefix.  The
older teacher-forced argmax helper remains available for reproducing previous
runs, but it is not the natural-loop training path anymore.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover - local manifest tools do not need transformers
    class LogitsProcessor:  # type: ignore[no-redef]
        def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
            raise NotImplementedError


@dataclass(frozen=True)
class RepeatSuppressionConfig:
    enabled: bool = False
    unlikelihood_weight: float = 0.1
    eos_weight: float = 0.05
    recent_window: int = 96
    min_cycle_length: int = 8
    max_cycle_length: int = 32
    cycle_repeats: int = 3
    cycle_penalty: float = 2.0
    force_eos_steps: int = 16
    continuation_escape: bool = False
    escape_budget: int = 16
    escape_clear_steps: int = 4
    escape_eos_suppression: float = 1.0
    escape_eos_boost: float = 0.5

    def __post_init__(self) -> None:
        if self.unlikelihood_weight < 0 or self.eos_weight < 0:
            raise ValueError("repeat-loss weights must be non-negative")
        if self.recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if self.min_cycle_length <= 0 or self.max_cycle_length < self.min_cycle_length:
            raise ValueError("invalid cycle length range")
        if self.cycle_repeats < 2:
            raise ValueError("cycle_repeats must be at least two")
        if self.cycle_penalty < 0 or self.force_eos_steps < 0:
            raise ValueError("cycle penalty and EOS steps must be non-negative")
        if self.escape_budget <= 0 or self.escape_clear_steps <= 0:
            raise ValueError("escape budget and clear steps must be positive")
        if self.escape_eos_suppression < 0 or self.escape_eos_boost < 0:
            raise ValueError("escape EOS adjustments must be non-negative")


def repeated_cycle_positions(
    labels: Tensor,
    *,
    min_cycle_length: int = 8,
    max_cycle_length: int = 32,
    cycle_repeats: int = 3,
    recent_window: int | None = None,
) -> Tensor:
    """Return labels completing a repeated cycle.

    ``labels`` has shape ``[batch, sequence]`` and uses ``-100`` for ignored
    prompt positions.  Matching is performed on the contiguous non-ignored
    target stream, then mapped back to the original tensor.  Only the final
    cycle is marked, so the loss does not punish the first occurrence of a
    legitimate repeated symbol.
    """

    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    if min_cycle_length <= 0 or max_cycle_length < min_cycle_length:
        raise ValueError("invalid cycle length range")
    if cycle_repeats < 2:
        raise ValueError("cycle_repeats must be at least two")
    if recent_window is not None and recent_window <= 0:
        raise ValueError("recent_window must be positive when provided")
    result = torch.zeros_like(labels, dtype=torch.bool)
    for batch_index in range(labels.shape[0]):
        valid_indices = torch.nonzero(labels[batch_index] != -100, as_tuple=False).flatten()
        if recent_window is not None:
            valid_indices = valid_indices[-recent_window:]
        values = labels[batch_index, valid_indices].tolist()
        if len(values) < min_cycle_length * cycle_repeats:
            continue
        for end in range(min_cycle_length * cycle_repeats, len(values) + 1):
            upper = min(max_cycle_length, end // cycle_repeats)
            for cycle_length in range(min_cycle_length, upper + 1):
                repeated = values[end - cycle_length : end]
                if any(
                    values[end - cycle_length * (repeat + 1) : end - cycle_length * repeat]
                    != repeated
                    for repeat in range(1, cycle_repeats)
                ):
                    continue
                final_indices = valid_indices[end - cycle_length : end]
                result[batch_index, final_indices] = True
    return result


def natural_predicted_loop_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    min_cycle_length: int = 8,
    max_cycle_length: int = 32,
    cycle_repeats: int = 3,
    recent_window: int | None = 96,
) -> dict[str, Tensor]:
    """Penalize wrong tokens that the model naturally predicts in a cycle.

    The detector is deliberately run on the detached argmax stream from the
    *single* normal teacher-forced forward.  Only the live logits at detected
    positions receive gradient.  Causal alignment is the same as the model's
    language-model loss: ``logits[..., :-1]`` predicts ``labels[..., 1:]``.
    Positions whose predicted cycle token is already the ground-truth token
    are excluded, so legitimate repeated symbols do not activate the term.

    The returned count tensors are detached diagnostics on the same device as
    ``logits``; ``loss`` remains connected to the live logits even when no
    position is active through a zero-valued scalar.
    """

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be [batch, sequence, vocab] and match labels")
    if logits.shape[1] < 2:
        zero = logits.sum() * 0.0
        empty = torch.zeros((), device=logits.device, dtype=torch.float32)
        return {
            "loss": zero,
            "active_tokens": empty,
            "cycle_tokens": empty,
            "active_pages": empty,
            "valid_tokens": empty,
        }

    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    valid = shift_labels != -100
    predicted = shift_logits.detach().argmax(dim=-1)
    prediction_stream = predicted.masked_fill(~valid, -100)
    cycle_positions = repeated_cycle_positions(
        prediction_stream,
        min_cycle_length=min_cycle_length,
        max_cycle_length=max_cycle_length,
        cycle_repeats=cycle_repeats,
        recent_window=recent_window,
    )
    cycle_positions &= valid
    mismatch = cycle_positions & predicted.ne(shift_labels)

    cycle_count = cycle_positions.sum().to(dtype=torch.float32)
    active_count = mismatch.sum().to(dtype=torch.float32)
    active_pages = mismatch.any(dim=1).sum().to(dtype=torch.float32)
    valid_count = valid.sum().to(dtype=torch.float32)
    if not bool(mismatch.any()):
        return {
            "loss": logits.sum() * 0.0,
            "active_tokens": active_count.detach(),
            "cycle_tokens": cycle_count.detach(),
            "active_pages": active_pages.detach(),
            "valid_tokens": valid_count.detach(),
        }

    # Select before casting to fp32; materializing a full page-vocabulary
    # tensor is unnecessarily expensive for long whole-page sequences.
    selected_logits = shift_logits[mismatch].float()
    candidate_ids = predicted[mismatch].long()
    log_probability = F.log_softmax(selected_logits, dim=-1).gather(
        -1, candidate_ids.unsqueeze(-1)
    ).squeeze(-1)
    probability = log_probability.exp().clamp(max=1.0 - 1e-6)
    loss = (-torch.log1p(-probability)).mean()
    return {
        "loss": loss,
        "active_tokens": active_count.detach(),
        "cycle_tokens": cycle_count.detach(),
        "active_pages": active_pages.detach(),
        "valid_tokens": valid_count.detach(),
    }


def natural_loop_rollout_loss(
    logits: Tensor,
    labels: Tensor,
    candidate_mask: Tensor,
    candidate_token_ids: Tensor,
    continuation_targets: Tensor,
    continuation_mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Score a detached free-run loop prefix with live decoder logits.

    The masks and targets come from a no-grad autoregressive rollout and are
    aligned to ``logits[..., :-1]``.  The unlikelihood mask marks tokens that
    actually extended the observed loop; the continuation mask can include
    the complete post-loop horizon and teaches that corrupted prefix to prefer
    the real target token.  A negative candidate equal to the real target is
    excluded so legal repeated symbols are not penalized.
    """

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be [batch, sequence, vocab] and match labels")
    shifted_shape = (logits.shape[0], logits.shape[1] - 1)
    if continuation_mask is None:
        continuation_mask = candidate_mask
    if any(
        tensor.shape != shifted_shape
        for tensor in (
            candidate_mask,
            candidate_token_ids,
            continuation_targets,
            continuation_mask,
        )
    ):
        raise ValueError("natural-loop rollout targets must align with logits[..., :-1]")

    shift_logits = logits[..., :-1, :]
    negative_valid = (
        candidate_mask
        & (candidate_token_ids >= 0)
        & (continuation_targets >= 0)
        & candidate_token_ids.ne(continuation_targets)
    )
    continuation_valid = continuation_mask & (continuation_targets >= 0)
    active_tokens = negative_valid.sum().to(dtype=torch.float32)
    active_pages = (negative_valid | continuation_valid).any(dim=1).sum().to(dtype=torch.float32)
    candidate_tokens = candidate_mask.sum().to(dtype=torch.float32)
    continuation_tokens = continuation_valid.sum().to(dtype=torch.float32)
    if not bool(negative_valid.any() or continuation_valid.any()):
        zero = logits.sum() * 0.0
        return {
            "loss": zero,
            "unlikelihood": zero,
            "continuation": zero,
            "active_tokens": active_tokens.detach(),
            "active_pages": active_pages.detach(),
            "candidate_tokens": candidate_tokens.detach(),
            "continuation_tokens": continuation_tokens.detach(),
        }

    zero = logits.sum() * 0.0
    if bool(negative_valid.any()):
        negative_logits = shift_logits[negative_valid].float()
        candidate_ids = candidate_token_ids[negative_valid].long()
        log_probability = F.log_softmax(negative_logits, dim=-1).gather(
            -1, candidate_ids.unsqueeze(-1)
        ).squeeze(-1)
        probability = log_probability.exp().clamp(max=1.0 - 1e-6)
        unlikelihood = (-torch.log1p(-probability)).mean()
    else:
        unlikelihood = zero
    if bool(continuation_valid.any()):
        continuation_logits = shift_logits[continuation_valid].float()
        target_ids = continuation_targets[continuation_valid].long()
        continuation = F.cross_entropy(continuation_logits, target_ids)
    else:
        continuation = zero
    terms = [term for term, active in ((unlikelihood, negative_valid), (continuation, continuation_valid)) if bool(active.any())]
    loss = sum(terms) / len(terms)
    return {
        "loss": loss,
        "unlikelihood": unlikelihood,
        "continuation": continuation,
        "active_tokens": active_tokens.detach(),
        "active_pages": active_pages.detach(),
        "candidate_tokens": candidate_tokens.detach(),
        "continuation_tokens": continuation_tokens.detach(),
    }


def unlikelihood_loss(
    logits: Tensor,
    labels: Tensor,
    *,
    min_cycle_length: int = 8,
    max_cycle_length: int = 32,
    cycle_repeats: int = 3,
    recent_window: int | None = None,
) -> Tensor:
    """Penalize only target tokens that complete an observed cycle."""

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be [batch, sequence, vocab] and match labels")
    positions = repeated_cycle_positions(
        labels,
        min_cycle_length=min_cycle_length,
        max_cycle_length=max_cycle_length,
        cycle_repeats=cycle_repeats,
        recent_window=recent_window,
    )
    positions &= labels != -100
    if not bool(positions.any()):
        return logits.sum() * 0.0
    # Index before casting: ``logits.float()[positions]`` would materialize a
    # full fp32 page-vocabulary tensor and can stall one DDP rank on long pages.
    selected_logits = logits[positions].float()
    selected_labels = labels[positions].long()
    log_probability = F.log_softmax(selected_logits, dim=-1).gather(
        -1, selected_labels.unsqueeze(-1)
    ).squeeze(-1)
    probability = log_probability.exp().clamp(max=1.0 - 1e-6)
    return (-torch.log1p(-probability)).mean()


def eos_focus_loss(logits: Tensor, labels: Tensor, eos_token_ids: Iterable[int]) -> Tensor:
    """Page-normalized EOS cross entropy for the text decoder."""

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be [batch, sequence, vocab] and match labels")
    eos_ids = {int(token_id) for token_id in eos_token_ids}
    if not eos_ids:
        return logits.sum() * 0.0
    page_losses: list[Tensor] = []
    for page_logits, page_labels in zip(logits, labels.long()):
        mask = page_labels == -100
        mask = ~mask
        eos_mask = torch.zeros_like(mask)
        for eos_id in eos_ids:
            eos_mask |= page_labels == eos_id
        mask &= eos_mask
        if bool(mask.any()):
            page_losses.append(
                F.cross_entropy(page_logits[mask].float(), page_labels[mask])
            )
    if not page_losses:
        return logits.sum() * 0.0
    return torch.stack(page_losses).mean()


class ContinuationStopHead(nn.Module):
    """Small trainable calibration head for loop escape and true EOS.

    The head consumes a few decoder-decision statistics instead of all decoder
    hidden states.  This keeps the GLM-OCR backbone unchanged and lets the same
    head run inside the generation logits processor.  Its outputs are
    ``[loop_risk, stop_probability]`` logits.
    """

    feature_count = 6

    def __init__(self, hidden_size: int = 32) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("continuation head hidden size must be positive")
        self.net = nn.Sequential(
            nn.LayerNorm(self.feature_count),
            nn.Linear(self.feature_count, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim < 2 or features.shape[-1] != self.feature_count:
            raise ValueError(
                f"continuation features must end with {self.feature_count} values"
            )
        return self.net(features.float())


def continuation_decision_features(
    scores: Tensor,
    eos_token_ids: Iterable[int],
    negative_token_ids: Tensor | None = None,
) -> Tensor:
    """Build compact features shared by training and generation.

    ``scores`` may be ``[N, vocab]`` or ``[batch, time, vocab]``.  Negative
    token ids identify the token that would extend a detected cycle; ``-1``
    means that no cycle token is active at that position.
    """

    if scores.ndim < 2:
        raise ValueError("scores must have at least rank 2")
    eos_ids = sorted({int(token_id) for token_id in eos_token_ids})
    flat_scores = scores.reshape(-1, scores.shape[-1])
    best = flat_scores.max(dim=-1).values.float()
    if eos_ids:
        eos_index = torch.tensor(eos_ids, device=scores.device, dtype=torch.long)
        eos_logits = flat_scores.index_select(-1, eos_index).float().max(dim=-1).values
    else:
        eos_logits = torch.zeros_like(best)
    if negative_token_ids is None:
        negative = torch.zeros_like(best)
        active = torch.zeros_like(best)
    else:
        negative_ids = negative_token_ids.reshape(-1).to(device=scores.device, dtype=torch.long)
        if negative_ids.numel() != flat_scores.shape[0]:
            raise ValueError("negative_token_ids must align with scores")
        active = (negative_ids >= 0).float()
        safe_ids = negative_ids.clamp(min=0, max=flat_scores.shape[-1] - 1)
        negative = flat_scores.gather(-1, safe_ids.unsqueeze(-1)).squeeze(-1).float()
        negative = negative * active
    features = torch.stack(
        (
            eos_logits,
            negative,
            best,
            best - negative,
            active,
            eos_logits - best,
        ),
        dim=-1,
    )
    return features.reshape(*scores.shape[:-1], ContinuationStopHead.feature_count)


def _causal_selection(
    logits: Tensor,
    labels: Tensor,
    selection: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be [batch, sequence, vocab] and match labels")
    target = labels[..., 1:]
    valid = target != -100
    if selection is not None:
        if selection.shape != labels.shape:
            raise ValueError("selection mask must match labels")
        valid &= selection[..., 1:]
    positions = valid.nonzero(as_tuple=False)
    if not positions.numel():
        empty_logits = logits[..., :1, :].reshape(0, logits.shape[-1])
        empty_labels = target.reshape(-1)[:0]
        return empty_logits, empty_labels, positions
    selected_logits = logits[..., :-1, :][valid].float()
    selected_labels = target[valid].long()
    return selected_logits, selected_labels, positions


def cycle_escape_losses(
    logits: Tensor,
    labels: Tensor,
    escape_mask: Tensor,
    negative_token_ids: Tensor,
    eos_token_ids: Iterable[int],
    *,
    margin: float = 0.5,
) -> dict[str, Tensor]:
    """Train a looped prefix to recover the real continuation.

    The loss is only active on positions following a deliberately looped
    prefix.  It rewards the true next token, separates it from the token that
    would extend the loop, and suppresses premature EOS while content remains.
    """

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be [batch, sequence, vocab] and match labels")
    if escape_mask.shape != labels.shape or negative_token_ids.shape != labels.shape:
        raise ValueError("escape masks and negative ids must match labels")
    target = labels[..., 1:]
    valid = (target != -100) & escape_mask[..., 1:]
    positions = valid.nonzero(as_tuple=False)
    selected_logits = logits[..., :-1, :][valid].float()
    selected_labels = target[valid].long()
    zero = logits.sum() * 0.0
    if not positions.numel():
        return {"escape": zero, "margin": zero, "continue": zero, "active": zero}
    negative = negative_token_ids[..., 1:][valid]
    keep = (negative >= 0) & (selected_labels != negative)
    if not bool(keep.any()):
        return {"escape": zero, "margin": zero, "continue": zero, "active": zero}
    selected_logits = selected_logits[keep]
    selected_labels = selected_labels[keep]
    negative = negative[keep].long().clamp(min=0, max=selected_logits.shape[-1] - 1)
    log_probs = F.log_softmax(selected_logits, dim=-1)
    gold_log_probability = log_probs.gather(-1, selected_labels.unsqueeze(-1)).squeeze(-1)
    negative_log_probability = log_probs.gather(-1, negative.unsqueeze(-1)).squeeze(-1)
    escape = F.cross_entropy(selected_logits, selected_labels)
    margin_loss = F.relu(margin - gold_log_probability + negative_log_probability).mean()
    eos_ids = sorted({int(token_id) for token_id in eos_token_ids})
    if eos_ids:
        eos_log_probability = log_probs[:, eos_ids].logsumexp(dim=-1)
        not_eos = ~torch.zeros_like(selected_labels, dtype=torch.bool)
        for eos_id in eos_ids:
            not_eos &= selected_labels != eos_id
        continue_loss = (-torch.log1p(-eos_log_probability.exp().clamp(max=1.0 - 1e-6)))[not_eos]
        continue_loss = continue_loss.mean() if continue_loss.numel() else zero
    else:
        continue_loss = zero
    return {
        "escape": escape,
        "margin": margin_loss,
        "continue": continue_loss,
        "active": torch.tensor(float(keep.sum().item()), device=logits.device),
    }


def continuation_head_loss(
    head: ContinuationStopHead,
    logits: Tensor,
    labels: Tensor,
    eos_token_ids: Iterable[int],
    *,
    loop_mask: Tensor | None = None,
    negative_token_ids: Tensor | None = None,
) -> Tensor:
    """Auxiliary BCE for loop-risk and true-stop decisions."""

    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits must be [batch, sequence, vocab] and match labels")
    target = labels[..., 1:]
    valid = target != -100
    # All causal targets correspond to ``labels[..., 1:]``.  Keep the
    # auxiliary targets on that same length; initializing from ``labels``
    # would leave one extra position when no loop mask is supplied.
    shifted_loop = torch.zeros_like(target, dtype=torch.bool)
    if loop_mask is not None:
        if loop_mask.shape != labels.shape:
            raise ValueError("loop_mask must match labels")
        shifted_loop = loop_mask[..., 1:]
    shifted_negative = None
    if negative_token_ids is not None:
        if negative_token_ids.shape != labels.shape:
            raise ValueError("negative_token_ids must match labels")
        shifted_negative = negative_token_ids[..., 1:]
    selected_logits = logits[..., :-1, :][valid]
    if not selected_logits.numel():
        return logits.sum() * 0.0
    selected_negative = shifted_negative[valid] if shifted_negative is not None else None
    features = continuation_decision_features(
        selected_logits,
        eos_token_ids,
        selected_negative,
    )
    predictions = head(features)
    loop_target = shifted_loop[valid].float()
    eos_ids = sorted({int(token_id) for token_id in eos_token_ids})
    stop_target = torch.zeros_like(loop_target)
    if eos_ids:
        for eos_id in eos_ids:
            stop_target = torch.maximum(
                stop_target, (target[valid] == eos_id).float()
            )
    targets = torch.stack((loop_target, stop_target), dim=-1)
    pos_weight = torch.tensor((2.0, 2.0), device=predictions.device, dtype=predictions.dtype)
    return F.binary_cross_entropy_with_logits(predictions, targets, pos_weight=pos_weight)


def _cycle_run(tokens: Sequence[int], min_cycle_length: int, max_cycle_length: int) -> tuple[int, int]:
    """Return ``(cycle_length, repeat_count)`` for the strongest suffix."""

    best = (0, 0)
    upper = min(max_cycle_length, len(tokens) // 2)
    for cycle_length in range(min_cycle_length, upper + 1):
        suffix = list(tokens[-cycle_length:])
        repeats = 1
        cursor = len(tokens) - cycle_length
        while cursor >= cycle_length and list(tokens[cursor - cycle_length : cursor]) == suffix:
            repeats += 1
            cursor -= cycle_length
        if repeats > best[1] or (repeats == best[1] and cycle_length > best[0]):
            best = (cycle_length, repeats)
    return best


def generated_cycle_window(
    tokens: Sequence[int],
    *,
    min_cycle_length: int = 8,
    max_cycle_length: int = 32,
    cycle_repeats: int = 3,
    recent_window: int | None = 96,
) -> dict[str, Any]:
    """Locate the first repeated-cycle suffix in a generated token stream.

    ``end`` is an exclusive generated-token index.  It identifies the prefix
    that is fed back to the model before the loop's next continuation token is
    predicted.  The function consumes only detached tokens and never enters
    autograd.
    """

    if min_cycle_length <= 0 or max_cycle_length < min_cycle_length:
        raise ValueError("invalid cycle length range")
    if cycle_repeats < 2:
        raise ValueError("cycle_repeats must be at least two")
    if recent_window is not None and recent_window <= 0:
        raise ValueError("recent_window must be positive when provided")

    values = [int(token) for token in tokens]
    window_size = recent_window or len(values)
    start = max(min_cycle_length * cycle_repeats, len(values) - window_size)
    for end in range(start, len(values) + 1):
        window_start = max(0, end - window_size)
        cycle_length, repeats = _cycle_run(
            values[window_start:end], min_cycle_length, max_cycle_length
        )
        if cycle_length and repeats >= cycle_repeats:
            cycle_start = end - cycle_length * repeats
            return {
                "detected": True,
                "start": int(cycle_start),
                "end": int(end),
                "length": int(cycle_length),
                "repeats": int(repeats),
                "cycle": values[end - cycle_length : end],
            }
    return {
        "detected": False,
        "start": None,
        "end": None,
        "length": 0,
        "repeats": 0,
        "cycle": [],
    }


def repetition_diagnostics(
    text: str,
    *,
    recent_window: int = 96,
    min_cycle_length: int = 8,
    max_cycle_length: int = 32,
    cycle_repeats: int = 3,
) -> dict[str, float | int | bool | None]:
    """Compute generation-loop diagnostics without changing the prediction."""

    tokens = list(text)
    suffix = tokens[-recent_window:]
    cycle_length, repeats = _cycle_run(suffix, min_cycle_length, max_cycle_length)
    return {
        "repeated_cycle_length": cycle_length or None,
        "repeated_cycle_count": repeats,
        "repeated_cycle_detected": repeats >= cycle_repeats,
        "repeated_cycle_rate": (
            min(1.0, cycle_length * repeats / max(1, len(suffix)))
            if cycle_length and repeats >= cycle_repeats
            else 0.0
        ),
    }


def loop_continuation_diagnostics(
    tokens: Sequence[int],
    eos_token_ids: Iterable[int],
    *,
    recent_window: int = 96,
    min_cycle_length: int = 8,
    max_cycle_length: int = 32,
    cycle_repeats: int = 3,
    continuation_tokens: int = 4,
) -> dict[str, float | int | bool | None]:
    """Measure whether a generated cycle is escaped before EOS.

    A page is a successful escape when at least ``continuation_tokens`` tokens
    after the first detected cycle are not the next token of that cycle.  This
    metric is diagnostic only and does not alter generation.
    """

    if continuation_tokens <= 0:
        raise ValueError("continuation_tokens must be positive")
    eos_ids = {int(token_id) for token_id in eos_token_ids}
    values = list(int(token) for token in tokens)
    detected_end: int | None = None
    detected_length: int | None = None
    start = max(min_cycle_length * cycle_repeats, len(values) - recent_window)
    for end in range(start, len(values) + 1):
        cycle_length, repeats = _cycle_run(
            values[max(0, end - recent_window) : end],
            min_cycle_length,
            max_cycle_length,
        )
        if cycle_length and repeats >= cycle_repeats:
            detected_end = end
            detected_length = cycle_length
            break
    if detected_end is None or detected_length is None:
        return {
            "loop_detected": False,
            "continued_non_cycle_tokens": 0,
            "post_loop_eos": False,
            "post_loop_early_eos": False,
        }
    cycle = values[detected_end - detected_length : detected_end]
    future = values[detected_end:]
    non_cycle = 0
    for index, token in enumerate(future[:continuation_tokens]):
        if token in eos_ids:
            break
        if token != cycle[index % detected_length]:
            non_cycle += 1
    eos_index = next((index for index, token in enumerate(future) if token in eos_ids), None)
    return {
        "loop_detected": True,
        "continued_non_cycle_tokens": non_cycle,
        "post_loop_eos": eos_index is not None,
        "post_loop_early_eos": eos_index is not None and eos_index < continuation_tokens,
    }


class AdaptiveCycleLogitsProcessor(LogitsProcessor):
    """Cycle penalty that prefers continuation over a forced EOS.

    ``force_eos_steps`` is retained for reproducing the historical A1 audit,
    but new runs set it to zero and enable ``continuation_escape``.  In that
    mode a cycle only lowers the token that would extend it and softly adjusts
    EOS using the optional continuation head; EOS is never hard-masked.
    """

    def __init__(
        self,
        *,
        prompt_length: int,
        eos_token_ids: Iterable[int],
        config: RepeatSuppressionConfig,
        continuation_head: ContinuationStopHead | None = None,
    ) -> None:
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        self.prompt_length = prompt_length
        self.eos_token_ids = tuple(sorted({int(token_id) for token_id in eos_token_ids}))
        self.config = config
        self.continuation_head = continuation_head
        self._persistent_steps: dict[int, int] = {}
        self._recovery_steps: dict[int, int] = {}
        self._clear_steps: dict[int, int] = {}

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        if input_ids.ndim != 2 or scores.ndim != 2 or input_ids.shape[0] != scores.shape[0]:
            raise ValueError("input_ids and scores must be compatible rank-2 tensors")
        adjusted = scores.clone()
        for batch_index in range(input_ids.shape[0]):
            generated = input_ids[batch_index, self.prompt_length :].tolist()
            suffix = generated[-self.config.recent_window :]
            cycle_length, repeats = _cycle_run(
                suffix,
                self.config.min_cycle_length,
                self.config.max_cycle_length,
            )
            key = int(batch_index)
            if repeats >= 2 and cycle_length:
                self._persistent_steps[key] = self._persistent_steps.get(key, 0) + 1
                self._recovery_steps[key] = self._recovery_steps.get(key, 0) + 1
                self._clear_steps[key] = 0
                expected = int(suffix[-cycle_length])
                expected_tensor = torch.tensor(
                    [expected], device=scores.device, dtype=torch.long
                )
                if self.continuation_head is not None:
                    features = continuation_decision_features(
                        scores[batch_index : batch_index + 1],
                        self.eos_token_ids,
                        expected_tensor,
                    )
                    with torch.no_grad():
                        head_logits = self.continuation_head(features.reshape(1, -1))[0]
                        loop_probability = torch.sigmoid(head_logits[0]).item()
                        stop_probability = torch.sigmoid(head_logits[1]).item()
                else:
                    loop_probability = 1.0
                    stop_probability = 0.0
                recovery_step = self._recovery_steps[key]
                decay = max(
                    0.25,
                    1.0 - (recovery_step - 1) / max(1, self.config.escape_budget),
                )
                penalty = self.config.cycle_penalty * decay * (0.5 + 0.5 * loop_probability)
                adjusted[batch_index, expected] -= penalty
                if self.config.continuation_escape and self.eos_token_ids:
                    # Keep EOS available, but do not let it win merely because
                    # a loop was detected while the head predicts continuation.
                    progress = min(
                        1.0,
                        recovery_step / max(1, self.config.escape_clear_steps),
                    )
                    eos_delta = (
                        self.config.escape_eos_boost * progress * stop_probability
                        - self.config.escape_eos_suppression
                        * (1.0 - progress)
                        * (1.0 - stop_probability)
                    )
                    adjusted[batch_index, list(self.eos_token_ids)] += eos_delta
            else:
                self._persistent_steps[key] = 0
                if self._recovery_steps.get(key, 0):
                    self._clear_steps[key] = self._clear_steps.get(key, 0) + 1
                    if self._clear_steps[key] >= self.config.escape_clear_steps:
                        self._recovery_steps[key] = 0
                        self._clear_steps[key] = 0
            if (
                self.config.force_eos_steps
                and self._persistent_steps.get(key, 0) >= self.config.force_eos_steps
                and self.eos_token_ids
            ):
                adjusted[batch_index].fill_(float("-inf"))
                adjusted[batch_index, list(self.eos_token_ids)] = 0.0
        return adjusted


def generate_with_loop_recovery(
    model: Any,
    inputs: dict[str, Any],
    *,
    prompt_length: int,
    eos_token_ids: Iterable[int],
    config: RepeatSuppressionConfig,
    max_new_tokens: int,
    mode: str = "loop_recovery",
    continuation_head: ContinuationStopHead | None = None,
) -> Tensor:
    """Run deterministic generation in plain or stateful recovery mode."""

    if mode not in {"plain", "loop_recovery"}:
        raise ValueError("generation mode must be plain or loop_recovery")
    processors = []
    if mode == "loop_recovery" and config.enabled:
        processors.append(
            AdaptiveCycleLogitsProcessor(
                prompt_length=prompt_length,
                eos_token_ids=eos_token_ids,
                config=config,
                continuation_head=continuation_head,
            )
        )
    return model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        **({"logits_processor": processors} if processors else {}),
    )
