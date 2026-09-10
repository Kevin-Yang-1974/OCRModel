"""Small, opt-in stabilization utilities for GLM-OCR generation.

The text path is deliberately separate from layout generation.  The
unlikelihood term is only enabled for tokens that complete a repeated cycle in
the teacher-forced target; ordinary repeated characters are left alone.  The
logits processor is an inference guard and never participates in checkpoint
selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

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


class AdaptiveCycleLogitsProcessor(LogitsProcessor):
    """Inference-only cycle penalty with a bounded EOS escape hatch."""

    def __init__(
        self,
        *,
        prompt_length: int,
        eos_token_ids: Iterable[int],
        config: RepeatSuppressionConfig,
    ) -> None:
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        self.prompt_length = prompt_length
        self.eos_token_ids = tuple(sorted({int(token_id) for token_id in eos_token_ids}))
        self.config = config
        self._persistent_steps: dict[int, int] = {}

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
                expected = int(suffix[-cycle_length])
                adjusted[batch_index, expected] -= self.config.cycle_penalty
            else:
                self._persistent_steps[key] = 0
            if (
                self.config.force_eos_steps
                and self._persistent_steps.get(key, 0) >= self.config.force_eos_steps
                and self.eos_token_ids
            ):
                adjusted[batch_index].fill_(float("-inf"))
                adjusted[batch_index, list(self.eos_token_ids)] = 0.0
        return adjusted
