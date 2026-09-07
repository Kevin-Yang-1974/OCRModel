from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .adapter import LayoutAdapterOutput
from .config import LayoutLossConfig


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.expand_as(values).sum().clamp_min(1)


def compute_layout_losses(
    output: LayoutAdapterOutput,
    *,
    target_boxes: Tensor,
    target_orders: Tensor,
    target_directions: Tensor,
    query_mask: Tensor,
    token_owners: Tensor | None = None,
    weights: LayoutLossConfig = LayoutLossConfig(),
) -> dict[str, Tensor]:
    """Compute auxiliary training losses after dataset-side query matching.

    Targets are aligned to query slots by the data/matching layer. ``token_owners``
    uses query indices for supervised visual tokens and ``-1`` for ignored tokens.
    """

    box = _masked_mean(F.smooth_l1_loss(output.boxes, target_boxes, reduction="none"), query_mask)
    order = _masked_mean(
        F.smooth_l1_loss(output.order_scores, target_orders, reduction="none"), query_mask
    )
    zero = output.boxes.sum() * 0.0
    direction_targets = target_directions.masked_fill(~query_mask, -100)
    direction = zero
    if query_mask.any():
        direction = F.cross_entropy(
            output.direction_logits.flatten(0, 1), direction_targets.flatten(), ignore_index=-100
        )
    assignment = zero
    entropy = zero
    if output.transport is not None:
        plan = output.transport.clamp_min(1e-12)
        entropy = -(plan * plan.log()).sum(dim=(1, 2)).mean()
        if token_owners is not None and (token_owners >= 0).any():
            probabilities = plan.transpose(1, 2)
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            assignment = F.nll_loss(
                probabilities.log().flatten(0, 1), token_owners.flatten(), ignore_index=-1
            )

    total = (
        weights.box * box
        + weights.order * order
        + weights.direction * direction
        + weights.assignment * assignment
        + weights.transport_entropy * entropy
    )
    return {
        "loss": total,
        "layout_box": box,
        "layout_order": order,
        "layout_direction": direction,
        "layout_assignment": assignment,
        "transport_entropy": entropy,
    }
