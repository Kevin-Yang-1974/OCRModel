from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .adapter import LayoutAdapterOutput
from .config import LayoutLossConfig, QueryAssignment


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum() / weights.expand_as(values).sum().clamp_min(1)


def _hungarian_assignment(cost: Tensor) -> list[tuple[int, int]]:
    """Solve a rectangular target-by-query assignment on a CPU cost matrix.

    The number of annotated regions is never larger than the number of queries,
    so this implementation only needs the ``rows <= columns`` case. Ties are
    resolved by ascending query index for deterministic diagnostics.
    """

    if cost.ndim != 2:
        raise ValueError("Hungarian cost must have shape [targets, queries]")
    rows, columns = cost.shape
    if rows > columns:
        raise ValueError("Hungarian matching requires at least as many queries as targets")
    if rows == 0:
        return []
    if not bool(torch.isfinite(cost).all()):
        raise ValueError("Hungarian cost contains non-finite values")

    values = cost.detach().float().cpu()
    potentials_row = [0.0] * (rows + 1)
    potentials_column = [0.0] * (columns + 1)
    assigned_column = [0] * (columns + 1)
    previous_column = [0] * (columns + 1)

    for row in range(1, rows + 1):
        assigned_column[0] = row
        column0 = 0
        minimum = [float("inf")] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column0] = True
            row0 = assigned_column[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = (
                    float(values[row0 - 1, column - 1])
                    - potentials_row[row0]
                    - potentials_column[column]
                )
                if current < minimum[column]:
                    minimum[column] = current
                    previous_column[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(columns + 1):
                if used[column]:
                    potentials_row[assigned_column[column]] += delta
                    potentials_column[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if assigned_column[column0] == 0:
                break
        while True:
            column1 = previous_column[column0]
            assigned_column[column0] = assigned_column[column1]
            column0 = column1
            if column0 == 0:
                break

    return [
        (assigned_column[column] - 1, column - 1)
        for column in range(1, columns + 1)
        if assigned_column[column] != 0
    ]


def match_layout_targets(
    output: LayoutAdapterOutput,
    targets: dict[str, Tensor],
    assignment: QueryAssignment = "fixed_order",
) -> dict[str, Tensor]:
    """Align region targets to predicted query slots without label leakage.

    ``fixed_order`` preserves the original reading-order slot contract. The
    ``hungarian`` mode uses detached predicted boxes and normalized order scores
    only to choose supervision slots; ground-truth geometry never enters the
    adapter forward/fusion path.
    """

    if assignment == "fixed_order":
        return targets
    if assignment != "hungarian":
        raise ValueError(f"unsupported query assignment: {assignment}")

    output_batch, query_count, _ = output.boxes.shape
    if targets["target_boxes"].shape[:2] != (output_batch, query_count):
        raise ValueError("target and adapter query shapes do not match")
    query_mask = targets["query_mask"].to(dtype=torch.bool)
    if query_mask.shape != (output_batch, query_count):
        raise ValueError("query_mask shape does not match adapter output")

    matched_boxes = torch.zeros_like(targets["target_boxes"])
    matched_orders = torch.zeros_like(targets["target_orders"])
    matched_directions = torch.zeros_like(targets["target_directions"])
    matched_mask = torch.zeros_like(query_mask)
    matched_owners = torch.full_like(targets["token_owners"], -1)

    for batch_index in range(output_batch):
        target_indices = torch.nonzero(query_mask[batch_index], as_tuple=False).flatten()
        if target_indices.numel() == 0:
            continue
        target_boxes = targets["target_boxes"][batch_index, target_indices].detach().float()
        predicted_boxes = output.boxes[batch_index].detach().float()
        box_cost = (target_boxes[:, None, :] - predicted_boxes[None, :, :]).abs().mean(dim=-1)
        target_orders = targets["target_orders"][batch_index, target_indices].detach().float()
        predicted_orders = output.order_scores[batch_index].detach().float().sigmoid()
        order_cost = (target_orders[:, None] - predicted_orders[None, :]).abs()
        cost = 0.7 * box_cost + 0.3 * order_cost
        pairs = _hungarian_assignment(cost)

        target_to_query = torch.full(
            (query_count,), -1, dtype=torch.long, device=query_mask.device
        )
        for target_local, query_index in pairs:
            target_index = int(target_indices[target_local])
            target_to_query[target_index] = query_index
            matched_boxes[batch_index, query_index] = targets["target_boxes"][
                batch_index, target_index
            ]
            matched_orders[batch_index, query_index] = targets["target_orders"][
                batch_index, target_index
            ]
            matched_directions[batch_index, query_index] = targets["target_directions"][
                batch_index, target_index
            ]
            matched_mask[batch_index, query_index] = True

        owners = targets["token_owners"][batch_index]
        valid_owners = (owners >= 0) & (owners < query_count)
        if bool(valid_owners.any()):
            matched_owners[batch_index, valid_owners] = target_to_query[owners[valid_owners]]

    return {
        "target_boxes": matched_boxes,
        "target_orders": matched_orders,
        "target_directions": matched_directions,
        "query_mask": matched_mask,
        "token_owners": matched_owners,
    }


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
    """Compute auxiliary training losses after query-target alignment.

    Targets are aligned to query slots by ``layout_targets`` and optionally
    ``match_layout_targets``. ``token_owners`` uses query indices for supervised
    visual tokens and ``-1`` for ignored tokens.
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
