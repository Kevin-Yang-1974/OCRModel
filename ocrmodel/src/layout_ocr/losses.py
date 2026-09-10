from __future__ import annotations

from typing import Any

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


def _validity_bce(logits: Tensor, targets: Tensor) -> Tensor:
    """Train valid/no-object as a real query classification problem.

    The reduction is over all queries, so the optimum for a constant predictor
    follows the page-level valid prior instead of the balanced-BCE value 0.5.
    """

    return F.binary_cross_entropy_with_logits(
        logits, targets.to(dtype=logits.dtype), reduction="mean"
    )


def _balanced_validity_bce(logits: Tensor, targets: Tensor) -> Tensor:
    """Compatibility wrapper for callers of the pre-fix private helper."""

    return _validity_bce(logits, targets)


def _validity_cardinality_loss(logits: Tensor, targets: Tensor) -> Tensor:
    predicted_fraction = logits.sigmoid().mean(dim=-1)
    target_fraction = targets.to(dtype=logits.dtype).mean(dim=-1)
    return (predicted_fraction - target_fraction).square().mean()


def _validity_ranking_loss(logits: Tensor, targets: Tensor, margin: float = 1.0) -> Tensor:
    """Separate matched queries from no-object queries by a positive margin."""

    targets = targets.to(dtype=torch.bool)
    page_losses: list[Tensor] = []
    for page_logits, page_targets in zip(logits, targets):
        positive = page_logits[page_targets]
        negative = page_logits[~page_targets]
        if positive.numel() and negative.numel():
            page_losses.append(
                F.softplus(margin - positive[:, None] + negative[None, :]).mean()
            )
    if page_losses:
        return torch.stack(page_losses).mean()
    return logits.sum() * 0.0


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
    *,
    return_info: bool = False,
) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, Any]]:
    """Align region targets to predicted query slots without label leakage.

    ``fixed_order`` preserves the original reading-order slot contract. The
    ``hungarian`` mode uses detached predicted boxes and normalized order scores
    only to choose supervision slots; ground-truth geometry never enters the
    adapter forward/fusion path.
    """

    if assignment == "fixed_order":
        if return_info:
            matched_query_indices = [
                torch.nonzero(mask, as_tuple=False).flatten().detach().cpu().tolist()
                for mask in targets["query_mask"]
            ]
            return targets, {
                "matched_query_indices": matched_query_indices,
                "matched_pairs": [
                    [[index, index] for index in indices]
                    for indices in matched_query_indices
                ],
                "matching_costs": [[] for _ in matched_query_indices],
            }
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
    matched_query_indices: list[list[int]] = []
    matched_pairs: list[list[list[int]]] = []
    matching_costs: list[list[float]] = []

    for batch_index in range(output_batch):
        target_indices = torch.nonzero(query_mask[batch_index], as_tuple=False).flatten()
        if target_indices.numel() == 0:
            matched_query_indices.append([])
            matched_pairs.append([])
            matching_costs.append([])
            continue
        target_boxes = targets["target_boxes"][batch_index, target_indices].detach().float()
        predicted_boxes = output.boxes[batch_index].detach().float()
        box_cost = (target_boxes[:, None, :] - predicted_boxes[None, :, :]).abs().mean(dim=-1)
        target_orders = targets["target_orders"][batch_index, target_indices].detach().float()
        predicted_orders = output.order_scores[batch_index].detach().float().sigmoid()
        order_cost = (target_orders[:, None] - predicted_orders[None, :]).abs()
        base_cost = 0.7 * box_cost + 0.3 * order_cost
        cost = base_cost
        if output.transport is not None:
            owners = targets["token_owners"][batch_index]
            raw_transport = output.transport[batch_index].detach().float()
            raw_transport = raw_transport / raw_transport.sum(dim=-1, keepdim=True).clamp_min(
                1e-12
            )
            support_cost = torch.zeros_like(box_cost)
            has_support = torch.zeros(
                target_indices.shape[0], dtype=torch.bool, device=box_cost.device
            )
            for target_local, target_index in enumerate(target_indices.tolist()):
                region_tokens = owners == target_index
                if bool(region_tokens.any()):
                    support = raw_transport[:, region_tokens].mean(dim=-1)
                    support_cost[target_local] = 1.0 - support
                    has_support[target_local] = True
            cost = torch.where(
                has_support[:, None],
                0.6 * box_cost + 0.2 * order_cost + 0.2 * support_cost,
                base_cost,
            )
        pairs = _hungarian_assignment(cost)
        matched_query_indices.append([query_index for _, query_index in pairs])
        matched_pairs.append(
            [[int(target_indices[target_local]), query_index] for target_local, query_index in pairs]
        )
        matching_costs.append(
            [float(cost[target_local, query_index]) for target_local, query_index in pairs]
        )

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

    result = {
        "target_boxes": matched_boxes,
        "target_orders": matched_orders,
        "target_directions": matched_directions,
        "query_mask": matched_mask,
        "token_owners": matched_owners,
    }
    if return_info:
        return result, {
            "matched_query_indices": matched_query_indices,
            "matched_pairs": matched_pairs,
            "matching_costs": matching_costs,
        }
    return result


def _assignment_nll(
    output: LayoutAdapterOutput,
    token_owners: Tensor,
) -> Tensor:
    """Compete for each owned token, including validity in the query softmax."""

    if output.transport is None:
        return output.boxes.sum() * 0.0
    transport = output.transport.clamp_min(1e-12)
    query_token_logits = transport.log()
    if output.validity_probs is not None:
        query_token_logits = query_token_logits + output.validity_probs.clamp_min(1e-12).log().unsqueeze(-1)
    log_probabilities = F.log_softmax(query_token_logits.transpose(1, 2), dim=-1)
    return F.nll_loss(
        log_probabilities.flatten(0, 1), token_owners.flatten(), ignore_index=-1
    )


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
    validity = zero
    validity_cardinality = zero
    validity_ranking = zero
    if output.validity_logits is not None:
        validity = _validity_bce(output.validity_logits, query_mask)
        validity_cardinality = _validity_cardinality_loss(output.validity_logits, query_mask)
        validity_ranking = _validity_ranking_loss(output.validity_logits, query_mask)
    plan_source = output.gated_transport if output.gated_transport is not None else output.transport
    if plan_source is not None:
        plan = plan_source.clamp_min(1e-12)
        entropy = -(plan * plan.log()).sum(dim=(1, 2)).mean()
        if token_owners is not None and (token_owners >= 0).any():
            assignment = _assignment_nll(output, token_owners)

    total = (
        weights.box * box
        + weights.order * order
        + weights.direction * direction
        + weights.assignment * assignment
        + weights.transport_entropy * entropy
        + weights.validity * validity
        + weights.validity_cardinality * validity_cardinality
        + weights.validity_ranking * validity_ranking
    )
    return {
        "loss": total,
        "layout_box": box,
        "layout_order": order,
        "layout_direction": direction,
        "layout_assignment": assignment,
        "transport_entropy": entropy,
        "layout_validity": validity,
        "layout_validity_bce": validity,
        "layout_validity_cardinality": validity_cardinality,
        "layout_validity_ranking": validity_ranking,
    }
