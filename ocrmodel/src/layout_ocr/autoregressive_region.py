"""Autoregressive region proposal and de-duplication head.

The head treats the 512 layout queries as a visual candidate bank.  It emits
an ordered pointer sequence and an EOS decision; a query is therefore not a
fixed ground-truth region slot.  The implementation is intentionally small so
it can be screened beside the existing adapter before any larger decoder is
considered.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class RegionDecoderConfig:
    input_hidden_size: int
    decoder_hidden_size: int = 256
    num_heads: int = 8
    num_layers: int = 2
    candidate_count: int = 512
    max_regions: int = 512
    num_directions: int = 3
    pointer_mask: bool = True
    spatial_penalty: float = 4.0
    spatial_iou_threshold: float = 0.8
    eos_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.input_hidden_size <= 0 or self.decoder_hidden_size <= 0:
            raise ValueError("region decoder hidden sizes must be positive")
        if self.decoder_hidden_size % self.num_heads:
            raise ValueError("decoder_hidden_size must be divisible by num_heads")
        if self.num_layers <= 0 or self.candidate_count <= 0 or self.max_regions <= 0:
            raise ValueError("region decoder sizes must be positive")
        if self.spatial_penalty < 0 or not 0.0 < self.spatial_iou_threshold <= 1.0:
            raise ValueError("invalid region duplicate penalty")
        if not 0.0 < self.eos_threshold < 1.0:
            raise ValueError("eos_threshold must be in (0, 1)")


@dataclass
class RegionDecoderOutput:
    pointer_logits: Tensor
    boxes: Tensor
    direction_logits: Tensor
    eos_logits: Tensor
    selected_indices: Tensor
    selected_mask: Tensor
    region_features: Tensor
    candidate_objectness: Tensor
    candidate_boxes: Tensor
    count_logits: Tensor
    target_indices: Tensor | None = None


def box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Pairwise IoU for normalized ``xyxy`` boxes."""

    top_left = torch.maximum(boxes_a[:, :, None, :2], boxes_b[:, None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, :, None, 2:], boxes_b[:, None, :, 2:])
    intersection = (bottom_right - top_left).clamp_min(0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    area_a = (boxes_a[..., 2] - boxes_a[..., 0]).clamp_min(0) * (
        boxes_a[..., 3] - boxes_a[..., 1]
    ).clamp_min(0)
    area_b = (boxes_b[..., 2] - boxes_b[..., 0]).clamp_min(0) * (
        boxes_b[..., 3] - boxes_b[..., 1]
    ).clamp_min(0)
    union = area_a[..., None] + area_b[..., None, :] - intersection_area
    return intersection_area / union.clamp_min(1e-8)


def _sort_boxes(boxes: Tensor) -> Tensor:
    xy_min = torch.minimum(boxes[..., :2], boxes[..., 2:])
    xy_max = torch.maximum(boxes[..., :2], boxes[..., 2:])
    return torch.cat((xy_min, xy_max), dim=-1).clamp(0.0, 1.0)


def assign_candidate_pointers(
    candidate_boxes: Tensor,
    target_boxes: Tensor,
    target_mask: Tensor,
) -> Tensor:
    """Greedily assign each ordered target to one distinct visual candidate."""

    if candidate_boxes.ndim != 3 or target_boxes.ndim != 3 or target_mask.ndim != 2:
        raise ValueError("candidate boxes, target boxes, and mask have invalid ranks")
    batch, candidates, _ = candidate_boxes.shape
    if target_boxes.shape[0] != batch or target_boxes.shape[2] != 4:
        raise ValueError("candidate and target batch shapes do not match")
    if target_mask.shape != target_boxes.shape[:2]:
        raise ValueError("target mask shape does not match target boxes")
    result = torch.full(
        target_mask.shape,
        -100,
        dtype=torch.long,
        device=candidate_boxes.device,
    )
    distances = torch.cdist(target_boxes.detach().float(), candidate_boxes.detach().float(), p=1)
    for batch_index in range(batch):
        available = torch.ones(candidates, dtype=torch.bool, device=candidate_boxes.device)
        for target_index in range(target_boxes.shape[1]):
            if not bool(target_mask[batch_index, target_index]):
                continue
            costs = distances[batch_index, target_index].masked_fill(~available, float("inf"))
            candidate_index = int(costs.argmin())
            if not bool(torch.isfinite(costs[candidate_index])):
                break
            result[batch_index, target_index] = candidate_index
            available[candidate_index] = False
    return result


class AutoregressiveRegionDecoder(nn.Module):
    """Decode ordered regions from a fixed visual candidate bank."""

    def __init__(self, config: RegionDecoderConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.decoder_hidden_size
        self.input_projection = nn.Linear(config.input_hidden_size, hidden)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.num_heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
        )
        self.candidate_encoder = nn.TransformerEncoder(encoder_layer, config.num_layers)
        self.start_token = nn.Parameter(torch.zeros(hidden))
        self.state_cell = nn.GRUCell(hidden, hidden)
        self.pointer_query = nn.Linear(hidden, hidden, bias=False)
        self.objectness_head = nn.Linear(hidden, 1)
        self.box_delta_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )
        self.direction_head = nn.Linear(hidden * 2, config.num_directions)
        self.eos_head = nn.Linear(hidden, 1)
        self.count_head = nn.Linear(hidden, 1)
        nn.init.constant_(self.eos_head.bias, -2.0)

    def _spatial_penalty(
        self,
        candidate_boxes: Tensor,
        previous_boxes: Tensor,
        previous_mask: Tensor,
    ) -> Tensor:
        if not bool(previous_mask.any()) or self.config.spatial_penalty == 0.0:
            return candidate_boxes.new_zeros(candidate_boxes.shape[:2])
        overlaps = box_iou(candidate_boxes, previous_boxes)
        valid = previous_mask[:, None, :].to(overlaps.dtype)
        overlaps = overlaps.masked_fill(valid == 0, 0.0)
        max_overlap = overlaps.max(dim=-1).values
        return self.config.spatial_penalty * F.relu(
            max_overlap - self.config.spatial_iou_threshold
        )

    def forward(
        self,
        candidates: Tensor,
        candidate_boxes: Tensor,
        *,
        targets: dict[str, Tensor] | None = None,
        enable_pointer_mask: bool | None = None,
        enable_spatial_penalty: bool | None = None,
    ) -> RegionDecoderOutput:
        if candidates.ndim != 3 or candidate_boxes.ndim != 3:
            raise ValueError("candidates and candidate_boxes must be [batch, queries, hidden/4]")
        batch, candidate_count, _ = candidates.shape
        if candidate_count != self.config.candidate_count:
            raise ValueError(
                f"expected {self.config.candidate_count} candidates, got {candidate_count}"
            )
        if candidate_boxes.shape != (batch, candidate_count, 4):
            raise ValueError("candidate_boxes must match the candidate bank")
        pointer_mask = self.config.pointer_mask if enable_pointer_mask is None else enable_pointer_mask
        spatial_penalty = (
            self.config.spatial_penalty != 0.0
            if enable_spatial_penalty is None
            else enable_spatial_penalty
        )
        memory = self.candidate_encoder(self.input_projection(candidates))
        objectness = self.objectness_head(memory).squeeze(-1)
        target_indices = None
        target_mask = None
        if targets is not None:
            target_mask = targets["query_mask"].to(device=candidates.device, dtype=torch.bool)
            target_indices = assign_candidate_pointers(
                candidate_boxes,
                targets["target_boxes"].to(device=candidates.device),
                target_mask,
            )
            if target_indices.shape[1] > self.config.max_regions:
                raise ValueError("region targets exceed the configured max_regions")
            if target_indices.shape[1] < self.config.max_regions:
                padding = self.config.max_regions - target_indices.shape[1]
                target_indices = F.pad(target_indices, (0, padding), value=-100)
                target_mask = F.pad(target_mask, (0, padding), value=False)

        state = self.start_token.unsqueeze(0).expand(batch, -1)
        used = torch.zeros(batch, candidate_count, dtype=torch.bool, device=candidates.device)
        active = torch.ones(batch, dtype=torch.bool, device=candidates.device)
        selected_indices: list[Tensor] = []
        selected_mask: list[Tensor] = []
        sequence_boxes: list[Tensor] = []
        sequence_pointers: list[Tensor] = []
        sequence_directions: list[Tensor] = []
        sequence_eos: list[Tensor] = []
        sequence_features: list[Tensor] = []
        history_boxes: list[Tensor] = []
        history_masks: list[Tensor] = []

        for step in range(self.config.max_regions):
            query = self.pointer_query(state)
            pointer_logits = torch.einsum("bd,bqd->bq", query, memory) / hidden_size_scale(memory)
            if pointer_mask:
                pointer_logits = pointer_logits.masked_fill(used, float("-inf"))
            if spatial_penalty:
                previous_boxes = (
                    torch.stack(history_boxes, dim=1)
                    if history_boxes
                    else candidate_boxes.new_zeros(batch, 0, 4)
                )
                previous_mask = (
                    torch.stack(history_masks, dim=1)
                    if history_masks
                    else torch.zeros(batch, 0, dtype=torch.bool, device=candidates.device)
                )
                pointer_logits = pointer_logits - self._spatial_penalty(
                    candidate_boxes, previous_boxes, previous_mask
                )
            eos_logits = self.eos_head(state).squeeze(-1)
            if target_indices is not None and target_mask is not None:
                valid_teacher = target_mask[:, step] if step < target_mask.shape[1] else torch.zeros_like(active)
                teacher_indices = (
                    target_indices[:, step]
                    if step < target_indices.shape[1]
                    else pointer_logits.argmax(dim=-1)
                )
                safe_teacher = teacher_indices.clamp(0, candidate_count - 1)
                predicted_indices = pointer_logits.argmax(dim=-1)
                chosen = torch.where(valid_teacher, safe_teacher, predicted_indices)
                current_mask = valid_teacher
            else:
                chosen = pointer_logits.argmax(dim=-1)
                current_mask = active.clone()
            selected = memory[torch.arange(batch, device=candidates.device), chosen]
            base_box = candidate_boxes[torch.arange(batch, device=candidates.device), chosen]
            pair = torch.cat((state, selected), dim=-1)
            predicted_box = _sort_boxes(base_box + 0.2 * torch.tanh(self.box_delta_head(pair)))
            direction = self.direction_head(pair)
            selected_indices.append(chosen)
            selected_mask.append(current_mask)
            sequence_pointers.append(pointer_logits)
            sequence_boxes.append(predicted_box)
            sequence_directions.append(direction)
            sequence_eos.append(eos_logits)
            sequence_features.append(selected)
            used = used | F.one_hot(chosen, candidate_count).to(torch.bool)
            history_boxes.append(predicted_box)
            history_masks.append(current_mask)
            state = self.state_cell(selected, state)
            if target_indices is None:
                active = active & (eos_logits.sigmoid() < self.config.eos_threshold)

        return RegionDecoderOutput(
            pointer_logits=torch.stack(sequence_pointers, dim=1),
            boxes=torch.stack(sequence_boxes, dim=1),
            direction_logits=torch.stack(sequence_directions, dim=1),
            eos_logits=torch.stack(sequence_eos, dim=1),
            selected_indices=torch.stack(selected_indices, dim=1),
            selected_mask=torch.stack(selected_mask, dim=1),
            region_features=torch.stack(sequence_features, dim=1),
            candidate_objectness=objectness,
            candidate_boxes=candidate_boxes,
            count_logits=self.count_head(memory.mean(dim=1)).squeeze(-1),
            target_indices=target_indices,
        )

    def _recompute_pointer_logits(
        self,
        candidates: Tensor,
        candidate_boxes: Tensor,
        *,
        targets: dict[str, Tensor] | None,
        enable_pointer_mask: bool,
        enable_spatial_penalty: bool,
    ) -> Tensor:
        """Re-run the light pointer path for loss reporting.

        Keeping the recurrent state in the public output would make the output
        contract needlessly large.  This pass is deterministic and retains
        gradients through the candidate encoder and pointer head.
        """

        batch, candidate_count, _ = candidates.shape
        memory = self.candidate_encoder(self.input_projection(candidates))
        target_indices = None if targets is None else assign_candidate_pointers(
            candidate_boxes,
            targets["target_boxes"].to(device=candidates.device),
            targets["query_mask"].to(device=candidates.device, dtype=torch.bool),
        )
        target_mask = None if targets is None else targets["query_mask"].to(
            device=candidates.device, dtype=torch.bool
        )
        state = self.start_token.unsqueeze(0).expand(batch, -1)
        used = torch.zeros(batch, candidate_count, dtype=torch.bool, device=candidates.device)
        previous_boxes = candidate_boxes.new_zeros(batch, self.config.max_regions, 4)
        previous_mask = torch.zeros(
            batch, self.config.max_regions, dtype=torch.bool, device=candidates.device
        )
        logits: list[Tensor] = []
        for step in range(self.config.max_regions):
            step_logits = torch.einsum(
                "bd,bqd->bq", self.pointer_query(state), memory
            ) / hidden_size_scale(memory)
            if enable_pointer_mask:
                step_logits = step_logits.masked_fill(used, float("-inf"))
            if enable_spatial_penalty:
                step_logits = step_logits - self._spatial_penalty(
                    candidate_boxes, previous_boxes, previous_mask
                )
            logits.append(step_logits)
            if target_indices is not None and target_mask is not None and step < target_indices.shape[1]:
                chosen = target_indices[:, step].clamp(0, candidate_count - 1)
                valid = target_mask[:, step]
                chosen = torch.where(valid, chosen, step_logits.argmax(dim=-1))
            else:
                chosen = step_logits.argmax(dim=-1)
            selected = memory[torch.arange(batch, device=candidates.device), chosen]
            base_box = candidate_boxes[torch.arange(batch, device=candidates.device), chosen]
            state = self.state_cell(selected, state)
            used = used | F.one_hot(chosen, candidate_count).to(torch.bool)
            if step < self.config.max_regions:
                previous_boxes[:, step] = base_box
                previous_mask[:, step] = (
                    target_mask[:, step]
                    if target_mask is not None and step < target_mask.shape[1]
                    else torch.ones(batch, dtype=torch.bool, device=candidates.device)
                )
        return torch.stack(logits, dim=1)


def hidden_size_scale(memory: Tensor) -> float:
    return float(memory.shape[-1]) ** 0.5


def _gather_sequence(values: Tensor, indices: Tensor) -> Tensor:
    safe = indices.clamp_min(0)
    return values.gather(1, safe.unsqueeze(-1).expand(*safe.shape, values.shape[-1]))


def compute_region_losses(
    output: RegionDecoderOutput,
    targets: dict[str, Tensor],
) -> dict[str, Tensor]:
    """Compute the B1 pointer, geometry, EOS, coverage and duplicate losses."""

    if output.target_indices is None:
        zero = output.pointer_logits.sum() * 0.0
        return {"loss": zero, "region_pointer": zero, "region_bbox": zero, "region_giou": zero,
                "region_direction": zero, "region_eos": zero, "region_coverage": zero,
                "region_duplicate": zero, "region_objectness": zero, "region_count": zero}
    target_indices = output.target_indices
    target_mask = targets["query_mask"].to(device=output.pointer_logits.device, dtype=torch.bool)
    target_boxes = targets["target_boxes"].to(device=output.pointer_logits.device)
    target_directions = targets["target_directions"].to(device=output.pointer_logits.device)
    if target_mask.shape[1] < output.boxes.shape[1]:
        padding = output.boxes.shape[1] - target_mask.shape[1]
        target_mask = F.pad(target_mask, (0, padding), value=False)
        target_boxes = F.pad(target_boxes, (0, 0, 0, padding), value=0.0)
        target_directions = F.pad(target_directions, (0, padding), value=0)
    pointer = F.cross_entropy(
        output.pointer_logits.flatten(0, 1), target_indices.flatten(), ignore_index=-100
    )
    mask = target_mask.unsqueeze(-1).to(output.boxes.dtype)
    bbox = (F.smooth_l1_loss(output.boxes, target_boxes, reduction="none") * mask).sum()
    bbox = bbox / mask.expand_as(output.boxes).sum().clamp_min(1.0)
    predicted = _gather_sequence(output.candidate_boxes, target_indices)
    candidate_bbox = (F.smooth_l1_loss(predicted, target_boxes, reduction="none") * mask).sum()
    candidate_bbox = candidate_bbox / mask.expand_as(predicted).sum().clamp_min(1.0)
    pred = output.boxes
    target = target_boxes
    top_left = torch.maximum(pred[..., :2], target[..., :2])
    bottom_right = torch.minimum(pred[..., 2:], target[..., 2:])
    intersection = (bottom_right - top_left).clamp_min(0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    pred_area = (pred[..., 2] - pred[..., 0]).clamp_min(0) * (
        pred[..., 3] - pred[..., 1]
    ).clamp_min(0)
    target_area = (target[..., 2] - target[..., 0]).clamp_min(0) * (
        target[..., 3] - target[..., 1]
    ).clamp_min(0)
    union = pred_area + target_area - intersection_area
    iou = intersection_area / union.clamp_min(1e-8)
    enclosing_top_left = torch.minimum(pred[..., :2], target[..., :2])
    enclosing_bottom_right = torch.maximum(pred[..., 2:], target[..., 2:])
    enclosing_size = (enclosing_bottom_right - enclosing_top_left).clamp_min(0)
    enclosing_area = enclosing_size[..., 0] * enclosing_size[..., 1]
    generalized = iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-8)
    giou = (1.0 - generalized)
    giou = (giou * mask.squeeze(-1)).sum() / target_mask.sum().clamp_min(1)
    direction_target = target_directions.masked_fill(~target_mask, -100)
    direction = F.cross_entropy(
        output.direction_logits.flatten(0, 1), direction_target.flatten(), ignore_index=-100
    )
    eos_target = torch.zeros_like(output.eos_logits)
    for batch_index in range(target_mask.shape[0]):
        count = int(target_mask[batch_index].sum())
        eos_target[batch_index, min(count, eos_target.shape[1] - 1)] = 1.0
    eos = F.binary_cross_entropy_with_logits(output.eos_logits, eos_target)
    candidate_target = torch.zeros_like(output.candidate_objectness)
    for batch_index in range(target_indices.shape[0]):
        valid = target_indices[batch_index][target_indices[batch_index] >= 0]
        if valid.numel():
            candidate_target[batch_index, valid.unique()] = 1.0
    objectness = F.binary_cross_entropy_with_logits(output.candidate_objectness, candidate_target)
    count_target = target_mask.sum(dim=-1).float() / target_mask.shape[-1]
    count = F.mse_loss(output.count_logits.sigmoid(), count_target)
    pairwise = box_iou(output.boxes, output.boxes)
    upper = torch.triu(torch.ones_like(pairwise), diagonal=1)
    valid_pairs = target_mask[:, :, None] & target_mask[:, None, :]
    duplicate = (F.relu(pairwise - 0.8).square() * upper * valid_pairs).sum()
    duplicate = duplicate / (upper * valid_pairs).sum().clamp_min(1.0)
    coverage = duplicate
    total = (
        pointer
        + 5.0 * (bbox + candidate_bbox)
        + 2.0 * giou
        + direction
        + 0.1 * count
        + 0.1 * coverage
        + 0.1 * duplicate
        + 0.1 * objectness
        + 0.1 * eos
    )
    return {
        "loss": total,
        "region_pointer": pointer,
        "region_bbox": bbox + candidate_bbox,
        "region_giou": giou,
        "region_direction": direction,
        "region_eos": eos,
        "region_coverage": coverage,
        "region_duplicate": duplicate,
        "region_objectness": objectness,
        "region_count": count,
    }
