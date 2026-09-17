from __future__ import annotations

import torch
from torch import Tensor


def box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Compute pairwise IoU for normalized ``xyxy`` boxes.

    Args:
        boxes_a: Tensor with shape ``[batch, boxes_a, 4]``.
        boxes_b: Tensor with shape ``[batch, boxes_b, 4]``.

    Returns:
        Pairwise IoU with shape ``[batch, boxes_a, boxes_b]``.
    """

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


def generalized_box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Compute pairwise generalized IoU for ``xyxy`` boxes.

    The returned values are in ``[-1, 1]`` for valid boxes.  Unlike IoU, the
    enclosing-box term supplies a gradient when two boxes do not overlap.
    """

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
    iou = intersection_area / union.clamp_min(1e-8)

    enclosing_top_left = torch.minimum(boxes_a[:, :, None, :2], boxes_b[:, None, :, :2])
    enclosing_bottom_right = torch.maximum(boxes_a[:, :, None, 2:], boxes_b[:, None, :, 2:])
    enclosing_size = (enclosing_bottom_right - enclosing_top_left).clamp_min(0)
    enclosing_area = enclosing_size[..., 0] * enclosing_size[..., 1]
    return iou - (enclosing_area - union) / enclosing_area.clamp_min(1e-8)


def box_giou_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Return per-box ``1 - GIoU`` for tensors shaped ``[batch, queries, 4]``."""

    pairwise_giou = generalized_box_iou(pred, target)
    return 1.0 - pairwise_giou.diagonal(dim1=-2, dim2=-1)
