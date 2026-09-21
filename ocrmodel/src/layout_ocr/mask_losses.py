"""Mask and stop losses for the decoder mask head.

The mask target is a soft rasterised box (or window hull) normalised so its best
cell is 1.0, and only a handful of cells are positive.  Two terms are used
together, and the combination matters more than either one:

* **Balanced BCE** keeps the positive and negative *groups* averaged apart, so
  the few positive cells are not drowned out by the page of negatives.  It is
  also, on its own, what lets the head lift a broad background floor: raising
  every cell to ~0.2 costs little in the averaged negative term while cutting
  the positive term a lot.
* **Soft Dice** is the region term that removes exactly that loophole.  It is a
  ratio over the whole map, so spreading the mask to cover the background lowers
  it directly -- which is what the measured 0.23 background floor needs.

Dice is averaged **per token** (macro), not summed over all cells (micro).  A
window target has 3-5x the positive area of a single-token target, so a micro
average would silently re-weight tokens between the two arms and confound the
comparison it is meant to inform.  Per-token Dice is scale-free.
"""

from __future__ import annotations

import torch
from torch import Tensor

EPS = 1e-7


def balanced_mask_bce(mask: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Balanced BCE on the soft mask: non-empty and empty groups averaged apart.

    ``mask`` and ``target`` are ``[B, T, N]``; ``valid`` is ``[B, T]`` bool.  A
    valid token with an all-zero target (EOS or a blank token) is an *empty*
    group and is scored only through the negative term -- it is a real training
    signal ("predict nothing here"), not a token to skip.
    """

    has_content = target.sum(dim=-1) > 0
    non_empty = valid & has_content
    empty = valid & ~has_content
    log_m = torch.log(mask.clamp_min(EPS))
    log_1m = torch.log1p(-mask.clamp_max(1.0 - EPS))
    non_empty_loss = mask.new_zeros(())
    if non_empty.any():
        m = mask[non_empty]
        g = target[non_empty]
        pos = -(g * log_m[non_empty]).sum() / g.sum().clamp_min(EPS)
        neg = -((1.0 - g) * log_1m[non_empty]).sum() / (1.0 - g).sum().clamp_min(EPS)
        non_empty_loss = 0.5 * pos + 0.5 * neg
    empty_loss = mask.new_zeros(())
    if empty.any():
        empty_loss = -log_1m[empty].mean()
    return non_empty_loss + empty_loss


def soft_dice(mask: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """``1 - mean_t Dice(mask_t, target_t)`` over valid tokens that carry a box.

    Tokens with an all-zero target are excluded: Dice is ``0/0`` there and any
    convention (0 or 1) would either punish a correct empty prediction or reward
    an empty one.  They are already covered by the BCE empty group above.
    """

    has_content = target.sum(dim=-1) > 0
    keep = valid & has_content
    if not bool(keep.any()):
        return mask.new_zeros(())
    prediction = mask[keep].flatten(1)
    truth = target[keep].flatten(1)
    numerator = 2.0 * (prediction * truth).sum(dim=-1)
    denominator = prediction.sum(dim=-1) + truth.sum(dim=-1)
    dice = numerator / denominator.clamp_min(EPS)
    return 1.0 - dice.mean()


def plain_mask_bce(mask: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Unbalanced BCE, kept only so the balanced/plain choice is measurable."""

    if not bool(valid.any()):
        return mask.new_zeros(())
    m = mask[valid]
    g = target[valid]
    loss = -(g * torch.log(m.clamp_min(EPS)) + (1.0 - g) * torch.log1p(-m.clamp_max(1.0 - EPS)))
    return loss.mean()


def balanced_stop_bce(stop_prob: Tensor, stop_target: Tensor) -> Tensor:
    stop_prob = stop_prob.clamp(EPS, 1.0 - EPS)
    log_e = torch.log(stop_prob)
    log_1e = torch.log1p(-stop_prob)
    pos = stop_target > 0.5
    pos_loss = -log_e[pos].mean() if pos.any() else stop_prob.new_zeros(())
    neg_loss = -log_1e[~pos].mean() if (~pos).any() else stop_prob.new_zeros(())
    return pos_loss + neg_loss


def mask_and_dice_loss(
    mask: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    dice_weight: float = 0.0,
    bce_mode: str = "balanced",
) -> tuple[Tensor, dict[str, float]]:
    """Total mask loss plus its parts, for logging.

    ``dice_weight=0`` and ``bce_mode='balanced'`` reproduce the previous
    behaviour exactly, so existing runs and tests stay comparable.
    """

    if bce_mode == "balanced":
        bce = balanced_mask_bce(mask, target, valid)
    elif bce_mode == "plain":
        bce = plain_mask_bce(mask, target, valid)
    else:
        raise ValueError(f"unknown bce_mode {bce_mode!r}")
    if dice_weight > 0.0:
        dice = soft_dice(mask, target, valid)
    else:
        dice = mask.new_zeros(())
    total = bce + dice_weight * dice
    has_content = target.sum(dim=-1) > 0
    keep = valid & has_content
    with torch.no_grad():
        parts = {
            "bce": float(bce.detach().item()),
            "dice": float(dice.detach().item()),
            "mask_mean": float(mask.detach().mean().item()),
            "mask_max": float(mask.detach().max().item()),
            "mask_p95": float(mask.detach().flatten().quantile(0.95).item()),
            "content_tokens": int(keep.sum().item()),
        }
    return total, parts
