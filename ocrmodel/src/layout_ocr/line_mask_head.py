"""Deterministic line masks: contextual patch scores and learned hold/update.

No boxes, line ids or reference strings enter forward. GT is used only by loss.
The stop auxiliary is independent of mask amplitude; the LM owns termination.
"""
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LineMaskConfig:
    hidden_size: int = 1536
    dim: int = 128
    bias: float = 1.0
    threshold: float = 0.5
    detach_every: int = 32


class LineMaskHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.dim
        self.visual = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, d))
        self.position = nn.Sequential(nn.Linear(4, d), nn.SiLU(), nn.Linear(d, d))
        self.horizontal = nn.Conv2d(d, d, (1, 9), padding=(0, 4), groups=d)
        self.vertical = nn.Conv2d(d, d, (9, 1), padding=(4, 0), groups=d)
        self.mix = nn.Conv2d(d, d, 1)
        self.query = nn.Sequential(nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, d), nn.SiLU(), nn.Linear(d, d))
        self.context = nn.Linear(d, d, bias=False)
        self.gate = nn.Sequential(nn.Linear(2*d, d), nn.SiLU(), nn.Linear(d, 1))
        self.stop = nn.Linear(d, 1)
        self.offset = nn.Parameter(torch.tensor(-2.0))
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode(self, visual, xywh, shape):
        k = self.visual(visual.float()) + self.position(xywh.float())
        grid = k.transpose(1, 2).reshape(k.shape[0], -1, *shape)
        grid = grid + self.mix(F.silu(self.horizontal(grid) + self.vertical(grid)))
        # Normalize once per page, not once per token. Repeated normalization
        # retains T copies of the N x D graph on dense full-training pages.
        return F.normalize(grid.flatten(2).transpose(1, 2), dim=-1)

    def step(self, hidden, keys, previous):
        q = self.query(hidden.float())
        context = torch.einsum('bn,bnd->bd', previous, keys) / previous.sum(-1, keepdim=True).clamp_min(1e-6)
        score_query = q + self.context(context)
        logits = torch.einsum('bd,bnd->bn', F.normalize(score_query, dim=-1), keys)
        logits = logits * self.log_scale.exp().clamp(max=30) + self.offset
        update_logits = self.gate(torch.cat((q, context), -1))
        update = update_logits.sigmoid()
        # Empty initial state must acquire a line. Later empty states may relocalize.
        update = torch.where(previous.sum(-1, keepdim=True) < 1e-6, torch.ones_like(update), update)
        mask = (1-update)*previous + update*logits.sigmoid()
        return mask, logits, update_logits, self.stop(q)

    def forward(self, hidden, visual, xywh, shape):
        keys = self.encode(visual, xywh, shape)
        previous = keys.new_zeros(keys.shape[:2])
        output = [[], [], [], []]
        for t in range(hidden.shape[1]):
            if t % self.config.detach_every == 0:
                previous = previous.detach()
            values = self.step(hidden[:, t], keys, previous)
            previous = values[0]
            for accumulator, value in zip(output, values):
                accumulator.append(value)
        return tuple(torch.stack(values, 1) for values in output)


def line_mask_loss(outputs, target, valid, stop_target):
    """Page-normalized spatial loss, candidate supervision and balanced transitions.

    BCE is computed on logits. Dice/area apply to the recurrent mask, not to a
    peak-normalized score. The area target is the GT line area, never zero L1.
    """
    mask, logits, update, stop = outputs
    keep = valid & (target.sum(-1) > 0)
    zero = sum(x.sum() for x in outputs)*0
    parts = {}
    if keep.any():
        p, g, z = mask[keep], target[keep], logits[keep]
        bce_cell = F.binary_cross_entropy_with_logits(z, g, reduction='none')
        positive = (bce_cell*g).sum(-1)/g.sum(-1).clamp_min(1)
        negative = (bce_cell*(1-g)).sum(-1)/(1-g).sum(-1).clamp_min(1)
        parts['bce'] = ((positive+negative)*0.5).mean()
        parts['dice'] = (1-(2*(p*g).sum(-1)+1)/(p.sum(-1)+g.sum(-1)+1)).mean()
        parts['area'] = F.smooth_l1_loss((p.sum(-1)+1).log(), (g.sum(-1)+1).log())
        # A normalized spatial distribution forces competition across patches.
        parts['location'] = -(g/g.sum(-1, keepdim=True).clamp_min(1)*F.log_softmax(z, -1)).sum(-1).mean()
    else:
        parts.update({name: zero for name in ('bce', 'dice', 'area', 'location')})
    transition_valid = keep[:, 1:] & keep[:, :-1]
    transition = (target[:, 1:]-target[:, :-1]).abs().sum(-1) > 0
    gate_loss = F.binary_cross_entropy_with_logits(update[:, 1:, 0], transition.float(), reduction='none')
    groups = [gate_loss[transition_valid & (transition == value)].mean() for value in (False, True) if (transition_valid & (transition == value)).any()]
    parts['transition'] = sum(groups)/len(groups) if groups else zero
    parts['stop'] = F.binary_cross_entropy_with_logits(stop[..., 0], stop_target)
    # EOS is an empty-mask example; blanks without geometry are ignored.
    eos = stop_target.bool()
    parts['empty'] = mask[eos].mean() if eos.any() else zero
    loss = parts['bce'] + parts['dice'] + .2*parts['location'] + .2*parts['area'] + .2*parts['transition'] + .05*parts['stop'] + .2*parts['empty']
    with torch.no_grad():
        binary = mask >= .5
        intersection = (binary*target).sum(-1)
        union = (binary | target.bool()).sum(-1).clamp_min(1)
        stats = {key: float(value.detach()) for key, value in parts.items()}
        stats['iou'] = float((intersection/union)[keep].mean()) if keep.any() else 0.
        stats['mass_ratio'] = float((mask.sum(-1)/target.sum(-1).clamp_min(1))[keep].mean()) if keep.any() else 0.
    return loss, stats
