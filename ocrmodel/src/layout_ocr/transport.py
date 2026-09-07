from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SemiRelaxedTransport(nn.Module):
    """Entropic transport with fixed query mass and a relaxed token marginal.

    The query marginal is projected exactly after every iteration. The token
    marginal is encouraged, but not forced, toward uniform coverage through a
    KL-proximal update controlled by ``relaxation``.
    """

    def __init__(self, epsilon: float = 0.1, relaxation: float = 0.5, iterations: int = 20):
        super().__init__()
        self.epsilon = epsilon
        self.relaxation = relaxation
        self.iterations = iterations

    def forward(self, scores: Tensor) -> Tensor:
        if scores.ndim != 3:
            raise ValueError("scores must have shape [batch, queries, tokens]")
        _, queries, tokens = scores.shape
        log_kernel = scores.float() / self.epsilon
        log_a = log_kernel.new_full((1, queries), -math.log(queries))
        log_b = log_kernel.new_full((1, tokens), -math.log(tokens))
        u = torch.zeros_like(log_a).expand(scores.shape[0], -1)
        v = torch.zeros_like(log_b).expand(scores.shape[0], -1)
        strength = self.relaxation / (self.relaxation + self.epsilon)

        for _ in range(self.iterations):
            u = log_a - torch.logsumexp(log_kernel + v[:, None, :], dim=-1)
            column_log_mass = torch.logsumexp(log_kernel + u[:, :, None], dim=1)
            v = strength * (log_b - column_log_mass)

        log_plan = log_kernel + u[:, :, None] + v[:, None, :]
        plan = torch.exp(log_plan)
        plan = plan / plan.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        plan = plan / queries
        return plan.to(scores.dtype)
