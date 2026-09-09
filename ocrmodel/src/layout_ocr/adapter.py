from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from .config import LayoutAdapterConfig
from .transport import SemiRelaxedTransport


@dataclass
class LayoutAdapterOutput:
    merged_tokens: Tensor
    layout_queries: Tensor
    boxes: Tensor
    order_scores: Tensor
    direction_logits: Tensor
    transport: Tensor | None
    validity_logits: Tensor | None = None
    validity_probs: Tensor | None = None
    gated_transport: Tensor | None = None
    valid_coverage: Tensor | None = None


class PreMergeLayoutAdapter(nn.Module):
    """Generate layout queries from whole-page visual tokens before VLM merging.

    ``patch_positions`` are deterministic vision-grid coordinates, not layout
    annotations. Ground-truth layout is deliberately absent from this interface.
    """

    def __init__(self, config: LayoutAdapterConfig):
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.query_seed = nn.Parameter(torch.empty(config.num_queries, d))
        nn.init.normal_(self.query_seed, std=0.02)
        self.query_attention = nn.MultiheadAttention(
            d, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(d)
        self.content_norm = nn.LayerNorm(d)
        self.box_head = nn.Linear(d, 4)
        self.order_head = nn.Linear(d, 1)
        self.direction_head = nn.Linear(d, config.num_directions)
        self.validity_head: nn.Linear | None = None
        if config.use_validity_head:
            self.validity_head = nn.Linear(d, 1)
            nn.init.zeros_(self.validity_head.weight)
            initial_probability = config.initial_valid_probability
            initial_logit = math.log(initial_probability / (1.0 - initial_probability))
            nn.init.constant_(self.validity_head.bias, initial_logit)
        initial_gate = math.atanh(config.initial_residual_scale)
        self.content_gate = nn.Parameter(torch.tensor(initial_gate, dtype=torch.float32))
        self.ot = SemiRelaxedTransport(
            config.ot_epsilon, config.ot_relaxation, config.ot_iterations
        )

    def effective_residual_scale(self) -> Tensor:
        """Return the checkpoint-configured residual scale.

        ``None`` preserves the original unbounded ``tanh(gate)`` behavior for
        legacy checkpoints.  New stabilization runs clamp the effective scale
        without changing the stored/raw gate parameter.
        """

        scale = self.content_gate.tanh()
        if self.config.max_residual_scale is not None:
            scale = scale.clamp(
                min=-self.config.max_residual_scale,
                max=self.config.max_residual_scale,
            )
        return scale

    def _queries(self, visual_tokens: Tensor) -> Tensor:
        seed = self.query_seed.unsqueeze(0).expand(visual_tokens.shape[0], -1, -1)
        update, _ = self.query_attention(seed, visual_tokens, visual_tokens, need_weights=False)
        return self.query_norm(seed + update)

    def _scores(
        self, queries: Tensor, visual_tokens: Tensor, boxes: Tensor, patch_positions: Tensor | None
    ) -> Tensor:
        scores = torch.matmul(queries, visual_tokens.transpose(-1, -2))
        scores = scores / self.config.hidden_size**0.5
        if self.config.mode in {"geometry", "layout_ot"}:
            if patch_positions is None:
                raise ValueError(f"patch_positions are required for mode={self.config.mode}")
            centers = (boxes[..., :2] + boxes[..., 2:]) * 0.5
            distance = torch.cdist(centers.float(), patch_positions.float(), p=2)
            scores = scores - distance.to(scores.dtype) / self.config.geometry_temperature
        return scores

    def forward(self, visual_tokens: Tensor, patch_positions: Tensor | None = None) -> LayoutAdapterOutput:
        if visual_tokens.ndim != 3:
            raise ValueError("visual_tokens must have shape [batch, tokens, hidden]")
        queries = self._queries(visual_tokens)
        raw_boxes = self.box_head(queries).sigmoid()
        xy_min = torch.minimum(raw_boxes[..., :2], raw_boxes[..., 2:])
        xy_max = torch.maximum(raw_boxes[..., :2], raw_boxes[..., 2:])
        boxes = torch.cat((xy_min, xy_max), dim=-1)
        order_scores = self.order_head(queries).squeeze(-1)
        direction_logits = self.direction_head(queries)
        validity_logits = None
        validity_probs = None
        if self.validity_head is not None:
            validity_logits = self.validity_head(queries).squeeze(-1)
            validity_probs = validity_logits.sigmoid()

        if self.config.mode == "content_only":
            transport = None
            gated_transport = None
            valid_coverage = None
            merged = visual_tokens
        else:
            scores = self._scores(queries, visual_tokens, boxes, patch_positions)
            if self.config.mode == "layout_ot":
                transport = self.ot(scores)
            else:
                transport = scores.softmax(dim=-1) / self.config.num_queries
            gated_transport = None
            valid_coverage = None
            fusion_transport = transport
            if validity_probs is not None:
                gated_transport = transport * validity_probs.unsqueeze(-1)
                raw_token_mass = transport.sum(dim=1)
                gated_token_mass = gated_transport.sum(dim=1)
                valid_coverage = gated_token_mass / raw_token_mass.clamp_min(1e-12)
                fusion_transport = gated_transport
            token_weights = fusion_transport.transpose(1, 2)
            token_weights = token_weights / token_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            layout_context = self.content_norm(torch.matmul(token_weights, queries))
            if valid_coverage is not None:
                layout_context = layout_context * valid_coverage.unsqueeze(-1)
            # Keep the zero-initialized gate an exact identity path.  This makes
            # the attention/geometry comparison attributable to the learned
            # layout context instead of an unconditional extra LayerNorm.
            merged = visual_tokens + self.effective_residual_scale() * layout_context

        return LayoutAdapterOutput(
            merged_tokens=merged,
            layout_queries=queries,
            boxes=boxes,
            order_scores=order_scores,
            direction_logits=direction_logits,
            transport=transport,
            validity_logits=validity_logits,
            validity_probs=validity_probs,
            gated_transport=gated_transport,
            valid_coverage=valid_coverage,
        )
