from __future__ import annotations

from dataclasses import dataclass

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
        self.content_gate = nn.Parameter(torch.tensor(0.0))
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

        if self.config.mode == "content_only":
            transport = None
            merged = visual_tokens
        else:
            scores = self._scores(queries, visual_tokens, boxes, patch_positions)
            if self.config.mode == "layout_ot":
                transport = self.ot(scores)
            else:
                transport = scores.softmax(dim=-1) / self.config.num_queries
            token_weights = transport.transpose(1, 2)
            token_weights = token_weights / token_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            layout_context = self.content_norm(torch.matmul(token_weights, queries))
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
        )
