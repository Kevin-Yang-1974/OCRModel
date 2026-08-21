from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VLQAOutput:
    visual_tokens: torch.Tensor
    layout_queries: torch.Tensor
    prediction_queries: torch.Tensor
    object_logits: torch.Tensor
    bbox_logits: torch.Tensor
    bbox_cxcywh: torch.Tensor
    bbox_xyxy: torch.Tensor
    direction_logits: torch.Tensor
    layout_residual: torch.Tensor
    layout_evidence: Optional[torch.Tensor] = None
    layout_memory_source: str = "visual_tokens"
    query_attention: Optional[torch.Tensor] = None
    writeback_attention: Optional[object] = None


@dataclass
class VLQALossOutput:
    loss: torch.Tensor
    object_loss: torch.Tensor
    bbox_l1_loss: torch.Tensor
    bbox_giou_loss: torch.Tensor
    direction_loss: torch.Tensor
    object_accuracy: torch.Tensor
    bbox_mean_iou: torch.Tensor
    direction_accuracy: torch.Tensor
    object_logit_abs_max: torch.Tensor
    direction_logit_abs_max: torch.Tensor
    bbox_pred_min: torch.Tensor
    bbox_pred_max: torch.Tensor
    query_abs_max: torch.Tensor
    prediction_query_abs_max: torch.Tensor
    bbox_logit_abs_max: torch.Tensor


class FeedForward(nn.Module):
    def __init__(self, dim: int, expansion: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class VisualQVLayoutConditionedAttention(nn.Module):
    """Route visual values with keys conditioned by layout context."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim < 1 or num_heads < 1 or dim % num_heads != 0:
            raise ValueError("dim must be positive and divisible by num_heads.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.layout_query_norm = nn.LayerNorm(dim)
        self.layout_key_norm = nn.LayerNorm(dim)
        self.layout_condition_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.visual_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.visual_query = nn.Linear(dim, dim)
        self.visual_key = nn.Linear(dim, dim)
        self.context_key = nn.Linear(dim, dim, bias=False)
        self.visual_value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, nn.MultiheadAttention):
                module._reset_parameters()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = tensor.shape
        return tensor.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        layout_vectors: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None,
        layout_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if visual_tokens.ndim != 3 or layout_vectors.ndim != 3:
            raise ValueError("visual_tokens and layout_vectors must be batch-first 3D tensors.")
        if visual_tokens.shape[0] != layout_vectors.shape[0]:
            raise ValueError("visual_tokens and layout_vectors batch sizes do not match.")
        if visual_tokens.shape[-1] != self.dim or layout_vectors.shape[-1] != self.dim:
            raise ValueError(f"VQLCA inputs must use hidden dimension {self.dim}.")

        layout_context, _ = self.layout_condition_attention(
            query=self.layout_query_norm(visual_tokens),
            key=self.layout_key_norm(layout_vectors),
            value=self.layout_key_norm(layout_vectors),
            key_padding_mask=layout_padding_mask,
            need_weights=False,
        )
        normalized_visual = self.visual_norm(visual_tokens)
        query = self._split_heads(self.visual_query(normalized_visual))
        key = self._split_heads(
            self.visual_key(normalized_visual)
            + self.context_key(self.context_norm(layout_context))
        )
        value = self._split_heads(self.visual_value(normalized_visual))
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if visual_padding_mask is not None:
            if visual_padding_mask.shape != visual_tokens.shape[:2]:
                raise ValueError("visual_padding_mask must have shape [B,N].")
            logits = logits.masked_fill(
                visual_padding_mask[:, None, None, :],
                torch.finfo(logits.dtype).min,
            )
        attention = logits.softmax(dim=-1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        routed = torch.matmul(attention, value)
        routed = routed.transpose(1, 2).reshape(visual_tokens.shape)
        routed = self.output(routed)
        if visual_padding_mask is not None:
            routed = routed.masked_fill(visual_padding_mask.unsqueeze(-1), 0.0)
        return routed, attention if return_attention else None


class VisualValueLayoutRouting(nn.Module):
    """Factorized visual-value routing conditioned on layout evidence.

    A conventional attention product requires the key and value sequence
    lengths to match.  The requested ``Q=V_i, K=A, V=V_i`` therefore cannot
    be written as one matrix product when ``L_v != K_p``.  This module keeps
    the intended semantics with two routing factors:

    ``R_va = softmax(Q(V_i) K(A)^T)`` has shape ``[B,H,L_v,K_p]`` and
    ``R_av = softmax(Q(A) K(V_i)^T)`` has shape ``[B,H,K_p,L_v]``.  Their
    product is a visual-to-visual route ``[B,H,L_v,L_v]`` which aggregates
    ``U(V_i)`` only.  Layout evidence is never a content Value.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim < 1 or num_heads < 1 or dim % num_heads:
            raise ValueError("dim must be positive and divisible by num_heads.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.visual_norm = nn.LayerNorm(dim)
        self.evidence_norm = nn.LayerNorm(dim)
        self.visual_query = nn.Linear(dim, dim)
        self.evidence_key = nn.Linear(dim, dim)
        self.evidence_query = nn.Linear(dim, dim)
        self.visual_key = nn.Linear(dim, dim)
        self.visual_value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _split(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    @staticmethod
    def _mask_logits(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is None:
            return logits
        return logits.masked_fill(mask[:, None, None, :], torch.finfo(logits.dtype).min)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        layout_evidence: torch.Tensor,
        visual_padding_mask: Optional[torch.Tensor] = None,
        layout_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        if visual_tokens.shape[0] != layout_evidence.shape[0]:
            raise ValueError("visual_tokens and layout_evidence batch sizes differ.")
        visual = self.visual_norm(visual_tokens)
        evidence = self.evidence_norm(layout_evidence)
        q_visual = self._split(self.visual_query(visual))
        k_evidence = self._split(self.evidence_key(evidence))
        q_evidence = self._split(self.evidence_query(evidence))
        k_visual = self._split(self.visual_key(visual))
        visual_value = self._split(self.visual_value(visual))

        visual_to_evidence = torch.matmul(q_visual, k_evidence.transpose(-2, -1)) / math.sqrt(self.head_dim)
        visual_to_evidence = self._mask_logits(visual_to_evidence, layout_padding_mask)
        visual_to_evidence = F.softmax(visual_to_evidence, dim=-1)
        evidence_to_visual = torch.matmul(q_evidence, k_visual.transpose(-2, -1)) / math.sqrt(self.head_dim)
        evidence_to_visual = self._mask_logits(evidence_to_visual, visual_padding_mask)
        evidence_to_visual = F.softmax(evidence_to_visual, dim=-1)
        route = torch.matmul(visual_to_evidence, evidence_to_visual)
        route = F.dropout(route, p=self.dropout, training=self.training)
        routed = torch.matmul(route, visual_value)
        routed = routed.transpose(1, 2).reshape_as(visual_tokens)
        routed = self.output(routed)
        if visual_padding_mask is not None:
            routed = routed.masked_fill(visual_padding_mask.unsqueeze(-1), 0.0)
        diagnostics = None
        if return_attention:
            diagnostics = {
                "visual_to_evidence": visual_to_evidence,
                "evidence_to_visual": evidence_to_visual,
                "visual_route": route,
            }
        return routed, diagnostics


class BoxHead(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(dim=-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(dim=-1)
    return torch.stack(
        (
            0.5 * (x0 + x1),
            0.5 * (y0 + y1),
            x1 - x0,
            y1 - y0,
        ),
        dim=-1,
    )


def aligned_generalized_box_iou(
    predicted_xyxy: torch.Tensor,
    target_xyxy: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    predicted_x0, predicted_y0, predicted_x1, predicted_y1 = predicted_xyxy.unbind(-1)
    target_x0, target_y0, target_x1, target_y1 = target_xyxy.unbind(-1)

    intersection_x0 = torch.maximum(predicted_x0, target_x0)
    intersection_y0 = torch.maximum(predicted_y0, target_y0)
    intersection_x1 = torch.minimum(predicted_x1, target_x1)
    intersection_y1 = torch.minimum(predicted_y1, target_y1)
    intersection = (
        (intersection_x1 - intersection_x0).clamp(min=0)
        * (intersection_y1 - intersection_y0).clamp(min=0)
    )

    predicted_area = (
        (predicted_x1 - predicted_x0).clamp(min=0)
        * (predicted_y1 - predicted_y0).clamp(min=0)
    )
    target_area = (
        (target_x1 - target_x0).clamp(min=0)
        * (target_y1 - target_y0).clamp(min=0)
    )
    union = predicted_area + target_area - intersection
    iou = intersection / union.clamp(min=eps)

    enclosing_x0 = torch.minimum(predicted_x0, target_x0)
    enclosing_y0 = torch.minimum(predicted_y0, target_y0)
    enclosing_x1 = torch.maximum(predicted_x1, target_x1)
    enclosing_y1 = torch.maximum(predicted_y1, target_y1)
    enclosing_area = (
        (enclosing_x1 - enclosing_x0).clamp(min=0)
        * (enclosing_y1 - enclosing_y0).clamp(min=0)
    )
    return iou - (enclosing_area - union) / enclosing_area.clamp(min=eps)


def aligned_box_iou(
    predicted_xyxy: torch.Tensor,
    target_xyxy: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    predicted_x0, predicted_y0, predicted_x1, predicted_y1 = predicted_xyxy.unbind(-1)
    target_x0, target_y0, target_x1, target_y1 = target_xyxy.unbind(-1)

    intersection = (
        (torch.minimum(predicted_x1, target_x1) - torch.maximum(predicted_x0, target_x0))
        .clamp(min=0)
        * (torch.minimum(predicted_y1, target_y1) - torch.maximum(predicted_y0, target_y0))
        .clamp(min=0)
    )
    predicted_area = (
        (predicted_x1 - predicted_x0).clamp(min=0)
        * (predicted_y1 - predicted_y0).clamp(min=0)
    )
    target_area = (
        (target_x1 - target_x0).clamp(min=0)
        * (target_y1 - target_y0).clamp(min=0)
    )
    union = predicted_area + target_area - intersection
    return intersection / union.clamp(min=eps)


def build_2d_sincos_position(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if height < 1 or width < 1:
        raise ValueError(f"Grid dimensions must be positive, got {(height, width)}.")
    quarter_dim = max(1, math.ceil(dim / 4))
    omega = torch.arange(quarter_dim, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(1, quarter_dim)))
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=torch.float32)
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    encoded_y = grid_y.reshape(-1, 1) * omega.reshape(1, -1)
    encoded_x = grid_x.reshape(-1, 1) * omega.reshape(1, -1)
    position = torch.cat(
        (encoded_y.sin(), encoded_y.cos(), encoded_x.sin(), encoded_x.cos()),
        dim=-1,
    )
    return position[:, :dim].to(dtype=dtype)


class VisualLayoutQueryAdapter(nn.Module):
    """Read soft page regions from visual tokens and write them back through a zero gate."""

    def __init__(
        self,
        visual_dim: int = 1024,
        layout_input_dim: int = 1024,
        adapter_dim: int = 256,
        num_queries: int = 16,
        num_heads: int = 8,
        ffn_expansion: int = 4,
        num_direction_classes: int = 5,
        dropout: float = 0.0,
        writeback_mode: str = "layout_value",
        writeback_num_heads: Optional[int] = None,
        writeback_dropout: Optional[float] = None,
        writeback_gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if visual_dim < 1 or layout_input_dim < 1 or adapter_dim < 1:
            raise ValueError("All feature dimensions must be positive.")
        if num_queries < 1:
            raise ValueError("num_queries must be positive.")
        if num_heads < 1 or adapter_dim % num_heads != 0:
            raise ValueError("adapter_dim must be divisible by num_heads.")
        if ffn_expansion < 1:
            raise ValueError("ffn_expansion must be positive.")
        if num_direction_classes < 2:
            raise ValueError("num_direction_classes must be at least 2.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if writeback_mode not in {"layout_value", "vqlca", "visual_value_layout_routing"}:
            raise ValueError("unsupported layout writeback mode.")
        writeback_num_heads = num_heads if writeback_num_heads is None else writeback_num_heads
        writeback_dropout = dropout if writeback_dropout is None else writeback_dropout
        if writeback_num_heads < 1 or adapter_dim % writeback_num_heads != 0:
            raise ValueError("adapter_dim must be divisible by writeback_num_heads.")
        if not 0.0 <= writeback_dropout < 1.0:
            raise ValueError("writeback_dropout must be in [0, 1).")

        self.visual_dim = visual_dim
        self.layout_input_dim = layout_input_dim
        self.adapter_dim = adapter_dim
        self.num_queries = num_queries
        self.num_direction_classes = num_direction_classes
        self.writeback_mode = writeback_mode
        self.writeback_gate_init = float(writeback_gate_init)

        self.query_embeddings = nn.Parameter(torch.empty(num_queries, adapter_dim))
        self.order_embeddings = nn.Parameter(torch.empty(num_queries, adapter_dim))

        self.memory_norm = nn.LayerNorm(layout_input_dim)
        self.memory_projection = nn.Linear(layout_input_dim, adapter_dim)
        self.query_norm = nn.LayerNorm(adapter_dim)
        self.query_cross_attention = nn.MultiheadAttention(
            embed_dim=adapter_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.query_ffn_norm = nn.LayerNorm(adapter_dim)
        self.query_ffn = FeedForward(adapter_dim, ffn_expansion, dropout)
        self.prediction_norm = nn.LayerNorm(adapter_dim)

        self.object_head = nn.Linear(adapter_dim, 1)
        self.box_head = BoxHead(adapter_dim)
        self.direction_head = nn.Linear(adapter_dim, num_direction_classes)

        self.visual_norm = nn.LayerNorm(visual_dim)
        self.visual_projection = nn.Linear(visual_dim, adapter_dim)
        if writeback_mode == "layout_value":
            self.writeback_query_norm = nn.LayerNorm(adapter_dim)
            self.writeback_key_norm = nn.LayerNorm(adapter_dim)
            self.writeback_attention = nn.MultiheadAttention(
                embed_dim=adapter_dim,
                num_heads=writeback_num_heads,
                dropout=writeback_dropout,
                batch_first=True,
            )
            self.vqlca_writeback = None
            self.visual_value_layout_routing = None
        elif writeback_mode == "vqlca":
            self.writeback_query_norm = None
            self.writeback_key_norm = None
            self.writeback_attention = None
            self.vqlca_writeback = VisualQVLayoutConditionedAttention(
                dim=adapter_dim,
                num_heads=writeback_num_heads,
                dropout=writeback_dropout,
            )
            self.visual_value_layout_routing = None
        else:
            self.writeback_query_norm = None
            self.writeback_key_norm = None
            self.writeback_attention = None
            self.vqlca_writeback = None
            self.visual_value_layout_routing = VisualValueLayoutRouting(
                dim=adapter_dim,
                num_heads=writeback_num_heads,
                dropout=writeback_dropout,
            )
        self.writeback_output = nn.Linear(adapter_dim, visual_dim)
        self.residual_gate = nn.Parameter(torch.zeros(()))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # The fast Hugging Face loading path does not reliably initialize direct
        # parameters on custom modules or MultiheadAttention projections.
        # Keep a complete explicit reset for a fresh VLQA attachment.
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, nn.MultiheadAttention):
                module._reset_parameters()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.query_embeddings, mean=0.0, std=0.02)
        nn.init.normal_(self.order_embeddings, mean=0.0, std=0.02)
        box_output = self.box_head.layers[-1]
        nn.init.normal_(box_output.weight, mean=0.0, std=0.001)
        if box_output.bias is not None:
            nn.init.zeros_(box_output.bias)
        nn.init.constant_(self.residual_gate, self.writeback_gate_init)

    def reset_writeback_parameters(self) -> None:
        if self.writeback_mode == "vqlca":
            assert self.vqlca_writeback is not None
            self.vqlca_writeback.reset_parameters()
        elif self.writeback_mode == "layout_value":
            assert self.writeback_attention is not None
            self.writeback_attention._reset_parameters()
        else:
            assert self.visual_value_layout_routing is not None
            self.visual_value_layout_routing.reset_parameters()
        nn.init.normal_(self.writeback_output.weight, mean=0.0, std=0.02)
        if self.writeback_output.bias is not None:
            nn.init.zeros_(self.writeback_output.bias)
        nn.init.constant_(self.residual_gate, self.writeback_gate_init)

    @staticmethod
    def _flatten_memory(
        layout_memory: torch.Tensor,
        memory_grid_size: Optional[tuple[int, int]],
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if layout_memory.ndim == 4:
            batch, channels, height, width = layout_memory.shape
            memory = layout_memory.flatten(2).transpose(1, 2)
            if memory_grid_size is not None and memory_grid_size != (height, width):
                raise ValueError(
                    "memory_grid_size disagrees with 4D layout_memory: "
                    f"declared={memory_grid_size}, actual={(height, width)}"
                )
            return memory, (height, width)
        if layout_memory.ndim != 3:
            raise ValueError(
                "layout_memory must have shape [B,N,C] or [B,C,H,W], got "
                f"{tuple(layout_memory.shape)}."
            )
        token_count = layout_memory.shape[1]
        if memory_grid_size is None:
            side = math.isqrt(token_count)
            if side * side != token_count:
                raise ValueError(
                    "memory_grid_size is required when layout token count is not square: "
                    f"N={token_count}."
                )
            memory_grid_size = (side, side)
        if memory_grid_size[0] * memory_grid_size[1] != token_count:
            raise ValueError(
                f"memory_grid_size={memory_grid_size} does not match N={token_count}."
            )
        return layout_memory, memory_grid_size

    def forward(
        self,
        visual_tokens: torch.Tensor,
        layout_memory: Optional[torch.Tensor] = None,
        memory_grid_size: Optional[tuple[int, int]] = None,
        visual_padding_mask: Optional[torch.Tensor] = None,
        layout_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> VLQAOutput:
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != self.visual_dim:
            raise ValueError(
                f"visual_tokens must have shape [B,N,{self.visual_dim}], got "
                f"{tuple(visual_tokens.shape)}."
            )
        layout_memory_source = "visual_tokens" if layout_memory is None else "high_resolution_visual_features"
        if layout_memory is None:
            layout_memory = visual_tokens
        memory, grid_size = self._flatten_memory(layout_memory, memory_grid_size)
        if memory.shape[0] != visual_tokens.shape[0]:
            raise ValueError("visual_tokens and layout_memory batch sizes do not match.")
        if memory.shape[-1] != self.layout_input_dim:
            raise ValueError(
                f"layout_memory feature dim must be {self.layout_input_dim}, got "
                f"{memory.shape[-1]}."
            )

        projected_memory = self.memory_projection(self.memory_norm(memory))
        position = build_2d_sincos_position(
            height=grid_size[0],
            width=grid_size[1],
            dim=self.adapter_dim,
            device=projected_memory.device,
            dtype=projected_memory.dtype,
        )
        projected_memory = projected_memory + position.unsqueeze(0)

        batch_size = visual_tokens.shape[0]
        queries = self.query_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        query_update, query_attention = self.query_cross_attention(
            query=self.query_norm(queries),
            key=projected_memory,
            value=projected_memory,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        queries = queries + query_update
        queries = queries + self.query_ffn(self.query_ffn_norm(queries))

        prediction_queries = self.prediction_norm(queries)
        object_logits = self.object_head(prediction_queries).squeeze(-1)
        bbox_logits = self.box_head(prediction_queries)
        bbox_cxcywh = bbox_logits.sigmoid()
        bbox_xyxy = cxcywh_to_xyxy(bbox_cxcywh)
        direction_logits = self.direction_head(prediction_queries)

        visual_queries = self.visual_projection(self.visual_norm(visual_tokens))
        ordered_layout_queries = prediction_queries + self.order_embeddings.unsqueeze(0)
        if self.writeback_mode == "layout_value":
            assert self.writeback_attention is not None
            assert self.writeback_query_norm is not None
            assert self.writeback_key_norm is not None
            writeback, writeback_attention = self.writeback_attention(
                query=self.writeback_query_norm(visual_queries),
                key=self.writeback_key_norm(ordered_layout_queries),
                value=ordered_layout_queries,
                key_padding_mask=layout_padding_mask,
                need_weights=return_attention,
                average_attn_weights=False,
            )
        elif self.writeback_mode == "vqlca":
            assert self.vqlca_writeback is not None
            writeback, writeback_attention = self.vqlca_writeback(
                visual_queries,
                ordered_layout_queries,
                visual_padding_mask=visual_padding_mask,
                layout_padding_mask=layout_padding_mask,
                return_attention=return_attention,
            )
        else:
            assert self.visual_value_layout_routing is not None
            writeback, writeback_attention = self.visual_value_layout_routing(
                visual_queries,
                queries,
                visual_padding_mask=visual_padding_mask,
                layout_padding_mask=layout_padding_mask,
                return_attention=return_attention,
            )
        layout_residual = self.writeback_output(writeback)
        output_tokens = visual_tokens + torch.tanh(self.residual_gate) * layout_residual

        return VLQAOutput(
            visual_tokens=output_tokens,
            layout_queries=queries,
            prediction_queries=prediction_queries,
            object_logits=object_logits,
            bbox_logits=bbox_logits,
            bbox_cxcywh=bbox_cxcywh,
            bbox_xyxy=bbox_xyxy,
            direction_logits=direction_logits,
            layout_residual=layout_residual,
            layout_evidence=queries,
            layout_memory_source=layout_memory_source,
            query_attention=query_attention if return_attention else None,
            writeback_attention=writeback_attention if return_attention else None,
        )


class VisualLayoutQueryLoss(nn.Module):
    """Ordered-slot object, box, and direction supervision; no standalone order loss."""

    def __init__(
        self,
        object_weight: float = 1.0,
        bbox_l1_weight: float = 5.0,
        bbox_giou_weight: float = 2.0,
        direction_weight: float = 1.0,
    ) -> None:
        super().__init__()
        weights = (object_weight, bbox_l1_weight, bbox_giou_weight, direction_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("VLQA loss weights must be non-negative.")
        self.object_weight = object_weight
        self.bbox_l1_weight = bbox_l1_weight
        self.bbox_giou_weight = bbox_giou_weight
        self.direction_weight = direction_weight

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_as_values = mask.to(dtype=values.dtype)
        numerator = (values * mask_as_values).sum()
        denominator = mask_as_values.sum().clamp(min=1.0)
        return numerator / denominator

    def forward(
        self,
        output: VLQAOutput,
        bbox_targets_xyxy: torch.Tensor,
        bbox_mask: torch.Tensor,
        object_targets: torch.Tensor,
        object_mask: torch.Tensor,
        direction_targets: torch.Tensor,
    ) -> VLQALossOutput:
        expected_box_shape = output.bbox_cxcywh.shape
        if bbox_targets_xyxy.shape != expected_box_shape:
            raise ValueError(
                f"bbox target shape must be {expected_box_shape}, got "
                f"{bbox_targets_xyxy.shape}."
            )
        expected_slot_shape = output.object_logits.shape
        for name, tensor in (
            ("bbox_mask", bbox_mask),
            ("object_targets", object_targets),
            ("object_mask", object_mask),
            ("direction_targets", direction_targets),
        ):
            if tensor.shape != expected_slot_shape:
                raise ValueError(
                    f"{name} shape must be {expected_slot_shape}, got {tensor.shape}."
                )

        # Keep all supervision numerically stable even when the adapter runs in BF16.
        object_logits = output.object_logits.float()
        bbox_cxcywh = output.bbox_cxcywh.float()
        bbox_xyxy = output.bbox_xyxy.float()
        direction_logits = output.direction_logits.float()
        bbox_targets_fp32 = bbox_targets_xyxy.float()
        object_targets_fp32 = object_targets.float()

        object_per_slot = F.binary_cross_entropy_with_logits(
            object_logits,
            object_targets_fp32,
            reduction="none",
        )
        object_loss = self._masked_mean(object_per_slot, object_mask.bool())

        target_cxcywh = xyxy_to_cxcywh(bbox_targets_fp32)
        bbox_l1_per_slot = F.l1_loss(
            bbox_cxcywh,
            target_cxcywh,
            reduction="none",
        ).sum(dim=-1)
        bbox_l1_loss = self._masked_mean(bbox_l1_per_slot, bbox_mask.bool())
        giou_per_slot = 1.0 - aligned_generalized_box_iou(
            bbox_xyxy,
            bbox_targets_fp32,
        )
        bbox_giou_loss = self._masked_mean(giou_per_slot, bbox_mask.bool())

        direction_mask = direction_targets.ne(-100)
        direction_per_slot = F.cross_entropy(
            direction_logits.reshape(-1, direction_logits.shape[-1]),
            direction_targets.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(direction_targets)
        direction_loss = self._masked_mean(direction_per_slot, direction_mask)

        with torch.no_grad():
            object_correct = object_logits.ge(0).eq(object_targets_fp32.ge(0.5)).float()
            object_accuracy = self._masked_mean(object_correct, object_mask.bool())
            bbox_mean_iou = self._masked_mean(
                aligned_box_iou(bbox_xyxy, bbox_targets_fp32),
                bbox_mask.bool(),
            )
            direction_correct = direction_logits.argmax(dim=-1).eq(direction_targets).float()
            direction_accuracy = self._masked_mean(direction_correct, direction_mask)

        total = (
            self.object_weight * object_loss
            + self.bbox_l1_weight * bbox_l1_loss
            + self.bbox_giou_weight * bbox_giou_loss
            + self.direction_weight * direction_loss
        )
        return VLQALossOutput(
            loss=total,
            object_loss=object_loss,
            bbox_l1_loss=bbox_l1_loss,
            bbox_giou_loss=bbox_giou_loss,
            direction_loss=direction_loss,
            object_accuracy=object_accuracy,
            bbox_mean_iou=bbox_mean_iou,
            direction_accuracy=direction_accuracy,
            object_logit_abs_max=object_logits.detach().abs().max(),
            direction_logit_abs_max=direction_logits.detach().abs().max(),
            bbox_pred_min=bbox_xyxy.detach().min(),
            bbox_pred_max=bbox_xyxy.detach().max(),
            query_abs_max=output.layout_queries.detach().float().abs().max(),
            prediction_query_abs_max=(
                output.prediction_queries.detach().float().abs().max()
            ),
            bbox_logit_abs_max=output.bbox_logits.detach().float().abs().max(),
        )
