from __future__ import annotations

import torch
import torch.nn as nn


class GenericVisualTransformerAdapter(nn.Module):
    """Non-layout token adaptor used as the VLQA capacity control."""

    def __init__(self, visual_dim: int = 1024, adapter_dim: int = 256,
                 num_heads: int = 8, ffn_expansion: int = 8,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if visual_dim < 1 or adapter_dim < 1:
            raise ValueError("visual_dim and adapter_dim must be positive.")
        if num_heads < 1 or adapter_dim % num_heads != 0:
            raise ValueError("adapter_dim must be divisible by num_heads.")
        if ffn_expansion < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("Invalid ffn_expansion or dropout.")
        self.visual_dim = visual_dim
        self.adapter_dim = adapter_dim
        self.input_norm = nn.LayerNorm(visual_dim)
        self.input_projection = nn.Linear(visual_dim, adapter_dim)
        self.attention_norm = nn.LayerNorm(adapter_dim)
        self.self_attention = nn.MultiheadAttention(
            adapter_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(adapter_dim)
        hidden_dim = adapter_dim * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(adapter_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, adapter_dim), nn.Dropout(dropout),
        )
        self.output_projection = nn.Linear(adapter_dim, visual_dim)
        self.residual_gate = nn.Parameter(torch.zeros(()))
        self.reset_parameters()

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
        nn.init.zeros_(self.residual_gate)

    def forward(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != self.visual_dim:
            raise ValueError(
                f"visual_tokens must have shape [B,N,{self.visual_dim}], got "
                f"{tuple(visual_tokens.shape)}."
            )
        hidden = self.input_projection(self.input_norm(visual_tokens))
        normalized = self.attention_norm(hidden)
        attended, _ = self.self_attention(
            normalized, normalized, normalized, need_weights=False
        )
        hidden = hidden + attended
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return visual_tokens + self.residual_gate.tanh() * self.output_projection(hidden)
