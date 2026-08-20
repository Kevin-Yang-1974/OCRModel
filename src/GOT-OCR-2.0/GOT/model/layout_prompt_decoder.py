"""Prompted variable-length layout decoding primitives.

This module is deliberately independent from ``layout_query.py``.  The
existing Fixed-Slot VLQA adapter remains the production baseline; this file
provides an opt-in prototype whose prompt bank is not indexed by regions and
whose number of records is determined by autoregressive EOS decoding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class VariableLayoutOutput:
    logits: Tensor
    hidden_states: Tensor
    prompt_context: Tensor
    prompt_attention: Optional[Tensor] = None
    region_positions: Optional[list[list[int]]] = None
    generated_eos: Optional[Tensor] = None
    truncated: Optional[Tensor] = None
    generated_ids: Optional[Tensor] = None


@dataclass
class LayoutRecordOutput:
    bbox: Tensor
    type_logits: Tensor
    direction_logits: Tensor
    count: Tensor


@dataclass
class VariableLayoutLossOutput:
    loss: Tensor
    sequence_loss: Tensor
    bbox_loss: Tensor
    type_loss: Tensor
    direction_loss: Tensor
    count_loss: Tensor
    prompt_diversity_loss: Tensor


class LayoutPromptBank(nn.Module):
    """Learnable global prompts with shape ``[1, K, D]``.

    Prompt index is an implementation slot in the attention bank only.  No
    parameter is assigned a first-column, second-region, or other layout
    meaning.
    """

    def __init__(self, hidden_size: int, num_prompts: int = 32) -> None:
        super().__init__()
        if hidden_size < 1 or num_prompts < 1:
            raise ValueError("hidden_size and num_prompts must be positive.")
        self.hidden_size = hidden_size
        self.num_prompts = num_prompts
        self.prompts = nn.Parameter(torch.empty(1, num_prompts, hidden_size))
        nn.init.normal_(self.prompts, mean=0.0, std=0.02)

    def forward(self, batch_size: int) -> Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        return self.prompts.expand(batch_size, -1, -1)


class LayoutPromptCrossAttention(nn.Module):
    """Read the complete page visual sequence with global layout prompts."""

    def __init__(
        self,
        visual_size: int,
        hidden_size: int,
        num_prompts: int = 32,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if visual_size < 1 or hidden_size < 1:
            raise ValueError("visual_size and hidden_size must be positive.")
        if num_heads < 1 or hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.visual_size = visual_size
        self.hidden_size = hidden_size
        self.num_prompts = num_prompts
        self.prompt_bank = LayoutPromptBank(hidden_size, num_prompts)
        self.visual_projection = (
            nn.Identity() if visual_size == hidden_size else nn.Linear(visual_size, hidden_size)
        )
        self.visual_norm = nn.LayerNorm(hidden_size)
        self.prompt_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        visual_tokens: Tensor,
        visual_padding_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        if visual_tokens.ndim != 3 or visual_tokens.shape[-1] != self.visual_size:
            raise ValueError(
                f"visual_tokens must have shape [B,L,{self.visual_size}], got "
                f"{tuple(visual_tokens.shape)}."
            )
        visual = self.visual_projection(visual_tokens)
        visual = self.visual_norm(visual)
        prompts = self.prompt_bank(visual.shape[0])
        update, attention = self.cross_attention(
            query=self.prompt_norm(prompts),
            key=visual,
            value=visual,
            key_padding_mask=visual_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        return self.output_norm(prompts + update), attention if return_attention else None


class LayoutVocabulary:
    """Small deterministic structural vocabulary used by the prototype decoder."""

    def __init__(self, type_names: tuple[str, ...] = ("COLUMN", "ROW", "REGION")) -> None:
        tokens = ["<PAD>", "<LAYOUT>", "<REGION>", "</REGION>", "<TYPE>", "</TYPE>", "<EOS>"]
        tokens.extend(type_names)
        self.tokens = tuple(dict.fromkeys(tokens))
        self.token_to_id = {token: index for index, token in enumerate(self.tokens)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_id = self.token_to_id["<PAD>"]
        self.layout_id = self.token_to_id["<LAYOUT>"]
        self.region_id = self.token_to_id["<REGION>"]
        self.end_region_id = self.token_to_id["</REGION>"]
        self.type_id = self.token_to_id["<TYPE>"]
        self.end_type_id = self.token_to_id["</TYPE>"]
        self.eos_id = self.token_to_id["<EOS>"]

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, tokens: list[str]) -> list[int]:
        unknown = [token for token in tokens if token not in self.token_to_id]
        if unknown:
            raise ValueError(f"Unknown layout token(s): {unknown}")
        return [self.token_to_id[token] for token in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id_to_token[int(token_id)] for token_id in ids]


def causal_mask(length: int, device: torch.device) -> Tensor:
    if length < 1:
        raise ValueError("causal mask length must be positive.")
    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)


class VariableLayoutDecoder(nn.Module):
    """Causal structural decoder with full-page and prompt memory access."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        visual_size: int,
        num_prompts: int = 32,
        num_layers: int = 2,
        num_heads: int = 8,
        feedforward_size: Optional[int] = None,
        max_layout_tokens: int = 2048,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or hidden_size < 1 or visual_size < 1:
            raise ValueError("vocab_size, hidden_size, and visual_size must be positive.")
        if num_layers < 1 or max_layout_tokens < 2:
            raise ValueError("num_layers must be positive and max_layout_tokens >= 2.")
        if num_heads < 1 or hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.visual_size = visual_size
        self.max_layout_tokens = max_layout_tokens
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_layout_tokens, hidden_size)
        self.prompt_attention = LayoutPromptCrossAttention(
            visual_size, hidden_size, num_prompts, num_heads, dropout
        )
        memory_projection = (
            nn.Identity() if visual_size == hidden_size else nn.Linear(visual_size, hidden_size)
        )
        self.memory_projection = memory_projection
        layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=feedforward_size or hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.token_head = nn.Linear(hidden_size, vocab_size)

    def _memory(
        self,
        visual_tokens: Tensor,
        prompt_context: Tensor,
        visual_padding_mask: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        visual = self.memory_projection(visual_tokens)
        memory = torch.cat((visual, prompt_context), dim=1)
        if visual_padding_mask is None:
            return memory, None
        prompt_mask = torch.zeros(
            (visual_padding_mask.shape[0], prompt_context.shape[1]),
            dtype=torch.bool,
            device=visual_padding_mask.device,
        )
        return memory, torch.cat((visual_padding_mask, prompt_mask), dim=1)

    def forward(
        self,
        input_ids: Tensor,
        visual_tokens: Tensor,
        visual_padding_mask: Optional[Tensor] = None,
        prompt_context: Optional[Tensor] = None,
        prompt_attention: Optional[Tensor] = None,
    ) -> VariableLayoutOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,T].")
        if input_ids.shape[1] > self.max_layout_tokens:
            raise ValueError("input_ids exceed max_layout_tokens.")
        if prompt_context is None:
            prompt_context, prompt_attention = self.prompt_attention(
                visual_tokens, visual_padding_mask, return_attention=False
            )
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        target = self.embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)
        memory, memory_padding_mask = self._memory(
            visual_tokens, prompt_context, visual_padding_mask
        )
        hidden = self.output_norm(
            self.decoder(
                target,
                memory,
                tgt_mask=causal_mask(input_ids.shape[1], input_ids.device),
                memory_key_padding_mask=memory_padding_mask,
            )
        )
        return VariableLayoutOutput(
            logits=self.token_head(hidden),
            hidden_states=hidden,
            prompt_context=prompt_context,
            prompt_attention=prompt_attention,
        )

    @torch.no_grad()
    def generate(
        self,
        visual_tokens: Tensor,
        *,
        layout_id: int,
        region_id: int,
        eos_id: int,
        pad_id: int,
        max_layout_tokens: Optional[int] = None,
        visual_padding_mask: Optional[Tensor] = None,
    ) -> VariableLayoutOutput:
        limit = max_layout_tokens or self.max_layout_tokens
        if limit < 2:
            raise ValueError("max_layout_tokens must be at least 2.")
        batch = visual_tokens.shape[0]
        device = visual_tokens.device
        prompt_context, prompt_attention = self.prompt_attention(
            visual_tokens, visual_padding_mask, return_attention=False
        )
        generated = torch.full((batch, 1), layout_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        all_hidden: list[Tensor] = []
        region_positions: list[list[int]] = [[] for _ in range(batch)]
        for _ in range(limit - 1):
            output = self.forward(
                generated,
                visual_tokens,
                visual_padding_mask,
                prompt_context,
                prompt_attention,
            )
            next_ids = output.logits[:, -1].argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, pad_id), next_ids)
            all_hidden.append(output.hidden_states[:, -1])
            for row, token_id in enumerate(next_ids.tolist()):
                if not finished[row] and token_id == region_id:
                    region_positions[row].append(generated.shape[1])
            generated = torch.cat((generated, next_ids[:, None]), dim=1)
            finished |= next_ids.eq(eos_id)
            if bool(finished.all()):
                break
        hidden_states = torch.cat(
            [self.forward(generated, visual_tokens, visual_padding_mask, prompt_context).hidden_states],
            dim=1,
        )
        return VariableLayoutOutput(
            logits=self.forward(generated, visual_tokens, visual_padding_mask, prompt_context).logits,
            hidden_states=hidden_states,
            prompt_context=prompt_context,
            prompt_attention=prompt_attention,
            region_positions=region_positions,
            generated_eos=finished,
            truncated=~finished,
            generated_ids=generated,
        )


class LayoutRecordHeads(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_types: int = 3,
        num_directions: int = 5,
        with_count: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_types < 1 or num_directions < 2:
            raise ValueError("invalid layout head dimensions")
        self.bbox_head = nn.Linear(hidden_size, 4)
        self.type_head = nn.Linear(hidden_size, num_types)
        self.direction_head = nn.Linear(hidden_size, num_directions)
        self.count_head = nn.Linear(hidden_size, 1) if with_count else None

    def forward(self, region_hidden: Tensor, page_hidden: Optional[Tensor] = None) -> LayoutRecordOutput:
        if region_hidden.ndim != 3:
            raise ValueError("region_hidden must have shape [B,R,D].")
        bbox = self.bbox_head(region_hidden).sigmoid()
        type_logits = self.type_head(region_hidden)
        direction_logits = self.direction_head(region_hidden)
        if self.count_head is None:
            count = region_hidden.new_zeros((region_hidden.shape[0], 1))
        else:
            if page_hidden is None:
                page_hidden = region_hidden[:, 0]
            count = self.count_head(page_hidden).squeeze(-1)
        return LayoutRecordOutput(bbox, type_logits, direction_logits, count)


def layout_prompt_diversity_loss(prompt_context: Tensor) -> Tensor:
    """Optional diagnostic regularizer; zero for a single prompt."""
    if prompt_context.shape[1] < 2:
        return prompt_context.new_zeros(())
    normalized = F.normalize(prompt_context.float(), dim=-1)
    similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
    eye = torch.eye(similarity.shape[-1], device=similarity.device, dtype=torch.bool)
    return similarity.masked_select(~eye).pow(2).mean()


class VariableLayoutLoss(nn.Module):
    def __init__(
        self,
        bbox_weight: float = 5.0,
        type_weight: float = 1.0,
        direction_weight: float = 1.0,
        count_weight: float = 0.1,
        prompt_diversity_weight: float = 0.0,
    ) -> None:
        super().__init__()
        weights = (bbox_weight, type_weight, direction_weight, count_weight, prompt_diversity_weight)
        if any(weight < 0 for weight in weights):
            raise ValueError("loss weights must be non-negative")
        self.weights = weights

    def forward(
        self,
        output: VariableLayoutOutput,
        target_ids: Tensor,
        record_output: LayoutRecordOutput,
        bbox_targets: Tensor,
        type_targets: Tensor,
        direction_targets: Tensor,
        record_mask: Tensor,
        count_targets: Tensor,
        pad_id: int,
    ) -> VariableLayoutLossOutput:
        sequence_loss = F.cross_entropy(
            output.logits[:, :-1].reshape(-1, output.logits.shape[-1]),
            target_ids[:, 1:].reshape(-1),
            ignore_index=pad_id,
        )
        mask = record_mask.to(dtype=record_output.bbox.dtype)
        bbox_l1 = (record_output.bbox - bbox_targets).abs().mean(dim=-1)
        bbox_loss = (bbox_l1 * mask).sum() / mask.sum().clamp_min(1.0)
        def masked_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
            valid = targets.ne(-100)
            if not bool(valid.any()):
                return logits.sum() * 0.0
            return F.cross_entropy(logits[valid], targets[valid])

        type_loss = masked_cross_entropy(
            record_output.type_logits, type_targets
        )
        direction_loss = masked_cross_entropy(
            record_output.direction_logits, direction_targets
        )
        count_loss = F.smooth_l1_loss(record_output.count.float(), count_targets.float())
        diversity = layout_prompt_diversity_loss(output.prompt_context)
        total = (
            sequence_loss
            + self.weights[0] * bbox_loss
            + self.weights[1] * type_loss
            + self.weights[2] * direction_loss
            + self.weights[3] * count_loss
            + self.weights[4] * diversity
        )
        return VariableLayoutLossOutput(
            total, sequence_loss, bbox_loss, type_loss, direction_loss, count_loss, diversity
        )
