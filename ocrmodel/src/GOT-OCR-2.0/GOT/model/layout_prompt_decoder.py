"""Prompted variable-length layout decoding and GOT2 integration primitives."""

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
    layout_evidence: Tensor
    prompt_attention: Optional[Tensor] = None
    region_positions: Optional[list[list[int]]] = None
    generated_eos: Optional[Tensor] = None
    truncated: Optional[Tensor] = None
    truncated_by_max_layout_tokens: Optional[Tensor] = None
    stopped_by_max_layout_records: Optional[Tensor] = None
    num_generated_regions: Optional[Tensor] = None
    num_layout_tokens: Optional[Tensor] = None
    region_token_probabilities: Optional[Tensor] = None
    sequence_log_probability: Optional[Tensor] = None
    coverage_summary: Optional[Tensor] = None
    coverage_region_counts: Optional[Tensor] = None
    generated_ids: Optional[Tensor] = None

    @property
    def prompt_context(self) -> Tensor:
        """Compatibility alias; new code should use ``layout_evidence``."""
        return self.layout_evidence


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
    bbox_l1_loss: Tensor
    bbox_giou_loss: Tensor
    type_loss: Tensor
    direction_loss: Tensor
    count_loss: Tensor
    prompt_diversity_loss: Tensor
    eos_accuracy: Tensor
    region_count_mae: Tensor


@dataclass
class PromptedVariableLayoutOutput:
    visual_tokens: Tensor
    layout_evidence: Tensor
    decoder_output: Optional[VariableLayoutOutput] = None
    record_output: Optional[LayoutRecordOutput] = None
    record_mask: Optional[Tensor] = None
    losses: Optional[VariableLayoutLossOutput] = None


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


class LayoutTokenFSM:
    """Finite-state grammar for the actual PVLD structural vocabulary."""

    EXPECT_LAYOUT = 0
    RECORD_BOUNDARY = 1
    EXPECT_TYPE_OPEN = 2
    EXPECT_TYPE_VALUE = 3
    EXPECT_TYPE_CLOSE = 4
    EXPECT_REGION_CLOSE = 5
    AFTER_EOS = 6

    def __init__(self, vocabulary: LayoutVocabulary) -> None:
        self.vocabulary = vocabulary
        self.type_value_ids = tuple(
            vocabulary.token_to_id[name]
            for name in ("COLUMN", "ROW", "REGION")
        )

    def allowed_token_ids(
        self,
        state: int,
        *,
        region_count: int,
        max_layout_records: Optional[int] = None,
    ) -> tuple[int, ...]:
        vocabulary = self.vocabulary
        if state == self.EXPECT_LAYOUT:
            return (vocabulary.layout_id,)
        if state == self.RECORD_BOUNDARY:
            if max_layout_records is not None and region_count >= max_layout_records:
                return (vocabulary.eos_id,)
            return (vocabulary.region_id, vocabulary.eos_id)
        if state == self.EXPECT_TYPE_OPEN:
            return (vocabulary.type_id,)
        if state == self.EXPECT_TYPE_VALUE:
            return self.type_value_ids
        if state == self.EXPECT_TYPE_CLOSE:
            return (vocabulary.end_type_id,)
        if state == self.EXPECT_REGION_CLOSE:
            return (vocabulary.end_region_id,)
        if state == self.AFTER_EOS:
            return (vocabulary.pad_id,)
        raise ValueError(f"Unknown layout FSM state: {state}.")

    def transition(self, state: int, token_id: int) -> tuple[int, int]:
        vocabulary = self.vocabulary
        if token_id not in self.allowed_token_ids(state, region_count=0):
            raise ValueError(
                f"Illegal layout token {vocabulary.id_to_token.get(token_id, token_id)!r} "
                f"in FSM state {state}."
            )
        if state == self.EXPECT_LAYOUT:
            return self.RECORD_BOUNDARY, 0
        if state == self.RECORD_BOUNDARY:
            return (
                (self.EXPECT_TYPE_OPEN, 1)
                if token_id == vocabulary.region_id
                else (self.AFTER_EOS, 0)
            )
        if state == self.EXPECT_TYPE_OPEN:
            return self.EXPECT_TYPE_VALUE, 0
        if state == self.EXPECT_TYPE_VALUE:
            return self.EXPECT_TYPE_CLOSE, 0
        if state == self.EXPECT_TYPE_CLOSE:
            return self.EXPECT_REGION_CLOSE, 0
        if state == self.EXPECT_REGION_CLOSE:
            return self.RECORD_BOUNDARY, 0
        return self.AFTER_EOS, 0

    def states_after_prefix(self, input_ids: Tensor) -> tuple[list[int], list[int]]:
        states: list[int] = []
        counts: list[int] = []
        for row in input_ids.detach().cpu().tolist():
            state = self.EXPECT_LAYOUT
            count = 0
            for token_id in row:
                state, increment = self.transition(state, int(token_id))
                count += increment
            states.append(state)
            counts.append(count)
        return states, counts

    def mask_logits(
        self,
        logits: Tensor,
        input_ids: Tensor,
        *,
        max_layout_records: Optional[int] = None,
    ) -> Tensor:
        if logits.shape[:2] != input_ids.shape:
            raise ValueError("FSM logits and input_ids must share [B,T].")
        masked = logits.clone()
        for row_index, row in enumerate(input_ids.detach().cpu().tolist()):
            state = self.EXPECT_LAYOUT
            count = 0
            for position, token_id in enumerate(row):
                state, increment = self.transition(state, int(token_id))
                count += increment
                allowed = self.allowed_token_ids(
                    state,
                    region_count=count,
                    max_layout_records=max_layout_records,
                )
                invalid = torch.ones(
                    logits.shape[-1], dtype=torch.bool, device=logits.device
                )
                invalid[list(allowed)] = False
                masked[row_index, position].masked_fill_(invalid, float("-inf"))
        return masked


def causal_mask(length: int, device: torch.device) -> Tensor:
    if length < 1:
        raise ValueError("causal mask length must be positive.")
    return torch.triu(torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1)


class CausalLayoutDecoderBlock(nn.Module):
    """Pre-norm causal self-attention, layout-memory attention, and FFN."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        feedforward_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.coverage_norm = nn.LayerNorm(hidden_size)
        self.coverage_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.cross_norm = nn.LayerNorm(hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, hidden_size),
        )

    @staticmethod
    def previous_region_memory(hidden: Tensor, region_mask: Tensor) -> Tensor:
        weighted = hidden * region_mask.unsqueeze(-1).to(hidden.dtype)
        cumulative = weighted.cumsum(dim=1) - weighted
        counts = region_mask.cumsum(dim=1) - region_mask.long()
        return cumulative / counts.clamp_min(1).unsqueeze(-1).to(hidden.dtype)

    def forward(
        self,
        hidden: Tensor,
        memory: Tensor,
        *,
        attention_mask: Tensor,
        target_padding_mask: Optional[Tensor],
        memory_padding_mask: Optional[Tensor],
        region_mask: Tensor,
    ) -> Tensor:
        normalized = self.self_norm(hidden)
        self_update, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            key_padding_mask=target_padding_mask,
            need_weights=False,
        )
        hidden_self = hidden + self_update
        coverage = self.previous_region_memory(hidden_self, region_mask)
        hidden_coverage = hidden_self + self.coverage_projection(
            self.coverage_norm(coverage)
        )
        normalized = self.cross_norm(hidden_coverage)
        memory_update, _ = self.cross_attention(
            normalized,
            memory,
            memory,
            key_padding_mask=memory_padding_mask,
            need_weights=False,
        )
        hidden_memory = hidden_coverage + memory_update
        return hidden_memory + self.ffn(self.ffn_norm(hidden_memory))


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
        vocabulary: Optional[LayoutVocabulary] = None,
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
        self.vocabulary = vocabulary or LayoutVocabulary()
        if self.vocabulary.vocab_size != vocab_size:
            raise ValueError("vocab_size does not match the supplied LayoutVocabulary.")
        self.fsm = LayoutTokenFSM(self.vocabulary)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_layout_tokens, hidden_size)
        self.prompt_attention = LayoutPromptCrossAttention(
            visual_size, hidden_size, num_prompts, num_heads, dropout
        )
        self.decoder_blocks = nn.ModuleList(
            CausalLayoutDecoderBlock(
                hidden_size,
                num_heads,
                feedforward_size or hidden_size * 4,
                dropout,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.token_head = nn.Linear(hidden_size, vocab_size)

    def _memory(
        self,
        layout_evidence: Tensor,
    ) -> tuple[Tensor, Optional[Tensor]]:
        return layout_evidence, None

    def forward(
        self,
        input_ids: Tensor,
        visual_tokens: Tensor,
        visual_padding_mask: Optional[Tensor] = None,
        layout_evidence: Optional[Tensor] = None,
        prompt_attention: Optional[Tensor] = None,
        target_padding_mask: Optional[Tensor] = None,
        max_layout_records: Optional[int] = None,
    ) -> VariableLayoutOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [B,T].")
        if input_ids.shape[1] > self.max_layout_tokens:
            raise ValueError("input_ids exceed max_layout_tokens.")
        if layout_evidence is None:
            layout_evidence, prompt_attention = self.prompt_attention(
                visual_tokens, visual_padding_mask, return_attention=False
            )
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        target = self.embedding(input_ids) + self.position_embedding(positions).unsqueeze(0)
        memory, memory_padding_mask = self._memory(layout_evidence)
        hidden = target
        region_mask = input_ids.eq(self.vocabulary.region_id)
        mask = causal_mask(input_ids.shape[1], input_ids.device)
        for block in self.decoder_blocks:
            hidden = block(
                hidden,
                memory,
                attention_mask=mask,
                target_padding_mask=target_padding_mask,
                memory_padding_mask=memory_padding_mask,
                region_mask=region_mask,
            )
        hidden = self.output_norm(hidden)
        if target_padding_mask is not None:
            hidden = hidden.masked_fill(target_padding_mask.unsqueeze(-1), 0.0)
        logits = self.fsm.mask_logits(
            self.token_head(hidden),
            input_ids,
            max_layout_records=max_layout_records,
        )
        coverage_counts = region_mask.sum(dim=1)
        coverage_summary = (
            (hidden * region_mask.unsqueeze(-1).to(hidden.dtype)).sum(dim=1)
            / coverage_counts.clamp_min(1).unsqueeze(-1).to(hidden.dtype)
        )
        return VariableLayoutOutput(
            logits=logits,
            hidden_states=hidden,
            layout_evidence=layout_evidence,
            prompt_attention=prompt_attention,
            coverage_summary=coverage_summary,
            coverage_region_counts=coverage_counts,
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
        max_layout_records: Optional[int] = None,
        visual_padding_mask: Optional[Tensor] = None,
        layout_evidence: Optional[Tensor] = None,
    ) -> VariableLayoutOutput:
        limit = max_layout_tokens or self.max_layout_tokens
        if limit < 2:
            raise ValueError("max_layout_tokens must be at least 2.")
        if limit > self.max_layout_tokens:
            raise ValueError("generation limit exceeds configured max_layout_tokens.")
        if max_layout_records is not None and max_layout_records < 0:
            raise ValueError("max_layout_records must be non-negative.")
        batch = visual_tokens.shape[0]
        device = visual_tokens.device
        prompt_attention = None
        if layout_evidence is None:
            layout_evidence, prompt_attention = self.prompt_attention(
                visual_tokens, visual_padding_mask, return_attention=False
            )
        generated = torch.full((batch, 1), layout_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        region_positions: list[list[int]] = [[] for _ in range(batch)]
        region_probabilities: list[list[float]] = [[] for _ in range(batch)]
        region_counts = torch.zeros(batch, dtype=torch.long, device=device)
        stopped_by_records = torch.zeros(batch, dtype=torch.bool, device=device)
        sequence_log_probability = torch.zeros(
            batch, dtype=layout_evidence.dtype, device=device
        )
        for _ in range(limit - 1):
            step_output = self.forward(
                generated,
                visual_tokens,
                visual_padding_mask,
                layout_evidence,
                prompt_attention,
                generated.eq(pad_id),
                max_layout_records=max_layout_records,
            )
            step_logits = step_output.logits[:, -1]
            probabilities = step_logits.softmax(dim=-1)
            states, counts = self.fsm.states_after_prefix(generated)
            if max_layout_records is not None:
                for row, (state, count) in enumerate(zip(states, counts)):
                    if (
                        not bool(finished[row])
                        and state == self.fsm.RECORD_BOUNDARY
                        and count >= max_layout_records
                    ):
                        stopped_by_records[row] = True
            next_ids = step_logits.argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, pad_id), next_ids)
            for row, token_id in enumerate(next_ids.tolist()):
                if not finished[row] and token_id == region_id:
                    region_positions[row].append(generated.shape[1])
                    region_probabilities[row].append(
                        float(probabilities[row, region_id].detach().cpu())
                    )
            active = ~finished
            selected_probability = probabilities.gather(1, next_ids[:, None]).squeeze(1)
            sequence_log_probability = sequence_log_probability + torch.where(
                active,
                selected_probability.clamp_min(torch.finfo(probabilities.dtype).tiny).log(),
                torch.zeros_like(selected_probability),
            )
            generated = torch.cat((generated, next_ids[:, None]), dim=1)
            region_counts += active & next_ids.eq(region_id)
            finished |= next_ids.eq(eos_id)
            if bool(finished.all()):
                break
        final_output = self.forward(
            generated,
            visual_tokens,
            visual_padding_mask,
            layout_evidence,
            prompt_attention,
            generated.eq(pad_id),
            max_layout_records=max_layout_records,
        )
        probability_width = max((len(row) for row in region_probabilities), default=0)
        region_probability_tensor = layout_evidence.new_zeros((batch, probability_width))
        for row, values in enumerate(region_probabilities):
            if values:
                region_probability_tensor[row, : len(values)] = torch.tensor(
                    values, dtype=layout_evidence.dtype, device=device
                )
        truncated = ~finished
        return VariableLayoutOutput(
            logits=final_output.logits,
            hidden_states=final_output.hidden_states,
            layout_evidence=layout_evidence,
            prompt_attention=prompt_attention,
            region_positions=region_positions,
            generated_eos=finished,
            truncated=truncated,
            truncated_by_max_layout_tokens=truncated,
            stopped_by_max_layout_records=stopped_by_records,
            num_generated_regions=region_counts,
            num_layout_tokens=generated.ne(pad_id).sum(dim=1),
            region_token_probabilities=region_probability_tensor,
            sequence_log_probability=sequence_log_probability,
            coverage_summary=final_output.coverage_summary,
            coverage_region_counts=final_output.coverage_region_counts,
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
        raw_bbox = self.bbox_head(region_hidden).sigmoid()
        x0 = torch.minimum(raw_bbox[..., 0], raw_bbox[..., 2])
        y0 = torch.minimum(raw_bbox[..., 1], raw_bbox[..., 3])
        x1 = torch.maximum(raw_bbox[..., 0], raw_bbox[..., 2])
        y1 = torch.maximum(raw_bbox[..., 1], raw_bbox[..., 3])
        bbox = torch.stack((x0, y0, x1, y1), dim=-1)
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
        bbox_giou_weight: float = 2.0,
        type_weight: float = 1.0,
        direction_weight: float = 1.0,
        count_weight: float = 0.1,
        prompt_diversity_weight: float = 0.0,
    ) -> None:
        super().__init__()
        weights = (
            bbox_weight,
            bbox_giou_weight,
            type_weight,
            direction_weight,
            count_weight,
            prompt_diversity_weight,
        )
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
        bbox_l1_loss = (bbox_l1 * mask).sum() / mask.sum().clamp_min(1.0)
        predicted = record_output.bbox
        target = bbox_targets
        intersection_x0 = torch.maximum(predicted[..., 0], target[..., 0])
        intersection_y0 = torch.maximum(predicted[..., 1], target[..., 1])
        intersection_x1 = torch.minimum(predicted[..., 2], target[..., 2])
        intersection_y1 = torch.minimum(predicted[..., 3], target[..., 3])
        intersection = (
            (intersection_x1 - intersection_x0).clamp_min(0)
            * (intersection_y1 - intersection_y0).clamp_min(0)
        )
        predicted_area = (
            (predicted[..., 2] - predicted[..., 0]).clamp_min(0)
            * (predicted[..., 3] - predicted[..., 1]).clamp_min(0)
        )
        target_area = (
            (target[..., 2] - target[..., 0]).clamp_min(0)
            * (target[..., 3] - target[..., 1]).clamp_min(0)
        )
        union = predicted_area + target_area - intersection
        iou = intersection / union.clamp_min(1e-7)
        enclosing = (
            (torch.maximum(predicted[..., 2], target[..., 2])
             - torch.minimum(predicted[..., 0], target[..., 0])).clamp_min(0)
            * (torch.maximum(predicted[..., 3], target[..., 3])
               - torch.minimum(predicted[..., 1], target[..., 1])).clamp_min(0)
        )
        giou_loss = 1.0 - (iou - (enclosing - union) / enclosing.clamp_min(1e-7))
        bbox_giou_loss = (giou_loss * mask).sum() / mask.sum().clamp_min(1.0)
        bbox_loss = self.weights[0] * bbox_l1_loss + self.weights[1] * bbox_giou_loss
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
        diversity = layout_prompt_diversity_loss(output.layout_evidence)
        eos_targets = target_ids[:, 1:].eq(6)
        eos_accuracy = (
            output.logits[:, :-1].argmax(dim=-1)[eos_targets].eq(6).float().mean()
            if bool(eos_targets.any())
            else output.logits.new_zeros(())
        )
        region_count_mae = (record_output.count.float() - count_targets.float()).abs().mean()
        total = (
            sequence_loss
            + bbox_loss
            + self.weights[2] * type_loss
            + self.weights[3] * direction_loss
            + self.weights[4] * count_loss
            + self.weights[5] * diversity
        )
        return VariableLayoutLossOutput(
            total,
            sequence_loss,
            bbox_loss,
            bbox_l1_loss,
            bbox_giou_loss,
            type_loss,
            direction_loss,
            count_loss,
            diversity,
            eos_accuracy,
            region_count_mae,
        )


class PromptedVariableLayoutAdapter(nn.Module):
    """End-to-end PVLD branch with visual-only OCR content routing.

    ``F -> layout_evidence`` and ``layout_evidence -> layout decoder`` form the
    layout branch.  OCR receives a zero-gated visual residual whose content
    Value is derived only from projected GOT2 visual tokens.
    """

    def __init__(
        self,
        visual_dim: int = 1024,
        high_resolution_dim: int = 1024,
        hidden_size: int = 256,
        num_prompt_queries: int = 32,
        decoder_layers: int = 2,
        num_heads: int = 8,
        max_layout_tokens: int = 2048,
        max_layout_records: int = 512,
        num_types: int = 3,
        num_directions: int = 5,
        dropout: float = 0.0,
        gate_init: float = 0.0,
        bbox_weight: float = 5.0,
        bbox_giou_weight: float = 2.0,
        type_weight: float = 1.0,
        direction_weight: float = 1.0,
        count_weight: float = 0.1,
        prompt_diversity_weight: float = 0.0,
    ) -> None:
        super().__init__()
        from GOT.model.layout_query import VisualValueLayoutRouting

        self.vocabulary = LayoutVocabulary()
        self.num_prompt_queries = num_prompt_queries
        self.max_layout_tokens = max_layout_tokens
        self.max_layout_records = max_layout_records
        self.decoder = VariableLayoutDecoder(
            vocab_size=self.vocabulary.vocab_size,
            hidden_size=hidden_size,
            visual_size=high_resolution_dim,
            num_prompts=num_prompt_queries,
            num_layers=decoder_layers,
            num_heads=num_heads,
            max_layout_tokens=max_layout_tokens,
            dropout=dropout,
            vocabulary=self.vocabulary,
        )
        self.record_heads = LayoutRecordHeads(
            hidden_size,
            num_types=num_types,
            num_directions=num_directions,
        )
        self.visual_norm = nn.LayerNorm(visual_dim)
        self.visual_projection = nn.Linear(visual_dim, hidden_size)
        self.visual_routing = VisualValueLayoutRouting(hidden_size, num_heads, dropout)
        self.writeback_output = nn.Linear(hidden_size, visual_dim)
        self.residual_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.criterion = VariableLayoutLoss(
            bbox_weight=bbox_weight,
            bbox_giou_weight=bbox_giou_weight,
            type_weight=type_weight,
            direction_weight=direction_weight,
            count_weight=count_weight,
            prompt_diversity_weight=prompt_diversity_weight,
        )

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                module._reset_parameters()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.decoder.prompt_attention.prompt_bank.prompts, mean=0.0, std=0.02)
        nn.init.zeros_(self.residual_gate)

    @staticmethod
    def _gather_records(
        hidden_states: Tensor,
        region_positions: Tensor,
        record_mask: Tensor,
    ) -> Tensor:
        indices = region_positions.clamp_min(0).unsqueeze(-1).expand(
            -1, -1, hidden_states.shape[-1]
        )
        gathered = hidden_states.gather(1, indices)
        return gathered.masked_fill(~record_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        visual_tokens: Tensor,
        high_resolution_features: Tensor,
        *,
        layout_input_ids: Optional[Tensor] = None,
        layout_attention_mask: Optional[Tensor] = None,
        layout_region_positions: Optional[Tensor] = None,
        layout_record_mask: Optional[Tensor] = None,
        layout_bbox_targets: Optional[Tensor] = None,
        layout_type_targets: Optional[Tensor] = None,
        layout_direction_targets: Optional[Tensor] = None,
        layout_count_targets: Optional[Tensor] = None,
        visual_padding_mask: Optional[Tensor] = None,
        high_resolution_padding_mask: Optional[Tensor] = None,
        generate_layout: bool = False,
    ) -> PromptedVariableLayoutOutput:
        layout_evidence, _ = self.decoder.prompt_attention(
            high_resolution_features,
            high_resolution_padding_mask,
            return_attention=False,
        )
        projected_visual = self.visual_projection(self.visual_norm(visual_tokens))
        routed, _ = self.visual_routing(
            projected_visual,
            layout_evidence,
            visual_padding_mask=visual_padding_mask,
        )
        visual_output = visual_tokens + torch.tanh(self.residual_gate) * self.writeback_output(routed)

        decoder_output = None
        record_output = None
        losses = None
        record_mask = layout_record_mask
        if layout_input_ids is not None:
            decoder_output = self.decoder(
                layout_input_ids,
                high_resolution_features,
                high_resolution_padding_mask,
                layout_evidence,
                target_padding_mask=(
                    ~layout_attention_mask.bool() if layout_attention_mask is not None else None
                ),
                max_layout_records=self.max_layout_records,
            )
            if layout_region_positions is None or layout_record_mask is None:
                raise ValueError("PVLD training requires REGION positions and record mask.")
            region_hidden = self._gather_records(
                decoder_output.hidden_states,
                layout_region_positions,
                layout_record_mask,
            )
            record_output = self.record_heads(region_hidden, layout_evidence.mean(dim=1))
            if layout_bbox_targets is not None:
                losses = self.criterion(
                    decoder_output,
                    layout_input_ids,
                    record_output,
                    layout_bbox_targets,
                    layout_type_targets,
                    layout_direction_targets,
                    layout_record_mask,
                    layout_count_targets,
                    self.vocabulary.pad_id,
                )
        elif generate_layout:
            decoder_output = self.decoder.generate(
                high_resolution_features,
                layout_id=self.vocabulary.layout_id,
                region_id=self.vocabulary.region_id,
                eos_id=self.vocabulary.eos_id,
                pad_id=self.vocabulary.pad_id,
                max_layout_tokens=self.max_layout_tokens,
                max_layout_records=self.max_layout_records,
                visual_padding_mask=high_resolution_padding_mask,
                layout_evidence=layout_evidence,
            )
            positions = decoder_output.region_positions or [[] for _ in range(visual_tokens.shape[0])]
            width = min(max((len(row) for row in positions), default=0), self.max_layout_records)
            region_positions = torch.zeros(
                (visual_tokens.shape[0], width), dtype=torch.long, device=visual_tokens.device
            )
            record_mask = torch.zeros_like(region_positions, dtype=torch.bool)
            for row, values in enumerate(positions):
                values = values[:width]
                if values:
                    region_positions[row, : len(values)] = torch.tensor(values, device=visual_tokens.device)
                    record_mask[row, : len(values)] = True
            region_hidden = self._gather_records(
                decoder_output.hidden_states, region_positions, record_mask
            )
            record_output = self.record_heads(region_hidden, layout_evidence.mean(dim=1))

        return PromptedVariableLayoutOutput(
            visual_tokens=visual_output,
            layout_evidence=layout_evidence,
            decoder_output=decoder_output,
            record_output=record_output,
            record_mask=record_mask,
            losses=losses,
        )
