#!/usr/bin/env python3
"""Bounded CUDA forward/backward smoke for the causal PVLD decoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def finite_gradient(parameter: torch.nn.Parameter, name: str) -> float:
    if parameter.grad is None or not torch.isfinite(parameter.grad).all():
        raise RuntimeError(f"missing or non-finite gradient: {name}")
    return float(parameter.grad.float().norm().item())


def padded(vocabulary, tokens: list[str], width: int) -> list[int]:
    values = vocabulary.encode(tokens)
    return values + [vocabulary.pad_id] * (width - len(values))


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.project_root.resolve()))
    from GOT.model.layout_prompt_decoder import PromptedVariableLayoutAdapter

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    torch.manual_seed(20260822)
    device = torch.device("cuda")
    adapter = PromptedVariableLayoutAdapter(
        visual_dim=32,
        high_resolution_dim=24,
        hidden_size=32,
        num_prompt_queries=32,
        decoder_layers=2,
        num_heads=4,
        max_layout_tokens=32,
        max_layout_records=3,
    ).to(device)
    vocabulary = adapter.vocabulary
    visual = torch.randn(3, 9, 32, device=device, requires_grad=True)
    high_resolution = torch.randn(3, 16, 24, device=device, requires_grad=True)

    with torch.no_grad():
        identity = adapter(visual, high_resolution)
        if not torch.equal(identity.visual_tokens, visual):
            raise RuntimeError("alpha=0 did not preserve the original GOT2 visual path.")
        adapter.residual_gate.fill_(0.25)

    width = 12
    input_ids = torch.tensor(
        [
            padded(vocabulary, ["<LAYOUT>", "<EOS>"], width),
            padded(
                vocabulary,
                ["<LAYOUT>", "<REGION>", "<TYPE>", "REGION", "</TYPE>", "</REGION>", "<EOS>"],
                width,
            ),
            padded(
                vocabulary,
                [
                    "<LAYOUT>", "<REGION>", "<TYPE>", "COLUMN", "</TYPE>", "</REGION>",
                    "<REGION>", "<TYPE>", "ROW", "</TYPE>", "</REGION>", "<EOS>",
                ],
                width,
            ),
        ],
        device=device,
    )
    region_positions = torch.tensor([[0, 0], [1, 0], [1, 6]], device=device)
    record_mask = torch.tensor(
        [[False, False], [True, False], [True, True]], device=device
    )
    bbox_targets = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.1, 0.1, 0.4, 0.5], [0.0, 0.0, 0.0, 0.0]],
            [[0.1, 0.1, 0.3, 0.8], [0.5, 0.2, 0.8, 0.9]],
        ],
        device=device,
    )
    type_targets = torch.tensor([[-100, -100], [2, -100], [0, 1]], device=device)
    direction_targets = torch.tensor([[-100, -100], [4, -100], [2, 3]], device=device)
    output = adapter(
        visual,
        high_resolution,
        layout_input_ids=input_ids,
        layout_attention_mask=input_ids.ne(vocabulary.pad_id),
        layout_region_positions=region_positions,
        layout_record_mask=record_mask,
        layout_bbox_targets=bbox_targets,
        layout_type_targets=type_targets,
        layout_direction_targets=direction_targets,
        layout_count_targets=torch.tensor([0.0, 1.0, 2.0], device=device),
    )
    if output.losses is None or not torch.isfinite(output.losses.loss):
        raise RuntimeError("PVLD loss is missing or non-finite.")
    if not torch.all((output.record_output.bbox >= 0) & (output.record_output.bbox <= 1)):
        raise RuntimeError("PVLD bbox escaped [0,1].")
    loss = output.losses.loss + output.visual_tokens.float().square().mean()
    loss.backward()
    first_block = adapter.decoder.decoder_blocks[0]
    gradients = {
        "causal_self_attention": finite_gradient(
            first_block.self_attention.in_proj_weight, "causal_self_attention"
        ),
        "cross_attention": finite_gradient(
            first_block.cross_attention.in_proj_weight, "cross_attention"
        ),
        "token_head": finite_gradient(adapter.decoder.token_head.weight, "token_head"),
        "coverage": finite_gradient(
            first_block.coverage_projection.weight, "coverage_projection"
        ),
        "region_bbox_head": finite_gradient(
            adapter.record_heads.bbox_head.weight, "bbox_head"
        ),
        "visual_value_routing": finite_gradient(
            adapter.visual_routing.visual_value.weight, "visual_value_routing"
        ),
    }

    with torch.no_grad():
        adapter.decoder.token_head.weight.zero_()
        adapter.decoder.token_head.weight[vocabulary.region_id, 0] = 0.25
        adapter.decoder.token_head.bias.zero_()
        adapter.decoder.token_head.bias[vocabulary.region_id] = 4.0
        adapter.decoder.token_head.bias[vocabulary.eos_id] = 1.0
        adapter.decoder.token_head.bias[vocabulary.token_to_id["REGION"]] = 0.5
    evidence, _ = adapter.decoder.prompt_attention(high_resolution[:1], return_attention=False)

    def generate(record_cap: int, token_cap: int = 32):
        return adapter.decoder.generate(
            high_resolution[:1],
            layout_id=vocabulary.layout_id,
            region_id=vocabulary.region_id,
            eos_id=vocabulary.eos_id,
            pad_id=vocabulary.pad_id,
            max_layout_tokens=token_cap,
            max_layout_records=record_cap,
            layout_evidence=evidence,
        )

    zero, one, multiple = generate(0), generate(1), generate(3)
    token_limited = generate(8, token_cap=3)
    for generated in (zero, one, multiple):
        adapter.decoder.fsm.states_after_prefix(generated.generated_ids)
        if not generated.generated_eos.item():
            raise RuntimeError("record-capped generation failed to emit EOS.")
    if [item.num_generated_regions.item() for item in (zero, one, multiple)] != [0, 1, 3]:
        raise RuntimeError("0/1/multiple REGION generation contract failed.")
    if not all(item.stopped_by_max_layout_records.item() for item in (zero, one, multiple)):
        raise RuntimeError("record-cap status was not reported.")
    if not token_limited.truncated_by_max_layout_tokens.item():
        raise RuntimeError("token-cap truncation status was not reported.")
    if token_limited.stopped_by_max_layout_records.item():
        raise RuntimeError("token cap was confused with record cap.")
    probabilities = multiple.region_token_probabilities
    if not torch.isfinite(probabilities).all() or not torch.all(
        (probabilities >= 0) & (probabilities <= 1)
    ):
        raise RuntimeError("REGION probabilities are invalid.")
    if probabilities.numel() > 1 and torch.allclose(
        probabilities, probabilities[:, :1].expand_as(probabilities)
    ):
        raise RuntimeError("REGION probabilities did not vary across decoding steps.")

    payload = {
        "status": "ok",
        "device": str(device),
        "loss": float(loss.detach().item()),
        "gradients": gradients,
        "generation": {
            "region_counts": [0, 1, 3],
            "generated_eos": [bool(item.generated_eos.item()) for item in (zero, one, multiple)],
            "stopped_by_max_layout_records": [
                bool(item.stopped_by_max_layout_records.item()) for item in (zero, one, multiple)
            ],
            "token_cap_truncated": bool(token_limited.truncated_by_max_layout_tokens.item()),
            "record_cap_distinct": not bool(token_limited.stopped_by_max_layout_records.item()),
            "fsm_sequences_valid": True,
            "region_probability_min": float(probabilities.min().item()),
            "region_probability_max": float(probabilities.max().item()),
            "region_probabilities_can_vary": True,
            "coverage_region_count": int(multiple.coverage_region_counts.item()),
        },
        "ocr_visual_value_source": "projected_visual_tokens_only",
        "alpha_zero_exact_identity": True,
        "formal_training_started": False,
        "frozen_test_started": False,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    main()
