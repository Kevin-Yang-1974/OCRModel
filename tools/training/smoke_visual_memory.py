#!/usr/bin/env python3
"""Bounded CPU/CUDA smoke for Vary intermediate memory and gradient paths."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from GOT.model.layout_prompt_decoder import LayoutVocabulary, PromptedVariableLayoutAdapter
from GOT.model.vision_encoder.vary_b import ImageEncoderViT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--ddp-backend", choices=("gloo", "nccl"), default=None)
    parser.add_argument("--max-layout-tokens", type=int, default=8)
    return parser.parse_args()


def configure_distributed(device: torch.device, backend: str | None) -> tuple[int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size == 1:
        return rank, world_size
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    selected_backend = backend or ("nccl" if device.type == "cuda" else "gloo")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=selected_backend)
    return rank, world_size


def make_layout_inputs(vocabulary: LayoutVocabulary, device: torch.device) -> tuple[torch.Tensor, ...]:
    ids = vocabulary.encode([
        "<LAYOUT>", "<REGION>", "<TYPE>", "COLUMN", "</TYPE>", "</REGION>", "<EOS>",
    ])
    layout_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention = torch.ones_like(layout_ids, dtype=torch.bool)
    positions = torch.tensor([[1]], dtype=torch.long, device=device)
    record_mask = torch.ones((1, 1), dtype=torch.bool, device=device)
    bbox = torch.tensor([[[0.1, 0.1, 0.4, 0.8]]], dtype=torch.float32, device=device)
    type_targets = torch.tensor([[0]], dtype=torch.long, device=device)
    direction_targets = torch.tensor([[0]], dtype=torch.long, device=device)
    count_targets = torch.tensor([1.0], dtype=torch.float32, device=device)
    return (
        layout_ids, attention, positions, record_mask, bbox,
        type_targets, direction_targets, count_targets,
    )


def run_resolution(
    encoder: ImageEncoderViT,
    resolution: str,
    image: torch.Tensor,
    device: torch.device,
    max_layout_tokens: int,
) -> dict[str, object]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    outputs = encoder(image, return_intermediate=True)
    ocr = outputs["ocr_features_16"].flatten(2).permute(0, 2, 1)
    high = outputs["layout_memory_64"] if resolution == "64" else outputs["ocr_features_16"]
    high = high.flatten(2).permute(0, 2, 1)
    expected_dim = 256 if resolution == "64" else 1024
    if high.shape != (1, 4096 if resolution == "64" else 256, expected_dim):
        raise RuntimeError(f"unexpected {resolution} memory shape: {tuple(high.shape)}")
    adapter = PromptedVariableLayoutAdapter(
        visual_dim=1024,
        high_resolution_dim=expected_dim,
        hidden_size=64,
        num_prompt_queries=4,
        decoder_layers=1,
        num_heads=4,
        max_layout_tokens=max_layout_tokens,
        max_layout_records=4,
        use_spatial_memory=True,
        gate_init=0.0,
    ).to(device=device, dtype=image.dtype)
    vocabulary = adapter.vocabulary
    layout_inputs = make_layout_inputs(vocabulary, device)
    visual_mask = torch.zeros((1, ocr.shape[1]), dtype=torch.bool, device=device)
    high_mask = torch.zeros((1, high.shape[1]), dtype=torch.bool, device=device)
    result = adapter(
        ocr,
        high,
        layout_input_ids=layout_inputs[0],
        layout_attention_mask=layout_inputs[1],
        layout_region_positions=layout_inputs[2],
        layout_record_mask=layout_inputs[3],
        layout_bbox_targets=layout_inputs[4],
        layout_type_targets=layout_inputs[5],
        layout_direction_targets=layout_inputs[6],
        layout_count_targets=layout_inputs[7],
        visual_padding_mask=visual_mask,
        high_resolution_padding_mask=high_mask,
    )
    if result.losses is None or not torch.isfinite(result.losses.loss):
        raise RuntimeError(f"{resolution} layout loss is non-finite")
    loss = result.losses.loss + result.visual_tokens.float().square().mean() * 0.01
    loss.backward()
    encoder_grad = encoder.patch_embed.proj.weight.grad
    adapter_grad = adapter.decoder.prompt_attention.prompt_bank.prompts.grad
    if encoder_grad is None or adapter_grad is None:
        raise RuntimeError(f"{resolution} gradient path is disconnected")
    encoder_grad_norm = float(encoder_grad.float().norm().detach().cpu())
    adapter_grad_norm = float(adapter_grad.float().norm().detach().cpu())
    if not all(torch.isfinite(torch.tensor(value)) and value > 0 for value in (encoder_grad_norm, adapter_grad_norm)):
        raise RuntimeError(f"{resolution} gradient is zero or non-finite")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "resolution": resolution,
        "ocr_tokens": list(ocr.shape),
        "layout_memory": list(high.shape),
        "encoder_gradient_norm": encoder_grad_norm,
        "adapter_gradient_norm": adapter_grad_norm,
        "loss": float(loss.detach().cpu()),
        "peak_memory_bytes": peak,
        "elapsed_ms": elapsed_ms,
    }


def main() -> int:
    args = parse_args()
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    rank, world_size = configure_distributed(device, args.ddp_backend)
    if device.type == "cuda" and world_size > 1:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    encoder = ImageEncoderViT(
        img_size=64,
        patch_size=1,
        embed_dim=32,
        depth=1,
        num_heads=4,
        out_chans=256,
    ).to(device=device, dtype=dtype)
    image = torch.randn(1, 3, 64, 64, device=device, dtype=dtype)
    smoke_results = []
    for resolution in ("16", "64"):
        encoder.zero_grad(set_to_none=True)
        smoke_results.append(run_resolution(encoder, resolution, image, device, args.max_layout_tokens))
    checkpoint_path = args.checkpoint or Path("/tmp/visual_memory_smoke.pt")
    torch.save(encoder.state_dict(), checkpoint_path)
    reloaded = ImageEncoderViT(
        img_size=64, patch_size=1, embed_dim=32, depth=1, num_heads=4, out_chans=256
    ).to(device=device, dtype=dtype)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if world_size != 1:
        value = torch.tensor([smoke_results[-1]["encoder_gradient_norm"]], device=device)
        torch.distributed.all_reduce(value)
        if rank == 0:
            smoke_results[-1]["ddp_gradient_sum"] = float(value.cpu())
    payload = {
        "status": "ok",
        "rank": rank,
        "device": str(device),
        "dtype": args.dtype,
        "results": smoke_results,
        "checkpoint": str(checkpoint_path),
        "world_size": world_size,
    }
    if rank == 0:
        print(json.dumps(payload, separators=(",", ":")))
    if world_size != 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
