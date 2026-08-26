#!/usr/bin/env python3
"""Small CUDA, NCCL, and ZeRO-2 checks for the BSCC ARM64 environment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cuda_bf16", "nccl", "zero2"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write(path: Path, payload: dict) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, separators=(",", ":")))


def require_gradient(parameter: torch.nn.Parameter) -> float:
    gradient = parameter.grad
    if gradient is None or not torch.isfinite(gradient).all():
        raise RuntimeError("missing or non-finite gradient")
    norm = float(gradient.float().norm().item())
    if norm <= 0.0:
        raise RuntimeError("zero gradient")
    return norm


def cuda_bf16(output: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    model = torch.nn.Linear(256, 128).to(device=device, dtype=torch.bfloat16)
    values = torch.randn(4, 256, device=device, dtype=torch.bfloat16)
    loss = model(values).float().square().mean()
    loss.backward()
    write(output, {
        "status": "ok", "mode": "cuda_bf16", "torch": torch.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
        "dtype": "bfloat16", "loss": float(loss.detach().item()),
        "gradient_norm": require_gradient(model.weight),
    })


def nccl(output: Path) -> None:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    value = torch.tensor(
        [rank + 1.0], device=torch.device("cuda", local_rank)
    )
    dist.all_reduce(value)
    expected = world_size * (world_size + 1) / 2
    if float(value.item()) != expected:
        raise RuntimeError("NCCL all_reduce returned an unexpected value")
    write(output, {
        "status": "ok", "mode": "nccl", "backend": dist.get_backend(),
        "world_size": world_size, "all_reduce_sum": float(value.item()),
    })
    dist.destroy_process_group()


def zero2(output: Path) -> None:
    import deepspeed

    model = torch.nn.Sequential(
        torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 64)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config={
            "train_micro_batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "bf16": {"enabled": True},
            "zero_optimization": {"stage": 2},
        },
    )
    values = torch.randn(1, 256, device=engine.device, dtype=torch.bfloat16)
    loss = engine(values).float().square().mean()
    engine.backward(loss)
    engine.step()
    write(output, {
        "status": "ok", "mode": "zero2", "deepspeed": deepspeed.__version__,
        "world_size": dist.get_world_size(), "zero_stage": 2,
        "loss": float(loss.detach().item()),
    })


def main() -> None:
    args = parse_args()
    if args.mode == "cuda_bf16":
        cuda_bf16(args.output)
    elif args.mode == "nccl":
        nccl(args.output)
    else:
        zero2(args.output)


if __name__ == "__main__":
    main()
