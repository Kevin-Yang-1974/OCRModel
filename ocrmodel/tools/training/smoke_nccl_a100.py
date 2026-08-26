#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-mib", type=int, default=0)
    args = parser.parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    value = torch.tensor([float(rank + 1)], device=torch.device("cuda", local_rank))
    dist.all_reduce(value)
    expected = world * (world + 1) / 2
    if float(value.item()) != expected:
        raise RuntimeError(f"all_reduce mismatch: {value.item()} != {expected}")
    tensor_elements = args.tensor_mib * 1024 * 1024 // 4
    if tensor_elements:
        payload = torch.full(
            (tensor_elements,),
            float(rank + 1),
            device=torch.device("cuda", local_rank),
            dtype=torch.float32,
        )
        dist.broadcast(payload, src=0)
        dist.all_reduce(payload)
        expected_payload = float(world)
        if not (
            float(payload[0].item()) == expected_payload
            and float(payload[-1].item()) == expected_payload
        ):
            raise RuntimeError("large collective payload mismatch")
    if rank == 0:
        summary = {
            "status": "ok",
            "world_size": world,
            "sum": float(value.item()),
            "tensor_mib": args.tensor_mib,
        }
        args.output.write_text(json.dumps(summary) + "\n", encoding="utf-8")
        print(json.dumps(summary, separators=(",", ":")))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
