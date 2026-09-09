from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedInfo:
    """Runtime information for the optional local multi-GPU DDP execution."""

    strategy: str
    rank: int
    local_rank: int
    world_size: int

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialize_distributed(strategy: str) -> DistributedInfo:
    """Initialize one local process group when launched by ``torchrun``."""

    if strategy == "none":
        return DistributedInfo(strategy="none", rank=0, local_rank=0, world_size=1)
    if strategy != "ddp":
        raise ValueError(f"unsupported distributed strategy: {strategy}")

    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "DDP requires torchrun environment variables: " + ", ".join(missing)
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 2:
        raise ValueError(f"DDP requires WORLD_SIZE >= 2, got {world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA")
    torch.cuda.set_device(local_rank)
    timeout_seconds = int(os.environ.get("GLMOCR_DDP_TIMEOUT_SECONDS", "600"))
    if timeout_seconds <= 0:
        raise ValueError("GLMOCR_DDP_TIMEOUT_SECONDS must be positive")
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=timeout_seconds),
        device_id=torch.device("cuda", local_rank),
    )
    return DistributedInfo(
        strategy="ddp", rank=rank, local_rank=local_rank, world_size=world_size
    )


def barrier(info: DistributedInfo) -> None:
    if info.enabled:
        dist.barrier(device_ids=[info.local_rank])


def destroy_distributed(info: DistributedInfo) -> None:
    if info.enabled and dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_adapter(adapter: nn.Module, info: DistributedInfo) -> nn.Module:
    if not info.enabled:
        return adapter
    return DistributedDataParallel(
        adapter,
        device_ids=[info.local_rank],
        output_device=info.local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def unwrap_module(module: nn.Module) -> nn.Module:
    """Return the underlying module for DDP-safe checkpoint and metadata access."""

    while hasattr(module, "module"):
        module = module.module  # type: ignore[assignment]
    return module


def all_finite(value: bool, info: DistributedInfo, device: torch.device) -> bool:
    """Return whether every rank reports a finite value."""

    if not info.enabled:
        return value
    flag = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def mean_scalar(value: float, info: DistributedInfo, device: torch.device) -> float:
    """Average a scalar across ranks for compact global training metrics."""

    if not info.enabled:
        return float(value)
    tensor = torch.tensor(float(value), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float((tensor / info.world_size).item())


def rank_epoch_indices(
    record_count: int,
    *,
    seed: int,
    epoch: int,
    rank: int,
    world_size: int,
    batch_size: int = 1,
) -> list[int]:
    """Make deterministic rank-local plans padded to complete global batches."""

    if record_count <= 0:
        raise ValueError("record_count must be positive")
    if not 0 <= rank < world_size:
        raise ValueError("rank must be within world_size")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    order = list(range(record_count))
    import random

    generator = random.Random(seed + epoch)
    generator.shuffle(order)
    global_batch = world_size * batch_size
    total = ((record_count + global_batch - 1) // global_batch) * global_batch
    order.extend(order[: total - record_count])
    return order[rank:total:world_size]
