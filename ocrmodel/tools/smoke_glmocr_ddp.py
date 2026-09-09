#!/usr/bin/env python3
"""Bounded five-process DDP smoke test for the full-query GLMOCR path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from layout_ocr.data import load_records, validate_records
from layout_ocr.distributed import (
    barrier,
    destroy_distributed,
    initialize_distributed,
    unwrap_module,
    wrap_adapter,
)
from layout_ocr.train_screen import (
    configure_deterministic_execution,
    load_adapter_checkpoint,
    load_model,
    save_adapter_checkpoint,
    train,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["geometry"], default="geometry")
    parser.add_argument("--num-queries", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--adapter-precision", choices=["fp32"], default="fp32")
    parser.add_argument(
        "--layout-loss-profile",
        choices=[
            "full",
            "ocr_only",
            "no_assignment",
            "no_assignment_validity",
            "no_geometry",
        ],
        default="full",
    )
    parser.add_argument("--query-assignment", choices=["hungarian"], default="hungarian")
    parser.add_argument("--residual-scale-cap", type=float, default=0.03)
    parser.add_argument("--initial-residual-scale", type=float, default=0.0)
    parser.add_argument("--use-validity-head", action="store_true")
    parser.add_argument("--initial-valid-probability", type=float, default=0.05)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--processor-mode", choices=["fast"], default="fast")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    configure_deterministic_execution()
    distributed = initialize_distributed("ddp")
    try:
        if distributed.is_main:
            args.output_dir.mkdir(parents=True)
        barrier(distributed)
        if not torch.cuda.is_available():
            raise RuntimeError("DDP smoke requires CUDA")
        device = torch.device("cuda", distributed.local_rank)
        torch.cuda.set_device(device)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        all_records = load_records(args.train_manifest)
        validate_records(all_records, split="train", num_queries=args.num_queries)
        # Select the most demanding pages so the smoke test exercises the
        # 512-query path and never hides high-region pages behind the legacy
        # 32-query fixture.
        records = sorted(
            all_records,
            key=lambda row: (len(row["regions"]), len(row["page_text"]), row["page_id"]),
            reverse=True,
        )[: max(distributed.world_size * 2, distributed.world_size)]
        if len(records) < distributed.world_size:
            raise ValueError("smoke fixture has fewer pages than DDP ranks")

        model_args = argparse.Namespace(
            model_path=args.model_path,
            processor_mode=args.processor_mode,
            max_pixels=args.max_pixels,
            mode=args.mode,
            num_queries=args.num_queries,
            residual_scale_cap=args.residual_scale_cap,
            initial_residual_scale=args.initial_residual_scale,
            use_validity_head=args.use_validity_head,
            initial_valid_probability=args.initial_valid_probability,
            adapter_precision=args.adapter_precision,
        )
        model, processor, bridge = load_model(model_args, device)
        bridge.adapter = wrap_adapter(bridge.adapter, distributed)  # type: ignore[assignment]
        train_args = argparse.Namespace(
            seed=args.seed,
            max_steps=args.steps,
            lr_schedule_steps=args.steps,
            learning_rate=5e-5,
            warmup_steps=1,
            min_lr_ratio=0.1,
            initial_residual_scale=args.initial_residual_scale,
            gate_freeze_steps=0,
            layout_loss_profile=args.layout_loss_profile,
            query_assignment=args.query_assignment,
            auxiliary_weight=0.2,
            auxiliary_weight_start=0.2,
            auxiliary_ramp_steps=0,
            max_grad_norm=1.0,
            diagnostic_steps=(),
            log_steps=1,
            validation_interval=args.steps + 1,
            output_dir=args.output_dir,
            num_queries=args.num_queries,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
        )
        training = train(
            train_args,
            model,
            processor,
            bridge,
            records,
            device,
            distributed=distributed,
        )
        barrier(distributed)
        if distributed.is_main:
            checkpoint_dir = args.output_dir / f"checkpoint-{args.steps}"
            load_adapter_checkpoint(checkpoint_dir, bridge)
            reloaded = unwrap_module(bridge.adapter)
            finite = all(bool(torch.isfinite(value).all()) for value in reloaded.state_dict().values())
            summary = {
                "status": "complete" if finite else "failed",
                "world_size": distributed.world_size,
                "global_batch_size": distributed.world_size,
                "per_device_batch_size": 1,
                "num_queries": args.num_queries,
                "steps": args.steps,
                "record_count": len(records),
                "max_regions_in_smoke": max(len(row["regions"]) for row in records),
                "checkpoint_reload": True,
                "parameters_finite": finite,
                "test_used_for_selection": False,
                "training": training,
            }
            write_json(args.output_dir / "smoke_summary.json", summary)
            (args.output_dir / "SMOKE_COMPLETED").touch()
            print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
        barrier(distributed)
    finally:
        destroy_distributed(distributed)


if __name__ == "__main__":
    main()
