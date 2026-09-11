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
    wrap_model,
)
from layout_ocr.train_screen import (
    configure_deterministic_execution,
    ContinuationStopHead,
    load_adapter_checkpoint,
    load_model,
    load_decoder_lora_checkpoint,
    load_continuation_head_checkpoint,
    natural_loop_config,
    save_adapter_checkpoint,
    save_continuation_head_checkpoint,
    train,
    write_json,
)
from layout_ocr.lora import inject_decoder_lora


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["geometry"], default="geometry")
    parser.add_argument("--num-queries", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decoder-adaptation", choices=["frozen", "lora"], default="frozen")
    parser.add_argument("--decoder-lora-rank", type=int, default=8)
    parser.add_argument("--decoder-lora-alpha", type=float, default=8.0)
    parser.add_argument("--decoder-lora-dropout", type=float, default=0.0)
    parser.add_argument("--decoder-learning-rate", type=float, default=1e-6)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--adapter-precision", choices=["fp32"], default="fp32")
    parser.add_argument(
        "--layout-loss-profile",
        choices=[
            "full",
            "ocr_only",
            "no_assignment",
            "no_assignment_validity",
            "validity_assignment",
            "no_geometry",
        ],
        default="full",
    )
    parser.add_argument("--query-assignment", choices=["hungarian"], default="hungarian")
    parser.add_argument("--residual-scale-cap", type=float, default=0.03)
    parser.add_argument("--initial-residual-scale", type=float, default=0.0)
    parser.add_argument("--use-validity-head", action="store_true")
    parser.add_argument("--initial-valid-probability", type=float, default=None)
    parser.add_argument(
        "--validity-gating-mode",
        choices=["legacy_normalized", "raw_mass"],
        default="legacy_normalized",
    )
    parser.add_argument("--validity-use-transport-evidence", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--processor-mode", choices=["fast"], default="fast")
    parser.add_argument("--text-repeat-suppression", action="store_true")
    parser.add_argument("--text-ul-weight", type=float, default=0.1)
    parser.add_argument("--text-eos-loss-weight", type=float, default=0.05)
    parser.add_argument("--repeat-recent-window", type=int, default=96)
    parser.add_argument("--repeat-min-cycle-length", type=int, default=8)
    parser.add_argument("--repeat-max-cycle-length", type=int, default=32)
    parser.add_argument("--repeat-cycle-repeats", type=int, default=3)
    parser.add_argument("--repeat-cycle-penalty", type=float, default=2.0)
    parser.add_argument("--repeat-force-eos-steps", type=int, default=16)
    parser.add_argument("--natural-loop-loss", action="store_true")
    parser.add_argument("--natural-loop-weight", type=float, default=0.05)
    parser.add_argument("--natural-loop-recent-window", type=int, default=96)
    parser.add_argument("--natural-loop-min-cycle-length", type=int, default=8)
    parser.add_argument("--natural-loop-max-cycle-length", type=int, default=32)
    parser.add_argument("--natural-loop-cycle-repeats", type=int, default=3)
    parser.add_argument("--continuation-escape", action="store_true")
    parser.add_argument("--escape-budget", type=int, default=16)
    parser.add_argument("--escape-clear-steps", type=int, default=4)
    parser.add_argument("--escape-eos-suppression", type=float, default=1.0)
    parser.add_argument("--escape-eos-boost", type=float, default=0.5)
    parser.add_argument("--loop-escape-training", action="store_true")
    parser.add_argument("--loop-escape-cycle-length", type=int, default=8)
    parser.add_argument("--loop-escape-horizon", type=int, default=8)
    parser.add_argument("--loop-escape-ramp-steps", type=int, default=128)
    parser.add_argument("--loop-escape-weight", type=float, default=0.1)
    parser.add_argument("--loop-escape-margin", type=float, default=0.5)
    parser.add_argument("--loop-escape-margin-weight", type=float, default=0.05)
    parser.add_argument("--loop-continue-weight", type=float, default=0.05)
    parser.add_argument("--continuation-head", action="store_true")
    parser.add_argument("--continuation-head-hidden-size", type=int, default=32)
    parser.add_argument("--continuation-head-weight", type=float, default=0.05)
    parser.add_argument("--continuation-head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--region-autoregressive", action="store_true")
    parser.add_argument("--region-decoder-hidden-size", type=int, default=256)
    parser.add_argument("--region-decoder-layers", type=int, default=2)
    parser.add_argument("--region-decoder-num-heads", type=int, default=8)
    parser.add_argument("--region-pointer-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--region-spatial-penalty", type=float, default=4.0)
    parser.add_argument("--region-spatial-iou-threshold", type=float, default=0.8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.layout_loss_profile == "validity_assignment":
        args.use_validity_head = True
        args.validity_gating_mode = "raw_mass"
        args.validity_use_transport_evidence = True
    if args.initial_valid_probability is None:
        args.initial_valid_probability = (
            0.066 if args.layout_loss_profile == "validity_assignment" else 0.05
        )
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.decoder_lora_rank <= 0 or args.decoder_lora_alpha <= 0:
        raise ValueError("decoder LoRA rank and alpha must be positive")
    if not 0.0 <= args.decoder_lora_dropout < 1.0:
        raise ValueError("decoder LoRA dropout must be in [0, 1)")
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be positive")
    if args.region_autoregressive and args.num_queries != 512:
        raise ValueError("--region-autoregressive requires --num-queries 512")
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
            validity_gating_mode=args.validity_gating_mode,
            validity_use_transport_evidence=args.validity_use_transport_evidence,
            adapter_precision=args.adapter_precision,
            region_autoregressive=args.region_autoregressive,
            region_decoder_hidden_size=args.region_decoder_hidden_size,
            region_decoder_layers=args.region_decoder_layers,
            region_decoder_num_heads=args.region_decoder_num_heads,
            region_pointer_mask=args.region_pointer_mask,
            region_spatial_penalty=args.region_spatial_penalty,
            region_spatial_iou_threshold=args.region_spatial_iou_threshold,
        )
        model, processor, bridge = load_model(model_args, device)
        continuation_head = (
            ContinuationStopHead(hidden_size=args.continuation_head_hidden_size).to(device)
            if args.continuation_head
            else None
        )
        decoder_lora_config = {"enabled": False, "target_count": 0}
        if args.decoder_adaptation == "lora":
            decoder_lora_config = inject_decoder_lora(
                model,
                rank=args.decoder_lora_rank,
                alpha=args.decoder_lora_alpha,
                dropout=args.decoder_lora_dropout,
            )
            model = wrap_model(model, distributed)
        else:
            bridge.adapter = wrap_adapter(bridge.adapter, distributed)  # type: ignore[assignment]
        if continuation_head is not None:
            continuation_head = wrap_model(continuation_head, distributed)
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
            decoder_adaptation=args.decoder_adaptation,
            decoder_learning_rate=args.decoder_learning_rate,
            text_repeat_suppression=args.text_repeat_suppression,
            text_ul_weight=args.text_ul_weight,
            text_eos_loss_weight=args.text_eos_loss_weight,
            repeat_recent_window=args.repeat_recent_window,
            repeat_min_cycle_length=args.repeat_min_cycle_length,
            repeat_max_cycle_length=args.repeat_max_cycle_length,
            repeat_cycle_repeats=args.repeat_cycle_repeats,
            repeat_cycle_penalty=args.repeat_cycle_penalty,
            repeat_force_eos_steps=args.repeat_force_eos_steps,
            natural_loop_loss=args.natural_loop_loss,
            natural_loop_weight=args.natural_loop_weight,
            natural_loop_recent_window=args.natural_loop_recent_window,
            natural_loop_min_cycle_length=args.natural_loop_min_cycle_length,
            natural_loop_max_cycle_length=args.natural_loop_max_cycle_length,
            natural_loop_cycle_repeats=args.natural_loop_cycle_repeats,
            continuation_escape=args.continuation_escape,
            escape_budget=args.escape_budget,
            escape_clear_steps=args.escape_clear_steps,
            escape_eos_suppression=args.escape_eos_suppression,
            escape_eos_boost=args.escape_eos_boost,
            loop_escape_training=args.loop_escape_training,
            loop_escape_cycle_length=args.loop_escape_cycle_length,
            loop_escape_horizon=args.loop_escape_horizon,
            loop_escape_ramp_steps=args.loop_escape_ramp_steps,
            loop_escape_weight=args.loop_escape_weight,
            loop_escape_margin=args.loop_escape_margin,
            loop_escape_margin_weight=args.loop_escape_margin_weight,
            loop_continue_weight=args.loop_continue_weight,
            continuation_head_learning_rate=args.continuation_head_learning_rate,
            continuation_head_weight=args.continuation_head_weight,
            region_autoregressive=args.region_autoregressive,
            region_decoder_hidden_size=args.region_decoder_hidden_size,
            region_decoder_layers=args.region_decoder_layers,
            region_decoder_num_heads=args.region_decoder_num_heads,
            region_pointer_mask=args.region_pointer_mask,
            region_spatial_penalty=args.region_spatial_penalty,
            region_spatial_iou_threshold=args.region_spatial_iou_threshold,
        )
        training = train(
            train_args,
            model,
            processor,
            bridge,
            continuation_head,
            records,
            device,
            distributed=distributed,
        )
        barrier(distributed)
        if distributed.is_main:
            checkpoint_dir = args.output_dir / f"checkpoint-{args.steps}"
            load_adapter_checkpoint(checkpoint_dir, bridge)
            if args.decoder_adaptation == "lora":
                load_decoder_lora_checkpoint(checkpoint_dir, model)
            if continuation_head is not None:
                load_continuation_head_checkpoint(checkpoint_dir, continuation_head)
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
                "decoder_adaptation": args.decoder_adaptation,
                "decoder_lora_config": decoder_lora_config,
                "natural_loop_config": natural_loop_config(args),
                }
            write_json(args.output_dir / "smoke_summary.json", summary)
            (args.output_dir / "SMOKE_COMPLETED").touch()
            print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
        barrier(distributed)
    finally:
        destroy_distributed(distributed)


if __name__ == "__main__":
    main()
