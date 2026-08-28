#!/usr/bin/env python3
"""Launch formal end-to-end PVLD-32 training on selected A100 GPUs."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def training_log_path(stage_root: Path, resume: bool) -> Path:
    if not resume:
        return stage_root / "train.log"
    candidate = stage_root / "train.recovery.log"
    attempt = 2
    while candidate.exists():
        candidate = stage_root / f"train.recovery.{attempt}.log"
        attempt += 1
    return candidate


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument(
        "--validation-image-root",
        type=Path,
        help="Image root for validation manifest; defaults to the manifest directory.",
    )
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--stages", choices=("p1", "p2", "p3", "p1,p2", "p2,p3", "p1,p2,p3"), default="p1,p2")
    parser.add_argument(
        "--ablation",
        choices=("vlqa_ocr_only", "vlqa_layout_direct", "vlqa_layout_p1_p2"),
        default="vlqa_layout_p1_p2",
    )
    parser.add_argument("--layout-loss-preset", choices=("layout_none", "layout_full"), default="layout_full")
    parser.add_argument("--num-layout-prompt-queries", type=int, default=32)
    parser.add_argument("--max-layout-records", type=int, default=512)
    parser.add_argument("--max-layout-tokens", type=int, default=2048)
    parser.add_argument("--layout-decoder-layers", type=int, default=2)
    parser.add_argument("--layout-decoder-hidden-size", type=int, default=256)
    parser.add_argument("--layout-decoder-num-heads", type=int, default=8)
    parser.add_argument("--layout-memory-resolution", choices=("16", "64"), default="16")
    parser.add_argument("--p1-max-steps", type=int, default=12000)
    parser.add_argument("--p2-max-steps", type=int, default=30000)
    parser.add_argument("--p3-max-steps", type=int, default=8000)
    parser.add_argument("--checkpoint-steps", type=int, default=2000)
    parser.add_argument("--p1-checkpoint-steps", type=int)
    parser.add_argument("--p2-checkpoint-steps", type=int)
    parser.add_argument("--checkpoint-retention", type=int, default=2)
    parser.add_argument("--layout-boundary-loss-weight", type=float, default=0.0)
    parser.add_argument("--layout-count-condition-strength", type=float, default=0.0)
    parser.add_argument("--pvld-use-spatial-memory", action="store_true")
    parser.add_argument("--pvld-shared-gradient-scale", type=float, default=1.0)
    parser.add_argument("--pvld-record-gradient-scale", type=float, default=1.0)
    parser.add_argument("--pvld-predicted-layout-routing", action="store_true")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--p1-learning-rate", type=float, default=1e-4)
    parser.add_argument("--p2-learning-rate", type=float, default=5e-5)
    parser.add_argument("--vision-learning-rate", type=float, default=1e-6)
    parser.add_argument("--projector-learning-rate", type=float, default=1e-5)
    parser.add_argument("--layout-learning-rate", type=float, default=1e-4)
    parser.add_argument("--qwen-learning-rate", type=float, default=1e-6)
    parser.add_argument("--gate-learning-rate", type=float, default=1e-5)
    parser.add_argument("--lm-head-learning-rate", type=float, default=0.0)
    for stage, defaults in {
        "p1": {"vision": 1e-6, "projector": 1e-5, "layout": 1e-4, "qwen": 0.0, "gate": 0.0, "lm_head": 0.0},
        "p2": {"vision": 5e-7, "projector": 5e-6, "layout": 5e-5, "qwen": 1e-6, "gate": 1e-5, "lm_head": 0.0},
        "p3": {"vision": 2e-7, "projector": 2e-6, "layout": 1e-5, "qwen": 5e-7, "gate": 1e-6, "lm_head": 0.0},
    }.items():
        for group, default in defaults.items():
            parser.add_argument(
                f"--{stage}-{group.replace('_', '-')}-learning-rate",
                type=float,
                default=default,
            )
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--replay-image-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpu-ids",
        default="",
        help="Comma-separated physical GPU ids. Omit to admit every GPU whose instantaneous utilization is below the limit.",
    )
    parser.add_argument("--gpu-utilization-limit", type=int, default=50)
    parser.add_argument(
        "--distributed-strategy",
        choices=("deepspeed_zero2", "ddp"),
        default="deepspeed_zero2",
    )
    parser.add_argument(
        "--nccl-p2p-disable",
        action="store_true",
        help="Disable NCCL GPU P2P for the known A100 topology issue.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--source-validation-selection",
        type=Path,
        help="Validation-only selection whose selected model initializes a standalone P2 or P3 stage.",
    )
    parser.add_argument("--resume-existing-run", action="store_true")
    parser.add_argument(
        "--skip-p1-validation-selection",
        action="store_true",
        help="Bounded P1 distributed smoke only; requires --stages p1 and <=10 steps.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(os.environ.get("GOT_TRAINING_RUNS", root.parent / "training_runs" / "GOT")),
    )
    parser.add_argument("--project-root", type=Path, default=root / "src" / "GOT-OCR-2.0")
    return parser.parse_args()


def _parse_gpu_utilization_output(output: str) -> dict[str, int]:
    observed: dict[str, int] = {}
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0] or not fields[1].isdigit():
            raise RuntimeError(f"cannot parse nvidia-smi GPU utilization row: {line!r}")
        gpu_id, utilization = fields
        if gpu_id in observed:
            raise RuntimeError(f"nvidia-smi returned duplicate GPU id: {gpu_id}")
        observed[gpu_id] = int(utilization)
    if not observed:
        raise RuntimeError("nvidia-smi returned no GPUs")
    return observed


def _query_gpu_utilization(gpu_ids: tuple[str, ...] | None = None) -> dict[str, int]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    if gpu_ids is not None:
        command[1:1] = ["-i", ",".join(gpu_ids)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()[:400]
        raise RuntimeError(f"nvidia-smi GPU query failed: {detail}")
    return _parse_gpu_utilization_output(result.stdout)


def target_gpus(raw: str, limit: int) -> tuple[tuple[str, ...], dict[str, int]]:
    if limit < 1 or limit > 100:
        raise ValueError("GPU utilization limit must be in [1, 100]")
    requested = tuple(part.strip() for part in raw.split(",") if part.strip())
    if len(set(requested)) != len(requested):
        raise ValueError("--gpu-ids must not contain duplicates")
    if requested:
        observed = _query_gpu_utilization(requested)
        missing = tuple(gpu_id for gpu_id in requested if gpu_id not in observed)
        if missing:
            raise RuntimeError(f"nvidia-smi did not report requested GPU ids: {missing}")
        observed = {gpu_id: observed[gpu_id] for gpu_id in requested}
        busy = {gpu: value for gpu, value in observed.items() if value >= limit}
        if busy:
            raise RuntimeError(f"GPU admission failed: {busy}; required utilization<{limit}")
        return requested, observed

    observed = _query_gpu_utilization()
    eligible = tuple(gpu_id for gpu_id, value in observed.items() if value < limit)
    if not eligible:
        raise RuntimeError(
            f"GPU admission failed: no GPU has utilization<{limit}; observed={observed}"
        )
    return eligible, observed


def stage_learning_rates(args: argparse.Namespace, stage: str) -> dict[str, float]:
    if stage not in {"p1", "p2", "p3"}:
        raise ValueError(stage)
    return {
        group: float(getattr(args, f"{stage}_{group}_learning_rate"))
        for group in ("vision", "projector", "layout", "qwen", "gate", "lm_head")
    }


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def training_command(
    args: argparse.Namespace,
    stage: str,
    source: Path,
    output: Path,
    gpu_ids: tuple[str, ...],
    source_validation_selection: Path | None = None,
) -> list[str]:
    steps = {
        "p1": args.p1_max_steps,
        "p2": args.p2_max_steps,
        "p3": args.p3_max_steps,
    }[stage]
    checkpoint_steps = (
        args.p1_checkpoint_steps if stage == "p1" else args.p2_checkpoint_steps
    ) or args.checkpoint_steps
    learning_rate = args.p1_learning_rate if stage == "p1" else args.p2_learning_rate
    ocr_weight = "0" if stage == "p1" else "1"
    shared_gradient_scale = args.pvld_shared_gradient_scale if stage == "p2" else 1.0
    record_gradient_scale = args.pvld_record_gradient_scale if stage == "p2" else 1.0
    predicted_layout_routing = args.pvld_predicted_layout_routing and stage == "p2"
    checkpoint_retention = args.checkpoint_retention
    if stage == "p1":
        checkpoint_retention = max(
            checkpoint_retention,
            math.ceil(args.p1_max_steps / checkpoint_steps),
        )
    if args.distributed_strategy == "deepspeed_zero2":
        deepspeed = Path(sys.executable).with_name("deepspeed")
        if not deepspeed.is_file():
            raise RuntimeError(f"DeepSpeed launcher is missing: {deepspeed}")
        command = [
            str(deepspeed), "--num_gpus", str(len(gpu_ids)), "--master_port",
            str(free_port()), str(args.project_root / "scripts" / "train_GOT_layout.py"),
            "--deepspeed", str(args.project_root / "zero_config" / "zero2.json"),
        ]
    else:
        torchrun = Path(sys.executable).with_name("torchrun")
        command = [
            str(torchrun), "--standalone", "--nproc_per_node",
            str(len(gpu_ids)), "--master_port", str(free_port()),
            str(args.project_root / "scripts" / "train_GOT_layout.py"),
        ]
        # PVLD P2 can legitimately leave visual-routing parameters unused
        # while residual_gate is initialized at zero. Let DDP detect and
        # reduce only parameters participating in each step; otherwise the
        # second optimizer step fails in reducer bucket rebuild.
        command.extend(["--ddp_find_unused_parameters", "True"])
    group_learning_rates = stage_learning_rates(args, stage)
    command.extend([
        "--model_name_or_path", str(source),
        "--tokenizer_name_or_path", str(args.tokenizer_model),
        "--layout_manifest", str(args.manifest),
        "--layout_image_root", str(args.dataset_root),
        "--layout_split", "train",
        "--layout_stage", stage,
        "--layout_architecture", "pvld",
        "--ablation_id", args.ablation,
        "--layout_loss_preset", args.layout_loss_preset,
        "--p2_train_scope", "adapter_projector",
        "--max_regions", str(args.max_layout_records),
        "--num_layout_prompt_queries", str(args.num_layout_prompt_queries),
        "--max_layout_records", str(args.max_layout_records),
        "--max_layout_tokens", str(args.max_layout_tokens),
        "--layout_decoder_layers", str(args.layout_decoder_layers),
        "--layout_decoder_hidden_size", str(args.layout_decoder_hidden_size),
        "--layout_decoder_num_heads", str(args.layout_decoder_num_heads),
        "--layout_memory_resolution", str(args.layout_memory_resolution),
        "--layout_boundary_loss_weight", str(args.layout_boundary_loss_weight),
        "--layout_count_condition_strength", str(args.layout_count_condition_strength),
        "--pvld_use_spatial_memory", str(args.pvld_use_spatial_memory),
        "--pvld_shared_gradient_scale", str(shared_gradient_scale),
        "--pvld_record_gradient_scale", str(record_gradient_scale),
        "--pvld_predicted_layout_routing", str(predicted_layout_routing),
        "--layout_writeback_mode", "visual_value_layout_routing",
        "--layout_writeback_source", "layout_evidence",
        "--layout_writeback_num_heads", str(args.layout_decoder_num_heads),
        "--layout_writeback_gate_init", "0",
        "--datasets", "layout-page-jsonl",
        "--conversation_version", "mpt",
        "--use_im_start_end", "True",
        "--bf16", "True",
        "--fp16", "False",
        "--gradient_accumulation_steps", str(args.gradient_accumulation_steps),
        "--max_grad_norm", str(args.max_grad_norm),
        "--per_device_train_batch_size", str(args.per_device_batch_size),
        "--optim", "adamw_torch",
        "--evaluation_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", str(checkpoint_steps),
        "--save_total_limit", str(checkpoint_retention),
        "--save_safetensors", "True",
        "--logging_steps", "1",
        "--model_max_length", "2048",
        "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "0",
        "--report_to", "none",
        "--remove_unused_columns", "False",
        "--max_steps", str(steps),
        "--learning_rate", str(learning_rate),
        "--lr_scheduler_type", "constant",
        "--warmup_ratio", "0",
        "--weight_decay", "0",
        "--object_loss_weight", "1" if args.layout_loss_preset == "layout_full" else "0",
        "--bbox_l1_loss_weight", "5" if args.layout_loss_preset == "layout_full" else "0",
        "--bbox_giou_loss_weight", "2" if args.layout_loss_preset == "layout_full" else "0",
        "--direction_loss_weight", "1" if args.layout_loss_preset == "layout_full" else "0",
        "--layout_loss_weight", "1" if args.layout_loss_preset == "layout_full" else "0",
        "--ocr_loss_weight", ocr_weight,
        "--primary_per_replay", "7",
        "--replay_ocr_loss_weight", "0.25",
        "--vision_learning_rate", str(group_learning_rates["vision"]),
        "--projector_learning_rate", str(group_learning_rates["projector"]),
        "--layout_learning_rate", str(group_learning_rates["layout"]),
        "--qwen_learning_rate", str(group_learning_rates["qwen"]),
        "--gate_learning_rate", str(group_learning_rates["gate"]),
        "--lm_head_learning_rate", str(group_learning_rates["lm_head"]),
        "--seed", str(args.seed),
        "--output_dir", str(output),
    ])
    if args.replay_manifest is not None:
        command.extend(["--replay_layout_manifest", str(args.replay_manifest)])
        if args.replay_image_root is not None:
            command.extend(["--replay_layout_image_root", str(args.replay_image_root)])
    if stage in {"p2", "p3"} and args.ablation == "vlqa_layout_p1_p2":
        if source_validation_selection is None:
            raise ValueError(
                f"PVLD C5 {stage.upper()} requires its validation-only preceding-stage selection."
            )
        command.extend(
            ["--source_validation_selection", str(source_validation_selection)]
        )
    return command


def p1_selection_command(
    args: argparse.Namespace,
    model_root: Path,
    output_dir: Path,
    gpu_ids: tuple[str, ...],
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "evaluation" / "select_layout_ablation_checkpoint.py"),
        "--ablation", args.ablation,
        "--model-root", str(model_root),
        "--model-kind", "pvld",
        "--selection-purpose", "p1_layout",
        "--tokenizer-model", str(args.tokenizer_model),
        "--validation-manifest", str(args.validation_manifest),
        "--validation-image-root", str(
            (args.validation_image_root or args.validation_manifest.parent).resolve()
        ),
        "--output-dir", str(output_dir),
        "--project-root", str(args.project_root),
        "--max-regions", str(args.max_layout_records),
        "--max-records", "0",
        "--max-new-tokens", str(args.max_layout_tokens),
        "--no-repeat-ngram-size", "20",
        "--parallel-gpu-ids", ",".join(gpu_ids),
        "--gpu-utilization-limit", str(args.gpu_utilization_limit),
    ]
    if (output_dir / "selection.json").is_file():
        command.append("--resume")
    return command


def main() -> int:
    args = parse_args()
    if args.source_validation_selection is not None and args.stages not in {"p2", "p3"}:
        raise ValueError(
            "--source-validation-selection is only valid for a standalone P2 or P3 stage."
        )
    if args.skip_p1_validation_selection and (
        args.stages != "p1" or args.p1_max_steps > 10
    ):
        raise ValueError(
            "--skip-p1-validation-selection is limited to P1 smokes of at most 10 steps"
        )
    ids, utilization = target_gpus(args.gpu_ids, args.gpu_utilization_limit)
    admission_mode = "explicit" if args.gpu_ids.strip() else "auto"
    run_root = args.runs_root.resolve() / args.run_id
    if run_root.exists() != args.resume_existing_run:
        expected = "existing" if args.resume_existing_run else "new"
        raise FileExistsError(f"expected {expected} run directory: {run_root}")
    metadata = run_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=args.resume_existing_run)
    status = {
        "status": "running",
        "run_id": args.run_id,
        "layout_architecture": "pvld",
        "input_granularity": "whole_page_image",
        "physical_gpu_ids": ids,
        "gpu_admission_mode": admission_mode,
        "gpu_utilization_at_admission": utilization,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "train_manifest": str(args.manifest.resolve()),
        "validation_manifest": str(args.validation_manifest.resolve()),
        "test_manifest": str(args.test_manifest.resolve()),
        "source_validation_selection": (
            str(args.source_validation_selection.resolve())
            if args.source_validation_selection is not None else None
        ),
        "resumed_from_existing_run": args.resume_existing_run,
        "p1_max_steps": args.p1_max_steps,
        "p2_max_steps": args.p2_max_steps,
        "p1_checkpoint_steps": args.p1_checkpoint_steps or args.checkpoint_steps,
        "p2_checkpoint_steps": args.p2_checkpoint_steps or args.checkpoint_steps,
        "checkpoint_retention": args.checkpoint_retention,
        "layout_memory_resolution": args.layout_memory_resolution,
        "replay_protocol": {
            "primary_per_replay": 7,
            "replay_ocr_loss_weight": 0.25,
            "manifest": str(args.replay_manifest.resolve()) if args.replay_manifest else None,
        },
        "learning_rate_groups": {
            stage: stage_learning_rates(args, stage)
            for stage in ("p1", "p2", "p3")
        },
        "layout_boundary_loss_weight": args.layout_boundary_loss_weight,
        "layout_count_condition_strength": args.layout_count_condition_strength,
        "pvld_use_spatial_memory": args.pvld_use_spatial_memory,
        "pvld_shared_gradient_scale_p2": args.pvld_shared_gradient_scale,
        "pvld_record_gradient_scale_p2": args.pvld_record_gradient_scale,
        "pvld_predicted_layout_routing_p2": args.pvld_predicted_layout_routing,
        "p1_forces_legacy_gradient_scale_and_routing": True,
        "distributed_strategy": args.distributed_strategy,
        "p1_validation_selection_skipped": args.skip_p1_validation_selection,
        "nccl_p2p_disable": args.nccl_p2p_disable and len(ids) > 1,
    }
    (metadata / "status.txt").write_text(compact(status) + "\n", encoding="utf-8")
    environment = dict(os.environ)
    # torchrun children use logical local ranks. Restrict the parent once to
    # the explicitly admitted physical cards, then let torchrun map 0..N-1.
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    if args.nccl_p2p_disable and len(ids) > 1:
        environment["NCCL_P2P_DISABLE"] = "1"
    source = args.source_model.resolve()
    source_selection_path: Path | None = (
        args.source_validation_selection.resolve()
        if args.source_validation_selection is not None else None
    )
    stage_metrics: dict[str, Any] = {}
    if args.resume_existing_run:
        p1_metrics = run_root / "p1" / "model" / "layout_training_metrics.json"
        if p1_metrics.is_file():
            metrics = json.loads(p1_metrics.read_text(encoding="utf-8"))
            stage_metrics["p1"] = {
                "global_step": int(metrics["global_step"]),
                "train_loss": float(metrics["train_loss"]),
                "model": str(p1_metrics.parent),
                "metrics": str(p1_metrics),
                "reused_completed_stage": True,
            }
        candidate_selection = run_root / "p1" / "validation_selection" / "selection.json"
        if candidate_selection.is_file() and source_selection_path is None:
            source_selection_path = candidate_selection.resolve()
    for stage in args.stages.split(","):
        output = run_root / stage / "model"
        output.mkdir(parents=True, exist_ok=args.resume_existing_run)
        log_path = training_log_path(run_root / stage, args.resume_existing_run)
        status.update({"stage": stage, "stage_status": "running"})
        (metadata / "status.txt").write_text(compact(status) + "\n", encoding="utf-8")
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                training_command(
                    args, stage, source, output,
                    ids,
                    source_validation_selection=source_selection_path,
                ),
                cwd=args.project_root.resolve(),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        metrics_path = output / "layout_training_metrics.json"
        if completed.returncode or not metrics_path.is_file():
            status.update({"status": "failed", "stage_status": "failed", "log": str(log_path)})
            (metadata / "status.txt").write_text(compact(status) + "\n", encoding="utf-8")
            write_json(run_root / "summary.json", status)
            raise RuntimeError(f"PVLD {stage} failed; see {log_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        loss = float(metrics.get("train_loss", float("nan")))
        if int(metrics.get("global_step", 0)) < 1 or not math.isfinite(loss):
            raise RuntimeError(f"PVLD {stage} produced no finite optimizer step")
        stage_metrics[stage] = {
            "global_step": int(metrics["global_step"]),
            "train_loss": loss,
            "model": str(output),
            "metrics": str(metrics_path),
            "log": str(log_path),
        }
        source = output
        if (
            stage == "p1"
            and args.ablation == "vlqa_layout_p1_p2"
            and not args.skip_p1_validation_selection
        ):
            selection_dir = run_root / "p1" / "validation_selection"
            selection_log = run_root / "p1" / "validation_selection.log"
            # Admission is intentionally repeated here.  A long P1 can change
            # the set of cards below the utilization threshold before its
            # validation-only selection begins.
            selection_gpu_ids, selection_utilization = target_gpus(
                args.gpu_ids, args.gpu_utilization_limit
            )
            status.update(
                {
                    "stage": "p1_validation_selection",
                    "stage_status": "running",
                    "p1_selection_physical_gpu_ids": selection_gpu_ids,
                    "p1_selection_gpu_utilization_at_admission": selection_utilization,
                }
            )
            (metadata / "status.txt").write_text(
                compact(status) + "\n", encoding="utf-8"
            )
            with selection_log.open("a", encoding="utf-8") as log:
                selected = subprocess.run(
                    p1_selection_command(args, output, selection_dir, selection_gpu_ids),
                    cwd=Path(__file__).resolve().parents[2],
                    env=dict(os.environ),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            selection_path = selection_dir / "selection.json"
            if selected.returncode or not selection_path.is_file():
                status.update(
                    {
                        "status": "failed",
                        "stage_status": "p1_validation_selection_failed",
                        "log": str(selection_log),
                    }
                )
                (metadata / "status.txt").write_text(
                    compact(status) + "\n", encoding="utf-8"
                )
                raise RuntimeError(
                    f"PVLD P1 validation selection failed; see {selection_log}"
                )
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if selection.get("selection_split") != "validation" or selection.get(
                "test_used_for_selection"
            ) is not False:
                raise RuntimeError("P1 selection did not preserve validation-only protocol.")
            source = Path(selection["selected"]["model_path"]).resolve()
            source_selection_path = selection_path.resolve()
            stage_metrics["p1"].update(
                {
                    "validation_selection": str(selection_path),
                    "selected_model": str(source),
                    "selected_optimizer_step": int(
                        selection["selected"]["optimizer_step"]
                    ),
                    "selection_physical_gpu_ids": list(selection_gpu_ids),
                    "selection_gpu_utilization_at_admission": selection_utilization,
                    "p2_source_is_validation_selected_p1": True,
                }
            )
    status.update({
        "status": "training_completed",
        "stage_status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stages": stage_metrics,
        "final_model": str(source),
        "completed_via_recovery": args.resume_existing_run,
    })
    (metadata / "status.txt").write_text(compact(status) + "\n", encoding="utf-8")
    write_json(run_root / "layout_training_metrics.json", stage_metrics)
    write_json(run_root / "summary.json", status)
    (run_root / "PVLD_TRAINING_FINISHED").touch()
    print(compact({"event": "pvld_training_completed", "run_id": args.run_id, "summary": str(run_root / "summary.json")}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(compact({"event": "pvld_training_failed", "error_type": type(exc).__name__, "error": str(exc)[:800]}), file=sys.stderr)
        raise SystemExit(1)
