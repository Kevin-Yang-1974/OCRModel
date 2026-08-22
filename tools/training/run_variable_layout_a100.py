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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--stages", choices=("p1", "p2", "p1,p2"), default="p1,p2")
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
    parser.add_argument("--p1-max-steps", type=int, default=12000)
    parser.add_argument("--p2-max-steps", type=int, default=30000)
    parser.add_argument("--checkpoint-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-retention", type=int, default=2)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--p1-learning-rate", type=float, default=1e-4)
    parser.add_argument("--p2-learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-ids", required=True)
    parser.add_argument("--gpu-utilization-limit", type=int, default=50)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume-existing-run", action="store_true")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(os.environ.get("GOT_TRAINING_RUNS", root.parent / "training_runs" / "GOT")),
    )
    parser.add_argument("--project-root", type=Path, default=root / "src" / "GOT-OCR-2.0")
    return parser.parse_args()


def target_gpus(raw: str, limit: int) -> tuple[tuple[str, ...], dict[str, int]]:
    ids = tuple(part.strip() for part in raw.split(",") if part.strip())
    observed: dict[str, int] = {}
    for gpu_id in ids:
        result = subprocess.run(
            ["nvidia-smi", "-i", gpu_id, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        value = result.stdout.strip()
        if result.returncode or not value.isdigit():
            raise RuntimeError(f"cannot query target GPU {gpu_id}")
        observed[gpu_id] = int(value)
    busy = {gpu: value for gpu, value in observed.items() if value >= limit}
    if busy:
        raise RuntimeError(f"GPU admission failed: {busy}; required utilization<{limit}")
    return ids, observed


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def training_command(args: argparse.Namespace, stage: str, source: Path, output: Path) -> list[str]:
    steps = args.p1_max_steps if stage == "p1" else args.p2_max_steps
    learning_rate = args.p1_learning_rate if stage == "p1" else args.p2_learning_rate
    ocr_weight = "0" if stage == "p1" else "1"
    deepspeed = Path(sys.executable).with_name("deepspeed")
    checkpoint_retention = args.checkpoint_retention
    if stage == "p1":
        checkpoint_retention = max(
            checkpoint_retention,
            math.ceil(args.p1_max_steps / args.checkpoint_steps),
        )
    command = [
        str(deepspeed), "--master_port", str(free_port()),
        str(args.project_root / "scripts" / "train_GOT_layout.py"),
        "--deepspeed", str(args.project_root / "zero_config" / "zero2.json"),
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
        "--per_device_train_batch_size", str(args.per_device_batch_size),
        "--optim", "adamw_torch",
        "--evaluation_strategy", "no",
        "--save_strategy", "steps",
        "--save_steps", str(args.checkpoint_steps),
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
        "--seed", str(args.seed),
        "--output_dir", str(output),
    ]
    return command


def p1_selection_command(
    args: argparse.Namespace,
    model_root: Path,
    output_dir: Path,
    gpu_id: str,
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
        "--validation-image-root", str(args.validation_manifest.parent),
        "--output-dir", str(output_dir),
        "--project-root", str(args.project_root),
        "--max-regions", str(args.max_layout_records),
        "--max-records", "0",
        "--max-new-tokens", str(args.max_layout_tokens),
        "--no-repeat-ngram-size", "20",
        "--gpu-id", gpu_id,
        "--gpu-utilization-limit", str(args.gpu_utilization_limit),
    ]
    if (output_dir / "selection.json").is_file():
        command.append("--resume")
    return command


def main() -> int:
    args = parse_args()
    ids, utilization = target_gpus(args.gpu_ids, args.gpu_utilization_limit)
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
        "gpu_utilization_at_admission": utilization,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "train_manifest": str(args.manifest.resolve()),
        "validation_manifest": str(args.validation_manifest.resolve()),
        "test_manifest": str(args.test_manifest.resolve()),
        "resumed_from_existing_run": args.resume_existing_run,
    }
    (metadata / "status.txt").write_text(compact(status) + "\n", encoding="utf-8")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    source = args.source_model.resolve()
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
    for stage in args.stages.split(","):
        output = run_root / stage / "model"
        output.mkdir(parents=True, exist_ok=args.resume_existing_run)
        log_name = "train.recovery.log" if args.resume_existing_run else "train.log"
        log_path = run_root / stage / log_name
        status.update({"stage": stage, "stage_status": "running"})
        (metadata / "status.txt").write_text(compact(status) + "\n", encoding="utf-8")
        with log_path.open("x", encoding="utf-8") as log:
            completed = subprocess.run(
                training_command(args, stage, source, output),
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
        if stage == "p1" and args.ablation == "vlqa_layout_p1_p2":
            selection_dir = run_root / "p1" / "validation_selection"
            selection_log = run_root / "p1" / "validation_selection.log"
            status.update({"stage": "p1_validation_selection", "stage_status": "running"})
            (metadata / "status.txt").write_text(
                compact(status) + "\n", encoding="utf-8"
            )
            with selection_log.open("a", encoding="utf-8") as log:
                selected = subprocess.run(
                    p1_selection_command(args, output, selection_dir, ids[0]),
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
            stage_metrics["p1"].update(
                {
                    "validation_selection": str(selection_path),
                    "selected_model": str(source),
                    "selected_optimizer_step": int(
                        selection["selected"]["optimizer_step"]
                    ),
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
