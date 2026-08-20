#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from run_layout_a100 import (
    EXIT_EXISTS,
    EXIT_FAILURE,
    EXIT_GPU_BUSY,
    EXIT_MISSING,
    EXIT_USAGE,
    RUN_ID_PATTERN,
    RunFailure,
    acquire_gpu_locks,
    bounded,
    compact_json,
    file_sha256,
    gpu_utilization,
    tail_lines,
    timestamp,
    write_json,
    write_status,
)


@dataclass(frozen=True)
class Settings:
    ocrmodel_root: Path
    project_root: Path
    workspace: Path
    dataset_root: Path
    train_manifest: Path
    validation_manifest: Path
    test_manifest: Path
    source_model: Path
    runs_root: Path
    run_id: str
    gpu_id: str
    max_steps: int
    max_train_records: int
    model_max_length: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    checkpoint_steps: int
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    train_scope: str
    seed: int
    skip_source_hash: bool

    @property
    def run_root(self) -> Path:
        return self.runs_root / self.run_id


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    workspace = Path(os.environ.get("OCR_WORKSPACE", root.parent))
    project = Path(os.environ.get("GOT_PROJECT_ROOT", root / "src" / "GOT-OCR-2.0"))
    parser = argparse.ArgumentParser(
        description="Train original GOT2 on whole-page OCR truth without VLQA."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument(
        "--source-model",
        type=Path,
        default=Path(os.environ.get("GOT_SOURCE_MODEL", workspace / "models" / "GOT-OCR2_0")),
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path(os.environ.get("GOT_TRAINING_RUNS", workspace / "training_runs" / "GOT")),
    )
    parser.add_argument("--ocrmodel-root", type=Path, default=root)
    parser.add_argument("--project-root", type=Path, default=project)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-id", default=os.environ.get("GOT_PHYSICAL_GPU", "0"))
    parser.add_argument("--max-steps", type=positive_int, default=12000)
    parser.add_argument("--max-train-records", type=nonnegative_int, default=0)
    parser.add_argument("--model-max-length", type=positive_int, default=2048)
    parser.add_argument("--per-device-batch-size", type=positive_int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--checkpoint-steps", type=positive_int, default=2000)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("constant", "cosine"),
        default="cosine",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--train-scope", choices=("projector", "decoder_projector"), default="decoder_projector")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-source-hash", action="store_true")
    return parser.parse_args(argv)


def resolve_settings(args: argparse.Namespace) -> Settings:
    gpu_id = str(args.gpu_id).strip()
    if not re.fullmatch(r"[0-9]+", gpu_id):
        raise RunFailure("--gpu-id must be one numeric physical GPU id.", exit_code=EXIT_USAGE)
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        raise RunFailure("Invalid --run-id.", exit_code=EXIT_USAGE)
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise RunFailure("--learning-rate must be positive.", exit_code=EXIT_USAGE)
    if not math.isfinite(args.warmup_ratio) or not 0.0 <= args.warmup_ratio < 1.0:
        raise RunFailure("--warmup-ratio must be in [0, 1).", exit_code=EXIT_USAGE)
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0.0:
        raise RunFailure("--weight-decay must be non-negative.", exit_code=EXIT_USAGE)
    settings = Settings(
        ocrmodel_root=args.ocrmodel_root.expanduser().resolve(),
        project_root=args.project_root.expanduser().resolve(),
        workspace=args.workspace.expanduser().resolve(),
        dataset_root=args.dataset_root.expanduser().resolve(),
        train_manifest=args.manifest.expanduser().resolve(),
        validation_manifest=args.validation_manifest.expanduser().resolve(),
        test_manifest=args.test_manifest.expanduser().resolve(),
        source_model=args.source_model.expanduser().resolve(),
        runs_root=args.runs_root.expanduser().resolve(),
        run_id=args.run_id,
        gpu_id=gpu_id,
        max_steps=args.max_steps,
        max_train_records=args.max_train_records,
        model_max_length=args.model_max_length,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        checkpoint_steps=args.checkpoint_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        train_scope=args.train_scope,
        seed=args.seed,
        skip_source_hash=args.skip_source_hash,
    )
    required = (
        settings.train_manifest,
        settings.validation_manifest,
        settings.test_manifest,
        settings.source_model / "model.safetensors",
        settings.source_model / "config.json",
        settings.project_root / "scripts" / "train_GOT_page_ocr.py",
        settings.project_root / "scripts" / "verify_linelevel_checkpoint.py",
        settings.project_root / "scripts" / "layout_page_dataset.py",
        settings.project_root / "zero_config" / "zero2.json",
        settings.ocrmodel_root / "tools" / "preprocessing" / "audit_synthetic_layout.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not settings.dataset_root.is_dir():
        raise RunFailure(f"Required C1 assets are missing: {missing}", exit_code=EXIT_MISSING)
    source_config = json.loads((settings.source_model / "config.json").read_text(encoding="utf-8"))
    if source_config.get("use_vlqa") is True:
        raise RunFailure("C1 source checkpoint must not contain VLQA.", exit_code=EXIT_USAGE)
    return settings


def environment(settings: Settings, *, cuda: bool = True) -> dict[str, str]:
    result = dict(os.environ)
    result.update(
        {
            "OCR_WORKSPACE": str(settings.workspace),
            "OCRMODEL_ROOT": str(settings.ocrmodel_root),
            "GOT_PROJECT_ROOT": str(settings.project_root),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": settings.gpu_id if cuda else "",
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLED": "true",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    return result


def require_gpu_free(settings: Settings) -> None:
    utilization = gpu_utilization(settings.gpu_id)
    if utilization >= 50:
        raise RunFailure(
            f"GPU{settings.gpu_id}_BUSY utilization={utilization} limit=50",
            exit_code=EXIT_GPU_BUSY,
        )


def free_master_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def training_command(settings: Settings) -> list[str]:
    deepspeed = Path(sys.executable).with_name("deepspeed")
    if not deepspeed.is_file():
        raise RunFailure(f"DeepSpeed is missing: {deepspeed}", exit_code=EXIT_MISSING)
    return [
        str(deepspeed), "--master_port", str(free_master_port()),
        str(settings.project_root / "scripts" / "train_GOT_page_ocr.py"),
        "--deepspeed", str(settings.project_root / "zero_config" / "zero2.json"),
        "--model_name_or_path", str(settings.source_model),
        "--layout_manifest", str(settings.train_manifest),
        "--layout_image_root", str(settings.dataset_root),
        "--layout_split", "train",
        "--max_regions", "16",
        "--max_train_records", str(settings.max_train_records),
        "--train_scope", settings.train_scope,
        "--datasets", "layout-page-jsonl",
        "--conversation_version", "mpt",
        "--use_im_start_end", "True",
        "--bf16", "True", "--fp16", "False",
        "--gradient_accumulation_steps", str(settings.gradient_accumulation_steps),
        "--optim", "adamw_torch",
        "--evaluation_strategy", "no", "--save_strategy", "steps",
        "--save_steps", str(settings.checkpoint_steps),
        "--save_safetensors", "True", "--weight_decay", str(settings.weight_decay),
        "--warmup_ratio", str(settings.warmup_ratio),
        "--lr_scheduler_type", settings.lr_scheduler_type,
        "--logging_steps", "1", "--tf32", "False",
        "--model_max_length", str(settings.model_max_length),
        "--gradient_checkpointing", "True", "--dataloader_num_workers", "0",
        "--report_to", "none", "--remove_unused_columns", "False",
        "--per_device_train_batch_size", str(settings.per_device_batch_size),
        "--max_steps", str(settings.max_steps),
        "--learning_rate", str(settings.learning_rate),
        "--seed", str(settings.seed), "--data_seed", str(settings.seed),
        "--output_dir", str(settings.run_root / "model"),
    ]


def execute(settings: Settings) -> int:
    locks = acquire_gpu_locks(settings.runs_root, (settings.gpu_id,))
    latest_log: Path | None = None
    try:
        require_gpu_free(settings)
        if settings.run_root.exists():
            raise RunFailure(f"Run output already exists: {settings.run_root}", exit_code=EXIT_EXISTS)
        metadata = settings.run_root / "metadata"
        metadata.mkdir(parents=True)
        status = {
            "status": "auditing",
            "started_at": timestamp(),
            "run_id": settings.run_id,
            "baseline": "c1_got2_ocr_only",
            "physical_gpu": settings.gpu_id,
            "max_steps": settings.max_steps,
            "source_model": str(settings.source_model),
        }
        write_status(metadata / "status.txt", status)
        resolved = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()}
        write_json(metadata / "resolved_settings.json", resolved)
        provenance = {
            "source_model": str(settings.source_model),
            "source_model_sha256": None if settings.skip_source_hash else file_sha256(settings.source_model / "model.safetensors"),
            "train_manifest_sha256": file_sha256(settings.train_manifest),
            "validation_manifest_sha256": file_sha256(settings.validation_manifest),
            "test_manifest_sha256": file_sha256(settings.test_manifest),
        }
        write_json(metadata / "provenance.json", provenance)

        audit_summary = metadata / "audit_summary.json"
        latest_log = metadata / "audit.log"
        audit_command = [
            sys.executable,
            str(settings.ocrmodel_root / "tools" / "preprocessing" / "audit_synthetic_layout.py"),
        ]
        for manifest in (settings.train_manifest, settings.validation_manifest, settings.test_manifest):
            audit_command.extend(("--manifest", str(manifest)))
        audit_command.extend(("--summary-json", str(audit_summary)))
        with latest_log.open("w", encoding="utf-8") as log:
            audit = subprocess.run(audit_command, env=environment(settings, cuda=False), stdout=log, stderr=subprocess.STDOUT, text=True)
        if audit.returncode != 0 or json.loads(audit_summary.read_text(encoding="utf-8")).get("status") != "ok":
            raise RunFailure("C1 group-isolated dataset audit failed.", log_path=latest_log)

        require_gpu_free(settings)
        status["status"] = "training"
        write_status(metadata / "status.txt", status)
        command = training_command(settings)
        write_json(metadata / "command.json", command)
        latest_log = settings.run_root / "train.log"
        with latest_log.open("w", encoding="utf-8") as log:
            trained = subprocess.run(command, cwd=settings.project_root, env=environment(settings), stdout=log, stderr=subprocess.STDOUT, text=True)
        metrics_path = settings.run_root / "model" / "page_ocr_training_metrics.json"
        if trained.returncode != 0 or not metrics_path.is_file():
            raise RunFailure(f"C1 training failed with exit code {trained.returncode}.", log_path=latest_log)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if int(metrics.get("global_step", -1)) != settings.max_steps:
            raise RunFailure("C1 optimizer step count does not match --max-steps.", log_path=latest_log)

        verification = metadata / "checkpoint_verification.json"
        verify_log = metadata / "checkpoint_verification.log"
        verify_command = [
            sys.executable,
            str(settings.project_root / "scripts" / "verify_linelevel_checkpoint.py"),
            "--source-model", str(settings.source_model),
            "--trained-model", str(settings.run_root / "model"),
            "--metrics-name", "page_ocr_training_metrics.json",
            "--expected-train-scope", settings.train_scope,
            "--output", str(verification),
        ]
        if settings.max_steps == 1:
            verify_command.append("--allow-no-observed-trainable-delta")
        with verify_log.open("w", encoding="utf-8") as log:
            checked = subprocess.run(verify_command, cwd=settings.project_root, env=environment(settings, cuda=False), stdout=log, stderr=subprocess.STDOUT, text=True)
        if checked.returncode != 0 or not verification.is_file():
            raise RunFailure("C1 checkpoint verification failed.", log_path=verify_log)

        summary = {
            "status": "ok",
            "baseline": "c1_got2_ocr_only",
            "run_id": settings.run_id,
            "model": str(settings.run_root / "model"),
            "physical_gpu": settings.gpu_id,
            "training_metrics": metrics,
            "checkpoint_verification": str(verification),
            "post_training_validation": "skipped",
        }
        write_json(settings.run_root / "summary.json", summary)
        status.update({"status": "completed", "finished_at": timestamp(), "model": summary["model"]})
        write_status(metadata / "status.txt", status)
        (settings.run_root / "GOT_PAGE_OCR_FINISHED").touch()
        print(compact_json({"event": "got2_page_ocr_completed", **summary}), flush=True)
        return 0
    except RunFailure as exc:
        print(compact_json({"event": "got2_page_ocr_failed", "run_id": settings.run_id, "exit_code": exc.exit_code, "error": bounded(str(exc)), "log": str(exc.log_path) if exc.log_path else None, "tail": tail_lines(exc.log_path or latest_log)}), flush=True)
        return exc.exit_code
    except Exception as exc:
        print(compact_json({"event": "got2_page_ocr_failed", "run_id": settings.run_id, "exit_code": EXIT_FAILURE, "error": bounded(str(exc)), "log": str(latest_log) if latest_log else None, "tail": tail_lines(latest_log)}), flush=True)
        return EXIT_FAILURE
    finally:
        for lock in reversed(locks):
            lock.close()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        settings = resolve_settings(parse_args(argv))
    except RunFailure as exc:
        print(compact_json({"event": "got2_page_ocr_rejected", "exit_code": exc.exit_code, "error": bounded(str(exc))}), flush=True)
        return exc.exit_code
    except Exception as exc:
        print(compact_json({"event": "got2_page_ocr_rejected", "exit_code": EXIT_FAILURE, "error": bounded(str(exc))}), flush=True)
        return EXIT_FAILURE
    return execute(settings)


if __name__ == "__main__":
    raise SystemExit(main())
