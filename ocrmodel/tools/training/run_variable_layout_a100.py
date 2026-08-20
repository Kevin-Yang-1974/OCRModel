#!/usr/bin/env python3
"""Bounded A100 launcher for the opt-in PVLD-32 prototype."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pretrain", "joint-train", "preflight"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--num-layout-prompt-queries", type=int, default=32)
    parser.add_argument("--max-layout-records", type=int, default=64)
    parser.add_argument("--max-layout-tokens", type=int, default=2048)
    parser.add_argument("--max-regions", type=int, default=32)
    parser.add_argument("--layout-decoder-layers", type=int, default=2)
    parser.add_argument("--layout-decoder-hidden-size", type=int, default=256)
    parser.add_argument("--layout-prompt-mode", default="global_prompt_full_page")
    parser.add_argument("--layout-loss-preset", default="sequence_bbox_direction")
    parser.add_argument("--layout-stage", choices=("p1", "p2"), required=True)
    parser.add_argument("--layout-bbox-loss-weight", type=float, default=5.0)
    parser.add_argument("--layout-type-loss-weight", type=float, default=1.0)
    parser.add_argument("--layout-direction-loss-weight", type=float, default=1.0)
    parser.add_argument("--layout-count-loss-weight", type=float, default=0.1)
    parser.add_argument("--layout-prompt-diversity-loss-weight", type=float, default=0.0)
    parser.add_argument("--ocr-loss-weight", type=float, default=0.0)
    parser.add_argument("--visual-feature-manifest", type=Path)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--gpu-id")
    group.add_argument("--gpu-ids")
    parser.add_argument("--gpu-utilization-limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-root", type=Path, default=Path(os.environ.get("GOT_TRAINING_RUNS", root.parent / "training_runs" / "GOT")))
    parser.add_argument("--project-root", type=Path, default=root / "src" / "GOT-OCR-2.0")
    return parser.parse_args()


def gpu_ids(args: argparse.Namespace) -> tuple[str, ...]:
    raw = args.gpu_ids if args.gpu_ids is not None else args.gpu_id
    return tuple(str(raw).split(","))


def require_target_gpus(ids: tuple[str, ...], limit: int) -> dict[str, int]:
    observed: dict[str, int] = {}
    for gpu_id in ids:
        completed = subprocess.run(
            ["nvidia-smi", "-i", gpu_id, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        value = completed.stdout.strip()
        if completed.returncode != 0 or not value.isdigit():
            raise RuntimeError(f"cannot query target GPU {gpu_id}")
        utilization = int(value)
        observed[gpu_id] = utilization
        if utilization >= limit:
            raise RuntimeError(f"GPU{gpu_id}_BUSY utilization={utilization} limit={limit}")
    return observed


def main() -> int:
    args = parse_args()
    ids = gpu_ids(args)
    utilization = require_target_gpus(ids, args.gpu_utilization_limit)
    run_root = args.runs_root.resolve() / args.run_id
    if run_root.exists():
        raise FileExistsError(f"run already exists; refusing duplicate launch: {run_root}")
    metadata = run_root / "metadata"
    metadata.mkdir(parents=True)
    started = datetime.now().astimezone().isoformat(timespec="seconds")
    status = {
        "status": "running",
        "run_id": args.run_id,
        "started_at": started,
        "physical_gpu_ids": ids,
        "gpu_utilization_at_admission": utilization,
        "layout_architecture": "PVLD-32",
        "implementation_scope": "standalone_feature_prototype",
    }
    (metadata / "status.txt").write_text(compact(status) + "\n", encoding="utf-8")
    trainer_output = run_root / "prototype"
    command = [
        sys.executable,
        str(args.project_root.resolve() / "scripts" / "train_GOT_variable_layout.py"),
        "--mode", args.mode,
        "--manifest", str(args.manifest.resolve()),
        "--model-name-or-path", str(args.source_model.resolve()),
        "--tokenizer-name-or-path", str(args.tokenizer_model.resolve()),
        "--num-layout-prompt-queries", str(args.num_layout_prompt_queries),
        "--max-layout-records", str(args.max_layout_records),
        "--max-layout-tokens", str(args.max_layout_tokens),
        "--max-regions", str(args.max_regions),
        "--layout-decoder-layers", str(args.layout_decoder_layers),
        "--layout-decoder-hidden-size", str(args.layout_decoder_hidden_size),
        "--layout-prompt-mode", args.layout_prompt_mode,
        "--layout-loss-preset", args.layout_loss_preset,
        "--layout-stage", args.layout_stage,
        "--layout-bbox-loss-weight", str(args.layout_bbox_loss_weight),
        "--layout-type-loss-weight", str(args.layout_type_loss_weight),
        "--layout-direction-loss-weight", str(args.layout_direction_loss_weight),
        "--layout-count-loss-weight", str(args.layout_count_loss_weight),
        "--layout-prompt-diversity-loss-weight", str(args.layout_prompt_diversity_loss_weight),
        "--ocr-loss-weight", str(args.ocr_loss_weight),
        "--max-steps", str(args.max_steps),
        "--seed", str(args.seed),
        "--run-id", args.run_id,
        "--gpu-ids", ",".join(ids),
        "--output-dir", str(trainer_output),
    ]
    if args.validation_manifest:
        command.extend(("--validation-manifest", str(args.validation_manifest.resolve())))
    if args.visual_feature_manifest:
        command.extend(("--visual-feature-manifest", str(args.visual_feature_manifest.resolve())))
    if args.smoke:
        command.append("--smoke")
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    log_path = run_root / "trainer.log"
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=args.project_root.resolve(), env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
    prototype_summary = trainer_output / "summary.json"
    if completed.returncode != 0 or not prototype_summary.is_file():
        status.update({"status": "failed", "returncode": completed.returncode, "log": str(log_path)})
        (metadata / "status.txt").write_text(compact(status) + "\n", encoding="utf-8")
        write_json(run_root / "summary.json", status)
        raise RuntimeError(f"PVLD prototype failed; see {log_path}")
    payload = json.loads(prototype_summary.read_text(encoding="utf-8"))
    final_status = "completed" if payload.get("training_executed") else "preflight_completed"
    status.update({"status": final_status, "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"), "prototype_summary": str(prototype_summary), "training_executed": bool(payload.get("training_executed", False))})
    (metadata / "status.txt").write_text(compact(status) + "\n", encoding="utf-8")
    write_json(run_root / "layout_training_metrics.json", payload)
    write_json(run_root / "summary.json", {**status, "metrics": payload})
    (run_root / "VARIABLE_LAYOUT_A100_FINISHED").touch()
    print(compact({"event": "variable_layout_a100_completed", "run_id": args.run_id, "status": final_status, "summary": str(run_root / "summary.json")}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(compact({"event": "variable_layout_a100_failed", "error_type": type(exc).__name__, "error": str(exc)[:800]}), file=sys.stderr)
        raise SystemExit(1)
