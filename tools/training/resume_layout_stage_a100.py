#!/usr/bin/env python3
"""Resume one failed fixed-slot layout training stage in its existing run."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import run_layout_a100 as runner


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    settings = runner.resolve_settings(runner.parse_args(argv))
    if settings.stages != ("p2",):
        raise runner.RunFailure("Recovery supports one direct P2 stage.")
    runner.validate_paths(settings)
    runner.require_gpu_free(settings)
    if not settings.run_root.is_dir():
        raise runner.RunFailure(f"Run output does not exist: {settings.run_root}")

    output_dir = settings.run_root / "p2" / "model"
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[1]),
    )
    if not checkpoints:
        raise runner.RunFailure(f"No recovery checkpoint exists in {output_dir}")

    log_path = settings.run_root / "p2" / "train.recovery.log"
    status_path = settings.run_root / "metadata" / "status.txt"
    runner.write_status(
        status_path,
        {
            "status": "running_p2_recovery",
            "run_id": settings.run_id,
            "resumed_at": runner.timestamp(),
            "recovery_checkpoint": str(checkpoints[-1]),
            "physical_gpus": list(settings.physical_gpu_ids),
        },
    )
    command = runner.build_training_command(
        settings,
        stage="p2",
        source_model=settings.source_model,
        output_dir=output_dir,
        master_port=runner.free_master_port(),
    )
    with log_path.open("x", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=settings.project_root,
            env=runner.training_environment(settings),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode:
        raise runner.RunFailure(
            f"P2 recovery failed with exit code {completed.returncode}.",
            log_path=log_path,
        )

    metrics_path = output_dir / "layout_training_metrics.json"
    metrics = load_json(metrics_path)
    runner.validate_stage_metrics(
        metrics,
        stage="p2",
        expected_steps=settings.p2_max_steps,
        max_regions=settings.max_regions,
        ablation_id=settings.ablation_id,
    )
    loss = float(metrics["train_loss"])
    if not math.isfinite(loss):
        raise runner.RunFailure("Recovered P2 train loss is not finite.")

    summary_path = settings.run_root / "summary.json"
    summary = load_json(summary_path)
    summary.update(
        {
            "status": "training_completed_after_recovery",
            "recovered_at": runner.timestamp(),
            "recovery_checkpoint": str(checkpoints[-1]),
            "p2": {
                "global_step": int(metrics["global_step"]),
                "train_loss": loss,
                "model": str(output_dir),
                "metrics": str(metrics_path),
                "log": str(log_path),
            },
        }
    )
    runner.write_json(summary_path, summary)
    runner.write_status(
        status_path,
        {
            "status": "training_completed_after_recovery",
            "run_id": settings.run_id,
            "finished_at": runner.timestamp(),
            "global_step": int(metrics["global_step"]),
            "train_loss": loss,
            "physical_gpus": list(settings.physical_gpu_ids),
        },
    )
    (settings.run_root / "LAYOUT_A100_RECOVERED").touch()
    runner.emit(
        "layout_stage_recovery_completed",
        run_id=settings.run_id,
        global_step=int(metrics["global_step"]),
        train_loss=loss,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        runner.emit(
            "layout_stage_recovery_failed",
            error_type=type(exc).__name__,
            error=runner.bounded(str(exc)),
        )
        raise SystemExit(1)
