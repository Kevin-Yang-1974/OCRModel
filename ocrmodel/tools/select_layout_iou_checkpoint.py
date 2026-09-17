#!/usr/bin/env python3
"""Select a layout checkpoint by validation box IoU only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _steps(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from exc
    if not parsed or any(step <= 0 for step in parsed) or parsed != tuple(sorted(set(parsed))):
        raise argparse.ArgumentTypeError("steps must be strictly increasing positive integers")
    return parsed


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--steps", type=_steps, required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--expected-world-size", type=int, default=5)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    validation_root = args.validation_root.resolve()
    group_root = run_dir.parent
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "metadata.json"
    if not (run_dir / "COMPLETED").is_file():
        raise RuntimeError(f"training run is not complete: {run_dir}")
    summary = _read(summary_path)
    metadata = _read(metadata_path)
    if summary.get("status") != "complete" or metadata.get("status") != "complete":
        raise RuntimeError("layout selection requires a complete training run")
    if summary.get("test_manifest_read") is not False or metadata.get("test_manifest_read") is not False:
        raise ValueError("layout selection requires a test-free training run")
    if summary.get("test_used_for_selection") is not False or metadata.get("test_used_for_selection") is not False:
        raise ValueError("layout selection must remain test-free")
    if metadata.get("distributed_strategy") != "ddp":
        raise ValueError("layout training run is not DDP")
    if args.expected_world_size > 0 and metadata.get("world_size") != args.expected_world_size:
        raise ValueError(
            f"world size mismatch: expected {args.expected_world_size}, got {metadata.get('world_size')!r}"
        )
    objective = ((summary.get("training") or {}).get("loss_objective") or {})
    if objective.get("layout_only") is not True:
        raise ValueError("training run did not disable the text-recognition objective")
    if objective.get("extra_terms") != []:
        raise ValueError("layout-only selection found extra objective terms")

    candidates: list[dict[str, Any]] = []
    for step in args.steps:
        checkpoint = run_dir / f"checkpoint-{step}"
        for filename in ("adapter.safetensors", "decoder_lora.safetensors"):
            if not (checkpoint / filename).is_file():
                raise FileNotFoundError(checkpoint / filename)
        health = _read(checkpoint / "checkpoint_health.json")
        if health.get("step") != step or health.get("checkpoint_finite") is not True:
            raise ValueError(f"checkpoint health failed at {checkpoint}")
        eval_dir = validation_root / f"step-{step}"
        if not (eval_dir / "COMPLETED").is_file():
            raise RuntimeError(f"validation evaluation is incomplete: {eval_dir}")
        eval_summary = _read(eval_dir / "summary.json")
        eval_metadata = _read(eval_dir / "metadata.json")
        if eval_summary.get("status") != "complete" or eval_summary.get("eval_only") is not True:
            raise ValueError(f"validation evaluation is not eval-only: {eval_dir}")
        if eval_summary.get("parameters_unchanged") is not True:
            raise ValueError(f"validation evaluation changed parameters: {eval_dir}")
        if eval_metadata.get("test_manifest_read") is not False:
            raise ValueError(f"validation evaluation reads test: {eval_dir}")
        validation = eval_summary.get("validation") or {}
        metric = validation.get("layout_box_iou")
        if not _finite(metric):
            raise ValueError(f"validation layout_box_iou is not finite at {eval_dir}: {metric!r}")
        candidates.append(
            {
                "step": step,
                "layout_box_iou": float(metric),
                "validation": validation,
                "checkpoint_health": health,
                "validation_run_dir": str(eval_dir),
                "test_used_for_selection": False,
            }
        )

    selected = max(candidates, key=lambda row: (float(row["layout_box_iou"]), -int(row["step"])))
    seed = int(metadata["seed"])
    selection = {
        "status": "complete",
        "dataset": args.dataset_label,
        "mode": metadata.get("mode"),
        "seed": seed,
        "seeds": [seed],
        "checkpoint_steps": list(args.steps),
        "selected_step": int(selected["step"]),
        "selection_metric": "validation_layout_box_iou",
        "selected_validation": selected,
        "candidates": candidates,
        "seed_runs": {str(seed): str(run_dir)},
        "validation_evaluated": True,
        "selection_performed": True,
        "selection_pending": False,
        "test_manifest_read": False,
        "test_used_for_selection": False,
    }
    serialized = json.dumps(selection, ensure_ascii=False, indent=2) + "\n"
    (run_dir / "selection.json").write_text(serialized, encoding="utf-8", newline="\n")
    (group_root / "selection.json").write_text(serialized, encoding="utf-8", newline="\n")
    summary.update(
        {
            "validation": selected,
            "validation_candidates": candidates,
            "selection_candidates": candidates,
            "selection": selection,
            "selection_pending": False,
            "validation_evaluated": True,
            "selection_performed": True,
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    metadata.update(
        {
            "selection_pending": False,
            "validation_evaluated": True,
            "selection_performed": True,
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    result = {
        "status": "complete",
        "run_dir": str(run_dir),
        "selected_step": int(selected["step"]),
        "selected_validation_layout_box_iou": float(selected["layout_box_iou"]),
        "checkpoint_steps": list(args.steps),
        "candidates": candidates,
        "test_used_for_selection": False,
    }
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
