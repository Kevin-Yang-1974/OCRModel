#!/usr/bin/env python3
"""Select one checkpoint from four parallel validation-only evaluations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_STEPS = (5000, 10000, 15000, 20000)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _parse_steps(value: str) -> tuple[int, ...]:
    try:
        steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("steps must be comma-separated positive integers") from exc
    if not steps or any(step <= 0 for step in steps) or steps != tuple(sorted(set(steps))):
        raise argparse.ArgumentTypeError("steps must be strictly increasing positive integers")
    return steps


def _finite(value: Any) -> bool:
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _core_lora_config(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("decoder_lora_config must be an object")
    required = ("rank", "alpha", "dropout", "learning_rate")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"decoder_lora_config is missing fields: {missing}")
    return {key: value[key] for key in required}


def _check_common_metadata(
    training_metadata: dict[str, Any],
    eval_metadata: dict[str, Any],
    *,
    context: str,
) -> None:
    keys = (
        "mode",
        "num_queries",
        "adapter_precision",
        "layout_loss_profile",
        "query_assignment",
        "processor_mode",
        "decoder_adaptation",
        "generation_mode",
        "max_eval_new_tokens",
        "test_manifest_read",
        "test_used_for_selection",
        "model_path",
        "code_sha256",
    )
    for key in keys:
        if eval_metadata.get(key) != training_metadata.get(key):
            raise ValueError(
                f"{context} metadata mismatch for {key}: "
                f"training={training_metadata.get(key)!r}, eval={eval_metadata.get(key)!r}"
            )
    if _core_lora_config(eval_metadata.get("decoder_lora_config")) != _core_lora_config(
        training_metadata.get("decoder_lora_config")
    ):
        raise ValueError(f"{context} decoder LoRA configuration does not match training")


def _checkpoint_health(path: Path, *, step: int, decoder_adaptation: str) -> dict[str, Any]:
    health_path = path / "checkpoint_health.json"
    health = _read_json(health_path)
    if health.get("step") != step:
        raise ValueError(f"checkpoint health step mismatch at {path}: {health.get('step')!r}")
    if health.get("parameters_finite") is not True or health.get("checkpoint_finite") is not True:
        raise ValueError(f"adapter checkpoint finite check failed at {path}")
    if decoder_adaptation == "lora":
        decoder_health = health.get("decoder_lora") or {}
        decoder_report = decoder_health.get("decoder_lora") or {}
        if decoder_report.get("parameters_finite") is not True:
            raise ValueError(f"decoder LoRA parameter finite check failed at {path}")
        if decoder_health.get("checkpoint_finite") is not True:
            raise ValueError(f"decoder LoRA checkpoint finite check failed at {path}")
    return health


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--group-root", type=Path)
    parser.add_argument("--steps", type=_parse_steps, default=DEFAULT_STEPS)
    parser.add_argument("--expected-world-size", type=int, default=4)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    validation_root = args.validation_root.resolve()
    group_root = (args.group_root or run_dir.parent).resolve()
    steps = tuple(args.steps)
    if args.expected_world_size <= 0:
        parser.error("--expected-world-size must be positive")
    if not (run_dir / "COMPLETED").is_file():
        raise RuntimeError(f"training run is not complete: {run_dir}")

    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "metadata.json"
    summary = _read_json(summary_path)
    metadata = _read_json(metadata_path)
    if summary.get("status") != "complete" or metadata.get("status") != "complete":
        raise RuntimeError("parallel validation requires a complete training run")
    if summary.get("test_manifest_read") is not False or metadata.get("test_manifest_read") is not False:
        raise ValueError("parallel validation requires a no-test training protocol")
    if summary.get("test_used_for_selection") is not False or metadata.get("test_used_for_selection") is not False:
        raise ValueError("parallel validation must remain validation-only")
    if metadata.get("distributed_strategy") != "ddp":
        raise ValueError("training run is not a DDP run")
    if metadata.get("world_size") != args.expected_world_size:
        raise ValueError(
            f"training world size mismatch: expected {args.expected_world_size}, "
            f"got {metadata.get('world_size')!r}"
        )
    if metadata.get("global_batch_size") not in (None, args.expected_world_size):
        raise ValueError("training global batch size does not match four-card protocol")
    if metadata.get("decoder_adaptation") != "lora":
        raise ValueError("this parallel validation workflow requires decoder LoRA training")
    training = summary.get("training") or {}
    natural_loop_config = training.get("natural_loop") or {}
    if natural_loop_config.get("enabled") is not False:
        raise ValueError("training run unexpectedly enabled natural-loop objective")
    for name in ("scheduled_sampling", "loop_escape", "continuation_head"):
        if (training.get(name) or {}).get("enabled") is not False:
            raise ValueError(f"training run enabled extra objective: {name}")
    objective = training.get("loss_objective") or {}
    if objective.get("formula") != "L_official + auxiliary_weight * L_layout":
        raise ValueError(f"unexpected training loss objective: {objective}")
    if not math.isclose(float(objective.get("layout_weight", -1.0)), 0.2):
        raise ValueError("training layout loss weight is not 0.2")
    if objective.get("extra_terms") != []:
        raise ValueError(f"unexpected extra loss terms: {objective.get('extra_terms')}")

    recorded_steps = tuple(int(step) for step in training.get("checkpoint_steps", []))
    if recorded_steps != steps:
        raise ValueError(
            f"training checkpoint steps mismatch: expected {list(steps)}, got {list(recorded_steps)}"
        )

    candidates: list[dict[str, Any]] = []
    for step in steps:
        checkpoint_dir = run_dir / f"checkpoint-{step}"
        for filename in ("adapter.safetensors", "decoder_lora.safetensors"):
            if not (checkpoint_dir / filename).is_file():
                raise FileNotFoundError(checkpoint_dir / filename)
        health = _checkpoint_health(
            checkpoint_dir,
            step=step,
            decoder_adaptation=metadata["decoder_adaptation"],
        )
        eval_dir = validation_root / f"step-{step}"
        eval_summary_path = eval_dir / "summary.json"
        eval_metadata_path = eval_dir / "metadata.json"
        if not (eval_dir / "COMPLETED").is_file():
            raise RuntimeError(f"validation evaluation is not complete: {eval_dir}")
        eval_summary = _read_json(eval_summary_path)
        eval_metadata = _read_json(eval_metadata_path)
        if eval_summary.get("status") != "complete" or eval_metadata.get("status") != "complete":
            raise RuntimeError(f"validation evaluation is not complete: {eval_dir}")
        if eval_summary.get("eval_only") is not True or eval_summary.get("parameters_unchanged") is not True:
            raise ValueError(f"validation evaluation is not eval-only: {eval_dir}")
        if eval_summary.get("test_used_for_selection") is not False:
            raise ValueError(f"validation evaluation is not test-free: {eval_dir}")
        if eval_metadata.get("test_manifest_read") is not False:
            raise ValueError(f"validation metadata reads test: {eval_dir}")
        if eval_summary.get("decoder_adaptation") != "lora":
            raise ValueError(f"validation evaluation did not use decoder LoRA: {eval_dir}")
        if eval_summary.get("decoder_lora_loaded") is not True:
            raise ValueError(f"validation decoder LoRA was not loaded: {eval_dir}")
        decoder_lora_finite = eval_summary.get("decoder_lora_finite") or eval_metadata.get(
            "decoder_lora_finite"
        ) or {}
        if decoder_lora_finite.get("enabled") is not True or decoder_lora_finite.get("parameters_finite") is not True:
            raise ValueError(f"validation decoder LoRA finite check failed: {eval_dir}")
        _check_common_metadata(metadata, eval_metadata, context=str(eval_dir))
        validation = eval_summary.get("validation") or {}
        cer = validation.get("cer")
        if not _finite(cer):
            raise ValueError(f"validation CER is non-finite at {eval_dir}: {cer!r}")
        if validation.get("test_used_for_selection") is not False:
            raise ValueError(f"validation metrics are not test-free: {eval_dir}")
        candidate = {
            "step": step,
            **validation,
            "validation": validation,
            "checkpoint_health": health,
            "validation_run_dir": str(eval_dir),
            "decoder_lora_loaded": True,
            "test_used_for_selection": False,
        }
        candidates.append(candidate)

    selected = min(candidates, key=lambda row: (float(row["cer"]), int(row["step"])))
    seed = int(metadata["seed"])
    selection = {
        "status": "complete",
        "dataset": "MTHv2",
        "mode": metadata["mode"],
        "seed": seed,
        "seeds": [seed],
        "checkpoint_steps": list(steps),
        "selected_step": int(selected["step"]),
        "selection_metric": "validation_cer",
        "selected_validation": selected,
        "candidates": candidates,
        "seed_runs": {str(seed): str(run_dir)},
        "validation_evaluated": True,
        "selection_performed": True,
        "selection_pending": False,
        "test_manifest_read": False,
        "test_used_for_selection": False,
    }
    parallel_summary = {
        "status": "complete",
        "run_dir": str(run_dir),
        "validation_root": str(validation_root),
        "world_size": args.expected_world_size,
        "checkpoint_steps": list(steps),
        "selected_step": int(selected["step"]),
        "selected_validation_cer": float(selected["cer"]),
        "selection_metric": "validation_cer",
        "candidates": candidates,
        "test_manifest_read": False,
        "test_used_for_selection": False,
    }

    run_dir_selection = run_dir / "selection.json"
    group_selection = group_root / "selection.json"
    run_dir_selection.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    group_selection.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "parallel_validation_summary.json").write_text(
        json.dumps(parallel_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    summary.update(
        {
            "validation": selected,
            "validation_candidates": candidates,
            "selection_candidates": candidates,
            "selection": selection,
            "selection_pending": False,
            "validation_evaluated": True,
            "selection_performed": True,
            "parallel_validation_summary": str(run_dir / "parallel_validation_summary.json"),
            "test_manifest_read": False,
            "test_used_for_selection": False,
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata.update(
        {
            "selection_pending": False,
            "validation_evaluated": True,
            "selection_performed": True,
            "test_manifest_read": False,
            "test_used_for_selection": False,
        }
    )
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(parallel_summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
