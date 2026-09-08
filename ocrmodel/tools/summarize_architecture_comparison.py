#!/usr/bin/env python3
"""Validation-only aggregation for the four-group architecture comparison."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


SEEDS = (42, 43, 44)
MODES = ("attention", "geometry", "layout_ot")
DEFAULT_MAX_STEPS = 256
DEFAULT_LR_SCHEDULE_STEPS = 128
DEFAULT_CHECKPOINT_STEPS = (64, 128, 192, 256)
# Kept as a public compatibility alias for callers that imported the old name.
CHECKPOINT_STEPS = DEFAULT_CHECKPOINT_STEPS
REPORT_METRICS = (
    "cer",
    "exact_page_rate",
    "generation_limit_hit_rate",
    "generation_eos_hit_rate",
    "generation_mean_new_tokens",
)
FINITE_TRAIN_FIELDS = (
    "ocr_loss",
    "auxiliary_loss",
    "total_loss",
    "gradient_norm",
    "learning_rate",
    "raw_content_gate",
    "effective_residual_scale",
)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_checkpoint_steps(value: str) -> tuple[int, ...]:
    """Parse actual checkpoint steps; step 0 is the identity diagnostic only."""

    if not value.strip():
        raise argparse.ArgumentTypeError("checkpoint steps must not be empty")
    parsed = []
    for item in value.replace(",", " ").split():
        try:
            step = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid checkpoint step: {item!r}") from exc
        if step <= 0:
            raise argparse.ArgumentTypeError(
                "checkpoint steps must be positive; step 0 is the identity diagnostic"
            )
        parsed.append(step)
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("checkpoint steps must be unique")
    return tuple(sorted(parsed))


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_train_metrics(path: Path) -> None:
    rows = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        rows += 1
        for field in FINITE_TRAIN_FIELDS:
            if not finite_number(row.get(field)):
                raise ValueError(f"non-finite or missing {field} at {path}:{line_number}")
        if row.get("parameters_finite") is not True:
            raise ValueError(f"parameter finite check failed at {path}:{line_number}")
    if not rows:
        raise ValueError(f"empty training metrics: {path}")


def validate_config(
    metadata: dict[str, Any],
    *,
    mode: str,
    seed: int,
    auxiliary_weight: float,
    max_steps: int,
    lr_schedule_steps: int,
) -> None:
    expected = {
        "mode": mode,
        "seed": seed,
        "max_steps": max_steps,
        "lr_schedule_steps": lr_schedule_steps,
        "auxiliary_weight": auxiliary_weight,
        "adapter_precision": "fp32",
        "layout_loss_profile": "full",
        "query_assignment": "hungarian",
        "test_used_for_selection": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"metadata {key} mismatch for {mode}/seed{seed}: {metadata.get(key)!r}")


def validate_candidate(candidate: dict[str, Any], path: Path) -> None:
    if candidate.get("test_used_for_selection") is not False:
        raise ValueError(f"candidate is not validation-only: {path}")
    health = candidate.get("checkpoint_health")
    if not isinstance(health, dict) or health.get("checkpoint_finite") is not True:
        raise ValueError(f"checkpoint finite check failed: {path}")
    if health.get("parameters_finite") is not True:
        raise ValueError(f"parameter finite check failed: {path}")
    for metric in REPORT_METRICS:
        if not finite_number(candidate.get(metric)):
            raise ValueError(f"non-finite or missing {metric}: {path}")


def read_training_run(
    root: Path,
    seed: int,
    mode: str,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    lr_schedule_steps: int = DEFAULT_LR_SCHEDULE_STEPS,
    checkpoint_steps: tuple[int, ...] = DEFAULT_CHECKPOINT_STEPS,
) -> list[dict[str, Any]]:
    run_dir = root / f"seed{seed}" / f"{mode}_aux0.2"
    metadata = read_json(run_dir / "metadata.json")
    validate_config(
        metadata,
        mode=mode,
        seed=seed,
        auxiliary_weight=0.2,
        max_steps=max_steps,
        lr_schedule_steps=lr_schedule_steps,
    )
    summary = read_json(run_dir / "summary.json")
    if summary.get("status") != "complete" or summary.get("test_used_for_selection") is not False:
        raise ValueError(f"incomplete or non-validation-only summary: {run_dir}")
    if summary.get("mode") != mode or summary.get("seed") != seed:
        raise ValueError(f"summary identity mismatch: {run_dir}")
    training = summary.get("training") or {}
    if training.get("steps") != max_steps or training.get("lr_schedule_steps") != lr_schedule_steps:
        raise ValueError(f"training horizon mismatch: {run_dir}")
    validate_train_metrics(run_dir / "train_metrics.jsonl")
    candidates = summary.get("selection_candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"missing validation candidates: {run_dir}")
    by_step = {candidate.get("step"): candidate for candidate in candidates}
    if set(by_step) != set(checkpoint_steps):
        raise ValueError(f"checkpoint set mismatch in {run_dir}: {sorted(by_step)}")
    rows = []
    for step in checkpoint_steps:
        candidate = by_step[step]
        validate_candidate(candidate, run_dir / f"checkpoint-{step}")
        rows.append({"mode": mode, "seed": seed, "step": step, **candidate})
    return rows


def read_baseline(
    root: Path,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    lr_schedule_steps: int = DEFAULT_LR_SCHEDULE_STEPS,
) -> dict[str, Any]:
    run_dir = root / "content_only_eval"
    metadata = read_json(run_dir / "metadata.json")
    validate_config(
        metadata,
        mode="content_only",
        seed=42,
        auxiliary_weight=0.0,
        max_steps=max_steps,
        lr_schedule_steps=lr_schedule_steps,
    )
    summary = read_json(run_dir / "summary.json")
    if (
        summary.get("status") != "complete"
        or summary.get("test_used_for_selection") is not False
        or summary.get("mode") != "content_only"
        or summary.get("eval_only") is not True
        or summary.get("training_updates") != 0
        or summary.get("parameters_unchanged") is not True
    ):
        raise ValueError(f"invalid content-only baseline: {run_dir}")
    validation = summary.get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"missing baseline validation: {run_dir}")
    for metric in REPORT_METRICS:
        if not finite_number(validation.get(metric)):
            raise ValueError(f"non-finite or missing baseline {metric}: {run_dir}")
    return {
        "mode": "content_only",
        "seed": 42,
        "eval_only": True,
        "training_updates": 0,
        **{metric: validation[metric] for metric in REPORT_METRICS},
    }


def aggregate_rows(
    records: list[dict[str, Any]],
    checkpoint_steps: tuple[int, ...] = DEFAULT_CHECKPOINT_STEPS,
) -> list[dict[str, Any]]:
    aggregates = []
    for mode in MODES:
        for step in checkpoint_steps:
            rows = [row for row in records if row["mode"] == mode and row["step"] == step]
            if [row["seed"] for row in rows] != list(SEEDS):
                raise ValueError(f"missing or reordered seeds for {mode} checkpoint-{step}")
            aggregate: dict[str, Any] = {
                "mode": mode,
                "step": step,
                "seeds": [
                    {"seed": row["seed"], **{metric: row[metric] for metric in REPORT_METRICS}}
                    for row in rows
                ],
            }
            for metric in REPORT_METRICS:
                values = [float(row[metric]) for row in rows]
                aggregate[f"mean_{metric}"] = statistics.fmean(values)
                aggregate[f"std_{metric}"] = statistics.pstdev(values)
            aggregates.append(aggregate)
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="expected training horizon; defaults to the original 256-step protocol",
    )
    parser.add_argument(
        "--lr-schedule-steps",
        type=int,
        default=DEFAULT_LR_SCHEDULE_STEPS,
        help="expected learning-rate schedule horizon",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=parse_checkpoint_steps,
        default=DEFAULT_CHECKPOINT_STEPS,
        help=(
            "comma- or space-separated actual checkpoint steps; step 0 is the "
            "identity diagnostic and is not a selection candidate"
        ),
    )
    args = parser.parse_args()
    if args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.lr_schedule_steps <= 0 or args.lr_schedule_steps > args.max_steps:
        parser.error("--lr-schedule-steps must be positive and no greater than --max-steps")
    if not args.checkpoint_steps:
        parser.error("--checkpoint-steps must not be empty")
    if any(step > args.max_steps for step in args.checkpoint_steps):
        parser.error("checkpoint steps must not exceed --max-steps")
    if 256 not in args.checkpoint_steps:
        parser.error("--checkpoint-steps must include 256 for the short-horizon comparison")

    records = [
        row
        for mode in MODES
        for seed in SEEDS
        for row in read_training_run(
            args.run_root,
            seed,
            mode,
            max_steps=args.max_steps,
            lr_schedule_steps=args.lr_schedule_steps,
            checkpoint_steps=args.checkpoint_steps,
        )
    ]
    aggregates = aggregate_rows(records, args.checkpoint_steps)
    baseline = read_baseline(
        args.run_root,
        max_steps=args.max_steps,
        lr_schedule_steps=args.lr_schedule_steps,
    )
    baseline_cer = float(baseline["cer"])
    short_horizon_steps = tuple(step for step in args.checkpoint_steps if step <= 256)
    long_horizon_steps = tuple(step for step in args.checkpoint_steps if step > 256)
    mode_results = []
    for mode in MODES:
        mode_rows = [row for row in aggregates if row["mode"] == mode]
        best = min(mode_rows, key=lambda row: (row["mean_cer"], row["step"]))
        short_rows = [row for row in mode_rows if row["step"] in short_horizon_steps]
        short_best = min(short_rows, key=lambda row: (row["mean_cer"], row["step"]))
        result = {
            "mode": mode,
            "best_step": best["step"],
            "best_mean_cer": best["mean_cer"],
            "best_gain_vs_content_only": baseline_cer - float(best["mean_cer"]),
            "short_horizon_best_step": short_best["step"],
            "short_horizon_best_mean_cer": short_best["mean_cer"],
        }
        for step in args.checkpoint_steps:
            if step in (256, max(args.checkpoint_steps)):
                row = next(row for row in mode_rows if row["step"] == step)
                result[f"step{step}_mean_cer"] = row["mean_cer"]
                result[f"step{step}_gain_vs_content_only"] = (
                    baseline_cer - float(row["mean_cer"])
                )
        if long_horizon_steps:
            long_rows = [row for row in mode_rows if row["step"] in long_horizon_steps]
            long_best = min(long_rows, key=lambda row: (row["mean_cer"], row["step"]))
            improved_seed_count = 0
            for seed in SEEDS:
                seed_short = min(
                    [
                        row
                        for row in records
                        if row["mode"] == mode
                        and row["seed"] == seed
                        and row["step"] in short_horizon_steps
                    ],
                    key=lambda row: (row["cer"], row["step"]),
                )
                seed_long = min(
                    [
                        row
                        for row in records
                        if row["mode"] == mode
                        and row["seed"] == seed
                        and row["step"] in long_horizon_steps
                    ],
                    key=lambda row: (row["cer"], row["step"]),
                )
                improved_seed_count += int(float(seed_long["cer"]) < float(seed_short["cer"]))
            result.update(
                {
                    "long_horizon_best_step": long_best["step"],
                    "long_horizon_best_mean_cer": long_best["mean_cer"],
                    "long_horizon_improved_seed_count": improved_seed_count,
                    "long_horizon_effective": (
                        float(long_best["mean_cer"]) < float(short_best["mean_cer"])
                        and improved_seed_count >= 2
                    ),
                }
            )
        else:
            result.update(
                {
                    "long_horizon_best_step": None,
                    "long_horizon_best_mean_cer": None,
                    "long_horizon_improved_seed_count": 0,
                    "long_horizon_effective": False,
                }
            )
        mode_results.append(result)
    selected = min(mode_results, key=lambda row: (row["best_mean_cer"], row["best_step"]))
    output = {
        "status": "complete",
        "comparison": "content_only_vs_attention_geometry_layout_ot",
        "config": {
            "max_steps": args.max_steps,
            "lr_schedule_steps": args.lr_schedule_steps,
            "adapter_precision": "fp32",
            "layout_loss_profile": "full",
            "query_assignment": "hungarian",
            "seeds": list(SEEDS),
            "checkpoint_steps": list(args.checkpoint_steps),
            "identity_diagnostic_step": 0,
        },
        "selection_metric": "mean_validation_cer_across_seeds",
        "selected_mode": selected["mode"],
        "selected_step": selected["best_step"],
        "selected_mean_cer": selected["best_mean_cer"],
        "aggregates": aggregates,
        "mode_results": mode_results,
        "content_only_baseline": baseline,
        "long_horizon_effective_for_any_mode": any(
            result["long_horizon_effective"] for result in mode_results
        ),
        "test_used_for_selection": False,
    }
    target = args.run_root / "architecture_comparison.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
