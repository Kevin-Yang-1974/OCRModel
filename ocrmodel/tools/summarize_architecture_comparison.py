#!/usr/bin/env python3
"""Validation-only aggregation for the four-group 256-step architecture comparison."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


SEEDS = (42, 43, 44)
MODES = ("attention", "geometry", "layout_ot")
CHECKPOINT_STEPS = (64, 128, 192, 256)
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


def validate_config(metadata: dict[str, Any], *, mode: str, seed: int, auxiliary_weight: float) -> None:
    expected = {
        "mode": mode,
        "seed": seed,
        "max_steps": 256,
        "lr_schedule_steps": 128,
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


def read_training_run(root: Path, seed: int, mode: str) -> list[dict[str, Any]]:
    run_dir = root / f"seed{seed}" / f"{mode}_aux0.2"
    metadata = read_json(run_dir / "metadata.json")
    validate_config(metadata, mode=mode, seed=seed, auxiliary_weight=0.2)
    summary = read_json(run_dir / "summary.json")
    if summary.get("status") != "complete" or summary.get("test_used_for_selection") is not False:
        raise ValueError(f"incomplete or non-validation-only summary: {run_dir}")
    if summary.get("mode") != mode or summary.get("seed") != seed:
        raise ValueError(f"summary identity mismatch: {run_dir}")
    training = summary.get("training") or {}
    if training.get("steps") != 256 or training.get("lr_schedule_steps") != 128:
        raise ValueError(f"training horizon mismatch: {run_dir}")
    validate_train_metrics(run_dir / "train_metrics.jsonl")
    candidates = summary.get("selection_candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"missing validation candidates: {run_dir}")
    by_step = {candidate.get("step"): candidate for candidate in candidates}
    if set(by_step) != set(CHECKPOINT_STEPS):
        raise ValueError(f"checkpoint set mismatch in {run_dir}: {sorted(by_step)}")
    rows = []
    for step in CHECKPOINT_STEPS:
        candidate = by_step[step]
        validate_candidate(candidate, run_dir / f"checkpoint-{step}")
        rows.append({"mode": mode, "seed": seed, "step": step, **candidate})
    return rows


def read_baseline(root: Path) -> dict[str, Any]:
    run_dir = root / "content_only_eval"
    metadata = read_json(run_dir / "metadata.json")
    validate_config(metadata, mode="content_only", seed=42, auxiliary_weight=0.0)
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


def aggregate_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates = []
    for mode in MODES:
        for step in CHECKPOINT_STEPS:
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
    args = parser.parse_args()

    records = [
        row
        for mode in MODES
        for seed in SEEDS
        for row in read_training_run(args.run_root, seed, mode)
    ]
    aggregates = aggregate_rows(records)
    baseline = read_baseline(args.run_root)
    baseline_cer = float(baseline["cer"])
    mode_results = []
    for mode in MODES:
        mode_rows = [row for row in aggregates if row["mode"] == mode]
        best = min(mode_rows, key=lambda row: (row["mean_cer"], row["step"]))
        final = next(row for row in mode_rows if row["step"] == 256)
        mode_results.append(
            {
                "mode": mode,
                "best_step": best["step"],
                "best_mean_cer": best["mean_cer"],
                "step256_mean_cer": final["mean_cer"],
                "best_gain_vs_content_only": baseline_cer - float(best["mean_cer"]),
                "step256_gain_vs_content_only": baseline_cer - float(final["mean_cer"]),
            }
        )
    selected = min(mode_results, key=lambda row: (row["best_mean_cer"], row["best_step"]))
    output = {
        "status": "complete",
        "comparison": "content_only_vs_attention_geometry_layout_ot",
        "config": {
            "max_steps": 256,
            "lr_schedule_steps": 128,
            "adapter_precision": "fp32",
            "layout_loss_profile": "full",
            "query_assignment": "hungarian",
            "seeds": list(SEEDS),
            "checkpoint_steps": list(CHECKPOINT_STEPS),
        },
        "selection_metric": "mean_validation_cer_across_seeds",
        "selected_mode": selected["mode"],
        "selected_step": selected["best_step"],
        "selected_mean_cer": selected["best_mean_cer"],
        "aggregates": aggregates,
        "mode_results": mode_results,
        "content_only_baseline": baseline,
        "test_used_for_selection": False,
    }
    target = args.run_root / "architecture_comparison.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
