#!/usr/bin/env python3
"""Validation-only selection for the stabilized three-seed confirmation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


SEEDS = (42, 43, 44)
MODES = ("attention", "geometry")
CHECKPOINT_STEPS = (256, 512, 768, 1024)
FINITE_TRAIN_FIELDS = (
    "ocr_loss",
    "auxiliary_loss",
    "total_loss",
    "gradient_norm",
    "learning_rate",
    "raw_content_gate",
    "effective_residual_scale",
)
REPORT_METRICS = (
    "cer",
    "exact_page_rate",
    "low_frequency_k1_recall",
    "low_frequency_k3_recall",
    "low_frequency_k5_recall",
    "generation_limit_hit_rate",
)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_complete_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError(f"run is not complete: {path}")
    if summary.get("test_used_for_selection") is not False:
        raise ValueError(f"summary does not assert validation-only selection: {path}")
    return summary


def validate_train_metrics(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
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


def read_training_run(root: Path, seed: int, mode: str) -> list[dict[str, Any]]:
    run_dir = root / f"seed{seed}" / f"{mode}_aux0.2"
    summary = read_complete_summary(run_dir / "summary.json")
    if (
        summary.get("mode") != mode
        or summary.get("seed") != seed
        or summary.get("auxiliary_weight") != 0.2
        or summary.get("eval_only") is True
    ):
        raise ValueError(f"run identity mismatch: {run_dir}")
    validate_train_metrics(run_dir / "train_metrics.jsonl")
    candidates = summary.get("selection_candidates")
    if candidates is None:
        selection_path = run_dir / "selection.json"
        if not selection_path.is_file():
            raise FileNotFoundError(selection_path)
        candidates = json.loads(selection_path.read_text(encoding="utf-8")).get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"missing validation candidates: {run_dir}")
    by_step = {candidate.get("step"): candidate for candidate in candidates}
    if set(by_step) != set(CHECKPOINT_STEPS):
        raise ValueError(f"checkpoint set mismatch in {run_dir}: {sorted(by_step)}")

    rows = []
    for step in CHECKPOINT_STEPS:
        candidate = by_step[step]
        if candidate.get("test_used_for_selection") is not False:
            raise ValueError(f"candidate {run_dir}/checkpoint-{step} is not validation-only")
        health = candidate.get("checkpoint_health")
        if not isinstance(health, dict) or health.get("checkpoint_finite") is not True:
            raise ValueError(f"checkpoint finite check failed: {run_dir}/checkpoint-{step}")
        if health.get("parameters_finite") is not True:
            raise ValueError(f"parameter finite check failed: {run_dir}/checkpoint-{step}")
        for metric in REPORT_METRICS:
            if not finite_number(candidate.get(metric)):
                raise ValueError(
                    f"non-finite or missing validation {metric}: {run_dir}/checkpoint-{step}"
                )
        rows.append({"mode": mode, "seed": seed, "step": step, **candidate})
    return rows


def read_baseline(root: Path) -> dict[str, Any]:
    path = root / "content_only_eval" / "summary.json"
    summary = read_complete_summary(path)
    if (
        summary.get("mode") != "content_only"
        or summary.get("eval_only") is not True
        or summary.get("training_updates") != 0
        or summary.get("parameters_unchanged") is not True
    ):
        raise ValueError(f"invalid content-only eval baseline: {path}")
    validation = summary.get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"missing baseline validation metrics: {path}")
    for metric in REPORT_METRICS:
        if not finite_number(validation.get(metric)):
            raise ValueError(f"non-finite or missing baseline {metric}: {path}")
    return {
        "mode": "content_only",
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
    mode_priority = {"attention": 0, "geometry": 1}
    selected = min(
        aggregates,
        key=lambda row: (row["mean_cer"], row["step"], mode_priority[row["mode"]]),
    )
    paired = []
    by_key = {(row["mode"], row["seed"], row["step"]): row for row in records}
    for step in CHECKPOINT_STEPS:
        for seed in SEEDS:
            attention = by_key[("attention", seed, step)]
            geometry = by_key[("geometry", seed, step)]
            paired.append(
                {
                    "step": step,
                    "seed": seed,
                    "geometry_minus_attention_cer": float(geometry["cer"])
                    - float(attention["cer"]),
                }
            )

    violations = []
    mode_stability = []
    for mode in MODES:
        mode_rows = [row for row in aggregates if row["mode"] == mode]
        best = min(mode_rows, key=lambda row: (row["mean_cer"], row["step"]))
        final = next(row for row in mode_rows if row["step"] == 1024)
        regression = float(final["mean_cer"]) - float(best["mean_cer"])
        row = {
            "mode": mode,
            "best_step": best["step"],
            "best_mean_cer": best["mean_cer"],
            "step1024_mean_cer": final["mean_cer"],
            "step1024_cer_regression": regression,
            "step1024_mean_generation_limit_hit_rate": final[
                "mean_generation_limit_hit_rate"
            ],
        }
        mode_stability.append(row)
        if regression > 0.05:
            violations.append(f"{mode}: step-1024 CER regression {regression:.6f} > 0.05")
        if float(final["mean_generation_limit_hit_rate"]) > 0.10:
            violations.append(
                f"{mode}: step-1024 generation limit hit rate "
                f"{final['mean_generation_limit_hit_rate']:.6f} > 0.10"
            )
    over_limit = [
        {"mode": row["mode"], "seed": row["seed"], "step": row["step"], "cer": row["cer"]}
        for row in records
        if float(row["cer"]) > 0.5
    ]
    if over_limit:
        violations.append(f"{len(over_limit)} seed/checkpoint CER values exceed 0.5")

    output = {
        "status": "complete",
        "selection_metric": "mean_validation_cer_across_seeds",
        "tie_break": ["earlier_checkpoint", "attention"],
        "selected_mode": selected["mode"],
        "selected_step": selected["step"],
        "selected_mean_cer": selected["mean_cer"],
        "selected_std_cer": selected["std_cer"],
        "aggregates": aggregates,
        "paired_geometry_minus_attention": paired,
        "content_only_baseline": read_baseline(args.run_root),
        "stability": {
            "passed": not violations,
            "eligible_for_formal_test": not violations,
            "mode_results": mode_stability,
            "cer_over_0_5": over_limit,
            "violations": violations,
        },
        "test_used_for_selection": False,
    }
    target = args.run_root / "selection.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
