#!/usr/bin/env python3
"""Aggregate the three full-MTHv2 geometry runs using validation only."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_STEPS = tuple(range(432, 3457, 432))
METRICS = (
    "cer",
    "exact_page_rate",
    "low_frequency_k1_recall",
    "low_frequency_k3_recall",
    "low_frequency_k5_recall",
    "generation_limit_hit_rate",
    "layout_box_mae",
    "layout_direction_accuracy",
    "residual_relative_norm",
    "writeback_residual_relative_norm",
    "transport_entropy",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _parse_int_list(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return result


def _candidate_map(summary: dict[str, Any], *, seed: int, expected_steps: tuple[int, ...]) -> dict[int, dict[str, Any]]:
    if summary.get("status") != "complete":
        raise ValueError(f"seed {seed} summary is not complete")
    if summary.get("test_used_for_selection") is not False:
        raise ValueError(f"seed {seed} summary does not prove test exclusion")
    candidates = summary.get("selection_candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"seed {seed} has no selection_candidates")
    by_step: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        step = int(candidate["step"])
        if step in by_step:
            raise ValueError(f"seed {seed} has duplicate candidate step {step}")
        health = candidate.get("checkpoint_health") or {}
        if health.get("parameters_finite") is not True or health.get("checkpoint_finite") is not True:
            raise ValueError(f"seed {seed} checkpoint finite check failed at step {step}")
        if candidate.get("test_used_for_selection") is not False:
            raise ValueError(f"seed {seed} candidate {step} does not prove test exclusion")
        for key in ("cer", "residual_relative_norm", "generation_limit_hit_rate"):
            if candidate.get(key) is not None and not math.isfinite(float(candidate[key])):
                raise ValueError(f"seed {seed} candidate {step} has non-finite {key}")
        by_step[step] = candidate
    missing = [step for step in expected_steps if step not in by_step]
    if missing:
        raise ValueError(f"seed {seed} is missing validation candidates: {missing}")
    return by_step


def _metric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, float] | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return {"mean": statistics.mean(values), "std": statistics.pstdev(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--steps", type=_parse_int_list, default=DEFAULT_STEPS)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--late-degradation-threshold",
        type=float,
        default=0.10,
        help="CER increase over a seed's best validation CER considered catastrophic",
    )
    args = parser.parse_args()
    if args.late_degradation_threshold <= 0:
        parser.error("--late-degradation-threshold must be positive")
    run_root = args.run_root.resolve()
    if len(args.seeds) != 3:
        parser.error("the full protocol requires exactly three seeds")
    expected_steps = tuple(sorted(set(args.steps)))
    if expected_steps != tuple(args.steps):
        parser.error("--steps must be strictly increasing")

    seed_candidates: dict[str, dict[int, dict[str, Any]]] = {}
    seed_summaries: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        seed_dir = run_root / f"seed{seed}"
        summary = _read_json(seed_dir / "summary.json")
        metadata = _read_json(seed_dir / "metadata.json")
        if metadata.get("status") != "complete":
            raise ValueError(f"seed {seed} metadata is not complete")
        if (
            metadata.get("world_size") != 5
            or metadata.get("num_queries") != 512
            or metadata.get("global_batch_size") not in (None, 5)
        ):
            raise ValueError(f"seed {seed} does not match the five-card/512-query protocol")
        if metadata.get("distributed_strategy") != "ddp":
            raise ValueError(f"seed {seed} is not a DDP run")
        if metadata.get("mode") != "geometry":
            raise ValueError(f"seed {seed} is not a geometry run")
        if (
            metadata.get("adapter_precision") != "fp32"
            or metadata.get("layout_loss_profile") != "full"
            or metadata.get("query_assignment") != "hungarian"
            or metadata.get("processor_mode") != "fast"
        ):
            raise ValueError(f"seed {seed} does not match the locked full-MTHv2 configuration")
        candidates = _candidate_map(summary, seed=seed, expected_steps=expected_steps)
        seed_candidates[str(seed)] = candidates
        seed_summaries[str(seed)] = summary

    by_step: dict[str, dict[str, Any]] = {}
    for step in expected_steps:
        rows = [seed_candidates[str(seed)][step] for seed in args.seeds]
        metrics = {key: _metric_stats(rows, key) for key in METRICS}
        metrics = {key: value for key, value in metrics.items() if value is not None}
        by_step[str(step)] = {
            "step": step,
            "seed_metrics": {
                str(seed): {key: row.get(key) for key in METRICS if row.get(key) is not None}
                for seed, row in zip(args.seeds, rows)
            },
            "metrics": metrics,
            "max_residual_relative_norm": max(
                float(row["residual_relative_norm"])
                for row in rows
                if row.get("residual_relative_norm") is not None
            ),
            "max_generation_limit_hit_rate": max(
                float(row["generation_limit_hit_rate"])
                for row in rows
                if row.get("generation_limit_hit_rate") is not None
            ),
        }

    selected = min(
        (by_step[str(step)] for step in expected_steps),
        key=lambda row: (row["metrics"]["cer"]["mean"], row["step"]),
    )
    selected_step = int(selected["step"])
    late_by_seed: dict[str, dict[str, Any]] = {}
    for seed in args.seeds:
        rows = [seed_candidates[str(seed)][step] for step in expected_steps]
        best_cer = min(float(row["cer"]) for row in rows)
        final_cer = float(rows[-1]["cer"])
        late_by_seed[str(seed)] = {
            "best_cer": best_cer,
            "final_cer": final_cer,
            "final_delta_from_best": final_cer - best_cer,
            "catastrophic": final_cer - best_cer >= args.late_degradation_threshold,
        }
    all_seed_late_catastrophic = all(row["catastrophic"] for row in late_by_seed.values())
    stability = {
        "passed": (
            selected["max_residual_relative_norm"] <= 0.01
            and selected["max_generation_limit_hit_rate"] < 0.10
            and not all_seed_late_catastrophic
        ),
        "selected_max_residual_relative_norm": selected["max_residual_relative_norm"],
        "selected_max_generation_limit_hit_rate": selected["max_generation_limit_hit_rate"],
        "late_degradation_threshold_cer": args.late_degradation_threshold,
        "late_by_seed": late_by_seed,
        "all_seed_late_catastrophic": all_seed_late_catastrophic,
        "checkpoint_health_passed": True,
    }
    result = {
        "status": "complete",
        "dataset": "MTHv2",
        "mode": "geometry",
        "seeds": list(args.seeds),
        "checkpoint_steps": list(expected_steps),
        "selected_step": selected_step,
        "selection_metric": "mean_validation_cer",
        "selected_validation": selected,
        "by_step": by_step,
        "stability": stability,
        "seed_runs": {str(seed): str((run_root / f"seed{seed}").resolve()) for seed in args.seeds},
        "test_used_for_selection": False,
    }
    output = (args.output or (run_root / "selection.json")).resolve()
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
