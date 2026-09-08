#!/usr/bin/env python3
"""Create a compact, identity-aware summary from diagnostic_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


LAYOUT_LOSS_KEYS = (
    "layout_box",
    "layout_order",
    "layout_direction",
    "layout_assignment",
    "transport_entropy",
)


def _diagnostic_path(source: Path) -> Path:
    if source.is_dir():
        return source / "diagnostic_summary.json"
    return source


def _compact_point(point: dict[str, Any], identity_cer: float | None) -> dict[str, Any]:
    validation = point.get("validation") or {}
    training = point.get("training") or {}
    loss_components = training.get("loss_components") or {}
    gradient_norms = training.get("gradient_norms")
    return {
        "step": point["step"],
        "cer": validation.get("cer"),
        "identity_delta_cer": (
            validation.get("cer") - identity_cer
            if identity_cer is not None and validation.get("cer") is not None
            else None
        ),
        "exact_page_rate": validation.get("exact_page_rate"),
        "generation_limit_hit_rate": validation.get("generation_limit_hit_rate"),
        "generation_eos_hit_rate": validation.get("generation_eos_hit_rate"),
        "generation_mean_new_tokens": validation.get("generation_mean_new_tokens"),
        "generation_lengths": validation.get("generation_lengths"),
        "training_loss_components": loss_components,
        "validation_layout_loss_means": validation.get("layout_loss_means"),
        "gradient_norms": gradient_norms,
        "teacher_forced_ocr_loss": validation.get("teacher_forced_ocr_loss"),
        "raw_content_gate": validation.get("raw_content_gate"),
        "effective_residual_scale": validation.get("effective_residual_scale"),
        "residual_relative_norm": validation.get("residual_relative_norm"),
        "writeback_residual_relative_norm": validation.get("writeback_residual_relative_norm"),
        "transport_entropy": validation.get("transport_entropy"),
        "transport_query_mass": validation.get("transport_query_mass"),
        "fusion_query_mass": validation.get("fusion_query_mass"),
        "invalid_query_transport_mass": validation.get("invalid_query_transport_mass"),
        "invalid_query_fusion_mass": validation.get("invalid_query_fusion_mass"),
        "training_dtypes": {
            "adapter": training.get("adapter_dtypes"),
            "loss": training.get("loss_dtypes"),
        },
        "validation_dtypes": {
            "adapter": validation.get("adapter_dtypes"),
            "loss": validation.get("layout_loss_dtypes"),
        },
    }


def summarize(source: Path) -> dict[str, Any]:
    path = _diagnostic_path(source)
    data = json.loads(path.read_text(encoding="utf-8"))
    points = sorted(data.get("points", []), key=lambda row: row["step"])
    if not points:
        raise ValueError(f"diagnostic summary has no points: {path}")
    identity = next((point for point in points if point["step"] == 0), None)
    identity_cer = identity.get("validation", {}).get("cer") if identity else None
    trained = [point for point in points if point["step"] > 0]
    best_trained = min(
        trained,
        key=lambda point: (point["validation"]["cer"], point["step"]),
    ) if trained else None
    best_vs_identity = min(
        points,
        key=lambda point: (point["validation"]["cer"], point["step"]),
    ) if identity else None
    compact_points = [_compact_point(point, identity_cer) for point in points]

    curves: dict[str, list[dict[str, Any]]] = {}
    for key in (
        "cer",
        "identity_delta_cer",
        "exact_page_rate",
        "generation_limit_hit_rate",
        "generation_eos_hit_rate",
        "teacher_forced_ocr_loss",
        "raw_content_gate",
        "effective_residual_scale",
        "residual_relative_norm",
        "writeback_residual_relative_norm",
        "transport_entropy",
        "invalid_query_transport_mass",
        "invalid_query_fusion_mass",
    ):
        curves[key] = [
            {"step": point["step"], "value": compact_point[key]}
            for point, compact_point in zip(points, compact_points)
        ]
    curves["transport_query_mass"] = [
        {"step": point["step"], "value": compact_point["transport_query_mass"]}
        for point, compact_point in zip(points, compact_points)
    ]
    curves["fusion_query_mass"] = [
        {"step": point["step"], "value": compact_point["fusion_query_mass"]}
        for point, compact_point in zip(points, compact_points)
    ]
    for key in LAYOUT_LOSS_KEYS:
        curves[f"training_{key}"] = [
            {
                "step": point["step"],
                "value": (compact_point["training_loss_components"] or {}).get(key),
            }
            for point, compact_point in zip(points, compact_points)
        ]
        curves[f"validation_{key}"] = [
            {
                "step": point["step"],
                "value": (compact_point["validation_layout_loss_means"] or {}).get(key),
            }
            for point, compact_point in zip(points, compact_points)
        ]

    return {
        "status": data.get("status", "complete"),
        "source": str(path),
        "steps": [point["step"] for point in points],
        "identity_baseline": (
            {"step": 0, "cer": identity_cer} if identity is not None else None
        ),
        "best_trained_step": best_trained["step"] if best_trained else None,
        "best_trained_cer": best_trained["validation"]["cer"] if best_trained else None,
        "best_vs_identity_step": best_vs_identity["step"] if best_vs_identity else None,
        "best_vs_identity_cer": (
            best_vs_identity["validation"]["cer"] if best_vs_identity else None
        ),
        "points": compact_points,
        "curves": curves,
        "triage": data.get("triage"),
        "test_used_for_selection": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = summarize(args.source)
    output = args.output or (_diagnostic_path(args.source).parent / "diagnostic_compact.json")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
