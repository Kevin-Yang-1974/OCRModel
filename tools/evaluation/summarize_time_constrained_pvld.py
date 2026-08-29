#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = "time_constrained_freeze_strategy_v1"
VARIANT = "original_pvld_freeze_strategy"


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def selection_metrics(selection: dict[str, Any]) -> dict[str, Any]:
    selected = selection["selected"]
    return {
        "optimizer_step": int(selected["optimizer_step"]),
        "model_path": selected["model_path"],
        "config_sha256": selected["config_sha256"],
        "weights_sha256": selected["weights_sha256"],
        "validation_metrics": selected["validation_metrics"],
    }


def training_stability(metrics: dict[str, Any]) -> dict[str, Any]:
    diagnostics = metrics.get("diagnostics") or {}
    tail = diagnostics.get("tail_mean") or {}
    train_loss = float(metrics.get("train_loss", float("nan")))
    return {
        "global_step": int(metrics.get("global_step", -1)),
        "train_loss": train_loss,
        "train_loss_finite": math.isfinite(train_loss),
        "diagnostic_log_count": int(diagnostics.get("log_count", 0)),
        "tail_mean_loss": tail.get("loss"),
        "tail_mean_ocr_loss": tail.get("ocr_loss"),
        "tail_mean_layout_loss": tail.get("layout_loss"),
        "tail_mean_vision_gradient_norm": tail.get("vision_gradient_norm"),
        "tail_mean_projector_gradient_norm": tail.get("projector_gradient_norm"),
        "tail_mean_layout_decoder_gradient_norm": tail.get("layout_decoder_gradient_norm"),
        "learning_rate_groups": metrics.get("learning_rate_groups"),
    }


def metric_delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for section in ("ocr", "layout"):
        current_section = current.get(section) or {}
        previous_section = previous.get(section) or {}
        for key in set(current_section) & set(previous_section):
            if isinstance(current_section[key], (int, float)) and isinstance(
                previous_section[key], (int, float)
            ):
                result[f"{section}.{key}"] = float(current_section[key]) - float(
                    previous_section[key]
                )
    return dict(sorted(result.items()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize the time-constrained PVLD run.")
    parser.add_argument("--validation-lock", type=Path, required=True)
    parser.add_argument("--p1-selection", type=Path, required=True)
    parser.add_argument("--p2-selection", type=Path, required=True)
    parser.add_argument("--p3-selection", type=Path, required=True)
    parser.add_argument("--p2-training-metrics", type=Path, required=True)
    parser.add_argument("--p3-training-metrics", type=Path, required=True)
    parser.add_argument("--test-summary", type=Path, required=True)
    parser.add_argument("--previous-test-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock = load(args.validation_lock)
    selections = {
        "p1": load(args.p1_selection),
        "p2": load(args.p2_selection),
        "p3": load(args.p3_selection),
    }
    for name, selection in selections.items():
        if (
            selection.get("selection_split") != "validation"
            or selection.get("test_used_for_selection") is not False
            or selection.get("validation_page_count") != 400
            or selection.get("protocol_version") != PROTOCOL_VERSION
            or selection.get("variant") != VARIANT
        ):
            raise ValueError(f"{name} selection violates the registered protocol.")
        if selection.get("validation_manifest_sha256") != lock.get(
            "validation_manifest_sha256"
        ):
            raise ValueError(f"{name} selection used a different validation manifest.")

    test = load(args.test_summary)
    test_metrics = test["metrics"]
    comparison: dict[str, Any] = {
        "status": "previous_result_not_supplied",
        "ocr_improved": None,
        "layout_degraded": None,
        "suitable_as_shared_training_strategy": None,
    }
    if args.previous_test_summary:
        previous = load(args.previous_test_summary)
        if previous.get("test_manifest_sha256") != test.get("test_manifest_sha256"):
            raise ValueError("Previous result used a different test manifest.")
        deltas = metric_delta(test_metrics, previous["metrics"])
        cer_delta = deltas.get("ocr.page_cer")
        f1_delta = deltas.get("layout.complete_region_f1")
        comparison = {
            "status": "compared",
            "previous_test_summary": str(args.previous_test_summary.resolve()),
            "metric_delta_current_minus_previous": deltas,
            "ocr_improved": cer_delta is not None and cer_delta < 0,
            "ocr_degraded": cer_delta is not None and cer_delta > 0,
            "layout_degraded": f1_delta is not None and f1_delta < 0,
            "suitable_as_shared_training_strategy": (
                cer_delta is not None and cer_delta <= 0 and f1_delta is not None and f1_delta >= 0
            ),
        }

    payload = {
        "status": "ok",
        "protocol_version": PROTOCOL_VERSION,
        "variant": VARIANT,
        "validation_page_count": 400,
        "validation_manifest": str(args.validation_lock.resolve()),
        "validation_manifest_sha256": lock["validation_manifest_sha256"],
        "test_used_for_selection": False,
        "test_used_for_training": False,
        "selected_checkpoints": {
            name: selection_metrics(selection) for name, selection in selections.items()
        },
        "training_stability": {
            "p2": training_stability(load(args.p2_training_metrics)),
            "p3": training_stability(load(args.p3_training_metrics)),
        },
        "selection_locked_test": {
            "summary": str(args.test_summary.resolve()),
            "metrics": test_metrics,
            "inference_failures": test.get("inference_failures"),
        },
        "comparison_to_previous_freeze_strategy": comparison,
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "time_constrained_pvld_summary_created", **payload}, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
