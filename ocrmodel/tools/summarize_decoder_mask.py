#!/usr/bin/env python3
"""Summarize the Gate C decoder-mask screen and select the best arm.

Reads each arm's training ``summary.json`` (which holds the per-step validation
metrics written by ``train_decoder_mask``), picks the best checkpoint *by free-
generation validation CER only*, and writes a compact comparison plus a
validation-only ``selection.json``.

The screen layout is ``<screen-root>/arms/<arm>/summary.json``.  Only validation
metrics enter the selection; any locked-test or intervention results are reported
alongside but are never used for selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_ARMS = ("B0", "B1", "B2", "B3")

METRIC_KEYS = (
    "cer",
    "exact_page_rate",
    "insertions",
    "deletions",
    "substitutions",
    "reference_characters",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object at {path}")
    return payload


def _validation_summary(summary: dict[str, Any]) -> dict[str, Any]:
    validation = summary.get("validation") or {}
    if not isinstance(validation, dict):
        raise ValueError("summary.json has no validation object")
    return validation


def _best_step(validation: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    """Select the step with the lowest free-generation validation CER."""

    best_step: str | None = None
    best_metrics: dict[str, Any] | None = None
    for step, metrics in validation.items():
        if not isinstance(metrics, dict) or metrics.get("cer") is None:
            continue
        if best_metrics is None or metrics["cer"] < best_metrics["cer"]:
            best_step, best_metrics = step, metrics
    return best_step, best_metrics


def _row(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if metrics is None:
        return {key: None for key in METRIC_KEYS}
    return {key: metrics.get(key) for key in METRIC_KEYS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="summarize the decoder-mask screen")
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=list(DEFAULT_ARMS))
    parser.add_argument("--output", type=Path, help="selection.json output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    screen_root = args.screen_root.resolve()
    arms_root = screen_root / "arms"
    results: list[dict[str, Any]] = []
    for arm in args.arms:
        summary_path = arms_root / arm / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = _read_json(summary_path)
        validation = _validation_summary(summary)
        best_step, best_metrics = _best_step(validation)
        results.append(
            {
                "arm": arm,
                "best_step": best_step,
                "selected_by_validation_cer": bool(best_step),
                **{key: best_metrics.get(key) if best_metrics else None for key in METRIC_KEYS},
            }
        )
    scored = [result for result in results if result["cer"] is not None]
    if not scored:
        raise RuntimeError("no arm reported a validation CER")
    best_arm = min(scored, key=lambda result: (result["cer"], -result["exact_page_rate"]))["arm"]
    selection = {
        "status": "complete",
        "selection_metric": "validation_cer",
        "selected_arm": best_arm,
        "best_step": {result["arm"]: result["best_step"] for result in scored},
        "results": results,
        "test_used_for_selection": False,
    }
    target = args.output or (screen_root / "selection.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
