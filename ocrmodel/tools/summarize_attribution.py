#!/usr/bin/env python3
"""Summarize the three GLMOCR decoder-capacity attribution runs.

This tool only reads the declared train/validation run artifacts.  It never
opens a test manifest and does not select a checkpoint or tune a threshold.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from layout_ocr.metrics import levenshtein_alignment


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_cer(reference: str, prediction: str) -> float:
    reference = "".join(reference.split())
    prediction = "".join(prediction.split())
    edits, _ = levenshtein_alignment(reference, prediction)
    return edits / max(1, len(reference))


def read_predictions(path: Path) -> dict[str, float]:
    result: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["page_id"])] = normalized_cer(
                str(row.get("reference", "")), str(row.get("prediction", ""))
            )
    if not result:
        raise ValueError(f"validation predictions are empty: {path}")
    return result


def paired_delta(left: dict[str, float], right: dict[str, float]) -> dict[str, Any]:
    pages = sorted(set(left) & set(right))
    deltas = [right[page] - left[page] for page in pages]
    return {
        "paired_pages": len(pages),
        "right_minus_left_mean_cer": sum(deltas) / max(1, len(deltas)),
        "right_wins": sum(value < 0 for value in deltas),
        "left_wins": sum(value > 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
    }


def last_ocr_rows(run_dir: Path, limit: int = 10) -> list[dict[str, Any]]:
    path = run_dir / "train_metrics.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append(
                    {
                        "step": row.get("step"),
                        "ocr_loss": row.get("ocr_loss"),
                        "total_loss": row.get("total_loss"),
                    }
                )
    return rows[-limit:]


def validation_prediction_path(run_dir: Path, summary: dict[str, Any]) -> Path:
    validation = summary.get("validation") or {}
    step = validation.get("step")
    candidates = [run_dir / "validation_predictions.jsonl"]
    if isinstance(step, int) and step >= 0:
        candidates.insert(0, run_dir / f"validation-{step}" / "validation_predictions.jsonl")
    for path in candidates:
        if path.is_file():
            return path
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"validation predictions missing; checked: {joined}")


def summarize_group(label: str, run_id: str, run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    completed_path = run_dir / "COMPLETED"
    if not summary_path.is_file() or not completed_path.is_file():
        raise FileNotFoundError(f"incomplete attribution group {label}: {run_dir}")
    summary = read_json(summary_path)
    validation = summary.get("validation") or {}
    training = summary.get("training") or {}
    parameter_report = summary.get("trainable_parameter_report") or {}
    return {
        "label": label,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "status": summary.get("status"),
        "seed": summary.get("seed"),
        "decoder_adaptation": summary.get("decoder_adaptation", "frozen"),
        "trainable_parameters": (
            0
            if summary.get("eval_only")
            else parameter_report.get("trainable_parameters", 0)
        ),
        "model_requires_grad_parameters": parameter_report.get("trainable_parameters", 0),
        "adapter_trainable_parameters": parameter_report.get("adapter_trainable_parameters", 0),
        "decoder_lora_trainable_parameters": parameter_report.get(
            "decoder_lora_trainable_parameters", 0
        ),
        "training_steps": training.get("steps", summary.get("training_updates", 0)),
        "validation": {
            key: validation.get(key)
            for key in (
                "step",
                "pages",
                "cer",
                "exact_page_rate",
                "teacher_forced_ocr_loss",
                "layout_box_mae",
                "validity_p_gap",
                "validity_auroc",
                "invalid_gated_context_share",
                "residual_relative_norm",
                "generation_limit_hit_rate",
            )
        },
        "last_ocr_rows": last_ocr_rows(run_dir),
        "test_manifest_read": bool(summary.get("test_manifest_read", True)),
        "test_used_for_selection": bool(summary.get("test_used_for_selection", True)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help="group label=run_id; provide exactly three groups",
    )
    args = parser.parse_args()
    if len(args.group) != 3:
        parser.error("exactly three --group entries are required")
    protocol = read_json(args.protocol_file)
    if protocol.get("test_manifest_read") is not False:
        raise ValueError("attribution protocol unexpectedly reads the test manifest")

    groups: list[dict[str, Any]] = []
    prediction_sets: dict[str, dict[str, float]] = {}
    for item in args.group:
        if "=" not in item:
            parser.error(f"invalid group specification: {item}")
        label, run_id = item.split("=", 1)
        run_dir = args.bundle_root / run_id / "seed42"
        group = summarize_group(label, run_id, run_dir)
        predictions = read_predictions(validation_prediction_path(run_dir, read_json(run_dir / "summary.json")))
        prediction_sets[label] = predictions
        groups.append(group)
    by_label = {group["label"]: group for group in groups}
    if not {"A_noop", "B_adapter_only", "C_decoder_lora"}.issubset(by_label):
        raise ValueError("groups must be labelled A_noop, B_adapter_only, C_decoder_lora")

    def validation_cer(label: str) -> float | None:
        value = by_label[label]["validation"].get("cer")
        return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None

    def delta(left: str, right: str) -> float | None:
        left_value, right_value = validation_cer(left), validation_cer(right)
        return None if left_value is None or right_value is None else right_value - left_value

    comparison = {
        "B_minus_A_validation_cer": delta("A_noop", "B_adapter_only"),
        "C_minus_B_validation_cer": delta("B_adapter_only", "C_decoder_lora"),
        "C_minus_A_validation_cer": delta("A_noop", "C_decoder_lora"),
        "paired_A_to_B": paired_delta(
            prediction_sets["A_noop"], prediction_sets["B_adapter_only"]
        ),
        "paired_B_to_C": paired_delta(
            prediction_sets["B_adapter_only"], prediction_sets["C_decoder_lora"]
        ),
        "paired_A_to_C": paired_delta(
            prediction_sets["A_noop"], prediction_sets["C_decoder_lora"]
        ),
        "checkpoint_selection": "not_performed",
        "test_used_for_selection": False,
    }
    output = {
        "status": "complete",
        "protocol": protocol,
        "groups": groups,
        "comparison": comparison,
        "test_manifest_read": False,
        "test_used_for_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
