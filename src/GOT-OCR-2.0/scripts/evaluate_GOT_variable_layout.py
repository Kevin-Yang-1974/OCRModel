#!/usr/bin/env python3
"""Evaluate PVLD-32 layout outputs without touching Fixed-Slot reports."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix0, iy0 = max(left[0], right[0]), max(left[1], right[1])
    ix1, iy1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 0.0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-granularity", default="whole_page_image")
    parser.add_argument("--max-layout-records", type=int, default=64)
    parser.add_argument("--max-layout-tokens", type=int, default=2048)
    parser.add_argument("--visual-feature-manifest", type=Path)
    parser.add_argument("--model-name-or-path", type=Path)
    parser.add_argument("--tokenizer-name-or-path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input_granularity != "whole_page_image":
        raise ValueError("PVLD formal protocol requires whole_page_image input")
    if args.max_layout_records < 1 or args.max_layout_tokens < 2:
        raise ValueError("max layout limits must be positive")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    records = read_records(args.manifest.resolve())
    if not records:
        raise ValueError("manifest is empty")
    predictions: list[dict[str, Any]] = []
    feature_records = read_records(args.visual_feature_manifest.resolve()) if args.visual_feature_manifest else None
    feature_by_page = {str(item.get("page_id")): item for item in feature_records or []}
    total_iou = 0.0
    matched = 0
    total_predicted = 0
    total_target = 0
    duplicate_pages = 0
    direction_matches = 0
    direction_compared = 0
    generated_eos = 0
    truncations = 0
    token_total = 0
    for record in records:
        target_regions = record.get("layout_regions", record.get("regions", []))
        target_regions = target_regions if isinstance(target_regions, list) else []
        feature = feature_by_page.get(str(record.get("page_id")))
        if feature is None:
            predicted_regions: list[dict[str, Any]] = []
            eos = False
            truncated = True
            tokens = 0
        else:
            # The standalone evaluator accepts precomputed predictions from a
            # feature manifest; it does not silently invent GOT2 vision output.
            predicted_regions = feature.get("predicted_regions", [])
            eos = bool(feature.get("generated_eos", True))
            truncated = bool(feature.get("truncated", False))
            tokens = int(feature.get("layout_tokens", 0))
        for pred, target in zip(predicted_regions, target_regions):
            if isinstance(pred.get("bbox"), list) and isinstance(target.get("bbox"), list):
                total_iou += bbox_iou(pred["bbox"], target["bbox"])
                matched += 1
            predicted_direction = pred.get("direction")
            target_direction = target.get("direction", target.get("writing_direction"))
            if predicted_direction is not None and target_direction is not None:
                direction_compared += 1
                direction_matches += int(predicted_direction == target_direction)
        total_predicted += len(predicted_regions)
        total_target += len(target_regions)
        duplicate = False
        for left in range(len(predicted_regions)):
            for right in range(left + 1, len(predicted_regions)):
                left_box = predicted_regions[left].get("bbox")
                right_box = predicted_regions[right].get("bbox")
                if isinstance(left_box, list) and isinstance(right_box, list) and bbox_iou(left_box, right_box) > 0.8:
                    duplicate = True
        duplicate_pages += int(duplicate)
        generated_eos += int(eos)
        truncations += int(truncated)
        token_total += tokens
        predictions.append({
            "page_id": record.get("page_id"),
            "input_granularity": args.input_granularity,
            "num_predicted_regions": len(predicted_regions),
            "regions": predicted_regions,
            "generated_eos": eos,
            "layout_tokens": tokens,
            "truncated": truncated,
            "layout_source": record.get("layout_source", "page_regions"),
        })
    page_count = len(records)
    target_counts = [len(record.get("layout_regions", record.get("regions", []))) for record in records]
    predicted_counts = [item["num_predicted_regions"] for item in predictions]
    count_abs = sum(abs(a - b) for a, b in zip(target_counts, predicted_counts)) / page_count
    precision = matched / total_predicted if total_predicted else 0.0
    recall = matched / total_target if total_target else 0.0
    region_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "status": "ok" if feature_records else "preflight_only",
        "schema_version": 1,
        "layout_architecture": "Prompted Variable-Length Layout Decoder",
        "layout_candidate_name": "PVLD-32",
        "input_granularity": args.input_granularity,
        "layout_metadata_as_model_input": False,
        "prompt_query_count": 32,
        "prompt_queries_are_region_slots": False,
        "max_layout_records": args.max_layout_records,
        "max_layout_tokens": args.max_layout_tokens,
        "pages": page_count,
        "metrics": {
            "page_cer": None,
            "page_edit_distance": None,
            "page_exact_match": None,
            "region_count_mae": count_abs,
            "count_exact_accuracy": sum(a == b for a, b in zip(target_counts, predicted_counts)) / page_count,
            "region_precision": precision,
            "region_recall": recall,
            "region_f1": region_f1,
            "bbox_iou": total_iou / matched if matched else 0.0,
            "direction_accuracy": direction_matches / direction_compared if direction_compared else None,
            "duplicate_region_rate": duplicate_pages / page_count,
            "eos_success_rate": generated_eos / page_count,
            "premature_eos_rate": None,
            "max_length_truncation_rate": truncations / page_count,
            "mean_layout_tokens": token_total / page_count,
            "inference_seconds_per_page": None,
            "prompt_query_extra_compute": {
                "num_prompt_queries": 32,
                "attention_score_elements_per_page": "32 * visual_token_count",
                "measured_flops": None,
            },
            "target_region_count_bins": {
                "0-8": sum(count <= 8 for count in target_counts),
                "9-16": sum(9 <= count <= 16 for count in target_counts),
                "17-32": sum(17 <= count <= 32 for count in target_counts),
                ">32": sum(count > 32 for count in target_counts),
            },
            "predicted_region_count_bins": {
                "0-8": sum(count <= 8 for count in predicted_counts),
                "9-16": sum(9 <= count <= 16 for count in predicted_counts),
                "17-32": sum(17 <= count <= 32 for count in predicted_counts),
                ">32": sum(count > 32 for count in predicted_counts),
            },
        },
        "notes": [
            "PVLD-32 prompt count is not a region limit; REGION records plus EOS determine predicted count.",
            "No whole-page GOT2 inference is claimed unless a visual-feature manifest or integrated model supplies predictions.",
        ],
    }
    (output / "layout_predictions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in predictions),
        encoding="utf-8",
    )
    (output / "layout_generation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "layout_report.md").write_text(
        "# PVLD-32 layout report\n\n"
        f"- status: `{summary['status']}`\n"
        f"- pages: `{page_count}`\n"
        f"- input: `{args.input_granularity}`\n"
        f"- prompt queries: `32` (global prompts, not region slots)\n\n"
        "This report is separate from Fixed-Slot VLQA-K16/K32. Whole-page OCR CER is "
        "reported only when an integrated GOT2 evaluator supplies OCR predictions.\n",
        encoding="utf-8",
    )
    print(json.dumps({"event": "variable_layout_evaluation_completed", "summary": str(output / "layout_generation_summary.json"), "status": summary["status"]}, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
