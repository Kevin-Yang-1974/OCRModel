#!/usr/bin/env python3
"""Merge disjoint multi-GPU GLMOCR locked-test shards.

Each shard is produced by ``evaluate_glmocr_locked_test`` with a distinct
``--test-shard-index``.  The merge is deliberately performed after all shard
processes finish, so the final ``locked-test`` directory is written exactly
once and remains selection-locked.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from layout_ocr.data import load_records, validate_records
from layout_ocr.metrics import aggregate_ocr_metrics


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON object in {path}")
                rows.append(value)
    return rows


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _region_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float((row.get("region_metrics") or {}).get(key))
        for row in rows
        if _numeric((row.get("region_metrics") or {}).get(key))
    ]
    return _mean(values)


def _prediction_metrics(
    rows: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    *,
    max_eval_new_tokens: int,
) -> dict[str, Any]:
    train_counts = Counter(
        character
        for record in train_records
        for character in record["page_text"]
        if not character.isspace()
    )
    pairs = [(str(row["reference"]), str(row["prediction"])) for row in rows]
    metrics = aggregate_ocr_metrics(pairs, train_counts)
    page_count = len(rows)
    generation_lengths = [int(row.get("generation_length", 0)) for row in rows]
    eos_values = [row.get("generation_eos_hit") for row in rows]
    eos_observable = any(value is not None for value in eos_values)
    generation_eos_hits = sum(bool(value) for value in eos_values if value is not None)
    limit_hits = sum(bool(row.get("generation_limit_hit")) for row in rows)
    loop_rows = [row.get("loop_continuation") or {} for row in rows]
    detected = sum(bool(row.get("loop_detected")) for row in loop_rows)
    escape_successes = sum(
        bool(row.get("loop_detected"))
        and int(row.get("continued_non_cycle_tokens", 0)) >= 4
        for row in loop_rows
    )
    post_eos = sum(bool(row.get("post_loop_eos")) for row in loop_rows)
    early_eos = sum(bool(row.get("post_loop_early_eos")) for row in loop_rows)
    buckets = {bucket: [] for bucket in ("sparse", "normal", "dense")}
    for row in rows:
        buckets.setdefault(str(row.get("density_bucket", "normal")), []).append(row)
    stratified: dict[str, Any] = {}
    region_keys = (
        "region_count",
        "region_pointer_reuse_rate",
        "region_spatial_duplicate_rate",
        "region_bbox_ap50",
        "region_bbox_precision50",
        "region_bbox_recall50",
        "region_reading_order_accuracy",
        "region_eos_hit",
        "region_limit_hit",
        "region_recall",
    )
    for bucket in ("sparse", "normal", "dense"):
        bucket_rows = buckets[bucket]
        stratified[bucket] = {
            "pages": len(bucket_rows),
            "ocr": aggregate_ocr_metrics(
                [(str(row["reference"]), str(row["prediction"])) for row in bucket_rows],
                train_counts,
            ),
            "region": {
                key: _region_mean(bucket_rows, key) for key in region_keys
            },
        }
    region_rows = [
        row for row in rows if isinstance(row.get("region_metrics"), dict)
    ]
    region_counts = [
        int((row["region_metrics"])["region_count"])
        for row in region_rows
        if _numeric((row["region_metrics"] or {}).get("region_count"))
    ]
    region_eos = sum(
        bool((row["region_metrics"] or {}).get("region_eos_hit"))
        for row in region_rows
        if (row["region_metrics"] or {}).get("region_eos_hit") is not None
    )
    region_limit = sum(
        bool((row["region_metrics"] or {}).get("region_limit_hit"))
        for row in region_rows
        if (row["region_metrics"] or {}).get("region_limit_hit") is not None
    )
    repeated_cycle_values = [
        float(row.get("repeated_cycle_rate", 0.0)) for row in rows
    ]
    repeated_trigram_values = [
        float(row.get("repeated_trigram_rate", 0.0)) for row in rows
    ]
    metrics.update(
        {
            "generation_max_new_tokens": max_eval_new_tokens,
            "generation_limit_hits": limit_hits,
            "generation_limit_hit_rate": limit_hits / max(1, page_count),
            "generation_lengths": generation_lengths,
            "generation_mean_new_tokens": sum(generation_lengths)
            / max(1, page_count),
            "generation_max_new_tokens_observed": max(generation_lengths, default=0),
            "repeated_trigram_rate": sum(repeated_trigram_values)
            / max(1, page_count),
            "repeated_cycle_page_rate": sum(
                bool(row.get("repeated_cycle_detected")) for row in rows
            )
            / max(1, page_count),
            "repeated_cycle_rate": sum(repeated_cycle_values)
            / max(1, page_count),
            "loop_detected_page_rate": detected / max(1, page_count),
            "loop_escape_success_rate": escape_successes / max(1, detected),
            "loop_post_eos_rate": post_eos / max(1, detected),
            "loop_early_eos_rate": early_eos / max(1, detected),
            "generation_eos_hits": generation_eos_hits,
            "generation_eos_observable": eos_observable,
            "generation_eos_hit_rate": (
                generation_eos_hits / max(1, page_count)
                if eos_observable
                else None
            ),
            "density_bucket_counts": {
                bucket: len(bucket_rows) for bucket, bucket_rows in buckets.items()
            },
            "stratified": stratified,
            "region_count_mean": _mean([float(value) for value in region_counts]),
            "region_pointer_reuse_rate": _region_mean(
                region_rows, "region_pointer_reuse_rate"
            ),
            "region_spatial_duplicate_rate": _region_mean(
                region_rows, "region_spatial_duplicate_rate"
            ),
            "region_eos_hit_rate": region_eos / max(1, len(region_rows)),
            "region_limit_hit_rate": region_limit / max(1, len(region_rows)),
            "region_recall": _region_mean(region_rows, "region_recall"),
            "region_bbox_ap50": _region_mean(region_rows, "region_bbox_ap50"),
            "region_bbox_precision50": _region_mean(
                region_rows, "region_bbox_precision50"
            ),
            "region_bbox_recall50": _region_mean(
                region_rows, "region_bbox_recall50"
            ),
            "region_reading_order_accuracy": _region_mean(
                region_rows, "region_reading_order_accuracy"
            ),
        }
    )
    return metrics


def _weighted_scalar(
    shard_metrics: list[dict[str, Any]],
    weights: list[int],
    key: str,
) -> float | None:
    values = [
        (float(metrics[key]), weight)
        for metrics, weight in zip(shard_metrics, weights)
        if _numeric(metrics.get(key))
    ]
    if not values:
        return None
    return sum(value * weight for value, weight in values) / max(
        1, sum(weight for _, weight in values)
    )


def _merge_shard_diagnostics(
    shard_metrics: list[dict[str, Any]], weights: list[int]
) -> dict[str, Any]:
    """Carry non-page-level diagnostics into the merged summary.

    OCR and generation metrics are recomputed from the merged prediction rows;
    this helper only preserves diagnostics that are not present per page.
    """

    keys = set().union(*(metrics.keys() for metrics in shard_metrics))
    derived = {
        "pages",
        "reference_characters",
        "character_errors",
        "insertions",
        "deletions",
        "substitutions",
        "insertion_errors",
        "deletion_errors",
        "substitution_errors",
        "cer",
        "exact_page_rate",
        "low_frequency_k1_character_types",
        "low_frequency_k1_reference_characters",
        "low_frequency_k1_recall",
        "low_frequency_k3_character_types",
        "low_frequency_k3_reference_characters",
        "low_frequency_k3_recall",
        "low_frequency_k5_character_types",
        "low_frequency_k5_reference_characters",
        "low_frequency_k5_recall",
        "r2_k1_reference_characters",
        "r2_k1_recall",
        "r2_k3_reference_characters",
        "r2_k3_recall",
        "r2_k5_reference_characters",
        "r2_k5_recall",
        "generation_max_new_tokens",
        "generation_limit_hits",
        "generation_limit_hit_rate",
        "generation_lengths",
        "generation_mean_new_tokens",
        "generation_max_new_tokens_observed",
        "repeated_trigram_rate",
        "repeated_cycle_page_rate",
        "repeated_cycle_rate",
        "loop_detected_page_rate",
        "loop_escape_success_rate",
        "loop_post_eos_rate",
        "loop_early_eos_rate",
        "generation_eos_hits",
        "generation_eos_hit_rate",
        "density_bucket_counts",
        "stratified",
        "region_count_mean",
        "region_pointer_reuse_rate",
        "region_spatial_duplicate_rate",
        "region_eos_hit_rate",
        "region_limit_hit_rate",
        "region_recall",
        "region_bbox_ap50",
        "region_bbox_precision50",
        "region_bbox_recall50",
        "region_reading_order_accuracy",
        "test_used_for_selection",
    }
    additive = {
        "layout_direction_regions",
        "generation_eos_hits",
        "generation_limit_hits",
    }
    merged: dict[str, Any] = {}
    for key in sorted(keys - derived):
        values = [metrics.get(key) for metrics in shard_metrics]
        if all(_numeric(value) for value in values if value is not None):
            if key in additive:
                merged[key] = sum(float(value) for value in values if value is not None)
            else:
                value = _weighted_scalar(shard_metrics, weights, key)
                if value is not None:
                    merged[key] = value
        elif key == "matcher_signatures":
            signatures: dict[str, Any] = {}
            for value in values:
                if isinstance(value, dict):
                    signatures.update(value)
            merged[key] = signatures
        elif key in {"layout_loss_means", "teacher_forced_layout_loss_means"}:
            result: dict[str, float] = {}
            subkeys = set().union(
                *(value.keys() for value in values if isinstance(value, dict))
            )
            for subkey in subkeys:
                subvalues = [
                    (float(value[subkey]), weight)
                    for value, weight in zip(values, weights)
                    if isinstance(value, dict) and _numeric(value.get(subkey))
                ]
                if subvalues:
                    result[subkey] = sum(v * w for v, w in subvalues) / max(
                        1, sum(w for _, w in subvalues)
                    )
            merged[key] = result or None
        elif isinstance(next((value for value in values if value is not None), None), list):
            merged[key] = values[0]
        elif isinstance(next((value for value in values if value is not None), None), dict):
            merged[key] = values[0]
        elif any(value is not None for value in values):
            merged[key] = next(value for value in values if value is not None)
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--selection-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-queries", type=int, default=512)
    parser.add_argument("--max-eval-new-tokens", type=int, default=1536)
    parser.add_argument("--gpu-ids", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    shards_dir = args.shards_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    summary = _read_json(run_dir / "summary.json")
    metadata = _read_json(run_dir / "metadata.json")
    selection = _read_json(args.selection_file.resolve())
    if summary.get("status") != "complete" or metadata.get("status") != "complete":
        raise RuntimeError("locked test merge requires a complete training run")
    if summary.get("test_used_for_selection") is not False:
        raise ValueError("training summary does not prove test exclusion")
    if selection.get("status") != "complete" or selection.get("test_used_for_selection") is not False:
        raise ValueError("selection file is not validation-only")
    train_records = load_records(args.train_manifest)
    test_records = load_records(args.test_manifest)
    validate_records(train_records, split="train", num_queries=args.num_queries)
    validate_records(test_records, split="test", num_queries=args.num_queries)
    protocol = _read_json(args.protocol_file)
    expected_test_pages = protocol.get("split_pages", {}).get("test")
    if expected_test_pages is not None and len(test_records) != expected_test_pages:
        raise ValueError(
            f"test page count mismatch: protocol={expected_test_pages}, "
            f"manifest={len(test_records)}"
        )
    expected_ids = {str(record["page_id"]): index for index, record in enumerate(test_records)}
    shard_dirs = sorted(
        path for path in shards_dir.glob("shard*") if path.is_dir()
    )
    if not shard_dirs:
        raise FileNotFoundError(f"no test shards found in {shards_dir}")
    shard_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        shard_summary = _read_json(shard_dir / "locked_test_summary.json")
        if shard_summary.get("status") != "complete":
            raise RuntimeError(f"incomplete test shard: {shard_dir}")
        if shard_summary.get("test_used_for_selection") is not False:
            raise ValueError(f"test shard is not selection-locked: {shard_dir}")
        shard_rows = _read_jsonl(shard_dir / "test_predictions.jsonl")
        expected_shard_pages = int(shard_summary.get("test_pages", len(shard_rows)))
        if len(shard_rows) != expected_shard_pages:
            raise ValueError(
                f"shard page count mismatch for {shard_dir}: "
                f"summary={expected_shard_pages}, rows={len(shard_rows)}"
            )
        shard_summaries.append(shard_summary)
        rows.extend(shard_rows)
    page_ids = [str(row.get("page_id", "")) for row in rows]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("test shards contain duplicate page_ids")
    if set(page_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(page_ids))[:5]
        extra = sorted(set(page_ids) - set(expected_ids))[:5]
        raise ValueError(f"test shard coverage mismatch: missing={missing}, extra={extra}")
    rows.sort(key=lambda row: expected_ids[str(row["page_id"])])
    weights = [int(summary.get("test_pages", 0)) for summary in shard_summaries]
    shard_metrics = [summary.get("metrics") or {} for summary in shard_summaries]
    metrics = _prediction_metrics(
        rows,
        train_records,
        max_eval_new_tokens=args.max_eval_new_tokens,
    )
    metrics.update(_merge_shard_diagnostics(shard_metrics, weights))
    metrics["test_used_for_selection"] = False
    metrics["pages"] = len(rows)
    output_dir.mkdir(parents=True)
    with (output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    locked_summary = {
        "status": "complete",
        "split": "test",
        "seed": args.seed,
        "mode": shard_summaries[0].get("mode", "geometry"),
        "selected_step": selection.get("selected_step"),
        "selection_file": str(args.selection_file.resolve()),
        "selection_metric": selection.get("selection_metric", "validation_cer"),
        "train_pages": len(train_records),
        "test_pages": len(rows),
        "test_pages_total": len(test_records),
        "test_shard_count": len(shard_dirs),
        "test_shard_gpu_ids": args.gpu_ids,
        "num_queries": args.num_queries,
        "max_eval_new_tokens": args.max_eval_new_tokens,
        "metrics": metrics,
        "test_used_for_selection": False,
    }
    (output_dir / "locked_test_summary.json").write_text(
        json.dumps(locked_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "LOCKED_TEST_COMPLETED").touch()
    print(json.dumps(locked_summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
