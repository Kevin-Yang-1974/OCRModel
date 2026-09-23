#!/usr/bin/env python3
"""Merge and audit the five disjoint shards of the locked 800-page test."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys


ARMS = ("baseline", "line_mask_epoch8_step3456")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def finite_tree(value, path="root"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite metric at {path}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite_tree(child, f"{path}[{index}]")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--mask-checkpoint", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path,
                        help="when given, populate the low-frequency diagnostics with real train counts")
    args = parser.parse_args()

    sys.path.insert(0, str(args.code_root.resolve() / "src"))
    from layout_ocr.data import load_records
    from layout_ocr.metrics import aggregate_ocr_metrics

    records = load_records(args.test_manifest)
    expected_ids = [str(row["page_id"]) for row in records]
    if len(records) != 800 or len(set(expected_ids)) != 800:
        raise ValueError("the official full test manifest must contain exactly 800 unique pages")
    if any(row.get("split", row.get("official_split")) != "test" for row in records):
        raise ValueError("manifest includes pages outside the official test split")
    manifest_sha = sha256(args.test_manifest)
    train_counts: Counter[str] = Counter()
    if args.train_manifest is not None:
        for train_record in load_records(args.train_manifest):
            train_counts.update(train_record["page_text"])

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if (
        selection.get("epoch") != 8
        or selection.get("step") != 3456
        or selection.get("test_manifest_read") is not False
        or selection.get("test_used_for_selection") is not False
        or Path(selection.get("checkpoint", "")).resolve() != args.mask_checkpoint.resolve()
    ):
        raise ValueError("selection provenance does not lock this test to epoch8/step3456")

    merged_rows = {arm: [] for arm in ARMS}
    seen = {arm: set() for arm in ARMS}
    shard_counts = []
    for index in range(5):
        shard_root = args.run_root / "shards" / f"shard-{index}"
        worker_status = json.loads((shard_root / "worker_status.json").read_text())
        if worker_status.get("status") != "complete" or worker_status.get("shard_index") != index:
            raise ValueError(f"worker shard {index} is incomplete")
        expected_shard_ids = expected_ids[index::5]
        shard_counts.append(len(expected_shard_ids))
        for arm in ARMS:
            shard_summary = json.loads(
                (shard_root / f"summary-{arm}.json").read_text(encoding="utf-8")
            )
            if (
                shard_summary.get("status") != "complete"
                or shard_summary.get("arm") != arm
                or shard_summary.get("test_manifest_sha256") != manifest_sha
                or shard_summary.get("test_manifest_read") is not True
                or shard_summary.get("test_used_for_selection") is not False
                or shard_summary.get("head_finite") is not True
            ):
                raise ValueError(f"protocol/status mismatch for {arm} shard {index}")
            rows = [
                json.loads(line)
                for line in (shard_root / f"{arm}.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if [row["page_id"] for row in rows] != expected_shard_ids:
                raise ValueError(f"missing, extra, reordered, or duplicate pages in {arm} shard {index}")
            for row in rows:
                page_id = row["page_id"]
                if page_id in seen[arm]:
                    raise ValueError(f"duplicate page {page_id} in {arm}")
                seen[arm].add(page_id)
                merged_rows[arm].append(row)

    manifest_text = {str(row["page_id"]): row["page_text"] for row in records}
    results = {}
    for arm in ARMS:
        if seen[arm] != set(expected_ids):
            raise ValueError(f"{arm} coverage is not exactly the complete test split")
        rows = merged_rows[arm]
        by_page = {row["page_id"]: row for row in rows}
        rows = [by_page[page_id] for page_id in expected_ids]
        for row in rows:
            if row["reference"] != manifest_text[row["page_id"]]:
                raise ValueError(f"reference mismatch for {arm} page {row['page_id']}")
            if row["generation_eos_hit"] == row["generation_limit_hit"]:
                raise ValueError(f"EOS/limit flags are inconsistent for {arm} page {row['page_id']}")
        metrics = aggregate_ocr_metrics(
            ((row["reference"], row["prediction"]) for row in rows), train_counts
        )
        generation_tokens = sum(int(row["generation_tokens"]) for row in rows)
        eos_pages = sum(bool(row["generation_eos_hit"]) for row in rows)
        limit_hits = sum(bool(row["generation_limit_hit"]) for row in rows)
        loop_pages = sum(
            bool(row.get("repetition", {}).get("repeated_cycle_detected")) for row in rows
        )
        metrics.update(
            {
                "generation_tokens": generation_tokens,
                "mean_generation_tokens": generation_tokens / len(rows),
                "eos_pages": eos_pages,
                "eos_rate": eos_pages / len(rows),
                "generation_limit_hits": limit_hits,
                "generation_limit_hit_rate": limit_hits / len(rows),
                "loop_pages": loop_pages,
                "loop_rate": loop_pages / len(rows),
            }
        )
        if not all(isinstance(row.get("prediction"), str) for row in rows):
            raise ValueError(f"non-text prediction found for {arm}")
        result_path = args.run_root / "results" / f"predictions-{arm}.jsonl"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        results[arm] = metrics

    baseline_cer = results["baseline"]["cer"]
    mask_cer = results["line_mask_epoch8_step3456"]["cer"]
    summary = {
        "status": "complete",
        "run_id": args.run_root.name,
        "protocol": {
            "dataset": "MTHv2 full official test",
            "test_pages": 800,
            "test_manifest": str(args.test_manifest),
            "test_manifest_sha256": manifest_sha,
            "test_manifest_read": True,
            "test_used_for_selection": False,
            "selection_locked": True,
            "selection_file": str(args.selection),
            "selected_epoch": selection["epoch"],
            "selected_step": selection["step"],
            "selected_validation_cer": selection["validation"]["cer"],
            "selected_checkpoint": str(args.mask_checkpoint),
            "selected_checkpoint_sha256": sha256(args.mask_checkpoint),
            "backbone_checkpoint": str(args.backbone_checkpoint),
            "backbone_lora_sha256": sha256(args.backbone_checkpoint / "decoder_lora.safetensors"),
            "generation": {
                "input": "full-page image + fixed Text Recognition prompt only",
                "max_pixels": 4000000,
                "max_new_tokens": 1536,
                "do_sample": False,
                "precision": "bfloat16",
                "attention_backend": "sdpa",
                "processor": "fast",
                "seed": 42,
            },
            "execution": {
                "physical_gpus": [0, 1, 2, 3, 4],
                "workers": 5,
                "pages_per_shard": shard_counts,
                "arms_per_worker": list(ARMS),
            },
            "arm_definition": {
                "baseline": "same checkpoint-3000 backbone LoRA, recurrent line-mask routing disabled",
                "line_mask_epoch8_step3456": "same backbone LoRA plus selected learned line-mask head",
                "only_difference": "learned line-mask routing enabled",
            },
        },
        "coverage": {
            "expected_pages": 800,
            "unique_pages_per_arm": {arm: len(seen[arm]) for arm in ARMS},
            "duplicates_per_arm": {arm: 0 for arm in ARMS},
            "missing_per_arm": {arm: 0 for arm in ARMS},
            "coverage_verified": True,
        },
        "results": results,
        "comparison": {
            "line_mask_minus_baseline_cer": mask_cer - baseline_cer,
            "relative_cer_reduction": (baseline_cer - mask_cer) / baseline_cer
            if baseline_cer
            else None,
        },
        "numerical_checks": {
            "selected_mask_head_finite": True,
            "metrics_finite": True,
            "nonfinite_or_nan_found": False,
        },
        "limitations": [
            "The test split was evaluated only after validation selection and did not affect checkpoint, threshold, or postprocessing selection.",
            "Low-frequency-character recall is unavailable in this run because it is not needed for the locked CER comparison.",
        ],
    }
    finite_tree(summary)
    write_json(args.run_root / "results" / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
