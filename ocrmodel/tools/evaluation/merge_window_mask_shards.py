#!/usr/bin/env python3
"""Verify disjoint complete shards, then recompute micro CER over all pages."""

import argparse
import json
from collections import Counter
from pathlib import Path

from evaluate_window_mask_routing import PROFILE, acceptance

from layout_ocr.metrics import aggregate_ocr_metrics


def merge_shards(root, count=5):
    summaries, predictions = [], {}
    keys = (
        "profile",
        "mode",
        "model_path",
        "backbone_checkpoint",
        "backbone_lora_sha256",
        "validation_sha256",
        "test_manifest_read",
        "test_used_for_selection",
        "reads_ground_truth_for_routing",
        "processor",
        "attention_backend",
        "precision",
        "layout_branch_present",
        "shard_count",
        "full_page_ids",
        "limited",
    )
    for index in range(count):
        folder = root / "shards" / str(index)
        summary = json.loads((folder / "summary.json").read_text())
        if (
            summary["status"] != "complete"
            or summary["shard_index"] != index
            or summary["shard_count"] != count
        ):
            raise ValueError(f"incomplete or mismatched shard {index}")
        if summaries and any(summary[k] != summaries[0][k] for k in keys):
            raise ValueError(f"protocol mismatch on shard {index}")
        rows = [
            json.loads(line)
            for line in (folder / "validation_predictions.jsonl").read_text().splitlines()
            if line.strip()
        ]
        expected = summary["full_page_ids"][index::count]
        if [r["page_id"] for r in rows] != expected or summary["page_ids"] != expected:
            raise ValueError(f"missing, reordered or unexpected pages in shard {index}")
        for row in rows:
            if row["page_id"] in predictions:
                raise ValueError("duplicate page across shards")
            predictions[row["page_id"]] = row
        summaries.append(summary)
    base = summaries[0]
    rows = [predictions[page] for page in base["full_page_ids"]]
    metrics = aggregate_ocr_metrics(((r["reference"], r["prediction"]) for r in rows), Counter())
    metrics["generation_limit_hits"] = sum(r["generation_limit_hit"] for r in rows)
    metrics["generation_tokens"] = sum(r["generation_tokens"] for r in rows)
    result = {k: base[k] for k in keys}
    result.update(
        status="complete",
        validation=metrics,
        shard_pages=[s["validation"]["pages"] for s in summaries],
        acceptance=acceptance(
            metrics,
            base["validation_sha256"],
            base["mode"],
            limited=base["limited"],
            legacy_layout=base["layout_branch_present"],
        ),
    )
    if base["mode"] == "gt" and not base["limited"] and len(rows) != PROFILE.validation_pages:
        raise ValueError("full GT acceptance requires all 149 pages")
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = merge_shards(args.run_root)
    out = args.run_root / "results"
    out.mkdir(exist_ok=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "validation_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
