#!/usr/bin/env python3
"""Merge five line-mask workers and score only pages with real transcripts."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


ARMS = ("baseline", "line_mask_epoch8_step3456")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pages = len(rows)
    eos_pages = sum(bool(row["generation_eos_hit"]) for row in rows)
    limit_hits = sum(bool(row["generation_limit_hit"]) for row in rows)
    loop_pages = sum(bool(row.get("repetition", {}).get("repeated_cycle_detected")) for row in rows)
    return {
        "pages": pages,
        "eos_pages": eos_pages,
        "eos_rate": eos_pages / max(1, pages),
        "generation_limit_hits": limit_hits,
        "generation_limit_rate": limit_hits / max(1, pages),
        "loop_pages": loop_pages,
        "loop_rate": loop_pages / max(1, pages),
        "mean_generation_tokens": sum(int(row["generation_tokens"]) for row in rows) / max(1, pages),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.shard_count != 5:
        raise ValueError("the extended evaluation protocol requires five shards")
    sys_path = args.code_root.resolve() / "src"
    import sys

    sys.path.insert(0, str(sys_path))
    from layout_ocr.metrics import aggregate_ocr_metrics

    run_root = args.run_root.resolve()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    records = read_jsonl(args.test_manifest)
    if len(records) != int(protocol["split_pages"]["expanded_test_total"]):
        raise ValueError("expanded manifest size differs from protocol")
    expected_ids = [str(row["page_id"]) for row in records]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("expanded test manifest has duplicate IDs")
    record_by_id = {str(row["page_id"]): row for row in records}
    merged: dict[str, list[dict[str, Any]]] = {}

    for arm in ARMS:
        observed: dict[str, dict[str, Any]] = {}
        for shard_index in range(args.shard_count):
            shard_root = run_root / "shards" / f"shard-{shard_index}"
            status_path = shard_root / "worker_status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") != "complete":
                raise ValueError(f"worker shard {shard_index} is incomplete")
            expected_shard_ids = expected_ids[shard_index :: args.shard_count]
            arm_summary = json.loads(
                (shard_root / f"summary-{arm}.json").read_text(encoding="utf-8")
            )
            if (
                arm_summary.get("status") != "complete"
                or arm_summary.get("test_used_for_selection") is not False
                or not arm_summary.get("head_finite")
            ):
                raise ValueError(f"invalid summary for {arm} shard {shard_index}")
            shard_rows = read_jsonl(shard_root / f"{arm}.jsonl")
            if [str(row["page_id"]) for row in shard_rows] != expected_shard_ids:
                raise ValueError(f"{arm} shard {shard_index} has missing/extra/reordered pages")
            for row in shard_rows:
                page_id = str(row["page_id"])
                if page_id in observed:
                    raise ValueError(f"duplicate {arm} prediction: {page_id}")
                if page_id not in record_by_id:
                    raise ValueError(f"unexpected {arm} prediction: {page_id}")
                manifest_row = record_by_id[page_id]
                if bool(row.get("reference_available")) != bool(
                    manifest_row.get("reference_available")
                ):
                    raise ValueError(f"label availability changed for {page_id}")
                if row.get("reference") != manifest_row.get("page_text"):
                    raise ValueError(f"reference mismatch for {page_id}")
                observed[page_id] = row
        if set(observed) != set(expected_ids):
            raise ValueError(f"{arm} total page coverage mismatch")
        rows = [observed[page_id] for page_id in expected_ids]
        merged[arm] = rows
        write_jsonl(run_root / f"predictions-{arm}.jsonl", rows)

    summary_arms: dict[str, Any] = {}
    for arm, rows in merged.items():
        labeled = [row for row in rows if row.get("reference_available")]
        unscored = [row for row in rows if not row.get("reference_available")]
        metrics = aggregate_ocr_metrics(
            ((row["reference"], row["prediction"]) for row in labeled), Counter()
        )
        by_domain: dict[str, Any] = {}
        for domain in sorted({str(row.get("domain") or "unknown") for row in labeled}):
            domain_rows = [row for row in labeled if str(row.get("domain") or "unknown") == domain]
            by_domain[domain] = {
                "metrics": aggregate_ocr_metrics(
                    ((row["reference"], row["prediction"]) for row in domain_rows), Counter()
                ),
                "generation": diagnostics(domain_rows),
            }
        summary_arms[arm] = {
            "labeled_test_metrics": metrics,
            "labeled_test_generation": diagnostics(labeled),
            "labeled_metrics_by_domain": by_domain,
            "new77_unscored_generation": diagnostics(unscored),
            "all136_generation": diagnostics(rows),
            "all_prediction_pages": len(rows),
            "labeled_reference_pages": len(labeled),
            "unscored_new_pages": len(unscored),
            "predictions_sha256": sha256(run_root / f"predictions-{arm}.jsonl"),
        }

    baseline = {row["page_id"]: row["prediction"] for row in merged["baseline"]}
    routed = {row["page_id"]: row["prediction"] for row in merged["line_mask_epoch8_step3456"]}
    changes = []
    for subset, rows in (
        ("labeled_existing_test", [row for row in records if row.get("reference_available")]),
        ("new77_unscored", [row for row in records if not row.get("reference_available")]),
        ("all_extended_test", records),
    ):
        pairs = [(baseline[str(row["page_id"])], routed[str(row["page_id"])]) for row in rows]
        divergence = aggregate_ocr_metrics(pairs, Counter()) if pairs else {"pages": 0}
        changes.append({
            "subset": subset,
            "pages": len(rows),
            "mask_output_vs_baseline_output_not_accuracy": divergence,
            "changed_pages": sum(
                baseline[str(row["page_id"])] != routed[str(row["page_id"])] for row in rows
            ),
        })

    result = {
        "status": "complete",
        "run_id": run_root.name,
        "dataset": protocol.get("dataset"),
        "protocol_label": protocol.get("protocol"),
        "expanded_test_pages": len(expected_ids),
        "labeled_test_pages": sum(bool(row.get("reference_available")) for row in records),
        "unscored_new_pages": sum(not bool(row.get("reference_available")) for row in records),
        "page_ids": expected_ids,
        "arms": summary_arms,
        "paired_output_change": changes,
        "locked_protocol": protocol,
        "test_manifest_read": True,
        "test_used_for_selection": False,
        "coverage": {
            "manifest_pages": len(expected_ids),
            "unique_page_ids": len(set(expected_ids)),
            "baseline_predictions": len(merged["baseline"]),
            "line_mask_predictions": len(merged["line_mask_epoch8_step3456"]),
            "duplicate_or_missing_pages": 0,
        },
    }
    write_json(run_root / "summary.json", result)
    print(json.dumps({
        "status": result["status"],
        "expanded_test_pages": result["expanded_test_pages"],
        "labeled_test_pages": result["labeled_test_pages"],
        "unscored_new_pages": result["unscored_new_pages"],
        "arms": {
            arm: {
                "cer": data["labeled_test_metrics"]["cer"],
                "I_D_S": [data["labeled_test_metrics"]["insertions"],
                          data["labeled_test_metrics"]["deletions"],
                          data["labeled_test_metrics"]["substitutions"]],
                "new77_generation": data["new77_unscored_generation"],
            }
            for arm, data in summary_arms.items()
        },
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
