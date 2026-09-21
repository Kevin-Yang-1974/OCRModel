#!/usr/bin/env python3
"""Merge disjoint eval-only attention shards into one run artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="merge GLM-OCR attention eval shards")
    parser.add_argument("--workers-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shards", type=int, default=4)
    parser.add_argument("--expected-pages", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker_summaries: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    report_ids: set[str] = set()
    prediction_ids: set[str] = set()
    for index in range(args.expected_shards):
        worker = args.workers_root / f"shard-{index}"
        summary_path = worker / "summary.json"
        probe_path = worker / "attention.jsonl"
        prediction_path = worker / "validation_predictions.jsonl"
        if not summary_path.is_file() or not probe_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"incomplete worker artifact: {worker}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise RuntimeError(f"worker {index} is not complete")
        validation = summary.get("validation") or {}
        if summary.get("test_manifest_read") is not False:
            raise RuntimeError(f"worker {index} is not test-free")
        worker_summaries.append({
            "index": index,
            "summary": str(summary_path),
            "pages": validation.get("pages"),
            "cer": validation.get("cer"),
            "generation_limit_hits": validation.get("generation_limit_hits"),
            "generation_mean_new_tokens": validation.get("generation_mean_new_tokens"),
        })
        for row in load_jsonl(probe_path):
            page_id = str(row.get("page_id"))
            if page_id in report_ids:
                raise RuntimeError(f"duplicate attention report page: {page_id}")
            report_ids.add(page_id)
            reports.append(row)
        for row in load_jsonl(prediction_path):
            page_id = str(row.get("page_id"))
            if page_id in prediction_ids:
                raise RuntimeError(f"duplicate prediction page: {page_id}")
            prediction_ids.add(page_id)
            predictions.append(row)

    if len(reports) != args.expected_pages or prediction_ids != report_ids:
        raise RuntimeError(
            f"merged page mismatch: reports={len(reports)} predictions={len(predictions)} "
            f"expected={args.expected_pages}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports.sort(key=lambda row: str(row["page_id"]))
    predictions.sort(key=lambda row: str(row["page_id"]))
    write_jsonl(args.output_dir / "attention.jsonl", reports)
    write_jsonl(args.output_dir / "validation_predictions.jsonl", predictions)
    payload = {
        "status": "complete",
        "pages": len(reports),
        "shards": worker_summaries,
        "attention": str(args.output_dir / "attention.jsonl"),
        "predictions": str(args.output_dir / "validation_predictions.jsonl"),
        "test_manifest_read": False,
        "test_used_for_selection": False,
    }
    (args.output_dir / "workers_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "event": "decoder_attention_shards_merged",
        "pages": len(reports),
        "shards": len(worker_summaries),
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
