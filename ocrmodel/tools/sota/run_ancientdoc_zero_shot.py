#!/usr/bin/env python3
"""Run one official external model on the AncientDoc split5 test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sota.adapters import AdapterError, DEFAULT_MAX_OUTPUT_TOKENS, build_adapter
from tools.sota.ancientdoc import (
    DATASET_ID,
    EVALUATION_SPLIT,
    SOURCE_SPLIT,
    iter_manifest,
    manifest_contract,
)
from tools.sota.registry import get_model
from tools.sota.schema import PredictionRecord, jsonl_record


PROMPT = "Read all text on this page in reading order. Return plain text only. Do not describe the image."


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--source-label", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args()

    if not args.allow_test:
        raise SystemExit("AncientDoc test is locked; pass --allow-test for the explicit requested run.")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive when provided")
    if args.max_output_tokens <= 0:
        raise SystemExit("--max-output-tokens must be positive")
    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        raise SystemExit("--shard-index must be in [0, shard-count)")
    if not args.source_label.is_file():
        raise SystemExit(f"source label does not exist: {args.source_label}")

    spec = get_model(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    contract = manifest_contract(args.manifest)
    protocol = {
        "dataset": {
            "dataset_id": DATASET_ID,
            "source_split": SOURCE_SPLIT,
            "evaluation_split": EVALUATION_SPLIT,
            "source_label": str(args.source_label.expanduser().resolve()),
            "source_label_sha256": _sha256(args.source_label),
            "manifest": contract,
        },
        "model": spec.to_dict(),
        "prompt": PROMPT,
        "generation": {"do_sample": False, "max_output_tokens": args.max_output_tokens},
        "model_inputs": ["whole_page_image", "ocr_prompt"],
        "bbox_as_input": False,
        "direction_as_input": False,
        "reading_order_as_input": False,
        "test_used_for_selection": False,
        "selection": None,
        "device": args.device,
        "dtype": args.dtype,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "evaluation_pages": args.limit or contract["expected_count"],
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    load_started = time.perf_counter()
    try:
        adapter = build_adapter(
            spec,
            args.model_root,
            device=args.device,
            dtype=args.dtype,
            max_output_tokens=args.max_output_tokens,
        )
        adapter.load()
        load_status = {"status": "loaded", **adapter.runtime, **adapter.parameter_stats()}
    except AdapterError as exc:
        load_status = {"status": "blocked", "error": str(exc)}
        adapter = None
    load_status["load_seconds"] = time.perf_counter() - load_started
    (args.output_dir / "load_status.json").write_text(
        json.dumps(load_status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    predictions_path = args.output_dir / "predictions.jsonl"
    seen = ok = failed = 0
    started = time.perf_counter()
    with predictions_path.open("w", encoding="utf-8", newline="\n") as output:
        if adapter is not None:
            for index, record in enumerate(iter_manifest(args.manifest, args.image_root, limit=args.limit)):
                if index % args.shard_count != args.shard_index:
                    continue
                page_started = time.perf_counter()
                try:
                    raw, normalized, runtime = adapter.predict(Path(record["image_path"]), PROMPT)
                    prediction = PredictionRecord(
                        record["page_id"],
                        record["image"],
                        spec.name,
                        raw,
                        normalized,
                        "ok",
                        {**runtime, "wall_seconds": time.perf_counter() - page_started, **adapter.parameter_stats()},
                        None,
                    )
                    ok += 1
                except AdapterError as exc:
                    prediction = PredictionRecord(
                        record["page_id"],
                        record["image"],
                        spec.name,
                        None,
                        "",
                        "failed",
                        {"error": str(exc), "wall_seconds": time.perf_counter() - page_started, **adapter.parameter_stats()},
                        None,
                    )
                    failed += 1
                output.write(jsonl_record(prediction) + "\n")
                output.flush()
                seen += 1
                if args.progress_every > 0 and seen % args.progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "event": "ancientdoc_zero_shot_progress",
                                "model": args.model,
                                "seen": seen,
                                "ok": ok,
                                "failed": failed,
                                "shard_index": args.shard_index,
                                "shard_count": args.shard_count,
                                "elapsed_seconds": round(time.perf_counter() - started, 1),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    if adapter is None:
        summary = {
            "event": "ancientdoc_zero_shot_complete",
            "dataset": DATASET_ID,
            "source_split": SOURCE_SPLIT,
            "split": EVALUATION_SPLIT,
            "model": args.model,
            "pages": 0,
            "ok": 0,
            "failed": 0,
            "status": "blocked",
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
            "load_seconds": load_status["load_seconds"],
            "predictions": str(predictions_path),
            "test_used_for_selection": False,
        }
        (args.output_dir / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 3

    summary = {
        "event": "ancientdoc_zero_shot_complete",
        "dataset": DATASET_ID,
        "source_split": SOURCE_SPLIT,
        "split": EVALUATION_SPLIT,
        "model": args.model,
        "pages": seen,
        "ok": ok,
        "failed": failed,
        "status": load_status["status"],
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "load_seconds": load_status["load_seconds"],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "predictions": str(predictions_path),
        "max_output_tokens": args.max_output_tokens,
        "test_used_for_selection": False,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
