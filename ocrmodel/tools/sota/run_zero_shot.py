"""Run official external SOTA zero-shot inference on the merged benchmark.

The zero-shot protocol has no trained checkpoint and therefore no validation
selection step: the official checkpoint, prompt and deterministic text
extraction are locked before any test page is read.  ``--allow-test`` must be
passed explicitly so that test access is always an intentional act and is
recorded in the protocol file as ``test_used_for_selection=false``.

Porting note (2026-09-12): independent copy of the archived
``tools/sota/run_zero_shot.py`` from branch ``archive/legacy-vlqa-chunk-20260829``.
Changes for the BSCC run: shard support for the 800-page test, ``--allow-test``
for zero-shot test, progress logging, and a stable page subset argument.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sota.adapters import AdapterError, DEFAULT_MAX_OUTPUT_TOKENS, build_adapter
from tools.sota.dataset import iter_manifest, manifest_contract
from tools.sota.registry import get_model
from tools.sota.schema import PredictionRecord, jsonl_record


PROMPT = "Read all text on this page in reading order. Return plain text only. Do not describe the image."


def _selected(index: int, shard_index: int, shard_count: int) -> bool:
    if shard_count <= 1:
        return True
    return index % shard_count == shard_index


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--allow-test", action="store_true", help="Explicitly unlock benchmark test for a zero-shot run.")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if args.split == "test" and not args.allow_test:
        raise SystemExit("benchmark test is locked; pass --allow-test for an explicit zero-shot test run.")
    if args.shard_count < 1 or not (0 <= args.shard_index < args.shard_count):
        raise SystemExit("shard-index must be in [0, shard-count).")
    if args.max_output_tokens <= 0:
        raise SystemExit("max-output-tokens must be positive.")

    spec = get_model(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = {
        "model": spec.to_dict(),
        "prompt": PROMPT,
        "generation": {"do_sample": False, "max_output_tokens": args.max_output_tokens},
        "manifest": manifest_contract(args.manifest, args.split),
        "split": args.split,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "test_used_for_selection": False,
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
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
        json.dumps(load_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    predictions_path = args.output_dir / "predictions.jsonl"
    seen = selected = ok = failed = blocked = 0
    started = time.perf_counter()
    with predictions_path.open("w", encoding="utf-8") as output:
        for index, record in enumerate(
            iter_manifest(
                args.manifest,
                args.image_root,
                split=args.split,
                limit=args.limit,
                allow_test=args.allow_test,
            )
        ):
            if not _selected(index, args.shard_index, args.shard_count):
                continue
            selected += 1
            page_started = time.perf_counter()
            if adapter is None:
                prediction = PredictionRecord(
                    record["page_id"], record["image"], spec.name, None, "", "blocked",
                    {"error": load_status["error"]},
                )
                blocked += 1
            else:
                try:
                    raw, normalized, runtime = adapter.predict(Path(record["image_path"]), PROMPT)
                    prediction = PredictionRecord(
                        record["page_id"], record["image"], spec.name, raw, normalized, "ok",
                        {**runtime, "wall_seconds": time.perf_counter() - page_started, **adapter.parameter_stats()},
                        None,
                    )
                    ok += 1
                except AdapterError as exc:
                    prediction = PredictionRecord(
                        record["page_id"], record["image"], spec.name, None, "", "failed",
                        {"error": str(exc), "wall_seconds": time.perf_counter() - page_started, **adapter.parameter_stats()},
                    )
                    failed += 1
            output.write(jsonl_record(prediction) + "\n")
            output.flush()
            seen += 1
            if args.progress_every > 0 and seen % args.progress_every == 0:
                print(json.dumps({
                    "event": "sota_zero_shot_progress",
                    "model": args.model,
                    "split": args.split,
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "seen": seen,
                    "ok": ok,
                    "failed": failed,
                    "blocked": blocked,
                    "elapsed_seconds": round(time.perf_counter() - started, 1),
                }, ensure_ascii=False), flush=True)

    summary = {
        "event": "sota_zero_shot_complete",
        "model": args.model,
        "split": args.split,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "pages": seen,
        "ok": ok,
        "failed": failed,
        "blocked": blocked,
        "status": load_status["status"],
        "load_seconds": load_status["load_seconds"],
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "predictions": str(predictions_path),
        "max_output_tokens": args.max_output_tokens,
        "test_used_for_selection": False,
    }
    (args.output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if (failed == 0 and blocked == 0) else 3


if __name__ == "__main__":
    raise SystemExit(main())
