"""Run official external SOTA zero-shot inference on MTHv2 validation only."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sota.adapters import AdapterError, build_adapter
from tools.sota.mthv2 import iter_manifest, manifest_contract
from tools.sota.registry import get_model
from tools.sota.schema import PredictionRecord, jsonl_record


PROMPT = "Read all text on this page in reading order. Return plain text only. Do not describe the image."


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
    args = parser.parse_args()
    if args.split == "test":
        raise SystemExit("MTHv2 test is locked; zero-shot smoke may use validation only.")
    spec = get_model(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "protocol.json").write_text(json.dumps({"model": spec.to_dict(), "prompt": PROMPT, "manifest": manifest_contract(args.manifest, args.split), "split": args.split}, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions_path = args.output_dir / "predictions.jsonl"
    try:
        adapter = build_adapter(spec, args.model_root, device=args.device, dtype=args.dtype)
        adapter.load()
        load_status = {"status": "loaded", **adapter.runtime, **adapter.parameter_stats()}
    except AdapterError as exc:
        load_status = {"status": "blocked", "error": str(exc)}
        adapter = None
    (args.output_dir / "load_status.json").write_text(json.dumps(load_status, ensure_ascii=False, indent=2), encoding="utf-8")
    count = 0
    with predictions_path.open("w", encoding="utf-8") as output:
        for record in iter_manifest(args.manifest, args.image_root, split=args.split, limit=args.limit):
            started = time.perf_counter()
            if adapter is None:
                prediction = PredictionRecord(record["page_id"], record["image"], spec.name, None, "", "blocked", {"error": load_status["error"]})
            else:
                try:
                    raw, normalized, runtime = adapter.predict(Path(record["image_path"]), PROMPT)
                    prediction = PredictionRecord(record["page_id"], record["image"], spec.name, raw, normalized, "ok", {**runtime, "wall_seconds": time.perf_counter() - started, **adapter.parameter_stats()}, None)
                except AdapterError as exc:
                    prediction = PredictionRecord(record["page_id"], record["image"], spec.name, None, "", "failed", {"error": str(exc), "wall_seconds": time.perf_counter() - started, **adapter.parameter_stats()})
            output.write(jsonl_record(prediction) + "\n")
            count += 1
    print(json.dumps({"event": "sota_zero_shot_complete", "model": args.model, "pages": count, "status": load_status["status"], "predictions": str(predictions_path), "test_used": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
