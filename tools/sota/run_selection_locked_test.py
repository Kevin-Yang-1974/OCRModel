"""Selection-locked MTHv2 test entrypoint.

The explicit ``--allow-formal-test`` switch is only used by the formally
authorized launcher.  Without it this command remains locked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--allow-formal-test", action="store_true")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("selection_split") != "validation" or selection.get("test_used_for_selection") is not False:
        raise SystemExit("Selection must be validation-only with test_used_for_selection=false")
    if not args.allow_formal_test:
        raise SystemExit("MTHv2 test is locked; pass --allow-formal-test from the authorized formal launcher.")
    if not args.test_manifest.is_file():
        raise FileNotFoundError(args.test_manifest)
    from tools.sota.adapters import AdapterError, build_adapter
    from tools.sota.mthv2 import iter_manifest
    from tools.sota.registry import get_model
    from tools.sota.schema import PredictionRecord, jsonl_record
    import time

    prompt = "Read all text on this page in reading order. Return plain text only. Do not describe the image."

    spec = get_model(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "protocol.json").write_text(json.dumps({"model": spec.to_dict(), "selection": selection, "split": "test", "test_used_for_selection": False}, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions_path = args.output_dir / "predictions.jsonl"
    try:
        adapter = build_adapter(spec, args.model_root, device=args.device, dtype=args.dtype)
        adapter.load()
        load_status = {"status": "loaded", **adapter.runtime, **adapter.parameter_stats()}
    except AdapterError as exc:
        adapter = None
        load_status = {"status": "blocked", "error": str(exc)}
    (args.output_dir / "load_status.json").write_text(json.dumps(load_status, ensure_ascii=False, indent=2), encoding="utf-8")
    count = 0
    with predictions_path.open("w", encoding="utf-8") as output:
        for record in iter_manifest(args.test_manifest, args.image_root, split="test", allow_test=True):
            started = time.perf_counter()
            if adapter is None:
                prediction = PredictionRecord(record["page_id"], record["image"], spec.name, None, "", "blocked", {"error": load_status["error"], "test_used": True})
            else:
                try:
                    raw, normalized, runtime = adapter.predict(Path(record["image_path"]), prompt)
                    prediction = PredictionRecord(record["page_id"], record["image"], spec.name, raw, normalized, "ok", {**runtime, "wall_seconds": time.perf_counter() - started, **adapter.parameter_stats(), "test_used": True}, None)
                except AdapterError as exc:
                    prediction = PredictionRecord(record["page_id"], record["image"], spec.name, None, "", "failed", {"error": str(exc), "wall_seconds": time.perf_counter() - started, "test_used": True}, None)
            output.write(jsonl_record(prediction) + "\n")
            count += 1
    summary = {"event": "sota_selection_locked_test_complete", "model": args.model, "pages": count, "status": load_status["status"], "predictions": str(predictions_path), "selection_split": selection["selection_split"], "test_used_for_selection": False, "test_used": True}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
