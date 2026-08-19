#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

def require_gpu_below_limit(gpu_id: str, utilization_limit: int) -> None:
    completed = subprocess.run(
        ["nvidia-smi", "-i", gpu_id, "--query-gpu=utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Cannot query GPU {gpu_id} utilization.")
    value = completed.stdout.strip()
    if not re.fullmatch(r"[0-9]+", value):
        raise RuntimeError(f"GPU{gpu_id} utilization is not numeric: {value!r}")
    utilization = int(value)
    if utilization >= utilization_limit:
        raise RuntimeError(
            f"GPU{gpu_id}_BUSY utilization={utilization} limit={utilization_limit}"
        )

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen test from a validation selection.")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--test-category", choices=("Synthetic-ID", "Synthetic-OOD", "Real-OOD"), required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--test-image-root", type=Path)
    parser.add_argument("--model-kind", choices=("baseline", "generic", "vlqa"), required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-regions", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=20)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--gpu-utilization-limit", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not re.fullmatch(r"[0-9]+", args.gpu_id):
        raise ValueError("--gpu-id must be one physical numeric GPU id.")
    if not 1 <= args.gpu_utilization_limit <= 100:
        raise ValueError("--gpu-utilization-limit must be an integer in 1..100.")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("purpose") != "layout_ablation_validation_selection" or selection.get("test_used_for_selection") is not False:
        raise RuntimeError("Invalid validation-only selection contract.")
    selected = selection["selected"]
    model = Path(selected["model_path"]).resolve()
    if sha256(model / "config.json") != selected["config_sha256"] or sha256(model / "model.safetensors") != selected["weights_sha256"]:
        raise RuntimeError("Selected checkpoint hashes changed after validation selection.")
    output = args.output_dir.resolve()
    summary_path = output / "summary.json"
    if summary_path.is_file():
        if not args.resume:
            raise FileExistsError(f"Formal test already completed: {summary_path}")
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        print(compact({"event": "layout_ablation_test_completed", "summary": str(summary_path), "resumed": True, "metrics": payload["metrics"]}))
        return 0
    if output.exists():
        raise FileExistsError(f"Incomplete test output exists; inspect before retry: {output}")
    evaluator_output = output / "evaluation"
    output.mkdir(parents=True)
    command = [
        sys.executable, str(args.project_root / "scripts" / "evaluate_GOT_layout.py"),
        "--model-name-or-path", str(model), "--model-kind", args.model_kind,
        "--tokenizer-name-or-path", str(args.tokenizer_model),
        "--layout-manifest", str(args.test_manifest),
        "--layout-image-root", str(args.test_image_root or args.test_manifest.parent),
        "--layout-split", "test", "--output-dir", str(evaluator_output),
        "--max-regions", str(args.max_regions), "--max-records", str(args.max_records),
        "--max-new-tokens", str(args.max_new_tokens),
        "--no-repeat-ngram-size", str(args.no_repeat_ngram_size),
    ]
    log_path = output / "evaluator.log"
    require_gpu_below_limit(args.gpu_id, args.gpu_utilization_limit)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=args.project_root, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
    evaluator_summary_path = evaluator_output / "layout_validation_metrics.json"
    if completed.returncode != 0 or not evaluator_summary_path.is_file():
        raise RuntimeError(f"Frozen test failed; see {log_path}.")
    evaluator = json.loads(evaluator_summary_path.read_text(encoding="utf-8"))
    payload = {
        "status": "ok", "purpose": "layout_ablation_frozen_test",
        "ablation_id": selection["ablation_id"], "test_category": args.test_category,
        "input_granularity": "whole_page_image", "selection": str(args.selection.resolve()),
        "selected_step": selected["optimizer_step"], "model": str(model),
        "config_sha256": selected["config_sha256"], "weights_sha256": selected["weights_sha256"],
        "test_manifest": str(args.test_manifest.resolve()),
        "test_manifest_sha256": sha256(args.test_manifest.resolve()),
        "inference_physical_gpu": args.gpu_id,
        "input_protocol": evaluator["input_protocol"], "metrics": evaluator["metrics"],
        "inference_failures": evaluator.get("inference_failures", 0),
        "evaluator_summary": str(evaluator_summary_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "TEST_FINISHED").touch()
    print(compact({"event": "layout_ablation_test_completed", "summary": str(summary_path), "metrics": payload["metrics"], "inference_failures": payload["inference_failures"]}))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(compact({"event": "layout_ablation_test_failed", "error_type": type(exc).__name__, "error": str(exc)[:800]}), file=sys.stderr)
        raise SystemExit(1)
