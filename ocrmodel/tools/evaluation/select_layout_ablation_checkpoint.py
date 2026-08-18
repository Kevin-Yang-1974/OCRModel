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

STEP_PATTERN = re.compile(r"checkpoint-(\d+)$")
DEFAULT_GPU_UTILIZATION_LIMIT = 50

def compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

def gpu_utilization(gpu_id: str) -> int:
    completed = subprocess.run(
        ["nvidia-smi", "-i", gpu_id, "--query-gpu=utilization.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Cannot query GPU {gpu_id} utilization.")
    value = completed.stdout.strip()
    if not re.fullmatch(r"[0-9]+", value) or int(value) > 100:
        raise RuntimeError(f"GPU{gpu_id} utilization is not numeric: {value!r}")
    return int(value)

def require_gpu_free(gpu_id: str, utilization_limit: int) -> None:
    utilization = gpu_utilization(gpu_id)
    if utilization >= utilization_limit:
        raise RuntimeError(
            f"GPU{gpu_id}_BUSY utilization={utilization} limit={utilization_limit}"
        )

def discover_candidates(model_root: Path, *, zero_shot: bool = False,
                        expected_ablation: str | None = None) -> list[tuple[int, Path]]:
    metrics_path = model_root / "layout_training_metrics.json"
    if zero_shot:
        if not (model_root / "model.safetensors").is_file():
            raise FileNotFoundError(model_root / "model.safetensors")
        return [(0, model_root.resolve())]
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if expected_ablation and metrics.get("ablation_id") != expected_ablation:
        raise RuntimeError("Training metrics ablation_id does not match selection request.")
    final_step = int(metrics["global_step"])
    candidates = [(final_step, model_root)]
    for path in model_root.glob("checkpoint-*"):
        match = STEP_PATTERN.search(path.name)
        if match and (path / "model.safetensors").is_file() and (path / "config.json").is_file():
            candidates.append((int(match.group(1)), path))
    unique: dict[str, tuple[int, Path]] = {}
    for step, path in sorted(candidates):
        digest = sha256(path / "model.safetensors")
        unique.setdefault(digest, (step, path.resolve()))
    return sorted(unique.values())

def select_best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return min(candidates, key=lambda item: (
        float(item["validation_metrics"]["page_cer"]),
        float(item["validation_metrics"]["whitespace_normalized_page_cer"]),
        int(item["optimizer_step"]),
    ))

def metric_value(metrics: dict[str, Any], canonical: str, legacy: str) -> Any:
    if canonical in metrics:
        return metrics[canonical]
    if legacy in metrics:
        return metrics[legacy]
    raise KeyError(f"Missing OCR metric: {canonical} or {legacy}")

def normalize_ocr_metrics(ocr: dict[str, Any]) -> dict[str, Any]:
    return {
        "page_cer": ocr["page_cer"],
        "whitespace_normalized_page_cer": ocr[
            "whitespace_normalized_page_cer"
        ],
        "total_edit_distance": metric_value(
            ocr, "total_edit_distance", "character_edits"
        ),
        "total_reference_characters": metric_value(
            ocr, "total_reference_characters", "reference_characters"
        ),
        "exact_matches": metric_value(
            ocr, "exact_matches", "page_exact_matches"
        ),
        "pages": ocr["pages"],
    }

def load_resumable_candidate_summary(
    summary_path: Path,
    *,
    model: Path,
    model_kind: str,
    validation_manifest: Path,
    max_new_tokens: int,
    no_repeat_ngram_size: int,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("status") != "ok":
        raise RuntimeError(f"Existing validation summary is not status=ok: {summary_path}")
    checks = {
        "model": Path(str(summary.get("model", ""))).resolve() == model.resolve(),
        "model_kind": summary.get("model_kind") == model_kind,
        "manifest": (
            Path(str(summary.get("manifest", ""))).resolve()
            == validation_manifest.resolve()
        ),
        "split": summary.get("split") == "validation",
        "inference_failures": summary.get("inference_failures") == 0,
    }
    decoding = summary.get("decoding")
    checks["decoding"] = (
        isinstance(decoding, dict)
        and decoding.get("do_sample") is False
        and decoding.get("num_beams") == 1
        and decoding.get("max_new_tokens") == max_new_tokens
        and decoding.get("no_repeat_ngram_size") == no_repeat_ngram_size
    )
    protocol = summary.get("input_protocol")
    checks["input_protocol"] = (
        isinstance(protocol, dict)
        and protocol.get("model_inputs") == ["whole_page_image", "ocr_prompt"]
        and protocol.get("layout_metadata_as_model_input") is False
    )
    metrics = summary.get("metrics")
    checks["ocr_metrics"] = (
        isinstance(metrics, dict) and isinstance(metrics.get("ocr"), dict)
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            f"Existing validation summary cannot be resumed; mismatched fields: {failed}"
        )
    return summary

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select one ablation checkpoint using validation only.")
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-kind", choices=("baseline", "generic", "vlqa"), required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-image-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--max-regions", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=20)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--gpu-utilization-limit", type=int, default=DEFAULT_GPU_UTILIZATION_LIMIT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not re.fullmatch(r"[0-9]+", args.gpu_id):
        raise ValueError("--gpu-id must be one physical numeric GPU id.")
    if not 1 <= args.gpu_utilization_limit <= 100:
        raise ValueError("--gpu-utilization-limit must be an integer in 1..100.")
    output = args.output_dir.resolve()
    selection_path = output / "selection.json"
    if selection_path.is_file():
        if not args.resume:
            raise FileExistsError(f"Selection already exists: {selection_path}")
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
        print(compact({"event": "layout_ablation_checkpoint_selected", "selection": str(selection_path), "selected": payload["selected"], "resumed": True}))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    candidate_results: list[dict[str, Any]] = []
    for step, model in discover_candidates(
        args.model_root.resolve(), zero_shot=args.ablation == "got2_zero_shot",
        expected_ablation=(None if args.ablation == "got2_zero_shot" else args.ablation),
    ):
        candidate_dir = output / f"step-{step:08d}"
        summary_path = candidate_dir / "layout_validation_metrics.json"
        command = [
            sys.executable, str(args.project_root / "scripts" / "evaluate_GOT_layout.py"),
            "--model-name-or-path", str(model), "--model-kind", args.model_kind,
            "--tokenizer-name-or-path", str(args.tokenizer_model),
            "--layout-manifest", str(args.validation_manifest),
            "--layout-image-root", str(args.validation_image_root or args.validation_manifest.parent),
            "--layout-split", "validation", "--output-dir", str(candidate_dir),
            "--max-regions", str(args.max_regions), "--max-records", str(args.max_records),
            "--max-new-tokens", str(args.max_new_tokens),
            "--no-repeat-ngram-size", str(args.no_repeat_ngram_size),
        ]
        log_path = candidate_dir / "evaluator.log"
        if summary_path.is_file():
            if not args.resume:
                raise FileExistsError(
                    f"Validation candidate already exists; use --resume: {summary_path}"
                )
            summary = load_resumable_candidate_summary(
                summary_path,
                model=model,
                model_kind=args.model_kind,
                validation_manifest=args.validation_manifest,
                max_new_tokens=args.max_new_tokens,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
        else:
            require_gpu_free(args.gpu_id, args.gpu_utilization_limit)
            candidate_dir.mkdir(parents=True, exist_ok=True)
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = args.gpu_id
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(command, cwd=args.project_root, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
            if completed.returncode != 0 or not summary_path.is_file():
                raise RuntimeError(f"Validation failed for checkpoint step {step}; see {log_path}.")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        ocr = summary["metrics"]["ocr"]
        candidate_results.append({
            "optimizer_step": step, "model_path": str(model),
            "config_sha256": sha256(model / "config.json"),
            "weights_sha256": sha256(model / "model.safetensors"),
            "validation_summary": str(summary_path),
            "validation_metrics": normalize_ocr_metrics(ocr),
        })
    selected = select_best(candidate_results)
    payload = {
        "status": "ok", "purpose": "layout_ablation_validation_selection",
        "ablation_id": args.ablation, "selection_split": "validation",
        "test_used_for_selection": False,
        "primary_metric": "page_cer",
        "tie_breakers": ["whitespace_normalized_page_cer", "earlier_optimizer_step"],
        "validation_manifest": str(args.validation_manifest.resolve()),
        "validation_manifest_sha256": sha256(args.validation_manifest.resolve()),
        "selection_physical_gpu": args.gpu_id,
        "selected": selected, "candidates": candidate_results,
    }
    write_json(selection_path, payload)
    (output / "SELECTION_FINISHED").touch()
    print(compact({"event": "layout_ablation_checkpoint_selected", "selection": str(selection_path), "selected": selected}))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(compact({"event": "layout_ablation_selection_failed", "error_type": type(exc).__name__, "error": str(exc)[:800]}), file=sys.stderr)
        raise SystemExit(1)
