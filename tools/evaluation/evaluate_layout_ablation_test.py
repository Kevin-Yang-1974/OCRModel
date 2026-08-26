#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import time
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


def require_gpu_free(gpu_id: str, utilization_limit: int = 50) -> None:
    """Compatibility name; admission is utilization-based, not process-based."""
    require_gpu_below_limit(gpu_id, utilization_limit)


def parse_gpu_ids(parallel_gpu_ids: str | None, gpu_id: str) -> tuple[str, ...]:
    raw = parallel_gpu_ids.split(",") if parallel_gpu_ids else [gpu_id]
    gpu_ids = tuple(item.strip() for item in raw if item.strip())
    if not gpu_ids or any(not item.isdigit() for item in gpu_ids):
        raise ValueError("GPU IDs must be a non-empty comma-separated numeric list.")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be unique.")
    return gpu_ids


def require_gpus_below_limit(
    gpu_ids: Sequence[str], utilization_limit: int
) -> None:
    for gpu_id in gpu_ids:
        require_gpu_below_limit(gpu_id, utilization_limit)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_manifest_shards(
    manifest: Path, output: Path, shard_count: int
) -> tuple[list[dict[str, Any]], list[Path]]:
    records = read_jsonl(manifest)
    shard_root = output / "metadata" / "manifest_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_paths = []
    for shard_index in range(shard_count):
        start = len(records) * shard_index // shard_count
        end = len(records) * (shard_index + 1) // shard_count
        shard_path = shard_root / f"test-shard-{shard_index:02d}-of-{shard_count:02d}.jsonl"
        shard_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in records[start:end]
            ),
            encoding="utf-8",
        )
        shard_paths.append(shard_path)
    return records, shard_paths


def merge_shard_evaluations(
    *,
    project_root: Path,
    manifest: Path,
    manifest_records: Sequence[dict[str, Any]],
    shard_outputs: Sequence[Path],
    output: Path,
    model_kind: str,
    object_threshold: float,
) -> dict[str, Any]:
    scripts_root = str((project_root / "scripts").resolve())
    if scripts_root not in sys.path:
        sys.path.insert(0, scripts_root)
    from layout_validation_metrics import (  # type: ignore
        DIRECTION_LABELS,
        LayoutValidationAccumulator,
        OCRValidationAccumulator,
    )

    shard_summaries = [
        json.loads((path / "layout_validation_metrics.json").read_text(encoding="utf-8"))
        for path in shard_outputs
    ]
    predictions_by_page: dict[str, dict[str, Any]] = {}
    for path in shard_outputs:
        for record in read_jsonl(path / "layout_validation_predictions.jsonl"):
            page_id = str(record["page_id"])
            if page_id in predictions_by_page:
                raise RuntimeError(f"Duplicate test page across shards: {page_id}")
            predictions_by_page[page_id] = record
    ordered_page_ids = [str(record["page_id"]) for record in manifest_records]
    if set(predictions_by_page) != set(ordered_page_ids):
        raise RuntimeError("Merged test predictions do not exactly cover the test manifest.")
    predictions = [predictions_by_page[page_id] for page_id in ordered_page_ids]

    layout_accumulator = (
        LayoutValidationAccumulator(object_threshold=object_threshold, iou_threshold=0.5)
        if model_kind in {"vlqa", "pvld"}
        else None
    )
    ocr_accumulator = OCRValidationAccumulator()
    confidence_accumulators = (
        {
            threshold: LayoutValidationAccumulator(
                object_threshold=threshold, iou_threshold=0.5
            )
            for threshold in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
        }
        if model_kind == "pvld"
        else {}
    )
    eos_pages = truncated_pages = record_cap_pages = premature_eos_pages = layout_tokens = 0
    for record in predictions:
        layout_predictions = record.get("layout_predictions")
        if layout_accumulator is None or layout_predictions is None:
            ocr_accumulator.add_page(record["reference_text"], record["predicted_text"])
            continue
        if model_kind == "pvld":
            scores = [float(item["score"]) for item in layout_predictions]
            boxes = [item["bbox"] for item in layout_predictions]
            directions = [DIRECTION_LABELS.index(item["direction"]) for item in layout_predictions]
        else:
            scores = [float(item["object_probability"]) for item in layout_predictions]
            boxes = [item["bbox_xyxy"] for item in layout_predictions]
            directions = [
                DIRECTION_LABELS.index(item["writing_direction"])
                for item in layout_predictions
            ]
        page_args = {
            "reference_text": record["reference_text"],
            "predicted_text": record["predicted_text"],
            "regions": record["regions"],
            "annotation_status": record["layout_annotation_status"],
            "object_scores": scores,
            "predicted_boxes": boxes,
            "predicted_directions": directions,
        }
        layout_accumulator.add_page(**page_args)
        for accumulator in confidence_accumulators.values():
            accumulator.add_page(**page_args)
        if model_kind == "pvld":
            generated_eos = bool(record["generated_eos"])
            eos_pages += int(generated_eos)
            truncated_pages += int(record["truncated_by_max_layout_tokens"])
            record_cap_pages += int(record["stopped_by_max_layout_records"])
            premature_eos_pages += int(
                generated_eos and len(layout_predictions) < len(record["regions"])
            )
            layout_tokens += int(record["num_layout_tokens"])

    combined_output = output / "evaluation"
    combined_output.mkdir(parents=True, exist_ok=True)
    predictions_path = combined_output / "layout_validation_predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in predictions),
        encoding="utf-8",
    )
    summary = dict(shard_summaries[0])
    page_count = len(predictions)
    summary.update({
        "manifest": str(manifest.resolve()),
        "pages": page_count,
        "metrics": (
            layout_accumulator.summary()
            if layout_accumulator is not None
            else {"ocr": ocr_accumulator.summary(), "layout": None}
        ),
        "inference_failures": sum(int(item.get("inference_failures", 0)) for item in shard_summaries),
        "predictions": str(predictions_path),
        "parallel_shards": len(shard_outputs),
    })
    summary["layout"] = dict(summary.get("layout") or {})
    if model_kind == "pvld":
        summary["layout"].update({
            "eos_success_rate": eos_pages / page_count,
            "premature_eos_rate": premature_eos_pages / page_count,
            "max_length_truncation_rate": truncated_pages / page_count,
            "stopped_by_max_layout_records_rate": record_cap_pages / page_count,
            "mean_layout_tokens": layout_tokens / page_count,
        })
        summary["confidence_threshold_scan"] = {
            str(threshold): accumulator.summary()["layout"]
            for threshold, accumulator in confidence_accumulators.items()
        }
    runtimes = [item["runtime"] for item in shard_summaries]
    total_layout = sum(float(item["layout_forward_seconds"]) for item in runtimes)
    total_generation = sum(float(item["ocr_generation_seconds"]) for item in runtimes)
    wall_seconds = max(float(item["evaluation_loop_seconds"]) for item in runtimes)
    token_slots = sum(int(item["generation_token_slots"]) for item in runtimes)
    summary["runtime"] = {
        "device": "parallel_cuda",
        "dtype": runtimes[0]["dtype"],
        "layout_forward_seconds": total_layout,
        "ocr_generation_seconds": total_generation,
        "total_inference_seconds": total_layout + total_generation,
        "evaluation_loop_seconds": wall_seconds,
        "mean_seconds_per_page": (total_layout + total_generation) / page_count,
        "pages_per_second": page_count / (total_layout + total_generation),
        "end_to_end_pages_per_second": page_count / wall_seconds,
        "generation_token_slots": token_slots,
        "generation_token_slots_per_second": token_slots / total_generation,
        "peak_cuda_memory_bytes": max(int(item["peak_cuda_memory_bytes"]) for item in runtimes),
        "per_shard": runtimes,
    }
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one frozen test from a validation selection.")
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--test-category", choices=("Synthetic-ID", "Synthetic-OOD", "Real-OOD"), required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--test-image-root", type=Path)
    parser.add_argument("--model-kind", choices=("baseline", "generic", "vlqa", "pvld"), required=True)
    parser.add_argument("--tokenizer-model", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-regions", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=20)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--parallel-gpu-ids")
    parser.add_argument("--gpu-utilization-limit", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
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
    gpu_ids = parse_gpu_ids(args.parallel_gpu_ids, args.gpu_id)
    if len(gpu_ids) > 1:
        output.mkdir(parents=True)
        manifest_records, shard_manifests = prepare_manifest_shards(
            args.test_manifest.resolve(), output, len(gpu_ids)
        )
        require_gpus_below_limit(gpu_ids, args.gpu_utilization_limit)
        shard_outputs = [output / "shards" / f"shard-{index:02d}" / "evaluation" for index in range(len(gpu_ids))]
        shard_logs = [output / "shards" / f"shard-{index:02d}" / "evaluator.log" for index in range(len(gpu_ids))]
        commands = []
        for shard_manifest, shard_output in zip(shard_manifests, shard_outputs):
            shard_output.parent.mkdir(parents=True, exist_ok=True)
            commands.append([
                sys.executable, str(args.project_root / "scripts" / "evaluate_GOT_layout.py"),
                "--model-name-or-path", str(model), "--model-kind", args.model_kind,
                "--tokenizer-name-or-path", str(args.tokenizer_model),
                "--layout-manifest", str(shard_manifest),
                "--layout-image-root", str(args.test_image_root or args.test_manifest.parent),
                "--layout-split", "test", "--output-dir", str(shard_output),
                "--max-regions", str(args.max_regions), "--max-records", str(args.max_records),
                "--max-new-tokens", str(args.max_new_tokens),
                "--no-repeat-ngram-size", str(args.no_repeat_ngram_size),
                "--object-threshold", str(selection["locked_object_threshold"]),
            ])

        started = time.perf_counter()
        def run_shard(index: int) -> int:
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu_ids[index]
            with shard_logs[index].open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    commands[index], cwd=args.project_root, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, text=True,
                )
            return completed.returncode

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            return_codes = list(executor.map(run_shard, range(len(gpu_ids))))
        failures = [
            {"shard": index, "gpu": gpu_ids[index], "log": str(shard_logs[index])}
            for index, code in enumerate(return_codes)
            if code != 0 or not (shard_outputs[index] / "layout_validation_metrics.json").is_file()
        ]
        if failures:
            raise RuntimeError(f"Parallel frozen test shard failure: {compact(failures)}")
        evaluator = merge_shard_evaluations(
            project_root=args.project_root,
            manifest=args.test_manifest,
            manifest_records=manifest_records,
            shard_outputs=shard_outputs,
            output=output,
            model_kind=args.model_kind,
            object_threshold=float(selection["locked_object_threshold"]),
        )
        evaluator["runtime"]["parallel_wall_seconds"] = time.perf_counter() - started
        evaluator_output = output / "evaluation"
        evaluator_summary_path = evaluator_output / "layout_validation_metrics.json"
        write_json(evaluator_summary_path, evaluator)
        if args.model_kind == "pvld":
            write_json(evaluator_output / "layout_generation_summary.json", evaluator)
            predictions = evaluator_output / "layout_validation_predictions.jsonl"
            (evaluator_output / "layout_predictions.jsonl").write_text(
                predictions.read_text(encoding="utf-8"), encoding="utf-8"
            )
        payload = {
            "status": "ok", "purpose": "layout_ablation_frozen_test",
            "ablation_id": selection["ablation_id"], "test_category": args.test_category,
            "input_granularity": "whole_page_image", "selection": str(args.selection.resolve()),
            "selected_step": selected["optimizer_step"], "model": str(model),
            "config_sha256": selected["config_sha256"], "weights_sha256": selected["weights_sha256"],
            "test_manifest": str(args.test_manifest.resolve()),
            "test_manifest_sha256": sha256(args.test_manifest.resolve()),
            "inference_physical_gpus": list(gpu_ids),
            "input_protocol": evaluator["input_protocol"], "metrics": evaluator["metrics"],
            "locked_object_threshold": selection["locked_object_threshold"],
            "test_threshold_source": "validation_selection",
            "inference_failures": evaluator.get("inference_failures", 0),
            "evaluator_summary": str(evaluator_summary_path),
            "parallel_shards": len(gpu_ids),
        }
        write_json(summary_path, payload)
        (output / "TEST_FINISHED").touch()
        print(compact({"event": "layout_ablation_test_completed", "summary": str(summary_path), "metrics": payload["metrics"], "inference_failures": payload["inference_failures"], "parallel_gpus": list(gpu_ids)}))
        return 0
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
        "--object-threshold", str(selection["locked_object_threshold"]),
    ]
    log_path = output / "evaluator.log"
    require_gpu_free(args.gpu_id, args.gpu_utilization_limit)
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
        "locked_object_threshold": selection["locked_object_threshold"],
        "test_threshold_source": "validation_selection",
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
