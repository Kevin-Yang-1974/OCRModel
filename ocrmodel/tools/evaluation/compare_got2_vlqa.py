#!/usr/bin/env python3
"""Run a bounded, same-protocol comparison of original GOT2 and VLQA.

The script intentionally evaluates the two checkpoints in separate processes. This
keeps CUDA allocator state and model-specific initialization independent while
writing only compact status to stdout; full logs and predictions remain in the
comparison run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


EXIT_FAILURE = 1
EXIT_USAGE = 64
EXIT_MISSING = 66
EXIT_EXISTS = 74
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ComparisonFailure(RuntimeError):
    def __init__(self, message: str, *, exit_code: int = EXIT_FAILURE, log: Path | None = None):
        super().__init__(message)
        self.exit_code = exit_code
        self.log = log


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def bounded(value: str, limit: int = 800) -> str:
    value = value.replace("\x00", "").strip()
    return value if len(value) <= limit else value[:limit] + "...[truncated]"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[1]
    ocrmodel_root = Path(os.environ.get("OCRMODEL_ROOT", script_root.parent))
    workspace = Path(os.environ.get("OCR_WORKSPACE", ocrmodel_root.parent))
    source_model = Path(
        os.environ.get("GOT_SOURCE_MODEL", workspace / "models" / "GOT-OCR2_0")
    )
    output_root = Path(
        os.environ.get("GOT_EVALUATION_RUNS", workspace / "runs" / "evaluation" / "GOT")
    )
    parser = argparse.ArgumentParser(
        description=(
            "Compare original whole-page GOT2 against a P2 VLQA checkpoint on "
            "the exact same manifest, prompt, and decoding protocol."
        )
    )
    parser.add_argument("--baseline-model", type=Path, default=source_model)
    parser.add_argument("--vlqa-model", type=Path, required=True)
    parser.add_argument("--tokenizer-model", type=Path)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--layout-image-root", type=Path)
    parser.add_argument("--layout-split", required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--output-root", type=Path, default=output_root)
    parser.add_argument("--run-id")
    parser.add_argument("--max-regions", type=positive_int, default=16)
    parser.add_argument("--max-records", type=nonnegative_int, default=0)
    parser.add_argument("--model-max-length", type=positive_int, default=2048)
    parser.add_argument("--max-new-tokens", type=positive_int, default=2048)
    parser.add_argument("--no-repeat-ngram-size", type=nonnegative_int, default=20)
    parser.add_argument("--object-threshold", type=probability, default=0.5)
    parser.add_argument("--iou-threshold", type=probability, default=0.5)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu-utilization-limit", type=positive_int, default=50)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument(
        "--audit-manifest",
        action="append",
        type=Path,
        default=[],
        help="Additional split manifests to include in the leakage audit.",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        help="Formal train manifest included in the cross-split leakage audit.",
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        help="Formal validation manifest included in the cross-split leakage audit.",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Diagnostic-only escape hatch; never use for a formal test result.",
    )
    parser.add_argument(
        "--allow-non-test-split",
        action="store_true",
        help="Permit a split name other than 'test' for local smoke diagnostics.",
    )
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ComparisonFailure(f"Expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tail_lines(path: Path, count: int = 20) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [bounded(line, 1000) for line in deque(handle, maxlen=count)]


def require_gpu_below_limit(utilization_limit: int = 50) -> None:
    command = [
        "nvidia-smi",
        "-i",
        "0",
        "--query-gpu=utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ComparisonFailure(f"Cannot query GPU 0: {exc}", exit_code=EXIT_MISSING) from exc
    if completed.returncode != 0:
        raise ComparisonFailure(
            f"nvidia-smi failed: {bounded(completed.stderr)}",
            exit_code=EXIT_MISSING,
        )
    value = completed.stdout.strip()
    if not value.isdigit():
        raise ComparisonFailure(f"GPU0 utilization is not numeric: {bounded(value)}", exit_code=EXIT_MISSING)
    utilization = int(value)
    if utilization >= utilization_limit:
        raise ComparisonFailure(f"GPU0_BUSY utilization={utilization} limit={utilization_limit}", exit_code=75)


def resolve_args(args: argparse.Namespace) -> dict[str, Any]:
    split = args.layout_split.strip()
    train_split = args.train_split.strip()
    validation_split = args.validation_split.strip()
    if not split or not train_split or not validation_split:
        raise ComparisonFailure(
            "--layout-split, --train-split, and --validation-split must be non-empty.",
            exit_code=EXIT_USAGE,
        )
    if split != "test" and not args.allow_non_test_split:
        raise ComparisonFailure(
            "Formal comparison requires --layout-split test; use "
            "--allow-non-test-split only for diagnostics.",
            exit_code=EXIT_USAGE,
        )
    if split == "test" and len({train_split, validation_split, split}) != 3:
        raise ComparisonFailure(
            "Formal train, validation, and test split names must be pairwise distinct.",
            exit_code=EXIT_USAGE,
        )
    baseline = args.baseline_model.expanduser().resolve()
    vlqa = args.vlqa_model.expanduser().resolve()
    manifest = args.layout_manifest.expanduser().resolve()
    image_root = (
        args.layout_image_root.expanduser().resolve()
        if args.layout_image_root is not None
        else manifest.parent
    )
    tokenizer = (
        args.tokenizer_model.expanduser().resolve()
        if args.tokenizer_model is not None
        else baseline
    )
    output_root = args.output_root.expanduser().resolve()
    run_id = args.run_id or f"got2_vlqa_compare_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ComparisonFailure(
            "--run-id may contain only letters, digits, period, underscore, and hyphen "
            "and must be at most 128 characters.",
            exit_code=EXIT_USAGE,
        )
    audit_manifests = [manifest]
    for path in (args.train_manifest, args.validation_manifest):
        if path is not None:
            resolved = path.expanduser().resolve()
            if resolved not in audit_manifests:
                audit_manifests.append(resolved)
    for path in args.audit_manifest:
        resolved = path.expanduser().resolve()
        if resolved not in audit_manifests:
            audit_manifests.append(resolved)
    if baseline == vlqa:
        raise ComparisonFailure("Baseline and VLQA model paths must differ.", exit_code=EXIT_USAGE)
    if (
        split == "test"
        and not args.skip_audit
        and (args.train_manifest is None or args.validation_manifest is None)
    ):
        raise ComparisonFailure(
            "Formal test comparison requires --train-manifest and "
            "--validation-manifest for one cross-split leakage audit.",
            exit_code=EXIT_USAGE,
        )
    return {
        "baseline_model": baseline,
        "vlqa_model": vlqa,
        "tokenizer_model": tokenizer,
        "manifest": manifest,
        "image_root": image_root,
        "split": split,
        "train_split": train_split,
        "validation_split": validation_split,
        "output_root": output_root,
        "run_id": run_id,
        "audit_manifests": audit_manifests,
    }


def runtime_environment(ocrmodel_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "OCRMODEL_ROOT": str(ocrmodel_root),
            "GOT_PROJECT_ROOT": str(ocrmodel_root / "src" / "GOT-OCR-2.0"),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLED": "true",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    return environment


def run_audit(
    *,
    ocrmodel_root: Path,
    manifests: Sequence[Path],
    output_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ocrmodel_root / "tools" / "preprocessing" / "audit_synthetic_layout.py"),
    ]
    for manifest in manifests:
        command.extend(("--manifest", str(manifest)))
    command.extend(("--summary-json", str(output_path)))
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ocrmodel_root,
            env=runtime_environment(ocrmodel_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0 or not output_path.is_file():
        raise ComparisonFailure("Comparison manifest audit failed.", log=log_path)
    summary = read_json(output_path)
    if summary.get("status") != "ok":
        raise ComparisonFailure("Comparison manifest audit did not report status=ok.", log=log_path)
    return summary


def run_evaluator(
    *,
    label: str,
    model_kind: str,
    model_path: Path,
    tokenizer_path: Path,
    manifest: Path,
    image_root: Path,
    split: str,
    output_dir: Path,
    args: argparse.Namespace,
    ocrmodel_root: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(ocrmodel_root / "src" / "GOT-OCR-2.0" / "scripts" / "evaluate_GOT_layout.py"),
        "--model-name-or-path",
        str(model_path),
        "--model-kind",
        model_kind,
        "--tokenizer-name-or-path",
        str(tokenizer_path),
        "--layout-manifest",
        str(manifest),
        "--layout-image-root",
        str(image_root),
        "--layout-split",
        split,
        "--output-dir",
        str(output_dir),
        "--max-regions",
        str(args.max_regions),
        "--max-records",
        str(args.max_records),
        "--model-max-length",
        str(args.model_max_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--no-repeat-ngram-size",
        str(args.no_repeat_ngram_size),
        "--object-threshold",
        str(args.object_threshold),
        "--iou-threshold",
        str(args.iou_threshold),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
    ]
    if model_kind == "vlqa":
        command.extend(("--require-vlqa-stage", "p2"))
    log_path = output_dir / "evaluate.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ocrmodel_root / "src" / "GOT-OCR-2.0",
            env=runtime_environment(ocrmodel_root),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    summary_path = output_dir / "layout_validation_metrics.json"
    if completed.returncode != 0 or not summary_path.is_file():
        raise ComparisonFailure(
            f"{label} evaluator failed with exit code {completed.returncode}.",
            log=log_path,
        )
    summary = read_json(summary_path)
    if summary.get("status") != "ok" or summary.get("model_kind") != model_kind:
        raise ComparisonFailure(f"{label} evaluator returned an invalid summary.", log=log_path)
    if int(summary.get("pages", 0)) < 1:
        raise ComparisonFailure(f"{label} evaluator processed no pages.", log=log_path)
    return {
        "model": str(model_path),
        "model_kind": model_kind,
        "summary": str(summary_path),
        "metrics": summary.get("metrics"),
        "runtime": summary.get("runtime"),
        "parameters": summary.get("parameters"),
        "input_protocol": summary.get("input_protocol"),
        "decoding": summary.get("decoding"),
        "pages": summary["pages"],
    }


def metric_value(summary: dict[str, Any], *keys: str) -> float | None:
    value: Any = summary.get("metrics", {})
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def compare(args: argparse.Namespace) -> dict[str, Any]:
    resolved = resolve_args(args)
    ocrmodel_root = Path(
        os.environ.get("OCRMODEL_ROOT", Path(__file__).resolve().parents[2])
    ).resolve()
    for key in ("baseline_model", "vlqa_model", "tokenizer_model", "manifest", "image_root"):
        path = resolved[key]
        if not path.exists():
            raise ComparisonFailure(f"Required path does not exist: {path}", exit_code=EXIT_MISSING)
    for path in resolved["audit_manifests"]:
        if not path.is_file():
            raise ComparisonFailure(f"Audit manifest does not exist: {path}", exit_code=EXIT_MISSING)
    run_root = resolved["output_root"] / resolved["run_id"]
    if run_root.exists():
        raise ComparisonFailure(f"Comparison output already exists: {run_root}", exit_code=EXIT_EXISTS)
    (run_root / "metadata").mkdir(parents=True, exist_ok=False)
    metadata = run_root / "metadata"
    manifest = resolved["manifest"]
    if not args.skip_audit:
        audit = run_audit(
            ocrmodel_root=ocrmodel_root,
            manifests=resolved["audit_manifests"],
            output_path=metadata / "audit_summary.json",
            log_path=metadata / "audit.log",
        )
        split_counts = audit.get("split_counts", {})
        required_splits = [(resolved["split"], "comparison")]
        if resolved["split"] == "test":
            required_splits.extend(
                (
                    (resolved["train_split"], "train"),
                    (resolved["validation_split"], "validation"),
                )
            )
        for split_name, label in required_splits:
            if int(split_counts.get(split_name, 0)) < 1:
                raise ComparisonFailure(
                    f"Formal {label} split {split_name!r} is empty after audit.",
                    log=metadata / "audit.log",
                )
    else:
        audit = {"status": "skipped", "reason": "explicit diagnostic escape hatch"}
        write_json(metadata / "audit_summary.json", audit)

    common = dict(
        manifest_sha256=sha256_file(manifest),
        split=resolved["split"],
        train_split=resolved["train_split"],
        validation_split=resolved["validation_split"],
        pages_expected=args.max_records or None,
        model_inputs=["whole_page_image", "ocr_prompt"],
        layout_metadata_as_model_input=False,
    )
    if args.device == "cuda":
        require_gpu_below_limit(args.gpu_utilization_limit)
    baseline = run_evaluator(
        label="baseline",
        model_kind="baseline",
        model_path=resolved["baseline_model"],
        tokenizer_path=resolved["tokenizer_model"],
        manifest=manifest,
        image_root=resolved["image_root"],
        split=resolved["split"],
        output_dir=run_root / "baseline",
        args=args,
        ocrmodel_root=ocrmodel_root,
    )
    if args.device == "cuda":
        require_gpu_below_limit(args.gpu_utilization_limit)
    vlqa = run_evaluator(
        label="vlqa",
        model_kind="vlqa",
        model_path=resolved["vlqa_model"],
        tokenizer_path=resolved["tokenizer_model"],
        manifest=manifest,
        image_root=resolved["image_root"],
        split=resolved["split"],
        output_dir=run_root / "vlqa",
        args=args,
        ocrmodel_root=ocrmodel_root,
    )
    if baseline["pages"] != vlqa["pages"]:
        raise ComparisonFailure(
            f"Evaluator page count mismatch: {baseline['pages']} != {vlqa['pages']}."
        )
    if baseline["input_protocol"] != vlqa["input_protocol"]:
        raise ComparisonFailure("Baseline and VLQA input protocols differ.")
    if baseline["decoding"] != vlqa["decoding"]:
        raise ComparisonFailure("Baseline and VLQA decoding configurations differ.")
    comparison = {
        "ocr_page_cer_delta_vlqa_minus_baseline": (
            metric_value(vlqa, "ocr", "page_cer")
            - metric_value(baseline, "ocr", "page_cer")
            if metric_value(vlqa, "ocr", "page_cer") is not None
            and metric_value(baseline, "ocr", "page_cer") is not None
            else None
        ),
        "ocr_whitespace_normalized_page_cer_delta_vlqa_minus_baseline": (
            metric_value(vlqa, "ocr", "whitespace_normalized_page_cer")
            - metric_value(baseline, "ocr", "whitespace_normalized_page_cer")
            if metric_value(vlqa, "ocr", "whitespace_normalized_page_cer") is not None
            and metric_value(baseline, "ocr", "whitespace_normalized_page_cer") is not None
            else None
        ),
        "page_exact_match_rate_delta_vlqa_minus_baseline": (
            metric_value(vlqa, "ocr", "page_exact_match_rate")
            - metric_value(baseline, "ocr", "page_exact_match_rate")
            if metric_value(vlqa, "ocr", "page_exact_match_rate") is not None
            and metric_value(baseline, "ocr", "page_exact_match_rate") is not None
            else None
        ),
        "vlqa_layout": vlqa["metrics"].get("layout"),
    }
    summary = {
        "schema_version": 1,
        "status": "ok",
        "purpose": "same-manifest whole-page GOT2 baseline versus VLQA comparison",
        "protocol": common,
        "audit": audit,
        "baseline": baseline,
        "vlqa": vlqa,
        "comparison": comparison,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    write_json(run_root / "summary.json", summary)
    (run_root / "LAYOUT_COMPARISON_FINISHED").touch(exist_ok=False)
    return {"run_root": str(run_root), "summary": str(run_root / "summary.json"), **summary}


def main(argv: Sequence[str] | None = None) -> int:
    try:
        summary = compare(parse_args(argv))
    except ComparisonFailure as exc:
        payload = {
            "event": "layout_comparison_failed",
            "exit_code": exc.exit_code,
            "error": bounded(str(exc)),
            "log": str(exc.log) if exc.log else None,
            "tail": tail_lines(exc.log) if exc.log else [],
        }
        print(compact_json(payload), flush=True)
        return exc.exit_code
    except Exception as exc:
        print(
            compact_json(
                {
                    "event": "layout_comparison_failed",
                    "exit_code": EXIT_FAILURE,
                    "error_type": type(exc).__name__,
                    "error": bounded(str(exc)),
                    "traceback": bounded(traceback.format_exc(), 1800),
                }
            ),
            flush=True,
        )
        return EXIT_FAILURE
    print(
        compact_json(
            {
                "event": "layout_comparison_completed",
                "run_root": summary["run_root"],
                "summary": summary["summary"],
                "pages": summary["baseline"]["pages"],
                "comparison": summary["comparison"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
