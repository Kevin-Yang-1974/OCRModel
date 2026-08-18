#!/usr/bin/env python3
"""Queue AncientDoc validation checkpoint selection and final test evaluation."""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


TRAINING_TOOLS = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_TOOLS))
from c4_selection_contract import load_c4_selection  # noqa: E402


LABELS = ("c0_got2_zero_shot", "c1_got2_ocr_only", "c4", "c5", "c6")
TRAINED_LABELS = LABELS[1:]
MODEL_OPTIONS = {
    "c1_got2_ocr_only": "c1_model",
    "c4": "c4_model",
    "c5": "c5_model",
    "c6": "c6_model",
}
CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


@dataclass(frozen=True)
class EvaluationJob:
    label: str
    model_kind: str
    model_path: Path
    output_dir: Path
    split: str
    step: int | None


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def nonnegative_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def parse_gpu_ids(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or any(not item.isdigit() for item in values):
        raise argparse.ArgumentTypeError("GPU ids must be a non-empty comma-separated list")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("GPU ids must be unique")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    workspace = Path(os.environ.get("OCR_WORKSPACE", root.parent))
    parser = argparse.ArgumentParser(
        description=(
            "Select AncientDoc checkpoints on validation and evaluate the frozen "
            "selection on test with a dynamic multi-GPU queue."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("select", "test", "select-and-test"),
        default="select-and-test",
    )
    parser.add_argument(
        "--ancient-dataset-id",
        default="ancientdoc_layout_260707_group_isolated_seed20260815",
    )
    parser.add_argument("--c0-model", type=Path, default=Path(os.environ.get("GOT_SOURCE_MODEL", "")))
    parser.add_argument(
        "--training-suite",
        type=Path,
        help="Training suite directory or suite_summary.jsonl; replaces four --cN-model arguments.",
    )
    for label, destination in MODEL_OPTIONS.items():
        parser.add_argument(f"--{label.split('_')[0]}-model", dest=destination, type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--c4-selection",
        type=Path,
        help="Required frozen C4 validation selection used by C4, C5, and C6.",
    )
    parser.add_argument("--run-prefix", default=f"ancientdoc_evaluation_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--suite-root", type=Path)
    parser.add_argument(
        "--gpu-ids",
        type=parse_gpu_ids,
        default=parse_gpu_ids(os.environ.get("GOT_VALIDATION_GPU_IDS", "0")),
    )
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--num-workers", type=nonnegative_int, default=4)
    parser.add_argument("--max-records", type=nonnegative_int, default=0)
    parser.add_argument("--model-max-length", type=positive_int, default=2048)
    parser.add_argument("--max-new-tokens", type=positive_int, default=2048)
    parser.add_argument("--no-repeat-ngram-size", type=nonnegative_int, default=20)
    parser.add_argument(
        "--max-ratio-deviation",
        type=nonnegative_finite_float,
        default=0.03,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ocrmodel-root", type=Path, default=root)
    parser.add_argument("--project-root", type=Path, default=root / "src" / "GOT-OCR-2.0")
    parser.add_argument("--workspace", type=Path, default=workspace)
    return parser.parse_args(argv)


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def model_dir(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").is_file() and any(
        (path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")
    )


def checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_RE.fullmatch(path.name)
    return int(match.group(1)) if match else None


def discover_candidates(model: Path) -> list[tuple[Path, int | None]]:
    model = model.expanduser().resolve()
    checkpoints = [
        (path, checkpoint_step(path))
        for path in model.glob("checkpoint-*")
        if checkpoint_step(path) is not None and model_dir(path)
    ]
    if checkpoints:
        return sorted(checkpoints, key=lambda item: int(item[1] or 0))
    if not model_dir(model):
        raise FileNotFoundError(f"No loadable model or checkpoints under {model}")
    metrics = next(
        (model / name for name in ("page_ocr_training_metrics.json", "layout_training_metrics.json")
         if (model / name).is_file()),
        None,
    )
    step = None
    if metrics is not None:
        value = read_json(metrics).get("global_step")
        step = int(value) if isinstance(value, (int, float)) else None
    return [(model, step)]


def validate_dataset(dataset_root: Path, max_ratio_deviation: float) -> None:
    audit = dataset_root / "audit" / "ancientdoc_split_leakage" / "split_leakage_audit.json"
    split_audit = dataset_root / "split_audit.json"
    verifier = (
        Path(__file__).resolve().parents[1]
        / "preprocessing"
        / "verify_ancientdoc_group_audit.py"
    )
    command = [sys.executable, str(verifier), str(audit), "--split-audit", str(split_audit), "--max-ratio-deviation", str(max_ratio_deviation)]
    checked = subprocess.run(command, text=True, capture_output=True)
    if checked.returncode != 0:
        raise RuntimeError(f"AncientDoc group audit failed: {checked.stdout.strip() or checked.stderr.strip()}")


def training_model_args(
    args: argparse.Namespace,
    c4_branch: dict[str, Any],
) -> dict[str, Path]:
    values = {"c0_got2_zero_shot": args.c0_model.expanduser().resolve()}
    suite_models: dict[str, Path] = {}
    if args.training_suite is not None:
        suite = args.training_suite.expanduser().resolve()
        summary = suite / "suite_summary.jsonl" if suite.is_dir() else suite
        if not summary.is_file():
            raise FileNotFoundError(f"Training suite summary does not exist: {summary}")
        aliases = {
            "c1_got2_ocr_only": "c1_got2_ocr_only",
            "c4_vlqa_ocr_only": "c4",
            "c5_vlqa_ocr_replay": "c5",
            "c6_vlqa_layout_replay": "c6",
        }
        for line in summary.read_text(encoding="utf-8").splitlines():
            payload = json.loads(line)
            if payload.get("event") != "baseline_finished":
                continue
            label = aliases.get(payload.get("baseline"))
            model = payload.get("model")
            if label and isinstance(model, str):
                suite_models[label] = Path(model).expanduser().resolve()
    for label in TRAINED_LABELS:
        path = (
            Path(c4_branch["selected_model_path"])
            if label == "c4"
            else suite_models.get(label) or getattr(args, MODEL_OPTIONS[label], None)
        )
        if path is None:
            raise ValueError(
                f"--{label.split('_')[0]}-model is required for --phase {args.phase}; "
                "alternatively pass --training-suite."
            )
        values[label] = path.expanduser().resolve()
    return values


def training_metrics_for_model(model_path: Path) -> dict[str, Any]:
    current = model_path.expanduser().resolve()
    for _ in range(5):
        metrics = current / "layout_training_metrics.json"
        if metrics.is_file():
            return read_json(metrics)
        current = current.parent
    raise FileNotFoundError(f"No layout_training_metrics.json for {model_path}")


def validate_replay_branch_consistency(
    model_paths: dict[str, Path],
    c4_branch: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "selection_path": str(Path(c4_branch["selection_path"]).resolve()),
        "selected_c4_step": int(c4_branch["selected_step"]),
        "selected_c4_model_path": str(Path(c4_branch["selected_model_path"]).resolve()),
        "selected_c4_weights_sha256": c4_branch["weights_sha256"],
    }
    observed: dict[str, Any] = {}
    replay_protocols: dict[str, dict[str, Any]] = {}
    for label in ("c5", "c6"):
        metrics = training_metrics_for_model(model_paths[label])
        branch = metrics.get("branch_initialization")
        if not isinstance(branch, dict):
            raise ValueError(f"{label} has no C4 branch_initialization provenance.")
        for key, value in expected.items():
            actual = branch.get(key)
            if key.endswith("_path") and isinstance(actual, str):
                actual = str(Path(actual).resolve())
            if actual != value:
                raise ValueError(
                    f"{label} C4 branch mismatch for {key}: {actual!r} != {value!r}"
                )
        if branch.get("optimizer_state_initialization") != "fresh":
            raise ValueError(f"{label} did not start with a fresh optimizer.")
        if branch.get("scheduler_state_initialization") != "fresh":
            raise ValueError(f"{label} did not start with a fresh scheduler.")
        budget = metrics.get("training_budget")
        if not isinstance(budget, dict):
            raise ValueError(f"{label} has no training_budget replay provenance.")
        replay_protocols[label] = {
            "primary_per_replay": budget.get("primary_per_replay"),
            "requested_replay_fraction": budget.get("requested_replay_fraction"),
            "replay_sample_exposures_estimate": budget.get(
                "replay_sample_exposures_estimate"
            ),
            "ancientdoc_sample_exposures_estimate": budget.get(
                "ancientdoc_sample_exposures_estimate"
            ),
        }
        observed[label] = branch
    for key in ("primary_per_replay", "requested_replay_fraction"):
        if replay_protocols["c5"].get(key) != replay_protocols["c6"].get(key):
            raise ValueError(f"C5/C6 replay protocol differs for {key}.")
    return {
        "expected": expected,
        "observed": observed,
        "replay_protocols": replay_protocols,
        "status": "consistent",
    }


def evaluator_command(args: argparse.Namespace, job: EvaluationJob, dataset_root: Path) -> list[str]:
    run_got2 = args.ocrmodel_root / "tools" / "environment" / "run_got2.sh"
    evaluator = args.project_root / "scripts" / "evaluate_GOT_layout.py"
    tokenizer = os.environ.get("GOT_TOKENIZER_MODEL", str(args.c0_model.expanduser().resolve()))
    return [
        "bash", str(run_got2), str(evaluator),
        "--model-name-or-path", str(job.model_path),
        "--model-kind", job.model_kind,
        "--tokenizer-name-or-path", tokenizer,
        "--layout-manifest", str(dataset_root / job.split / "manifest.jsonl"),
        "--layout-image-root", str(dataset_root / job.split),
        "--layout-split", job.split,
        "--output-dir", str(job.output_dir),
        "--max-regions", "16",
        "--max-records", str(args.max_records),
        "--model-max-length", str(args.model_max_length),
        "--max-new-tokens", str(args.max_new_tokens),
        "--no-repeat-ngram-size", str(args.no_repeat_ngram_size),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--object-threshold", "0.5",
        "--iou-threshold", "0.5",
        "--dtype", "bfloat16",
        "--device", "cuda",
    ]


def summary_path(output_dir: Path) -> Path:
    return output_dir / "layout_validation_metrics.json"


def run_queue(args: argparse.Namespace, jobs: Iterable[EvaluationJob], dataset_root: Path, event_log: Path) -> list[dict[str, Any]]:
    pending = collections.deque(jobs)
    free_gpus = collections.deque(args.gpu_ids)
    active: dict[int, tuple[subprocess.Popen[str], str, EvaluationJob, Any]] = {}
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    event_log.parent.mkdir(parents=True, exist_ok=True)

    def emit(payload: dict[str, Any]) -> None:
        line = compact_json(payload)
        print(line, flush=True)
        with event_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def attach_job(payload: dict[str, Any], job: EvaluationJob) -> dict[str, Any]:
        enriched = dict(payload)
        enriched["_label"] = job.label
        enriched["_step"] = job.step
        enriched["_summary_path"] = str(summary_path(job.output_dir))
        return enriched

    while pending or active:
        while pending and free_gpus:
            job = pending.popleft()
            output_dir = job.output_dir
            metric_file = summary_path(output_dir)
            if metric_file.is_file():
                results.append(attach_job(read_json(metric_file), job))
                emit({"event": "evaluation_reused", "label": job.label, "step": job.step, "output": str(output_dir)})
                continue
            if output_dir.exists() and any(output_dir.iterdir()):
                existing = {path.name for path in output_dir.iterdir()}
                if not args.resume or not existing.issubset({"launcher.log"}):
                    raise FileExistsError(
                        "Refusing to overwrite incomplete evaluation output: "
                        f"{output_dir}; existing={sorted(existing)}"
                    )
            output_dir.mkdir(parents=True, exist_ok=True)
            gpu = free_gpus.popleft()
            log_path = output_dir / "launcher.log"
            log = log_path.open("w", encoding="utf-8")
            environment = dict(os.environ)
            environment.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpu})
            process = subprocess.Popen(
                evaluator_command(args, job, dataset_root),
                cwd=args.project_root,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active[process.pid] = (process, gpu, job, log)
            emit({"event": "evaluation_started", "label": job.label, "step": job.step, "gpu": gpu, "pid": process.pid})

        finished: list[int] = []
        for pid, (process, gpu, job, log) in active.items():
            code = process.poll()
            if code is None:
                continue
            log.close()
            finished.append(pid)
            free_gpus.append(gpu)
            metric_file = summary_path(job.output_dir)
            if code != 0 or not metric_file.is_file():
                failure = {"event": "evaluation_failed", "label": job.label, "step": job.step, "gpu": gpu, "exit_code": code, "log": str(job.output_dir / "launcher.log")}
                failures.append(failure)
                emit(failure)
                continue
            payload = read_json(metric_file)
            results.append(attach_job(payload, job))
            emit({"event": "evaluation_finished", "label": job.label, "step": job.step, "gpu": gpu, "pages": payload.get("pages"), "pages_per_second": payload.get("runtime", {}).get("pages_per_second")})
        for pid in finished:
            process, _, _, _ = active.pop(pid)
            if process.returncode != 0:
                # Let remaining jobs finish, then report one aggregate failure.
                pass
        if active and not finished:
            time.sleep(0.5)
    if failures:
        failed_labels = ", ".join(
            f"{item['label']}@{item.get('step')}" for item in failures
        )
        raise RuntimeError(f"{len(failures)} evaluation job(s) failed: {failed_labels}")
    if any(payload.get("status") != "ok" for payload in results):
        raise RuntimeError("At least one evaluation result is not ok.")
    if not results:
        raise RuntimeError("Evaluation queue produced no results.")
    return results


def ocr_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("metrics", {}).get("ocr")
    if not isinstance(metrics, dict):
        raise ValueError("Evaluation output has no OCR metrics")
    return metrics


def select_best(results: list[dict[str, Any]], label: str) -> dict[str, Any]:
    candidates = [payload for payload in results if payload.get("_label") == label]
    if not candidates:
        raise ValueError(f"No validation candidates for {label}")
    selected = min(
        candidates,
        key=lambda payload: (
            float(ocr_metrics(payload)["page_cer"]),
            float(ocr_metrics(payload).get("whitespace_normalized_page_cer", 1e9)),
            int(payload.get("_step") if payload.get("_step") is not None else 10**18),
        ),
    )
    return {
        "label": label,
        "step": selected.get("_step"),
        "model": selected.get("model"),
        "validation_summary": selected.get("_summary_path"),
        "metrics": ocr_metrics(selected),
    }


def delta(results: dict[str, dict[str, Any]], model: str, reference: str) -> float:
    return float(ocr_metrics(results[model])["page_cer"]) - float(
        ocr_metrics(results[reference])["page_cer"]
    )


def training_provenance(label: str, model_path: Path, selected_step: int | None) -> dict[str, Any]:
    if label == "c0_got2_zero_shot":
        return {
            "role": "zero_shot_reference",
            "selected_step": 0,
            "initial_checkpoint": str(model_path),
            "optimizer_steps": 0,
        }
    candidates = []
    current = model_path
    for _ in range(4):
        candidates.extend(
            current / name
            for name in ("page_ocr_training_metrics.json", "layout_training_metrics.json")
        )
        current = current.parent
    metrics_path = next((path for path in candidates if path.is_file()), None)
    if metrics_path is None:
        return {"role": "trained_baseline", "selected_step": selected_step, "metrics_status": "missing"}
    payload = read_json(metrics_path)
    budget = payload.get("training_budget", {})
    return {
        "role": payload.get("comparison_role", "trained_baseline"),
        "selected_step": selected_step,
        "maximum_optimizer_steps": payload.get("optimizer_steps", payload.get("global_step")),
        "trainable_parameters": payload.get("trainable_parameters"),
        "total_parameters": payload.get("total_parameters"),
        "train_scope": payload.get("train_scope"),
        "frozen_modules": payload.get("frozen_modules"),
        "optimizer": payload.get("optimizer"),
        "learning_rate": payload.get("learning_rate"),
        "lr_scheduler_type": payload.get("lr_scheduler_type"),
        "weight_decay": payload.get("weight_decay"),
        "initial_checkpoint": payload.get("initial_checkpoint", payload.get("source_model")),
        "upstream_training_history": payload.get("upstream_training_history"),
        "ancientdoc_sample_exposures_estimate_at_max_steps": budget.get("ancientdoc_sample_exposures_estimate"),
        "replay_sample_exposures_estimate_at_max_steps": budget.get("replay_sample_exposures_estimate"),
        "primary_per_replay": budget.get("primary_per_replay"),
        "requested_replay_fraction": budget.get("requested_replay_fraction"),
        "supervised_token_exposures_estimate_at_max_steps": budget.get("supervised_token_exposures_estimate"),
        "metrics_path": str(metrics_path),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    ocr = {label: summary[label]["metrics"]["ocr"] for label in LABELS}
    lines = [
        "# AncientDoc 选择后评估报告",
        "",
        "本报告将 trained checkpoints 只在 group-isolated validation 上选择，随后对冻结的最佳 checkpoint 和 C0 zero-shot reference 进行一次 test 评估。",
        "布局 metadata 不作为模型输入；模型输入仍为整页图像和 OCR prompt。",
        "",
        "| 模型 | test pages | page CER | 去空白 CER | 完全匹配页 |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        metric = ocr[label]
        lines.append(
            f"| {label} | {metric['pages']} | {metric['page_cer']:.6f} | "
            f"{metric.get('whitespace_normalized_page_cer', float('nan')):.6f} | "
            f"{metric['page_exact_matches']}/{metric['pages']} |"
        )
    lines.extend(
        [
            "",
            "## Checkpoint selection and training scope",
            "",
            "| 模型 | selected step | train scope | trainable parameters | initial checkpoint |",
            "|---|---:|---|---:|---|",
        ]
    )
    for label in LABELS:
        provenance = summary["training_provenance"][label]
        lines.append(
            f"| {label} | {provenance.get('selected_step', '-')} | "
            f"{provenance.get('train_scope', '-')} | "
            f"{provenance.get('trainable_parameters', '-')} | "
            f"{provenance.get('initial_checkpoint', '-')} |"
        )
    lines.extend(
        [
            "",
            "## Required deltas",
            "",
            "| delta | page CER difference |",
            "|---|---:|",
        ]
    )
    for name, value in summary["deltas"].items():
        lines.append(f"| {name} | {value:+.6f} |")
    lines.extend(
        [
            "",
            "## Fairness notes",
            "",
            "- 相同步数不自动等于相同训练预算；报告保留 checkpoint step、可训练参数、样本/token 暴露量和初始 checkpoint。",
            "- C1 与 C4 的上游训练历史和结构仍不同，C4-C1 只能解释为完整适配路线对照。",
            "- C4 由独立 validation-only selection 冻结；C5/C6 都从该同一 C4-best 独立启动。",
            "- C5-C4 衡量 OCR replay；C6-C4 衡量带布局监督 replay 的整体变化；C6-C5 比较 synthetic layout supervision。",
            "- validation 只用于 checkpoint 选择，test 不参与选择。",
            "- C0 是 zero-shot reference，不是同预算训练 baseline。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ocrmodel_root = args.ocrmodel_root.expanduser().resolve()
    args.ocrmodel_root = ocrmodel_root
    args.project_root = args.project_root.expanduser().resolve()
    paths = dict(
        GOT_LAYOUT_DATA=Path(os.environ.get("GOT_LAYOUT_DATA", args.workspace / "training_data" / "got_layout_pages")),
        GOT_EVALUATION_RUNS=Path(os.environ.get("GOT_EVALUATION_RUNS", args.workspace / "evaluation_runs" / "GOT")),
    )
    dataset_root = paths["GOT_LAYOUT_DATA"] / args.ancient_dataset_id
    validate_dataset(dataset_root, args.max_ratio_deviation)
    if args.c4_selection is None:
        raise ValueError(
            "--c4-selection is required so final C4 and the C5/C6 branch point cannot diverge."
        )
    c4_branch = load_c4_selection(args.c4_selection.expanduser().resolve())
    if args.phase == "test":
        if args.selection is None:
            raise ValueError("--selection is required for --phase test")
        selection = read_json(args.selection.expanduser().resolve())
        selected = selection["selected"]
        model_paths = {label: Path(selected[label]["model"]) for label in TRAINED_LABELS}
        model_paths["c0_got2_zero_shot"] = args.c0_model.expanduser().resolve()
        if model_paths["c4"].resolve() != Path(c4_branch["selected_model_path"]).resolve():
            raise ValueError("Final selection C4 differs from the frozen C4 branch selection.")
    else:
        model_paths = training_model_args(args, c4_branch)
    branch_consistency = validate_replay_branch_consistency(model_paths, c4_branch)
    suite_root = (args.suite_root or paths["GOT_EVALUATION_RUNS"] / args.run_prefix).expanduser().resolve()
    if suite_root.exists() and not args.resume:
        raise FileExistsError(f"Evaluation suite already exists: {suite_root}; use --resume or a new --run-prefix")
    suite_root.mkdir(parents=True, exist_ok=True)
    event_log = suite_root / "queue.jsonl"

    selected: dict[str, Any]
    if args.phase in {"select", "select-and-test"}:
        validation_jobs: list[EvaluationJob] = []
        for label in LABELS:
            if label == "c0_got2_zero_shot":
                candidates = [(model_paths[label], None)]
            elif label == "c4":
                candidates = [(model_paths[label], int(c4_branch["selected_step"]))]
            else:
                candidates = discover_candidates(model_paths[label])
            for model_path, step in candidates:
                candidate_name = "final" if step is None else f"step-{step:06d}"
                validation_jobs.append(
                    EvaluationJob(
                        label=label,
                        model_kind="baseline" if label.startswith("c0") or label.startswith("c1") else "vlqa",
                        model_path=model_path,
                        output_dir=suite_root / "validation" / label / candidate_name,
                        split="validation",
                        step=step,
                    )
                )
        raw_results = run_queue(args, validation_jobs, dataset_root, event_log)
        validation_results = raw_results
        selected = {label: select_best(validation_results, label) for label in TRAINED_LABELS}
        selected["c0_got2_zero_shot"] = next(
            item for item in validation_results if item["_label"] == "c0_got2_zero_shot"
        )
        selection_payload = {
            "status": "ok",
            "split": "validation",
            "selected": selected,
            "candidates": validation_results,
            "c4_branch_selection": c4_branch,
            "replay_branch_consistency": branch_consistency,
            "test_used_for_selection": False,
        }
        (suite_root / "selection.json").write_text(
            json.dumps(selection_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if args.phase == "select":
            print(compact_json({"status": "ok", "phase": "select", "suite_root": str(suite_root), "selection": str(suite_root / "selection.json")}))
            return 0
        model_paths = {label: Path(selected[label]["model"]) for label in TRAINED_LABELS}
        model_paths["c0_got2_zero_shot"] = args.c0_model.expanduser().resolve()

    test_jobs = [
        EvaluationJob(
            label=label,
            model_kind="baseline" if label.startswith("c0") or label.startswith("c1") else "vlqa",
            model_path=model_paths[label],
            output_dir=suite_root / "test" / label,
            split="test",
            step=(selected.get(label, {}).get("step") if isinstance(selected, dict) else None),
        )
        for label in LABELS
    ]
    test_results = run_queue(args, test_jobs, dataset_root, event_log)
    by_label = {item["_label"]: item for item in test_results}
    if set(by_label) != set(LABELS):
        raise RuntimeError(f"Missing final test results: {sorted(set(LABELS) - set(by_label))}")
    output = {
        "status": "ok",
        "suite_root": str(suite_root),
        **by_label,
        "selection": str(suite_root / "selection.json") if (suite_root / "selection.json").is_file() else str(args.selection),
        "deltas": {
            "c1_minus_c0": delta({label: by_label[label] for label in LABELS}, "c1_got2_ocr_only", "c0_got2_zero_shot"),
            "c4_minus_c1": delta({label: by_label[label] for label in LABELS}, "c4", "c1_got2_ocr_only"),
            "c5_minus_c4": delta({label: by_label[label] for label in LABELS}, "c5", "c4"),
            "c6_minus_c4": delta({label: by_label[label] for label in LABELS}, "c6", "c4"),
            "c6_minus_c5": delta({label: by_label[label] for label in LABELS}, "c6", "c5"),
        },
        "training_provenance": {
            label: training_provenance(
                label,
                model_paths[label],
                next((job.step for job in test_jobs if job.label == label), None),
            )
            for label in LABELS
        },
        "c4_branch_selection": c4_branch,
        "replay_branch_consistency": branch_consistency,
        "fairness": {
            "same_steps_is_same_budget": False,
            "c0_is_zero_shot_reference": True,
            "validation_used_only_for_checkpoint_selection": True,
            "test_used_for_final_report_only": True,
            "dynamic_gpu_queue": True,
            "c4_selected_before_replay_on_validation": True,
            "c5_c6_share_identical_c4_branch_point": True,
        },
    }
    (suite_root / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_report(suite_root / "report.md", output)
    print(compact_json({"status": "ok", "phase": "test", "suite_root": str(suite_root), "summary": str(suite_root / "summary.json"), "report": str(suite_root / "report.md")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
