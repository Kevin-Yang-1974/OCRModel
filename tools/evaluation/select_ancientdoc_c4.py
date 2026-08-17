#!/usr/bin/env python3
"""Select the AncientDoc C4 branch checkpoint on validation only."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
OCRMODEL_ROOT = SCRIPT_DIR.parents[1]
TRAINING_TOOLS = OCRMODEL_ROOT / "tools" / "training"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(TRAINING_TOOLS))

import run_ancientdoc_evaluation as evaluation  # noqa: E402
from c4_selection_contract import (  # noqa: E402
    CHECKPOINT_RE,
    checkpoint_provenance,
    file_sha256,
    read_json,
)


@dataclass(frozen=True)
class C4Candidate:
    candidate_id: str
    model_path: Path
    optimizer_step: int
    provenance: dict[str, Any]
    source: str


def discover_c4_candidates(c4_run_root: Path) -> tuple[list[C4Candidate], list[dict[str, Any]]]:
    c4_run_root = c4_run_root.expanduser().resolve()
    model_root = c4_run_root / "p2" / "model"
    if not c4_run_root.is_dir() or not model_root.is_dir():
        raise FileNotFoundError(f"C4 run or p2/model is missing: {c4_run_root}")
    if not (c4_run_root / "LAYOUT_A100_FINISHED").is_file():
        raise RuntimeError(f"C4 run is not complete: {c4_run_root}")
    parent_metrics_path = model_root / "layout_training_metrics.json"
    parent_metrics = read_json(parent_metrics_path)
    if parent_metrics.get("layout_stage") != "p2":
        raise ValueError("C4 parent metrics must report layout_stage='p2'.")

    checkpoint_dirs = sorted(
        (path for path in model_root.glob("checkpoint-*") if path.is_dir()),
        key=lambda path: int(CHECKPOINT_RE.fullmatch(path.name).group(1))
        if CHECKPOINT_RE.fullmatch(path.name)
        else 10**18,
    )
    malformed = [path.name for path in checkpoint_dirs if CHECKPOINT_RE.fullmatch(path.name) is None]
    if malformed:
        raise ValueError(f"Malformed C4 checkpoint directories: {malformed}")
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint-* directories under {model_root}")

    candidates: list[C4Candidate] = []
    for path in checkpoint_dirs:
        match = CHECKPOINT_RE.fullmatch(path.name)
        assert match is not None
        step = int(match.group(1))
        trainer_state_path = path / "trainer_state.json"
        if not trainer_state_path.is_file():
            raise FileNotFoundError(f"Checkpoint has no trainer_state.json: {path}")
        trainer_state = read_json(trainer_state_path)
        if int(trainer_state.get("global_step", -1)) != step:
            raise ValueError(f"Checkpoint step mismatch in {trainer_state_path}")
        provenance = checkpoint_provenance(path)
        provenance.update(
            {
                "trainer_state_path": str(trainer_state_path),
                "trainer_state_sha256": file_sha256(trainer_state_path),
                "parent_metrics_path": str(parent_metrics_path),
                "parent_max_optimizer_steps": int(parent_metrics.get("global_step", -1)),
                "layout_stage": "p2",
            }
        )
        candidates.append(
            C4Candidate(
                candidate_id=f"step-{step:06d}",
                model_path=path.resolve(),
                optimizer_step=step,
                provenance=provenance,
                source="periodic_checkpoint",
            )
        )

    final_step = int(parent_metrics.get("global_step", -1))
    if final_step < 1:
        raise ValueError(f"Invalid C4 final optimizer step in {parent_metrics_path}")
    final_provenance = checkpoint_provenance(model_root)
    duplicate = next(
        (
            candidate
            for candidate in candidates
            if candidate.provenance["weights_sha256"] == final_provenance["weights_sha256"]
        ),
        None,
    )
    excluded: list[dict[str, Any]] = []
    if duplicate is not None:
        excluded.append(
            {
                "model_path": str(model_root),
                "optimizer_step": final_step,
                "reason": "byte_identical_to_periodic_checkpoint",
                "duplicate_of": str(duplicate.model_path),
                "provenance": final_provenance,
            }
        )
    else:
        final_provenance.update(
            {
                "parent_metrics_path": str(parent_metrics_path),
                "parent_metrics_sha256": file_sha256(parent_metrics_path),
                "parent_max_optimizer_steps": final_step,
                "layout_stage": "p2",
            }
        )
        candidates.append(
            C4Candidate(
                candidate_id=f"final-step-{final_step:06d}",
                model_path=model_root.resolve(),
                optimizer_step=final_step,
                provenance=final_provenance,
                source="final_model_distinct_from_periodic_checkpoints",
            )
        )
    return candidates, excluded


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("No evaluated C4 candidates.")
    return min(
        candidates,
        key=lambda item: (
            float(item["validation_metrics"]["page_cer"]),
            float(item["validation_metrics"]["whitespace_normalized_page_cer"]),
            int(item["optimizer_step"]),
            str(item["model_path"]),
        ),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    workspace = Path(os.environ.get("OCR_WORKSPACE", OCRMODEL_ROOT.parent))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c4-run", type=Path, required=True)
    parser.add_argument(
        "--ancient-dataset-id",
        default="ancientdoc_layout_260707_group_isolated_seed20260815",
    )
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=Path(os.environ.get("GOT_TOKENIZER_MODEL", os.environ.get("GOT_SOURCE_MODEL", ""))),
    )
    parser.add_argument("--gpu-ids", type=evaluation.parse_gpu_ids, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--run-id",
        default=f"ancientdoc_c4_selection_{datetime.now():%Y%m%d_%H%M%S}",
    )
    parser.add_argument("--num-workers", type=evaluation.nonnegative_int, default=4)
    parser.add_argument("--max-records", type=evaluation.nonnegative_int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--ocrmodel-root", type=Path, default=OCRMODEL_ROOT)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=OCRMODEL_ROOT / "src" / "GOT-OCR-2.0",
    )
    return parser.parse_args(argv)


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# AncientDoc C4 validation checkpoint selection",
        "",
        "本报告只使用书籍隔离 validation；test 未参与 checkpoint 选择。",
        "模型输入为原始整页图像和 `OCR: ` prompt，布局 metadata 不作为输入。",
        "",
        "| candidate | step | page CER | 去空白 CER | exact match | pages/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    selected_path = payload["selected"]["model_path"]
    for item in payload["candidates"]:
        metrics = item["validation_metrics"]
        marker = " **selected**" if item["model_path"] == selected_path else ""
        lines.append(
            f"| {item['candidate_id']}{marker} | {item['optimizer_step']} | "
            f"{metrics['page_cer']:.6f} | "
            f"{metrics['whitespace_normalized_page_cer']:.6f} | "
            f"{metrics['page_exact_matches']}/{metrics['pages']} | "
            f"{item['runtime'].get('end_to_end_pages_per_second', float('nan')):.6f} |"
        )
    lines.extend(
        [
            "",
            "选择规则固定为 page CER、去空白 page CER、较早 optimizer step。",
            f"选中：`{payload['selected']['model_path']}`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.ocrmodel_root = args.ocrmodel_root.expanduser().resolve()
    args.project_root = args.project_root.expanduser().resolve()
    args.c0_model = args.tokenizer_model.expanduser().resolve()
    args.batch_size = 1
    args.model_max_length = 2048
    args.max_new_tokens = 2048
    args.no_repeat_ngram_size = 20
    args.max_ratio_deviation = 0.03
    layout_data = Path(
        os.environ.get(
            "GOT_LAYOUT_DATA",
            args.workspace / "training_data" / "got_layout_pages",
        )
    ).expanduser().resolve()
    evaluation_runs = Path(
        os.environ.get(
            "GOT_EVALUATION_RUNS",
            args.workspace / "evaluation_runs" / "GOT",
        )
    ).expanduser().resolve()
    dataset_root = layout_data / args.ancient_dataset_id
    evaluation.validate_dataset(dataset_root, args.max_ratio_deviation)
    manifest = dataset_root / "validation" / "manifest.jsonl"
    candidates, excluded = discover_c4_candidates(args.c4_run)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else evaluation_runs / args.run_id
    )
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"C4 selection output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        evaluation.EvaluationJob(
            label="c4",
            model_kind="vlqa",
            model_path=candidate.model_path,
            output_dir=output_dir / "validation" / candidate.candidate_id,
            split="validation",
            step=candidate.optimizer_step,
        )
        for candidate in candidates
    ]
    results = evaluation.run_queue(args, jobs, dataset_root, output_dir / "queue.jsonl")
    result_by_model = {str(Path(item["model"]).resolve()): item for item in results}
    candidate_payloads: list[dict[str, Any]] = []
    for candidate in candidates:
        result = result_by_model.get(str(candidate.model_path))
        if result is None:
            raise RuntimeError(f"Missing C4 evaluation result for {candidate.model_path}")
        ocr = evaluation.ocr_metrics(result)
        candidate_payloads.append(
            {
                "candidate_id": candidate.candidate_id,
                "source": candidate.source,
                "optimizer_step": candidate.optimizer_step,
                "model_path": str(candidate.model_path),
                "validation_metrics": {
                    "pages": ocr["pages"],
                    "page_cer": ocr["page_cer"],
                    "whitespace_normalized_page_cer": ocr["whitespace_normalized_page_cer"],
                    "page_exact_matches": ocr["page_exact_matches"],
                    "page_exact_match_rate": ocr["page_exact_match_rate"],
                },
                "runtime": result.get("runtime", {}),
                "metrics_path": result["_summary_path"],
                "predictions_path": result.get("predictions"),
                "launcher_log": str(
                    output_dir / "validation" / candidate.candidate_id / "launcher.log"
                ),
                "provenance": candidate.provenance,
            }
        )
    selected = select_candidate(candidate_payloads)
    evaluator_path = args.project_root / "scripts" / "evaluate_GOT_layout.py"
    payload = {
        "schema_version": 1,
        "status": "ok",
        "purpose": "c4_checkpoint_selection",
        "c4_run_root": str(args.c4_run.expanduser().resolve()),
        "c4_model_root": str(args.c4_run.expanduser().resolve() / "p2" / "model"),
        "dataset_root": str(dataset_root),
        "manifest": str(manifest),
        "manifest_sha256": file_sha256(manifest),
        "selection_split": "validation",
        "test_used_for_selection": False,
        "selection_rule": [
            "minimum_page_cer",
            "minimum_whitespace_normalized_page_cer",
            "earlier_optimizer_step",
        ],
        "evaluator": {
            "script": str(evaluator_path),
            "script_sha256": file_sha256(evaluator_path),
            "model_input": ["whole_page_image", "ocr_prompt"],
            "prompt": "OCR: ",
            "layout_metadata_as_model_input": False,
            "layout_explanation_forward_skipped_without_annotations": True,
            "decoding": "greedy",
            "max_new_tokens": 2048,
            "no_repeat_ngram_size": 20,
            "batch_size": 1,
            "dtype": "bfloat16",
            "image_processor": "BlipImageEvalProcessor(image_size=1024)",
            "tokenizer_model": str(args.c0_model),
            "max_records": args.max_records,
        },
        "gpu_queue": {
            "gpu_ids": args.gpu_ids,
            "one_evaluator_per_gpu": True,
            "dynamic_no_wave_barrier": True,
        },
        "excluded_duplicates": excluded,
        "candidates": candidate_payloads,
        "selected": selected,
    }
    selection_path = output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    selected_metadata = {
        "status": "ok",
        "selection": str(selection_path),
        "c4_run_root": payload["c4_run_root"],
        "selected_model_path": selected["model_path"],
        "selected_optimizer_step": selected["optimizer_step"],
        "validation_metrics": selected["validation_metrics"],
        "provenance": selected["provenance"],
        "weights_modified": False,
    }
    (output_dir / "selected_checkpoint_metadata.json").write_text(
        json.dumps(selected_metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(output_dir / "report.md", payload)
    print(
        json.dumps(
            {
                "status": "ok",
                "event": "ancientdoc_c4_checkpoint_selected",
                "output_dir": str(output_dir),
                "selection": str(selection_path),
                "selected_step": selected["optimizer_step"],
                "selected_model": selected["model_path"],
                "validation_metrics": selected["validation_metrics"],
                "candidate_count": len(candidate_payloads),
                "excluded_duplicate_count": len(excluded),
                "test_used_for_selection": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
