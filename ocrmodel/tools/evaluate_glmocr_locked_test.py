#!/usr/bin/env python3
"""Run the selection-locked MTHv2 test evaluation for one completed seed run.

The evaluator deliberately lives outside ``train_screen``'s training path.  It
requires a completed validation selection and never changes that selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support both ``python -m tools.evaluate_glmocr_locked_test`` with the
# launcher-provided PYTHONPATH and direct execution from the tools directory.
if __package__ is None:
    source_root = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(source_root))

import torch

from layout_ocr.data import load_records, validate_records
from layout_ocr.train_screen import (
    configure_deterministic_execution,
    evaluate,
    load_adapter_checkpoint,
    load_model,
    write_json,
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-file",
        type=Path,
        help="validation-only selection file; defaults to run-dir/selection.json",
    )
    parser.add_argument("--mode", choices=["geometry"], default="geometry")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-queries", type=int, default=512)
    parser.add_argument("--adapter-precision", choices=["fp32"], default="fp32")
    parser.add_argument("--layout-loss-profile", choices=["full"], default="full")
    parser.add_argument("--query-assignment", choices=["hungarian"], default="hungarian")
    parser.add_argument("--residual-scale-cap", type=float, default=0.03)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--processor-mode", choices=["fast"], default="fast")
    parser.add_argument("--max-eval-new-tokens", type=int, default=1536)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    metadata_path = run_dir / "metadata.json"
    selection_path = (args.selection_file or (run_dir / "selection.json")).resolve()
    output_dir = run_dir / "locked-test"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not (run_dir / "COMPLETED").is_file():
        raise RuntimeError(f"training run is not complete: {run_dir}")
    summary = _read_json(summary_path)
    metadata = _read_json(metadata_path)
    selection = _read_json(selection_path)
    if summary.get("status") != "complete" or metadata.get("status") != "complete":
        raise RuntimeError("locked test requires a complete training run")
    if summary.get("test_used_for_selection") is not False:
        raise ValueError("training summary does not prove test exclusion")
    if selection.get("status") != "complete":
        raise RuntimeError("selection file is not complete")
    if selection.get("test_used_for_selection") is not False:
        raise ValueError("selection file is not validation-only")
    selected_seeds = {str(value) for value in selection.get("seeds", [])}
    if selected_seeds and str(args.seed) not in selected_seeds:
        raise ValueError(f"selection file does not include seed {args.seed}")
    seed_runs = selection.get("seed_runs") or {}
    if seed_runs and str(args.seed) not in seed_runs:
        raise ValueError(f"selection file has no run for seed {args.seed}")
    selected_step = selection.get("selected_step")
    if not isinstance(selected_step, int) or selected_step <= 0:
        raise ValueError(f"invalid selected_step in {selection_path}: {selected_step!r}")
    checkpoint_dir = run_dir / f"checkpoint-{selected_step}"
    if not (checkpoint_dir / "adapter.safetensors").is_file():
        raise FileNotFoundError(checkpoint_dir / "adapter.safetensors")

    protocol = _read_json(args.protocol_file)
    expected_test_pages = protocol.get("split_pages", {}).get("test")
    train_records = load_records(args.train_manifest)
    test_records = load_records(args.test_manifest)
    validate_records(train_records, split="train", num_queries=args.num_queries)
    validate_records(test_records, split="test", num_queries=args.num_queries)
    if expected_test_pages is not None and len(test_records) != expected_test_pages:
        raise ValueError(
            f"test page count mismatch: protocol={expected_test_pages}, "
            f"manifest={len(test_records)}"
        )

    configure_deterministic_execution()
    if not torch.cuda.is_available():
        raise RuntimeError("locked test requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    adapter_config = metadata.get("adapter_config") or {}
    if not isinstance(adapter_config, dict):
        raise ValueError("metadata.adapter_config must be an object")
    model_args = argparse.Namespace(
        model_path=args.model_path,
        processor_mode=args.processor_mode,
        max_pixels=args.max_pixels,
        mode=args.mode,
        num_queries=args.num_queries,
        residual_scale_cap=args.residual_scale_cap,
        initial_residual_scale=float(
            (summary.get("training") or {}).get(
                "initial_residual_scale",
                metadata.get("initial_residual_scale", 0.0),
            )
        ),
        use_validity_head=bool(adapter_config.get("use_validity_head", False)),
        initial_valid_probability=float(
            adapter_config.get("initial_valid_probability", 0.05)
        ),
        validity_gating_mode=adapter_config.get(
            "validity_gating_mode", "legacy_normalized"
        ),
        validity_use_transport_evidence=bool(
            adapter_config.get("validity_use_transport_evidence", False)
        ),
        adapter_precision=args.adapter_precision,
    )
    model, processor, bridge = load_model(model_args, device)
    load_adapter_checkpoint(checkpoint_dir, bridge)
    output_dir.mkdir(parents=True)
    eval_args = argparse.Namespace(
        mode=args.mode,
        num_queries=args.num_queries,
        layout_loss_profile=metadata.get("layout_loss_profile", args.layout_loss_profile),
        query_assignment=metadata.get("query_assignment", args.query_assignment),
        max_eval_new_tokens=args.max_eval_new_tokens,
        output_dir=output_dir,
        diagnostic_steps=(),
        generation_mode=metadata.get("generation_mode", "plain"),
        audit_prompt_prefix=False,
        eval_disable_repeat_guard=False,
    )
    metrics = evaluate(
        eval_args,
        model,
        processor,
        bridge,
        None,
        test_records,
        train_records,
        device,
        output_dir=output_dir,
        split_name="test",
    )
    locked_summary = {
        "status": "complete",
        "split": "test",
        "seed": args.seed,
        "mode": args.mode,
        "selected_step": selected_step,
        "selection_file": str(selection_path),
        "selection_metric": selection.get("selection_metric", "validation_cer"),
        "train_pages": len(train_records),
        "test_pages": len(test_records),
        "num_queries": args.num_queries,
        "max_eval_new_tokens": args.max_eval_new_tokens,
        "metrics": metrics,
        "test_used_for_selection": False,
    }
    write_json(output_dir / "locked_test_summary.json", locked_summary)
    (output_dir / "LOCKED_TEST_COMPLETED").touch()
    print(json.dumps(locked_summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
