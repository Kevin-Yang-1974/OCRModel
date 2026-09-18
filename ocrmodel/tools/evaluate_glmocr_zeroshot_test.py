#!/usr/bin/env python3
"""Score the untouched GLM-OCR checkpoint on the locked test split.

This is the zero-shot counterpart of ``tools.evaluate_glmocr_locked_test``: it
runs the same probe pages through the same ``evaluate`` / ``aggregate_ocr_metrics``
code path, but loads no adapter and no decoder LoRA checkpoint.  The adapter is
installed at ``content_gate = 0`` in ``content_only`` mode, where the adapter
forward returns ``merged = visual_tokens`` unchanged, so the visual features
reaching the language model are exactly the official ones.  Whatever gap appears
between this run and a trained locked test is therefore attributable to the
adapter plus decoder LoRA, not to the metric harness.

The tool deliberately has no ``--run-dir`` and no checkpoint arguments: there is
no training run behind it, and its summary says so (``zero_shot: true``,
``eval_checkpoint_dir: null``, ``test_used_for_selection: false``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support both ``python -m tools.evaluate_glmocr_zeroshot_test`` with the
# launcher-provided PYTHONPATH and direct execution from the tools directory.
if __package__ is None:
    source_root = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(source_root))

import torch

from layout_ocr.data import load_records, validate_records
from layout_ocr.train_screen import (
    configure_deterministic_execution,
    evaluate,
    load_model,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["content_only"],
        default="content_only",
        help="only content_only is accepted; it is the identity adapter path",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--adapter-precision", choices=["fp32"], default="fp32")
    parser.add_argument("--layout-loss-profile", choices=["ocr_only"], default="ocr_only")
    parser.add_argument("--query-assignment", choices=["hungarian"], default="hungarian")
    parser.add_argument("--residual-scale-cap", type=float, default=0.03)
    parser.add_argument(
        "--initial-residual-scale",
        type=float,
        default=0.0,
        help="must stay 0 so the semantic residual is exactly disabled",
    )
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--processor-mode", choices=["fast"], default="fast")
    parser.add_argument("--max-eval-new-tokens", type=int, default=1536)
    parser.add_argument("--generation-mode", default="loop_recovery")
    parser.add_argument(
        "--test-shard-index",
        type=int,
        default=0,
        help="zero-based test shard index for parallel zero-shot evaluation",
    )
    parser.add_argument(
        "--test-shard-count",
        type=int,
        default=1,
        help="number of disjoint test shards for parallel zero-shot evaluation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if args.mode != "content_only":
        raise ValueError("zero-shot evaluation requires --mode content_only")
    if args.initial_residual_scale != 0.0:
        raise ValueError(
            "--initial-residual-scale must be 0 for zero-shot; a non-zero gate "
            "would inject an untrained semantic residual"
        )
    if args.test_shard_count <= 0:
        raise ValueError("test-shard-count must be positive")
    if not 0 <= args.test_shard_index < args.test_shard_count:
        raise ValueError("test-shard-index must satisfy 0 <= index < test-shard-count")

    protocol = json.loads(args.protocol_file.read_text(encoding="utf-8"))
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
    test_pages_total = len(test_records)
    if args.test_shard_count > 1:
        test_records = test_records[args.test_shard_index :: args.test_shard_count]
        if not test_records:
            raise ValueError(
                f"test shard {args.test_shard_index} is empty for "
                f"{test_pages_total} pages and {args.test_shard_count} shards"
            )

    configure_deterministic_execution()
    if not torch.cuda.is_available():
        raise RuntimeError("zero-shot test requires CUDA")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model_args = argparse.Namespace(
        model_path=args.model_path,
        processor_mode=args.processor_mode,
        max_pixels=args.max_pixels,
        mode=args.mode,
        num_queries=args.num_queries,
        box_head_mlp=False,
        box_head_hidden=0,
        sem_adapter_mlp=False,
        sem_adapter_hidden=0,
        query_refine_layers=0,
        residual_scale_cap=args.residual_scale_cap,
        initial_residual_scale=args.initial_residual_scale,
        use_validity_head=False,
        initial_valid_probability=0.05,
        validity_gating_mode="legacy_normalized",
        validity_use_transport_evidence=False,
        adapter_precision=args.adapter_precision,
    )
    model, processor, bridge = load_model(model_args, device)
    gate = float(torch.tanh(bridge.adapter.content_gate.detach()).item())
    if gate != 0.0:
        raise RuntimeError(f"zero-shot adapter gate is not exactly 0: {gate!r}")

    output_dir.mkdir(parents=True)
    eval_args = argparse.Namespace(
        mode=args.mode,
        num_queries=args.num_queries,
        layout_loss_profile=args.layout_loss_profile,
        query_assignment=args.query_assignment,
        max_eval_new_tokens=args.max_eval_new_tokens,
        output_dir=output_dir,
        diagnostic_steps=(),
        generation_mode=args.generation_mode,
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
    summary = {
        "status": "complete",
        "split": "test",
        "zero_shot": True,
        "seed": args.seed,
        "mode": args.mode,
        "model_path": str(args.model_path.resolve()),
        "selected_step": 0,
        "selection_file": None,
        "selection_metric": None,
        "eval_checkpoint_dir": None,
        "adapter_gate_after_load": gate,
        "train_pages": len(train_records),
        "test_pages": len(test_records),
        "test_pages_total": test_pages_total,
        "test_shard_index": args.test_shard_index,
        "test_shard_count": args.test_shard_count,
        "num_queries": args.num_queries,
        "max_eval_new_tokens": args.max_eval_new_tokens,
        "generation_mode": args.generation_mode,
        "decoder_adaptation": "frozen",
        "decoder_lora_loaded": False,
        "parameters_unchanged": True,
        "training_updates": 0,
        "metrics": metrics,
        "test_manifest_read": True,
        "test_used_for_selection": False,
    }
    write_json(output_dir / "zeroshot_test_summary.json", summary)
    (output_dir / "ZEROSHOT_TEST_COMPLETED").touch()
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
