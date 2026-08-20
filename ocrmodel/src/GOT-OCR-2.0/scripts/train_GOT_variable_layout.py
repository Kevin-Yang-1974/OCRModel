#!/usr/bin/env python3
"""PVLD-32 prototype trainer and preflight contract.

The existing GOT2 Trainer is intentionally untouched.  This prototype can
train on a feature manifest containing ``visual_features`` (``[L,D]`` arrays)
and otherwise performs a strict manifest/config preflight.  It never reports
preflight as model training success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pretrain", "joint-train", "preflight"), default="preflight")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--validation-manifest", type=Path)
    parser.add_argument("--model-name-or-path", type=Path)
    parser.add_argument("--tokenizer-name-or-path", type=Path)
    parser.add_argument("--num-layout-prompt-queries", type=int, default=32)
    parser.add_argument("--max-layout-tokens", type=int, default=2048)
    parser.add_argument("--max-layout-records", type=int, default=64)
    parser.add_argument("--layout-decoder-layers", type=int, default=2)
    parser.add_argument("--layout-decoder-hidden-size", type=int, default=256)
    parser.add_argument("--layout-prompt-mode", default="global_prompt_full_page")
    parser.add_argument("--layout-loss-preset", default="sequence_bbox_direction")
    parser.add_argument("--layout-stage", choices=("p1", "p2"), default="p1")
    parser.add_argument("--layout-bbox-loss-weight", type=float, default=5.0)
    parser.add_argument("--layout-type-loss-weight", type=float, default=1.0)
    parser.add_argument("--layout-direction-loss-weight", type=float, default=1.0)
    parser.add_argument("--layout-count-loss-weight", type=float, default=0.1)
    parser.add_argument("--layout-prompt-diversity-loss-weight", type=float, default=0.0)
    parser.add_argument("--ocr-loss-weight", type=float, default=0.0)
    parser.add_argument("--max-regions", type=int, default=32, help="Fixed-Slot comparator only; PVLD does not use it as a region cap.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--visual-feature-manifest", type=Path, help="Optional JSONL with visual_features for standalone tensor training.")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Run one CPU/GPU tensor step without claiming GOT training.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.layout_stage == "p1" and args.ocr_loss_weight != 0.0:
        raise ValueError("P1 requires --ocr-loss-weight 0")
    if args.layout_stage == "p2" and args.ocr_loss_weight <= 0.0:
        raise ValueError("P2 requires positive --ocr-loss-weight")


def make_run_dirs(output: Path) -> tuple[Path, Path]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / "metadata"
    metadata.mkdir(exist_ok=True)
    return output, metadata


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_feature_smoke(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    from GOT.model.layout_prompt_decoder import (
        LayoutRecordHeads,
        LayoutVocabulary,
        VariableLayoutDecoder,
        VariableLayoutLoss,
    )

    feature_records = read_manifest(args.visual_feature_manifest) if args.visual_feature_manifest else records
    first = feature_records[0]
    features = first.get("visual_features")
    if features is None:
        raise RuntimeError(
            "PVLD standalone training requires --visual-feature-manifest with visual_features; "
            "GOT2 vision-tower wiring is intentionally not implied by this prototype."
        )
    visual = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    vocabulary = LayoutVocabulary()
    target_tokens = first.get("layout_target_tokens", ["<LAYOUT>", "<EOS>"])
    target_ids = torch.tensor([vocabulary.encode(target_tokens)], dtype=torch.long)
    decoder = VariableLayoutDecoder(
        vocab_size=vocabulary.vocab_size,
        hidden_size=args.layout_decoder_hidden_size,
        visual_size=visual.shape[-1],
        num_prompts=args.num_layout_prompt_queries,
        num_layers=args.layout_decoder_layers,
        max_layout_tokens=args.max_layout_tokens,
    )
    heads = LayoutRecordHeads(args.layout_decoder_hidden_size)
    output = decoder(target_ids, visual)
    region_hidden = output.hidden_states[:, :1]
    record_output = heads(region_hidden)
    loss = VariableLayoutLoss(
        bbox_weight=args.layout_bbox_loss_weight,
        type_weight=args.layout_type_loss_weight,
        direction_weight=args.layout_direction_loss_weight,
        count_weight=args.layout_count_loss_weight,
        prompt_diversity_weight=args.layout_prompt_diversity_loss_weight,
    )(
        output,
        target_ids,
        record_output,
        torch.zeros_like(record_output.bbox),
        torch.zeros(record_output.type_logits.shape[:2], dtype=torch.long),
        torch.zeros(record_output.direction_logits.shape[:2], dtype=torch.long),
        torch.zeros(record_output.bbox.shape[:2], dtype=torch.bool),
        torch.zeros(1),
        vocabulary.pad_id,
    )
    loss.loss.backward()
    return {
        "status": "tensor_smoke_ok",
        "loss": float(loss.loss.detach()),
        "visual_shape": list(visual.shape),
        "prompt_shape": [1, args.num_layout_prompt_queries, args.layout_decoder_hidden_size],
        "layout_target_tokens": len(target_tokens),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    output, metadata = make_run_dirs(args.output_dir.resolve())
    records = read_manifest(args.manifest.resolve())
    validation_count = len(read_manifest(args.validation_manifest.resolve())) if args.validation_manifest else 0
    status = {
        "status": "preflight",
        "run_id": args.run_id,
        "layout_architecture": "Prompted Variable-Length Layout Decoder",
        "layout_candidate_name": "PVLD-32",
        "num_layout_prompt_queries": args.num_layout_prompt_queries,
        "prompt_queries_are_region_slots": False,
        "max_layout_records": args.max_layout_records,
        "max_layout_tokens": args.max_layout_tokens,
        "max_regions_comparator": args.max_regions,
        "input_protocol": "whole_page_image_plus_ocr_prompt",
        "bbox_direction_order_as_model_input": False,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest.resolve()),
        "train_records": len(records),
        "validation_records": validation_count,
        "layout_stage": args.layout_stage,
        "mode": args.mode,
        "gpu_ids": args.gpu_ids,
        "implementation_scope": "standalone_feature_prototype",
    }
    (metadata / "status.txt").write_text(json.dumps(status, ensure_ascii=False) + "\n", encoding="utf-8")
    result: dict[str, Any] = status.copy()
    if args.smoke or args.visual_feature_manifest:
        result.update(run_feature_smoke(args, records))
    else:
        result.update({
            "status": "preflight_ok",
            "training_executed": False,
            "reason": "GOT2 vision-tower integration is a separate follow-up; no fabricated training metrics were produced.",
        })
    result["completed_at"] = time.time()
    write_json(output / "layout_training_metrics.json", result)
    write_json(output / "summary.json", result)
    (output / "PVLD_PROTOTYPE_FINISHED").touch()
    print(json.dumps({"event": "variable_layout_run_completed", "status": result["status"], "summary": str(output / "summary.json")}, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
