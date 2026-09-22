#!/usr/bin/env python3
"""line100 -> 3--5 window GT acceptance, or prompt-only learned-mask evaluation."""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from safetensors.torch import load_file

from layout_ocr.data import load_records, prepare_inference_inputs, prepare_training_inputs
from layout_ocr.decoder_mask_checkpoint import load_config, load_fingerprint, restore_router
from layout_ocr.decoder_mask_router import _normalized_grid_xywh
from layout_ocr.lora import inject_decoder_lora, load_lora_state_dict
from layout_ocr.mask_targets import build_mask_targets
from layout_ocr.metrics import aggregate_ocr_metrics
from layout_ocr.window_mask_routing import FirstLayerWindowRuntime, WindowRoutingProfile

PROFILE = WindowRoutingProfile()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_backbone(model_path, checkpoint, device):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from layout_ocr.train_screen import configure_deterministic_execution

    configure_deterministic_execution()
    random.seed(PROFILE.seed)
    torch.manual_seed(PROFILE.seed)
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, local_files_only=True)
    processor.image_processor.size = {
        **processor.image_processor.size,
        "longest_edge": PROFILE.max_pixels,
    }
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    )
    model.to(device)
    inject_decoder_lora(model, rank=8, alpha=8, dropout=0)
    weights = load_file(str(checkpoint / "decoder_lora.safetensors"))
    if not all(torch.isfinite(t).all() for t in weights.values()):
        raise FloatingPointError("backbone LoRA contains non-finite values")
    load_lora_state_dict(model, weights)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model, processor


def eos_ids(model, processor):
    ids = set()
    for value in (processor.tokenizer.eos_token_id, model.generation_config.eos_token_id):
        if value is not None:
            ids.update(value if isinstance(value, (tuple, list)) else [value])
    return sorted(ids)


def targets_for(processor, record, device, eos, merge, target_mode="window"):
    inputs = prepare_training_inputs(processor, record, device, set(eos))
    length = int((inputs["labels"] == -100).int().cumprod(1).sum())
    target_ids = inputs["input_ids"][0, length:]
    xywh, _ = _normalized_grid_xywh(inputs["image_grid_thw"], merge)
    targets = build_mask_targets(
        processor.tokenizer,
        record,
        target_ids,
        eos,
        xywh,
        target_mode=target_mode,
        window_min=3,
        window_max=5,
        line_source="annotation",
        raster_mode="hard",
    )
    return targets, target_ids


def acceptance(
    metrics,
    manifest_hash,
    mode,
    *,
    limited=False,
    legacy_layout=False,
    target_mode="window",
    bias=None,
):
    """Eligibility for the one pre-registered configuration.

    The recorded criterion -- window target, B=1.0, all 149 pages -- was fixed
    before the run.  Any other target shape or bias is a diagnostic and returns
    eligible=False, so a swept or stronger configuration can never be read as
    the recorded configuration having passed.
    """

    eligible = (
        mode == "gt"
        and target_mode == "window"
        and (bias is None or float(bias) == float(PROFILE.bias))
        and not limited
        and not legacy_layout
        and metrics["pages"] == PROFILE.validation_pages
        and manifest_hash == PROFILE.validation_sha256
    )
    return {
        "eligible": eligible,
        "criterion": "GT-window validation CER < 0.13",
        "passed": bool(metrics["cer"] < PROFILE.acceptance_cer) if eligible else None,
        "threshold": PROFILE.acceptance_cer,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        required=True,
        help="line100 trained checkpoint-3000, including decoder_lora.safetensors",
    )
    parser.add_argument("--mask-checkpoint", type=Path)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("gt", "predicted", "legacy-line"), default="gt")
    parser.add_argument(
        "--legacy-layout-control",
        action="store_true",
        help="GT diagnostic only: keep old geometry branch to isolate its removal",
    )
    parser.add_argument(
        "--target-mode",
        choices=("window", "anchored", "line", "token"),
        default="window",
        help="GT spatial target shape; only 'window' is acceptance-eligible. "
        "'anchored' is the in-line window union the rest of its line",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=None,
        help="per-key additive logit bias; defaults to the profile's recorded 1.0. "
        "The recorded window value was inherited from the whole-line arm and has "
        "never been swept for this target",
    )
    parser.add_argument("--pages", type=int, default=0, help="smoke subset, never acceptance")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard-index must be in [0, shard-count)")
    return args


def shard_records(records, count, index):
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid shard count/index")
    return records[index::count]


def main():
    args = parse_args()
    if args.mode == "predicted" and (args.mask_checkpoint is None or args.legacy_layout_control):
        raise ValueError("predicted mode needs a trained mask checkpoint and no layout control")
    records = load_records(args.validation_manifest)
    if any(r.get("split", r.get("official_split")) == "test" for r in records):
        raise ValueError("this entry point is validation-only")
    if args.pages:
        records = records[: args.pages]
    full_page_ids = [r["page_id"] for r in records]
    records = shard_records(records, args.shard_count, args.shard_index)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model, processor = load_backbone(args.model_path, args.backbone_checkpoint, args.device)
    eos = eos_ids(model, processor)
    metadata = {
        "profile": asdict(PROFILE),
        "mode": args.mode,
        "target_mode": args.target_mode,
        "bias": float(PROFILE.bias if args.bias is None else args.bias),
        "model_path": str(args.model_path),
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "backbone_lora_sha256": sha256(args.backbone_checkpoint / "decoder_lora.safetensors"),
        "validation_sha256": sha256(args.validation_manifest),
        "test_manifest_read": False,
        "test_used_for_selection": False,
        "reads_ground_truth_for_routing": args.mode != "predicted",
        "usable_for_selection": args.mode == "predicted",
        "processor": "fast",
        "attention_backend": "math-sdpa",
        "precision": "bfloat16",
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "full_page_ids": full_page_ids,
        "page_ids": [r["page_id"] for r in records],
        "limited": bool(args.pages),
    }
    legacy_bridge = None
    if args.mode == "legacy-line" or args.legacy_layout_control:
        from layout_ocr.glm_bridge import install_layout_adapter
        from layout_ocr.train_screen import load_adapter_checkpoint

        legacy_bridge = install_layout_adapter(
            model,
            "geometry",
            num_queries=32,
            max_residual_scale=0.03,
            initial_residual_scale=0,
            adapter_precision="fp32",
        )
        load_adapter_checkpoint(args.backbone_checkpoint, legacy_bridge)
        model.eval()
    metadata["layout_branch_present"] = legacy_bridge is not None
    route = None
    fusion = None
    if args.mode == "legacy-line":
        from layout_ocr.attention_routing import install_attention_routing

        route, _ = install_attention_routing(
            model,
            legacy_bridge,
            bias=PROFILE.bias,
            tokenizer=processor.tokenizer,
            pointer="synced",
            box_source="line",
        )
    else:
        config = load_config(args.mask_checkpoint) if args.mask_checkpoint else None
        profile = (
            PROFILE
            if args.bias is None
            else replace(PROFILE, bias=float(args.bias))
        )
        fusion = FirstLayerWindowRuntime(model, processor.tokenizer, config, profile)
        if args.mode == "predicted":
            fingerprint = load_fingerprint(args.mask_checkpoint)
            if (
                fingerprint.get("architecture") != "line100-first-layer-window-v1"
                or fingerprint.get("backbone_lora_sha256") != metadata["backbone_lora_sha256"]
            ):
                raise ValueError(
                    "mask checkpoint was trained with a different backbone or mask timing"
                )
            saved = json.loads((args.mask_checkpoint / "window_routing_profile.json").read_text())
            if saved != asdict(PROFILE):
                raise ValueError("mask checkpoint routing profile differs from line100-window")
            restore_router(model, fusion.runtime, args.mask_checkpoint)
            metadata["mask_sha256"] = sha256(args.mask_checkpoint / "decoder_mask.safetensors")
        fusion.head.eval()
    (args.output_dir / "protocol.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pairs, limit_hits = [], 0
    start = time.time()
    with (args.output_dir / "validation_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            # Predicted mode passes image+prompt only to preprocessing and routing.
            visual_record = {k: record[k] for k in ("image_path", "prompt") if k in record}
            inputs = prepare_inference_inputs(processor, visual_record, torch.device(args.device))
            length = inputs["input_ids"].shape[1]
            if legacy_bridge is not None:
                legacy_bridge.set_grid_thw(inputs["image_grid_thw"])
            if route is not None:
                route.set_page(
                    record["page_id"],
                    record.get("characters"),
                    length,
                    inputs["input_ids"],
                    reference=record["page_text"],
                    regions=record["regions"],
                )
            elif args.mode == "gt":
                targets, _ = targets_for(
                    processor,
                    record,
                    torch.device(args.device),
                    eos,
                    fusion.runtime.spatial_merge_size,
                    args.target_mode,
                )
                fusion.set_page(
                    inputs, record["page_id"], gt_targets=targets, reference=record["page_text"]
                )
            else:
                fusion.set_page(inputs, record["page_id"])
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=PROFILE.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=eos,
                )
            tokens = generated[0, length:]
            prediction = processor.tokenizer.decode(tokens, skip_special_tokens=True)
            pairs.append((record["page_text"], prediction))
            hit = not any(int(t) in eos for t in tokens.tolist())
            limit_hits += hit
            result = {
                "page_id": record["page_id"],
                "reference": record["page_text"],
                "prediction": prediction,
                "generation_tokens": len(tokens),
                "generation_limit_hit": hit,
                "routing": route.report() if route else fusion.report(),
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {"pages": len(pairs), "total": len(records), "page_id": record["page_id"]}
                ),
                flush=True,
            )
    metrics = aggregate_ocr_metrics(pairs, Counter())
    metrics["generation_limit_hits"] = limit_hits
    summary = {
        **metadata,
        "status": "complete",
        "validation": metrics,
        "elapsed_seconds": time.time() - start,
        "acceptance": acceptance(
            metrics,
            metadata["validation_sha256"],
            args.mode,
            limited=bool(args.pages) or args.shard_count > 1,
            legacy_layout=args.legacy_layout_control,
            target_mode=args.target_mode,
            bias=metadata["bias"],
        ),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
