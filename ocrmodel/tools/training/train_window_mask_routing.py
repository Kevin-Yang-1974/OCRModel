#!/usr/bin/env python3
"""Train only the first-layer window head over the frozen line100 backbone.

Sequential teacher forcing uses the same hard-mask/cache timing as inference.
The target for the head at q is the window at q+2 because it drives q+1.
GT supplies loss targets only; every applied bias comes from the learned head.
"""

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

import torch
from evaluate_window_mask_routing import PROFILE, eos_ids, load_backbone, sha256, targets_for
from torch.nn import functional as F

from layout_ocr.data import load_records, prepare_inference_inputs
from layout_ocr.decoder_mask_checkpoint import save_decoder_mask_checkpoint
from layout_ocr.mask_losses import mask_and_dice_loss
from layout_ocr.window_mask_routing import FirstLayerWindowRuntime


def teacher_forced_page(model, fusion, inputs, target_ids, targets, chunk_size):
    """Accumulate head gradients; caller updates weights only after the page."""
    fusion.set_page(inputs)
    ids = inputs["input_ids"]
    kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}
    kwargs.update(use_cache=True, cache_position=torch.arange(ids.shape[1], device=ids.device))
    kwargs["position_ids"] = model._prepare_position_ids_for_generation(ids, kwargs)
    steps = len(target_ids) - 1
    if steps <= 0:
        raise ValueError("a page needs at least one content token and EOS")
    chunk, total = [], 0.0
    for step in range(steps):
        prepared = model.prepare_inputs_for_generation(ids, is_first_iteration=step == 0, **kwargs)
        output = model(**prepared, return_dict=True)
        index = step + 1  # prefill head predicts the window used to generate target token 1
        rt = fusion.runtime
        mask_loss, _ = mask_and_dice_loss(
            rt.last_mask,
            targets.mask[:, index : index + 1],
            targets.spatial_valid[:, index : index + 1],
        )
        stop_loss = F.binary_cross_entropy(
            rt.last_stop.squeeze(-1), targets.stop_target[:, index : index + 1]
        )
        loss = mask_loss + stop_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite window loss at target {index}")
        total += float(loss.detach())
        chunk.append(loss / steps)
        kwargs = model._update_model_kwargs_for_generation(output, kwargs, is_encoder_decoder=False)
        ids = torch.cat((ids, target_ids[step : step + 1].view(1, 1)), dim=1)
        if len(chunk) == chunk_size or step == steps - 1:
            torch.stack(chunk).sum().backward()
            chunk = []
            fusion.detach_recurrence()
    return total / steps


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if min(args.max_steps, args.chunk_size, args.save_every) <= 0 or args.learning_rate <= 0:
        parser.error("steps, chunk size, checkpoint interval and learning rate must be positive")
    return args


def main():
    args = parse_args()
    records = load_records(args.train_manifest)
    if any(r.get("split", r.get("official_split")) in ("test", "validation") for r in records):
        raise ValueError("head training accepts train records only")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model, processor = load_backbone(args.model_path, args.backbone_checkpoint, args.device)
    fusion = FirstLayerWindowRuntime(model, processor.tokenizer)
    fusion.head.train()
    optimizer = torch.optim.AdamW(
        fusion.head.parameters(), lr=args.learning_rate, weight_decay=0.01
    )
    eos = eos_ids(model, processor)
    fingerprint = {
        "architecture": "line100-first-layer-window-v1",
        "profile": asdict(PROFILE),
        "train_sha256": sha256(args.train_manifest),
        "train_pages": len(records),
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "backbone_lora_sha256": sha256(args.backbone_checkpoint / "decoder_lora.safetensors"),
        "model_path": str(args.model_path),
        "test_manifest_read": False,
        "test_used_for_selection": False,
        "validation_manifest_read": False,
        "head_only": True,
        "decoder_lora_frozen": True,
        "mask_target_shift": 2,
        "per_device_batch": 1,
        "gpu_count": 1,
        "effective_global_batch": 1,
        "ddp": False,
        "seed": PROFILE.seed,
        "learning_rate": args.learning_rate,
        "decoder_learning_rate": 0,
        "warmup_steps": args.warmup_steps,
        "schedule": "cosine, min ratio 0.1",
        "max_steps": args.max_steps,
        "chunk_size": args.chunk_size,
        "mask_loss": "balanced BCE weight 1",
        "stop_loss_weight": 1,
        "dice_weight": 0,
        "auxiliary_weight": 0,
        "gate": "none",
        "precision": "head fp32; backbone bf16",
    }
    (args.output_dir / "protocol.json").write_text(
        json.dumps(fingerprint, indent=2), encoding="utf-8"
    )
    rng = random.Random(PROFILE.seed)
    order = list(range(len(records)))
    with (args.output_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for step in range(args.max_steps):
            if step % len(order) == 0:
                rng.shuffle(order)
            record = records[order[step % len(order)]]
            inputs = prepare_inference_inputs(
                processor, {"image_path": record["image_path"]}, torch.device(args.device)
            )
            targets, target_ids = targets_for(
                processor, record, torch.device(args.device), eos, fusion.runtime.spatial_merge_size
            )
            warmup = max(1, args.warmup_steps)
            ratio = (
                (step + 1) / warmup
                if step < args.warmup_steps
                else 0.1
                + 0.9
                * 0.5
                * (
                    1
                    + math.cos(
                        math.pi
                        * (step - args.warmup_steps)
                        / max(1, args.max_steps - args.warmup_steps)
                    )
                )
            )
            optimizer.param_groups[0]["lr"] = args.learning_rate * ratio
            optimizer.zero_grad(set_to_none=True)
            loss = teacher_forced_page(model, fusion, inputs, target_ids, targets, args.chunk_size)
            torch.nn.utils.clip_grad_norm_(fusion.head.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in fusion.head.parameters()):
                raise FloatingPointError("non-finite mask checkpoint")
            row = {
                "step": step + 1,
                "page_id": record["page_id"],
                "loss": loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            if (step + 1) % args.save_every == 0 or step + 1 == args.max_steps:
                out = args.output_dir / f"step-{step + 1}"
                save_decoder_mask_checkpoint(
                    out,
                    config=fusion.runtime.config,
                    router_state=fusion.head.state_dict(),
                    fingerprint=fingerprint,
                    training_state={"step": step + 1, "optimizer": optimizer.state_dict()},
                )
                (out / "window_routing_profile.json").write_text(
                    json.dumps(asdict(PROFILE), indent=2), encoding="utf-8"
                )
    (args.output_dir / "summary.json").write_text(
        json.dumps({"status": "complete", **fingerprint}), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
