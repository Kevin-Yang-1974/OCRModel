#!/usr/bin/env python3
"""Is the learned attention bias actually landing where recognition needs it?

Three questions, answered on real pages rather than by inspecting the head's own
loss:

1. **Mechanical.** Is a non-zero additive bias reaching the attention logits, and
   is it exactly ``beta * M`` placed on the image keys of the target query rows?
   Two teacher-forced forwards -- one at the trained ``beta``, one at ``beta = 0``
   -- are differenced, so the answer does not rely on re-deriving the model's own
   causal mask.

2. **Alignment.** Does the predicted mask ``M_t`` point at the cell that holds the
   character token ``t`` is supposed to write?  Reported as top-1 hit rate, soft
   IoU, in-box/out-box contrast and centroid distance against the rasterised
   ground-truth box, each with the *chance* level that a random in-grid guess
   would score (a hit rate is meaningless without it).

3. **Attention mass.** The plan is explicit that mask values must not stand in for
   attention mass, so this measures the real thing: the attention weights the
   eager backend returns, captured through a forward hook on a few layers, and
   compared between ``beta`` and ``beta = 0``.  If the bias does not move
   attention onto the masked cells, it cannot be doing anything.

Nothing here reads the test manifest, and the mask head is never fed bbox input:
the ground truth is used only to *score* the prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import torch

from layout_ocr.data import load_records, prepare_training_inputs
from layout_ocr.decoder_mask_checkpoint import load_config, restore_router
from layout_ocr.decoder_mask_model import enable_eager_backend, install_decoder_mask_router
from layout_ocr.decoder_mask_router import _normalized_grid_xywh
from layout_ocr.lora import inject_decoder_lora, load_lora_state_dict
from layout_ocr.decoder_mask_checkpoint import load_lora_state
from layout_ocr.mask_targets import build_mask_targets


def _eos_ids(model: Any, processor: Any) -> set[int]:
    ids: set[int] = set()
    tokenizer = getattr(processor, "tokenizer", None)
    value = getattr(tokenizer, "eos_token_id", None)
    if value is not None:
        if isinstance(value, (list, tuple, set)):
            ids = {int(item) for item in value if item is not None}
        else:
            ids = {int(value)}
    gen = getattr(model, "generation_config", None)
    gvalue = getattr(gen, "eos_token_id", None)
    if gvalue is not None:
        if isinstance(gvalue, (list, tuple, set)):
            ids.update(int(item) for item in gvalue if item is not None)
        else:
            ids.add(int(gvalue))
    return ids


def _image_token_id(model: Any) -> int:
    value = getattr(model.config, "image_token_id", None)
    if value is None:
        value = getattr(getattr(model.config, "text_config", None), "image_token_id", None)
    if value is None:
        raise RuntimeError("could not resolve the model image token id")
    return int(value)


def _prompt_length(inputs: dict[str, Any]) -> int:
    labels = inputs["labels"]
    return int((labels == -100).to(torch.int32).cumprod(dim=1).sum().item())


def _alignment_stats(
    predicted: torch.Tensor, ground_truth: torch.Tensor, has_box: torch.Tensor
) -> dict[str, float]:
    """Top-1 hit, soft IoU, in/out contrast and centroid error over boxed tokens."""

    if not bool(has_box.any()):
        return {}
    pred = predicted[0][has_box]  # [K, N]
    gt = ground_truth[0][has_box]  # [K, N]
    inside = gt > 0.0
    n_cells = pred.shape[1]

    top1 = pred.argmax(dim=-1)
    hits = inside.gather(1, top1.unsqueeze(1)).squeeze(1).float().mean()

    inter = torch.minimum(pred, gt).sum(dim=-1)
    union = torch.maximum(pred, gt).sum(dim=-1).clamp_min(1e-9)
    iou = (inter / union).mean()

    inside_mean = (pred * inside).sum(dim=-1) / inside.sum(dim=-1).clamp_min(1)
    outside_mean = (pred * ~inside).sum(dim=-1) / (~inside).sum(dim=-1).clamp_min(1)

    # Centroid distance in grid units, converting the flat cell index to (row, col).
    side = int(round(n_cells**0.5))
    coords = torch.stack(
        [torch.arange(n_cells, device=pred.device) // side, torch.arange(n_cells, device=pred.device) % side],
        dim=-1,
    ).to(pred.dtype)
    pred_centroid = (pred.unsqueeze(-1) * coords).sum(dim=1) / pred.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    gt_centroid = (gt.unsqueeze(-1) * coords).sum(dim=1) / gt.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    centroid = (pred_centroid - gt_centroid).norm(dim=-1).mean()

    box_share = inside.float().mean()
    return {
        "boxed_tokens": int(has_box.sum().item()),
        "cells": n_cells,
        "top1_hit_rate": float(hits.item()),
        "top1_chance": float(box_share.item()),
        "soft_iou": float(iou.item()),
        "in_box_mask_mean": float(inside_mean.mean().item()),
        "out_box_mask_mean": float(outside_mean.mean().item()),
        "centroid_distance_cells": float(centroid.item()),
    }


class _AttentionProbe:
    """Capture the real attention weights and the mask handed to attention."""

    def __init__(
        self, runtime: Any, layers: list[Any], watch_layers: list[int], split_layer: int
    ) -> None:
        self.handles: list[Any] = []
        self.mass: dict[int, dict[str, float]] = {}
        self.received_mask: torch.Tensor | None = None
        self._watch = set(watch_layers)
        self._split = split_layer
        self._layers = layers
        # The head runs in the split layer's *pre*-hook, so by the time any biased
        # layer's attention sees the mask, ``runtime.last_mask`` is already the
        # current page's prediction.  Reading it live avoids handing the probe a
        # tensor captured after the forward has finished.
        self._runtime = runtime

    def attach(self) -> None:
        for index in self._watch:
            self.handles.append(
                self._layers[index].self_attn.register_forward_hook(self._attn_hook(index))
            )
        # The mask the first biased layer actually hands to attention.
        self.handles.append(
            self._layers[self._split].self_attn.register_forward_pre_hook(
                self._mask_hook, with_kwargs=True
            )
        )

    def _mask_hook(self, module: Any, args: Any, kwargs: dict) -> None:
        mask = kwargs.get("attention_mask")
        if mask is not None:
            self.received_mask = mask.detach().clone()

    def _attn_hook(self, index: int):
        def hook(module: Any, args: Any, output: Any) -> None:
            if not isinstance(output, tuple) or len(output) < 2 or output[1] is None:
                return None
            weights = output[1].detach()  # [B, H, q_len, kv_len]
            if weights.ndim != 4:
                return None
            image_positions = self._runtime.image_positions
            query_positions = self._runtime.query_positions
            mask = self._runtime.last_mask
            if image_positions is None or query_positions is None or mask is None:
                return None
            mask = mask.detach()
            # Target-token query rows only; the prompt rows have no mask target.
            rows = weights[0][:, query_positions, :]  # [H, T, kv_len]
            image_mass = rows[:, :, image_positions].sum(dim=-1)  # [H, T]
            self.mass.setdefault(index, {})
            self.mass[index]["image_mass_mean"] = float(image_mass.mean().item())
            self.mass[index]["image_mass_max"] = float(image_mass.max().item())

            # Attention placed on the cells the mask actually points at, against
            # the share of cells those k entries represent.
            attention_on_cells = rows[:, :, image_positions]  # [H, T, N]
            _, top = mask[0].topk(k=min(8, mask.shape[-1]), dim=-1)  # [T, k]
            gathered = attention_on_cells.gather(
                2, top.unsqueeze(0).expand(attention_on_cells.shape[0], -1, -1)
            )
            self.mass[index]["topk_mask_mass"] = float(gathered.sum(dim=-1).mean().item())
            self.mass[index]["topk_mask_chance"] = float(top.shape[-1] / attention_on_cells.shape[-1])
            self.mass[index]["rows"] = int(attention_on_cells.shape[1])
            del weights
            return None

        return hook

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="diagnose where the decoder mask bias lands")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--attention-layers", default="", help="comma list; default split_layer and the last two")
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False, local_files_only=True)
    size = dict(processor.image_processor.size)
    size["longest_edge"] = args.max_pixels
    processor.image_processor.size = size

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    model.eval()

    eos_ids = _eos_ids(model, processor)
    image_token_id = _image_token_id(model)
    spatial_merge_size = int(model.model.visual.spatial_merge_size)

    inject_decoder_lora(model, rank=8, alpha=8.0)
    lora_state = load_lora_state(checkpoint_dir)
    if lora_state is not None:
        load_lora_state_dict(model, lora_state)

    config = load_config(checkpoint_dir)
    enable_eager_backend(model)
    runtime = install_decoder_mask_router(model, config, image_token_id, spatial_merge_size)
    restore_router(model, runtime, checkpoint_dir)
    runtime.router.eval()
    runtime.set_noise(0.0, 0.0)

    text_model = model.model.language_model
    layers = list(text_model.layers)
    split_layer = int(config.split_layer)
    if args.attention_layers.strip():
        watch = [int(part) for part in args.attention_layers.split(",") if part.strip()]
    else:
        watch = sorted({split_layer, max(0, len(layers) - 2), len(layers) - 1})

    records = load_records(args.manifest)[: args.pages]
    beta = float(config.bias_max)

    reports: list[dict[str, Any]] = []
    for record in records:
        inputs = prepare_training_inputs(processor, record, device, eos_ids)
        prompt_length = _prompt_length(inputs)
        target_ids = inputs["input_ids"][0, prompt_length:]
        xywh, _ = _normalized_grid_xywh(inputs["image_grid_thw"], spatial_merge_size)
        targets = build_mask_targets(processor.tokenizer, record, target_ids, eos_ids, xywh)
        runtime.set_page(inputs["image_grid_thw"], inputs["input_ids"], prompt_length, targets)

        report: dict[str, Any] = {
            "page_id": record["page_id"],
            "prompt_length": prompt_length,
            "target_tokens": int(target_ids.numel()),
            "alignment_report": targets.alignment_report,
            "beta": beta,
            "biased_layers": len(layers) - split_layer,
            "attention_layers": watch,
        }

        passes: dict[str, dict[str, Any]] = {}
        for label, strength in (("bias_on", beta), ("bias_off", 0.0)):
            runtime.set_bias_strength(strength)
            probe = _AttentionProbe(runtime, layers, watch, split_layer)
            probe.attach()
            with torch.no_grad():
                model(**inputs)
            probe.detach()
            passes[label] = {
                "bias_tensor": runtime._bias.detach().clone() if runtime._bias is not None else None,
                "mask": runtime.last_mask.detach().clone(),
                "stop": runtime.last_stop.detach().clone(),
                "received_mask": probe.received_mask,
                "mass": probe.mass,
            }

        on, off = passes["bias_on"], passes["bias_off"]
        bias_tensor = on["bias_tensor"]
        observed = (on["received_mask"] - off["received_mask"]) if (
            on["received_mask"] is not None and off["received_mask"] is not None
        ) else None

        # --- 1. mechanical -----------------------------------------------------
        predicted_bias = bias_tensor
        mechanical: dict[str, Any] = {
            "bias_nonzero": int((bias_tensor != 0).sum().item()) if bias_tensor is not None else 0,
            "bias_abs_max": float(bias_tensor.abs().max().item()) if bias_tensor is not None else 0.0,
        }
        if observed is not None and predicted_bias is not None:
            difference = (observed - predicted_bias)
            mechanical["received_minus_expected_abs_max"] = float(difference.abs().max().item())
            mechanical["received_nonzero"] = int((observed != 0).sum().item())
            mechanical["received_matches_bias"] = bool(float(difference.abs().max().item()) < 1e-3)
        report["mechanical"] = mechanical

        # --- 2. alignment ------------------------------------------------------
        ground_truth = targets.mask
        has_box = targets.spatial_valid[0] & (ground_truth[0].sum(dim=-1) > 0)
        report["alignment_bias_on"] = _alignment_stats(on["mask"], ground_truth, has_box)
        report["alignment_bias_off"] = _alignment_stats(off["mask"], ground_truth, has_box)
        report["stop_head_mean"] = float(on["stop"].mean().item())
        report["stop_head_target_mean"] = float(targets.stop_target.mean().item())

        # --- 3. real attention mass -------------------------------------------
        mass: dict[str, Any] = {}
        for index in watch:
            on_stats = on["mass"].get(index, {})
            off_stats = off["mass"].get(index, {})
            mass[str(index)] = {
                "image_mass_bias_on": on_stats.get("image_mass_mean"),
                "image_mass_bias_off": off_stats.get("image_mass_mean"),
                "image_mass_delta": (
                    (on_stats.get("image_mass_mean") or 0.0) - (off_stats.get("image_mass_mean") or 0.0)
                    if on_stats and off_stats
                    else None
                ),
                "topk_mask_mass_bias_on": on_stats.get("topk_mask_mass"),
                "topk_mask_mass_bias_off": off_stats.get("topk_mask_mass"),
                "topk_mask_chance": on_stats.get("topk_mask_chance"),
            }
        report["attention_mass"] = mass
        reports.append(report)
        runtime.clear_page()
        print(json.dumps({k: report[k] for k in ("page_id", "mechanical", "alignment_bias_on")}, ensure_ascii=False))

    payload = {
        "status": "complete",
        "checkpoint_dir": str(checkpoint_dir),
        "manifest": str(args.manifest),
        "split_layer": split_layer,
        "bias_max": beta,
        "pages": reports,
    }
    (output_dir / "bias_diagnosis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "bias_diagnosis_complete", "output": str(output_dir / "bias_diagnosis.json")}))


if __name__ == "__main__":
    main()
