#!/usr/bin/env python3
"""Independent prompt-only evaluation of a decoder-mask checkpoint.

This is deliberately separate from the training loop so a checkpoint can be
scored on a manifest the trainer never opened (the selection-locked test) and so
a fixed ``--bias-max 0`` intervention can be applied to a *learned* checkpoint
without retraining it (plan section 8, Gate C).

Two routing modes are supported and auto-detected from the checkpoint contents:

* ``learned`` -- the checkpoint carries ``decoder_mask_config.json`` and
  ``decoder_mask.safetensors``; the head is reinstalled, restored, and the text
  decoder is switched to the eager backend so the additive mask is applied at
  inference time.
* ``none``    -- a LoRA-only baseline checkpoint (B0); only the decoder LoRA is
  reloaded and no mask head is installed.

The head is *not* read from the manifest: generation is prompt-only, matching the
deployment constraint that bbox/reading-order/token-region are never inference
inputs.  The mask diagnostics reported here are coarse per-page summaries of the
last emitted token's predicted mask; the full aligned localization report (first
character hit, cross-line hit, soft IoU) needs the ground-truth character channel
and is deferred to a follow-up.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import torch

from layout_ocr.data import load_records, prepare_inference_inputs
from layout_ocr.decoder_mask_checkpoint import (
    load_config,
    load_fingerprint,
    load_lora_state,
    restore_router,
)
from layout_ocr.decoder_mask_model import enable_eager_backend, install_decoder_mask_router
from layout_ocr.lora import inject_decoder_lora, load_lora_state_dict
from layout_ocr.metrics import aggregate_ocr_metrics


def _eos_ids(model: object, processor: object) -> set[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    ids: set[int] = set()
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


def _image_token_id(model: object) -> int:
    value = getattr(model.config, "image_token_id", None)
    if value is None:
        value = getattr(getattr(model.config, "text_config", None), "image_token_id", None)
    if value is None:
        raise RuntimeError("could not resolve the model image token id")
    return int(value)


def _decode_tokens(tokenizer: object, tokens: torch.Tensor, eos_ids: set[int]) -> str:
    ids = tokens.tolist()
    first_eos = next((i for i, t in enumerate(ids) if int(t) in eos_ids), None)
    if first_eos is not None:
        ids = ids[: first_eos + 1]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _train_character_counts(records: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record["page_text"])
    return counts


def _mask_diagnostics(runtime: object) -> dict[str, float]:
    """Coarse per-page summary of the last emitted token's predicted mask."""

    mask = runtime.last_mask
    stop = runtime.last_stop
    report: dict[str, float] = {}
    if mask is not None:
        flat = mask.reshape(-1)
        report["mask_mean"] = float(flat.mean().item())
        report["mask_max"] = float(flat.max().item())
        report["mask_sparsity"] = float((flat < 0.05).float().mean().item())
        clamped = flat.clamp_min(1e-9)
        report["mask_entropy"] = float(-(clamped * clamped.log()).sum().item())
    else:
        report["mask_mean"] = None
        report["mask_max"] = None
        report["mask_sparsity"] = None
        report["mask_entropy"] = None
    report["stop_mean"] = float(stop.mean().item()) if stop is not None else None
    return report


def _detect_routing_mode(checkpoint_dir: Path) -> str:
    if (checkpoint_dir / "decoder_mask_config.json").is_file():
        return "learned"
    return "none"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="score a decoder-mask checkpoint prompt-only")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, help="train manifest for character counts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--routing-mode", choices=("none", "learned"))
    parser.add_argument(
        "--bias-max",
        type=float,
        help="override the head bias strength; pass 0 for the bias-off intervention",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--processor-mode", choices=("fast", "slow"), default="slow")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)

    fingerprint = load_fingerprint(checkpoint_dir)
    routing_mode = args.routing_mode or fingerprint.get("routing_mode") or _detect_routing_mode(checkpoint_dir)
    if routing_mode not in {"none", "learned"}:
        raise ValueError(f"unsupported routing mode: {routing_mode}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(
        args.model_path, use_fast=args.processor_mode == "fast", local_files_only=True
    )
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
    if not eos_ids:
        raise RuntimeError("could not resolve EOS token ids")
    image_token_id = _image_token_id(model)
    spatial_merge_size = int(model.model.visual.spatial_merge_size)

    lora_rank = int(fingerprint.get("lora_rank", args.lora_rank))
    lora_alpha = float(fingerprint.get("lora_alpha", args.lora_alpha))
    inject_decoder_lora(model, rank=lora_rank, alpha=lora_alpha)
    lora_state = load_lora_state(checkpoint_dir)
    if lora_state is None:
        raise FileNotFoundError(checkpoint_dir / "lora.safetensors")
    load_lora_state_dict(model, lora_state)

    runtime = None
    bias_max = None
    if routing_mode == "learned":
        config = load_config(checkpoint_dir)
        enable_eager_backend(model)
        runtime = install_decoder_mask_router(model, config, image_token_id, spatial_merge_size)
        restore_router(model, runtime, checkpoint_dir)
        runtime.router.eval()
        runtime.set_noise(0.0, 0.0)
        bias_max = args.bias_max if args.bias_max is not None else config.bias_max
        runtime.set_bias_strength(bias_max)

    records = load_records(args.manifest)
    train_counts = (
        _train_character_counts(load_records(args.train_manifest))
        if args.train_manifest
        else Counter()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, str]] = []
    predictions: list[dict] = []
    total_tokens = 0
    limit_hits = 0
    mask_reports: list[dict] = []
    start = time.time()
    for record in records:
        inputs = prepare_inference_inputs(processor, record, device)
        prompt_length = int(inputs["input_ids"].shape[1])
        if runtime is not None:
            runtime.set_page(inputs["image_grid_thw"], inputs["input_ids"], prompt_length, None)
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=sorted(eos_ids),
        )
        tokens = generated[0, prompt_length:]
        total_tokens += int(tokens.numel())
        hit_limit = not any(int(t) in eos_ids for t in tokens.tolist())
        limit_hits += int(hit_limit)
        prediction = _decode_tokens(processor.tokenizer, tokens, eos_ids)
        pairs.append((record["page_text"], prediction))
        entry = {
            "page_id": record["page_id"],
            "reference": record["page_text"],
            "prediction": prediction,
            "generated_tokens": int(tokens.numel()),
            "generation_limit_hit": hit_limit,
        }
        if runtime is not None:
            entry["mask"] = _mask_diagnostics(runtime)
            mask_reports.append({"page_id": record["page_id"], **entry["mask"]})
        predictions.append(entry)
        if runtime is not None:
            runtime.clear_page()
    elapsed = time.time() - start

    metrics = aggregate_ocr_metrics(pairs, train_counts)
    cuda_peak_gb = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        cuda_peak_gb = float(torch.cuda.max_memory_allocated(device)) / (1024**3)
    summary = {
        "status": "complete",
        "routing_mode": routing_mode,
        "checkpoint_dir": str(checkpoint_dir),
        "model_path": str(args.model_path),
        "manifest": str(args.manifest),
        "pages": len(records),
        "bias_max": bias_max,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "metrics": metrics,
        "resource": {
            "elapsed_seconds": elapsed,
            "generated_tokens": total_tokens,
            "tokens_per_second": total_tokens / elapsed if elapsed > 0 else None,
            "generation_limit_hits": limit_hits,
            "generation_limit_hit_rate": limit_hits / max(1, len(records)),
            "cuda_peak_memory_gb": cuda_peak_gb,
        },
        "mask_pages": len(mask_reports),
        "test_used_for_selection": False,
    }
    (output_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in predictions) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
