"""Training CLI for the learned decoder mask head (plan section 7, file 4).

This is an *independent* entry point: it loads the native GLM-OCR model and, for
``--routing-mode learned``, installs the mask head; it never installs the layout
adapter and never computes ``layout_targets``.  The native ``--routing-mode none``
path is the B0 baseline (same LoRA budget, no head).

The arms of the Gate C screen are expressed as flag compositions:

* B0  ``--routing-mode none``
* B1  ``--routing-mode learned --router-bias-max 0``   (mask loss on, bias never applied)
* B2  ``--routing-mode learned`` + ``--router-use-prev-mask`` off
* B3  ``--routing-mode learned`` (defaults: prev-mask + noise channels)
* B5  ``--router-scheduled-sampling > 0`` (not implemented in the first cut; refused)

Only the head and the decoder LoRA train; the vision tower and the frozen decoder
backbone stay frozen.  ``beta`` ramps from 0 to ``--router-bias-max`` over
``--router-bias-warmup-steps`` updates, and the two noise channels anneal to 0
over ``--router-noise-warmup-steps`` updates.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .data import load_records, prepare_inference_inputs, prepare_training_inputs
from .decoder_mask_checkpoint import (
    restore_router,
    save_decoder_mask_checkpoint,
)
from .decoder_mask_model import enable_eager_backend, install_decoder_mask_router
from .decoder_mask_router import DecoderMaskConfig, _normalized_grid_xywh
from .lora import inject_decoder_lora, iter_lora_parameters, lora_state_dict
from .mask_targets import build_mask_targets
from .metrics import aggregate_ocr_metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train the learned decoder mask head")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--protocol-file", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--processor-mode", choices=("fast", "slow"), default="slow")
    # LoRA
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    # routing
    parser.add_argument("--routing-mode", choices=("none", "learned"), default="none")
    parser.add_argument("--router-split-layer", type=int, default=8)
    parser.add_argument("--router-dim", type=int, default=256)
    parser.add_argument("--router-bias-max", type=float, default=2.0)
    parser.add_argument("--router-bias-warmup-steps", type=int, default=100)
    parser.add_argument("--router-mask-loss-weight", type=float, default=0.2)
    parser.add_argument("--router-stop-loss-weight", type=float, default=0.05)
    parser.add_argument("--router-detach-every", type=int, default=64)
    parser.add_argument("--router-mask-feedback-noise", type=float, default=0.15)
    parser.add_argument("--router-input-noise", type=float, default=0.05)
    parser.add_argument("--router-noise-warmup-steps", type=int, default=200)
    parser.add_argument("--router-scheduled-sampling", type=float, default=0.0)
    parser.add_argument("--router-use-prev-mask", action="store_true", default=True)
    parser.add_argument("--router-no-prev-mask", action="store_true")
    # optimization
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--validation-every", type=int, default=256)
    parser.add_argument("--validation-steps", nargs="*", type=int, default=[256, 512, 1024])
    # generation / eval
    parser.add_argument("--max-eval-new-tokens", type=int, default=512)
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _eos_ids(model: Any, processor: Any) -> set[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    ids: set[int] = set()
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        ids = set()
    elif isinstance(value, (list, tuple, set)):
        ids = {int(item) for item in value}
    else:
        ids = {int(value)}
    gen = getattr(model, "generation_config", None)
    gvalue = getattr(gen, "eos_token_id", None)
    if gvalue is not None:
        if isinstance(gvalue, (list, tuple, set)):
            ids.update(int(item) for item in gvalue)
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


def _balanced_mask_bce(mask: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Balanced BCE on the soft mask (plan 5): non-empty and empty groups averaged apart."""

    eps = 1e-7
    has_content = target.sum(dim=-1) > 0
    non_empty = valid & has_content
    empty = valid & ~has_content
    log_m = torch.log(mask.clamp_min(eps))
    log_1m = torch.log1p(-mask.clamp_max(1.0 - eps))
    ne_loss = mask.new_zeros(())
    if non_empty.any():
        m = mask[non_empty]
        g = target[non_empty]
        pos = -(g * log_m[non_empty]).sum() / g.sum().clamp_min(eps)
        neg = -((1.0 - g) * log_1m[non_empty]).sum() / (1.0 - g).sum().clamp_min(eps)
        ne_loss = 0.5 * pos + 0.5 * neg
    empty_loss = mask.new_zeros(())
    if empty.any():
        empty_loss = -log_1m[empty].mean()
    return ne_loss + empty_loss


def _balanced_stop_bce(stop_prob: torch.Tensor, stop_target: torch.Tensor) -> torch.Tensor:
    eps = 1e-7
    stop_prob = stop_prob.clamp(eps, 1.0 - eps)
    log_e = torch.log(stop_prob)
    log_1e = torch.log1p(-stop_prob)
    pos = stop_target > 0.5
    pos_loss = -log_e[pos].mean() if pos.any() else stop_prob.new_zeros(())
    neg_loss = -log_1e[~pos].mean() if (~pos).any() else stop_prob.new_zeros(())
    return pos_loss + neg_loss


def _beta_at(step: int, args: argparse.Namespace) -> float:
    if args.routing_mode != "learned":
        return 0.0
    if args.router_bias_warmup_steps <= 0:
        return float(args.router_bias_max)
    frac = min(1.0, step / args.router_bias_warmup_steps)
    return float(args.router_bias_max) * frac


def _noise_at(step: int, initial: float, args: argparse.Namespace) -> float:
    if args.router_noise_warmup_steps <= 0:
        return 0.0
    return initial * max(0.0, 1.0 - step / args.router_noise_warmup_steps)


def _load(processor: Any, record: dict[str, Any], device: torch.device, eos_ids: set[int]) -> dict[str, Any]:
    return prepare_training_inputs(processor, record, device, eos_ids)


def _prompt_length(inputs: dict[str, Any]) -> int:
    labels = inputs["labels"]
    return int((labels == -100).to(torch.int32).cumprod(dim=1).sum().item())


def _build_targets(processor: Any, record: dict[str, Any], inputs: dict[str, Any], spatial_merge_size: int, eos_ids: set[int]):
    prompt_length = _prompt_length(inputs)
    target_ids = inputs["input_ids"][0, prompt_length:]
    xywh, _ = _normalized_grid_xywh(inputs["image_grid_thw"], spatial_merge_size)
    return build_mask_targets(processor.tokenizer, record, target_ids, eos_ids, xywh)


def _decode_tokens(tokenizer: Any, tokens: torch.Tensor, eos_ids: set[int]) -> str:
    ids = tokens.tolist()
    first_eos = next((i for i, t in enumerate(ids) if int(t) in eos_ids), None)
    if first_eos is not None:
        ids = ids[: first_eos + 1]
    return tokenizer.decode(ids, skip_special_tokens=True)


def _train_character_counts(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record["page_text"])
    return counts


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    _set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    eos_ids = _eos_ids(model, processor)
    if not eos_ids:
        raise RuntimeError("could not resolve EOS token ids")
    spatial_merge_size = int(model.model.visual.spatial_merge_size)
    image_token_id = _image_token_id(model)

    inject_decoder_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)

    runtime = None
    config = None
    use_prev_mask = args.router_use_prev_mask and not args.router_no_prev_mask
    if args.routing_mode == "learned":
        if args.router_scheduled_sampling > 0.0:
            raise NotImplementedError(
                "token-level scheduled sampling (B5) is not implemented in the first cut"
            )
        config = DecoderMaskConfig(
            router_dim=args.router_dim,
            split_layer=args.router_split_layer,
            bias_max=args.router_bias_max,
            mask_feedback_noise=args.router_mask_feedback_noise,
            input_noise=args.router_input_noise,
            detach_every=args.router_detach_every,
            use_prev_mask=use_prev_mask,
        )
        enable_eager_backend(model)
        runtime = install_decoder_mask_router(model, config, image_token_id, spatial_merge_size)
    model.config.use_cache = False

    train_records = load_records(Path(args.train_manifest))
    validation_records = load_records(Path(args.validation_manifest))
    train_counts = _train_character_counts(train_records)

    # Optimizer over LoRA + (if installed) router parameters.
    parameter_groups = [{"params": list(iter_lora_parameters(model))}]
    if runtime is not None:
        parameter_groups.append({"params": list(runtime.router.parameters())})
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.learning_rate, weight_decay=args.weight_decay)

    trainable_report = {
        "decoder_lora_parameters": sum(p.numel() for p in iter_lora_parameters(model)),
        "router_parameters": runtime.router.trainable_parameter_count() if runtime is not None else 0,
    }
    with open(output_dir / "trainable_report.json", "w", encoding="utf-8") as handle:
        json.dump(trainable_report, handle, indent=2)

    model.train()
    step = 0
    cursor = 0
    log_lines: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    start = time.time()
    random.shuffle(train_records)

    while step < args.max_steps:
        record = train_records[cursor % len(train_records)]
        cursor += 1
        inputs = _load(processor, record, device, eos_ids)
        prompt_length = _prompt_length(inputs)
        beta = _beta_at(step, args)
        mask_targets = None
        if runtime is not None:
            mask_targets = _build_targets(processor, record, inputs, spatial_merge_size, eos_ids)
            runtime.set_page(inputs["image_grid_thw"], inputs["input_ids"], prompt_length, mask_targets)
            runtime.set_bias_strength(beta)
            runtime.set_noise(
                _noise_at(step, config.mask_feedback_noise, args),
                _noise_at(step, config.input_noise, args),
            )
        outputs = model(**inputs)
        loss = outputs.loss
        if runtime is not None and mask_targets is not None:
            mask_loss = _balanced_mask_bce(
                runtime.last_mask, mask_targets.mask, mask_targets.spatial_valid
            )
            stop_loss = _balanced_stop_bce(
                runtime.last_stop.squeeze(-1), mask_targets.stop_target
            )
            loss = loss + args.router_mask_loss_weight * mask_loss + args.router_stop_loss_weight * stop_loss
        else:
            mask_loss = torch.zeros((), device=device)
            stop_loss = torch.zeros((), device=device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite total loss at step {step}")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        log_lines.append(
            {
                "step": step,
                "loss": float(loss.detach().item()),
                "mask_loss": float(mask_loss.detach().item()),
                "stop_loss": float(stop_loss.detach().item()),
                "beta": beta,
            }
        )
        if step % args.checkpoint_every == 0 or step in args.validation_steps or step == args.max_steps:
            # The B0 baseline (routing-mode none) still writes a LoRA-only
            # checkpoint so the locked test can score it symmetrically.
            save_decoder_mask_checkpoint(
                output_dir / f"step-{step}",
                config=config,
                router_state=runtime.router.state_dict() if runtime is not None else None,
                lora_state=lora_state_dict(model),
                training_state={"optimizer": optimizer.state_dict(), "step": step},
                fingerprint={
                    "seed": args.seed,
                    "routing_mode": args.routing_mode,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                },
            )
        if step in args.validation_steps or step == args.max_steps:
            val = run_validation(args, model, processor, runtime, validation_records, device, eos_ids, spatial_merge_size, image_token_id, train_counts, step)
            summary[f"step-{step}"] = val
            with open(output_dir / "validation.json", "w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            (output_dir / "train_log.jsonl").write_text(
                "\n".join(json.dumps(line, ensure_ascii=False) for line in log_lines) + "\n",
                encoding="utf-8",
            )

    elapsed = time.time() - start
    final = {
        "steps": step,
        "elapsed_seconds": elapsed,
        "trainable_report": trainable_report,
        "validation": summary,
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(final, handle, ensure_ascii=False, indent=2)
    return final


@torch.no_grad()
def run_validation(
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    runtime: Any,
    records: list[dict[str, Any]],
    device: torch.device,
    eos_ids: set[int],
    spatial_merge_size: int,
    image_token_id: int,
    train_counts: Counter[str],
    step: int,
) -> dict[str, Any]:
    model.eval()
    if runtime is not None:
        runtime.router.eval()
        runtime.set_bias_strength(args.router_bias_max)
        runtime.set_noise(0.0, 0.0)
    pairs: list[tuple[str, str]] = []
    mask_report: list[dict[str, Any]] = []
    model.config.use_cache = True
    for record in records:
        inputs = prepare_inference_inputs(processor, record, device)
        prompt_length = inputs["input_ids"].shape[1]
        if runtime is not None:
            runtime.set_page(inputs["image_grid_thw"], inputs["input_ids"], prompt_length, None)
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_eval_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=sorted(eos_ids),
        )
        tokens = generated[0, prompt_length:]
        prediction = _decode_tokens(processor.tokenizer, tokens, eos_ids)
        pairs.append((record["page_text"], prediction))
        if runtime is not None and runtime.last_mask is not None:
            mask_report.append(
                {
                    "page_id": record["page_id"],
                    "mean_mask": float(runtime.last_mask.mean().item()),
                    "mean_stop": float(runtime.last_stop.mean().item()),
                }
            )
        if runtime is not None:
            runtime.clear_page()
    model.config.use_cache = False
    model.train()
    if runtime is not None:
        runtime.router.train()
    metrics = aggregate_ocr_metrics(pairs, train_counts)
    metrics["mask_pages"] = len(mask_report)
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
