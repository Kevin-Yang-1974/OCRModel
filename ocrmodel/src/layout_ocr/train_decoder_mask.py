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
from .decoder_mask_router import (
    DecoderMaskConfig,
    _normalized_grid_xywh,
    fine_grid_xywh,
    pool_to_merged,
)
from .lora import inject_decoder_lora, iter_lora_parameters, lora_state_dict
from .mask_losses import balanced_stop_bce, mask_and_dice_loss
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
    # visual source / grid
    parser.add_argument("--router-visual-source", choices=("merged", "fine"), default="merged")
    parser.add_argument(
        "--router-pool-mode",
        choices=("max", "mean"),
        default="max",
        help="how the fine mask is reduced onto the merged grid the bias lives on",
    )
    parser.add_argument("--router-head", choices=("mlp", "vae"), default="mlp")
    parser.add_argument("--router-target-mode", choices=("token", "window"), default="token")
    parser.add_argument("--router-window-size", nargs=2, type=int, default=[3, 5], metavar=("MIN", "MAX"))
    parser.add_argument("--router-window-line-source", choices=("annotation", "geometry", "auto"), default="auto")
    parser.add_argument("--router-dice-weight", type=float, default=0.0)
    parser.add_argument("--router-mask-bce", choices=("balanced", "plain"), default="balanced")
    parser.add_argument("--router-vae-latent-channels", type=int, default=4)
    parser.add_argument("--router-vae-latent-size", type=int, default=16)
    parser.add_argument("--router-vae-kl-weight", type=float, default=1.0)
    parser.add_argument("--router-vae-kl-warmup-steps", type=int, default=200)
    parser.add_argument("--router-vae-kl-free-bits", type=float, default=0.05)
    parser.add_argument("--router-vae-inference", choices=("mean", "sample"), default="mean")
    parser.add_argument("--probe", action="store_true", help="print the seam/grid probe and exit")
    # Head-only regime: freeze everything except the mask head and train it on
    # mask supervision alone.  No LoRA is injected (nothing would train it), the
    # backbone builds no autograd graph, and the language-model CE is skipped --
    # which is what lets a 4M-resolution run fit on one 40GB card.
    parser.add_argument("--head-only", action="store_true")
    # Page sharding: run N processes over disjoint page shards, one per card, and
    # average the resulting heads afterwards.
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    # optimization
    # The LoRA rate is the corrected project value: 1e-4 is 10x the established
    # lr1e5 recipe and produced severe over-generation in the first screen.
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--router-learning-rate",
        type=float,
        default=None,
        help="head LR; defaults to --learning-rate. The head trains from scratch, so "
        "it usually needs its own (higher) rate than a LoRA that must stay near its base.",
    )
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


def _beta_at(step: int, args: argparse.Namespace) -> float:
    if args.routing_mode != "learned":
        return 0.0
    if args.router_bias_warmup_steps <= 0:
        return float(args.router_bias_max)
    frac = min(1.0, step / args.router_bias_warmup_steps)
    return float(args.router_bias_max) * frac


def _kl_weight_at(step: int, args: argparse.Namespace) -> float:
    """Linear KL warmup; the anti-collapse pair with the decoder's small init."""

    if args.router_head != "vae":
        return 0.0
    if args.router_vae_kl_warmup_steps <= 0:
        return float(args.router_vae_kl_weight)
    return float(args.router_vae_kl_weight) * min(1.0, step / args.router_vae_kl_warmup_steps)


def _pooled_peak(runtime: Any) -> float:
    """Mean per-token peak of the mask *after* it is pooled onto the merged grid.

    The bias the decoder actually receives is ``beta * pooled(M)``, so this is the
    number that says whether ``beta`` still means what it did on the merged grid.
    """

    mask = runtime.last_mask
    if mask is None:
        return 0.0
    pooled = mask
    if runtime.config.visual_source == "fine":
        pooled = pool_to_merged(
            mask, runtime.merged_shape, runtime.spatial_merge_size, runtime.config.pool_mode
        )
    return float(pooled.detach().amax(dim=-1).mean().item())


def _noise_at(step: int, initial: float, args: argparse.Namespace) -> float:
    if args.router_noise_warmup_steps <= 0:
        return 0.0
    return initial * max(0.0, 1.0 - step / args.router_noise_warmup_steps)


def _load(processor: Any, record: dict[str, Any], device: torch.device, eos_ids: set[int]) -> dict[str, Any]:
    return prepare_training_inputs(processor, record, device, eos_ids)


def _prompt_length(inputs: dict[str, Any]) -> int:
    labels = inputs["labels"]
    return int((labels == -100).to(torch.int32).cumprod(dim=1).sum().item())


def _router_xywh(grid_thw: torch.Tensor, spatial_merge_size: int, visual_source: str):
    """The head's own grid: one cell per patch in fine mode, else the merged grid."""

    if visual_source == "fine":
        return fine_grid_xywh(grid_thw, 1)
    return _normalized_grid_xywh(grid_thw, spatial_merge_size)


def _build_targets(
    processor: Any,
    record: dict[str, Any],
    inputs: dict[str, Any],
    spatial_merge_size: int,
    eos_ids: set[int],
    args: argparse.Namespace,
):
    prompt_length = _prompt_length(inputs)
    target_ids = inputs["input_ids"][0, prompt_length:]
    xywh, _ = _router_xywh(inputs["image_grid_thw"], spatial_merge_size, args.router_visual_source)
    return build_mask_targets(
        processor.tokenizer,
        record,
        target_ids,
        eos_ids,
        xywh,
        target_mode=args.router_target_mode,
        window_min=args.router_window_size[0],
        window_max=args.router_window_size[1],
        line_source=args.router_window_line_source,
    )


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

    # Head-only: inject no LoRA at all.  A frozen LoRA that never receives a
    # gradient would only add parameters to the checkpoint and a backward pass
    # nothing needs; leaving the backbone untouched keeps it graph-free.
    if not args.head_only:
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
            visual_source=args.router_visual_source,
            pool_mode=args.router_pool_mode,
            head=args.router_head,
            target_mode=args.router_target_mode,
            window_min=args.router_window_size[0],
            window_max=args.router_window_size[1],
            line_source=args.router_window_line_source,
            dice_weight=args.router_dice_weight,
            bce_mode=args.router_mask_bce,
            vae_latent_channels=args.router_vae_latent_channels,
            vae_latent_size=args.router_vae_latent_size,
            vae_kl_weight=args.router_vae_kl_weight,
            vae_kl_warmup_steps=args.router_vae_kl_warmup_steps,
            vae_kl_free_bits=args.router_vae_kl_free_bits,
            vae_inference=args.router_vae_inference,
        )
        enable_eager_backend(model)
        runtime = install_decoder_mask_router(model, config, image_token_id, spatial_merge_size)
    model.config.use_cache = False

    train_records = load_records(Path(args.train_manifest))
    validation_records = load_records(Path(args.validation_manifest))
    if args.shard_count > 1:
        if not 0 <= args.shard_index < args.shard_count:
            raise ValueError("shard_index must be in [0, shard_count)")
        # Deterministic contiguous shards in the manifest's own order, so the
        # five processes together cover exactly the full page set with no
        # overlap and no dependence on which one starts first.
        shard = train_records[args.shard_index :: args.shard_count]
        if not shard:
            raise ValueError(
                f"shard {args.shard_index}/{args.shard_count} is empty for "
                f"{len(train_records)} pages; reduce the shard count"
            )
        train_records = shard
    train_counts = _train_character_counts(train_records)

    # Optimizer over LoRA + (if installed) router parameters.
    # The LoRA must stay near its base (a high rate causes over-generation, since
    # teacher forcing never teaches recovery from the model's own errors), while
    # the head trains from scratch and needs a rate it can actually move at.  One
    # shared rate cannot serve both, so they are separate parameter groups.
    head_lr = args.router_learning_rate if args.router_learning_rate is not None else args.learning_rate
    parameter_groups = []
    if not args.head_only:
        parameter_groups.append({"params": list(iter_lora_parameters(model)), "lr": args.learning_rate})
    if runtime is not None:
        parameter_groups.append({"params": list(runtime.router.parameters()), "lr": head_lr})
    if not parameter_groups:
        raise RuntimeError("nothing to optimize: head-only mode needs --routing-mode learned")
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.learning_rate, weight_decay=args.weight_decay)

    trainable_report = {
        "decoder_lora_parameters": sum(p.numel() for p in iter_lora_parameters(model)),
        "router_parameters": runtime.router.trainable_parameter_count() if runtime is not None else 0,
        "lora_learning_rate": args.learning_rate,
        "router_learning_rate": head_lr if runtime is not None else None,
        "visual_source": args.router_visual_source,
        # Read the *installed* config: the installer re-derives the model-dependent
        # widths, so the local one still carries the placeholder zeros.
        "visual_hidden_size": int(getattr(runtime.config, "visual_hidden_size", 0)) if runtime else None,
        "head": args.router_head,
        "target_mode": args.router_target_mode,
        "pool_mode": args.router_pool_mode,
        "dice_weight": args.router_dice_weight,
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
            mask_targets = _build_targets(processor, record, inputs, spatial_merge_size, eos_ids, args)
            runtime.set_page(inputs["image_grid_thw"], inputs["input_ids"], prompt_length, mask_targets)
            runtime.set_bias_strength(beta)
            runtime.set_noise(
                _noise_at(step, config.mask_feedback_noise, args),
                _noise_at(step, config.input_noise, args),
            )
        # Head-only skips the language-model CE entirely: the head is trained by
        # mask supervision, and the CE would only add a [1, L, vocab] logit tensor
        # (hundreds of MB at 4M resolution) that no gradient needs.
        model_inputs = dict(inputs)
        if args.head_only:
            model_inputs.pop("labels", None)
        outputs = model(**model_inputs)
        loss = outputs.loss if outputs.loss is not None else torch.zeros((), device=device)
        parts: dict[str, Any] = {}
        if runtime is not None and mask_targets is not None:
            mask_loss, parts = mask_and_dice_loss(
                runtime.last_mask,
                mask_targets.mask,
                mask_targets.spatial_valid,
                dice_weight=args.router_dice_weight,
                bce_mode=args.router_mask_bce,
            )
            stop_loss = balanced_stop_bce(
                runtime.last_stop.squeeze(-1), mask_targets.stop_target
            )
            supervision = (
                args.router_mask_loss_weight * mask_loss
                + args.router_stop_loss_weight * stop_loss
            )
            # Head-only trains on supervision alone; otherwise the CE comes along
            # and the head also sees the language-model signal through the bias.
            loss = supervision if args.head_only else loss + supervision
            if args.router_head == "vae":
                kl_weight = _kl_weight_at(step, args)
                kl = runtime.router.kl_loss()
                loss = loss + kl_weight * kl
                parts.update(runtime.router.kl_diagnostics())
                parts["kl_term"] = float(kl.detach().item())
                parts["kl_weight"] = kl_weight
            # The pooled peak is the empirical answer to "does beta stay anchored":
            # the bias is beta * pooled(M), so a peak well below 1 means the fine
            # grid weakened the intervention instead of sharpening it.
            parts["pooled_peak"] = _pooled_peak(runtime)
            if mask_targets.window_report:
                parts["window"] = mask_targets.window_report
            parts["stop_loss"] = float(stop_loss.detach().item())
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
                **parts,
            }
        )
        if step % args.checkpoint_every == 0 or step in args.validation_steps or step == args.max_steps:
            # The B0 baseline (routing-mode none) still writes a LoRA-only
            # checkpoint so the locked test can score it symmetrically.
            save_decoder_mask_checkpoint(
                output_dir / f"step-{step}",
                config=runtime.config if runtime is not None else config,
                router_state=runtime.router.state_dict() if runtime is not None else None,
                # No LoRA was injected in head-only mode, so there is nothing to
                # save -- an empty file would make the scorer try to load a LoRA
                # into a model that has none.
                lora_state=None if args.head_only else lora_state_dict(model),
                training_state={"optimizer": optimizer.state_dict(), "step": step},
                fingerprint={
                    "seed": args.seed,
                    "routing_mode": args.routing_mode,
                    "head_only": args.head_only,
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "lora_rank": args.lora_rank,
                    "lora_alpha": args.lora_alpha,
                    # The architecture knobs are recorded so an arm can be
                    # identified without loading the config, and so a mismatched
                    # scorer can be caught rather than silently scoring a
                    # different head.
                    "head": args.router_head,
                    "visual_source": args.router_visual_source,
                    "target_mode": args.router_target_mode,
                    "pool_mode": args.router_pool_mode,
                    "bias_max": args.router_bias_max,
                    "dice_weight": args.router_dice_weight,
                    "learning_rate": args.learning_rate,
                    "router_learning_rate": head_lr,
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


def run_probe(args: argparse.Namespace) -> int:
    """Print the vision seam and both grids, then exit.

    The fine-grid experiment rests entirely on the pre-merge features being four
    times the merged token count and wider than the text hidden size.  If that is
    false, four cards would be spent rediscovering it, so this is a pre-flight
    rather than a convenience.
    """

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    vision = getattr(getattr(model, "model", None), "visual", None)
    captured: dict[str, Any] = {}
    handles: list[Any] = []
    post_layernorm = getattr(vision, "post_layernorm", None)
    if post_layernorm is not None:
        def capture_output(module: Any, call_args: Any, output: Any) -> None:
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(tensor):
                captured["post_layernorm_out"] = list(tensor.shape)

        handles.append(post_layernorm.register_forward_hook(capture_output))
    for name, module in (("downsample_in", getattr(vision, "downsample", None)), ("merger_in", getattr(vision, "merger", None))):
        if module is None:
            continue

        def capture_input(module_: Any, call_args: Any, _name: str = name) -> None:
            tensor = call_args[0] if call_args else None
            if torch.is_tensor(tensor):
                captured[_name] = list(tensor.shape)

        handles.append(module.register_forward_pre_hook(capture_input))

    records = load_records(Path(args.train_manifest))
    eos_ids = _eos_ids(model, processor)
    inputs = prepare_training_inputs(processor, records[0], device, eos_ids)
    with torch.no_grad():
        model(**inputs)
    for handle in handles:
        handle.remove()

    grid = [int(value) for value in inputs["image_grid_thw"][0].tolist()]
    _, height, width = grid
    merge = int(getattr(vision, "spatial_merge_size", 0) or 0)
    vision_config = getattr(vision, "config", None)
    payload = {
        "event": "visual_probe",
        "grid_thw": grid,
        "N_fine": height * width,
        "N_merged": ((height // merge) * (width // merge)) if merge else None,
        "image_tokens_in_sequence": int((inputs["input_ids"][0] == _image_token_id(model)).sum()),
        "vision_hidden_size": int(getattr(vision_config, "hidden_size", 0) or 0),
        "vision_out_hidden_size": int(getattr(vision_config, "out_hidden_size", 0) or 0),
        "spatial_merge_size": merge,
        "captured": captured,
    }
    payload["fine_grid_available"] = bool(payload["N_merged"]) and payload["N_fine"] != payload["N_merged"]
    print(json.dumps(payload, ensure_ascii=False))
    if args.router_visual_source == "fine" and not payload["fine_grid_available"]:
        raise SystemExit("fine grid requested but N_fine == N_merged; the premise does not hold")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.probe:
        return run_probe(args)
    run_training(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
