from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from .config import LayoutLossConfig
from .data import layout_targets, load_records, prepare_inference_inputs, prepare_training_inputs
from .glm_bridge import LayoutAwarePatchMerger, install_layout_adapter
from .losses import compute_layout_losses
from .metrics import aggregate_ocr_metrics


def optional_positive_float(value: str) -> float | None:
    if value.lower() in {"none", "off"}:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive, 'none', or 'off'")
    return parsed


def learning_rate_at_step(
    step: int,
    *,
    peak_learning_rate: float,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float,
) -> float:
    """Linear warmup followed by cosine decay to an exact terminal ratio."""

    if not 0 <= step <= max_steps:
        raise ValueError("step must be between zero and max_steps")
    if not 0 <= warmup_steps < max_steps:
        raise ValueError("warmup_steps must be non-negative and smaller than max_steps")
    if peak_learning_rate <= 0 or not 0 <= min_lr_ratio <= 1:
        raise ValueError("invalid learning-rate schedule")
    if warmup_steps and step <= warmup_steps:
        return peak_learning_rate * step / warmup_steps
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_learning_rate * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def adapter_finite_report(adapter: torch.nn.Module) -> dict[str, Any]:
    non_finite = [
        name
        for name, value in adapter.state_dict().items()
        if not bool(torch.isfinite(value).all())
    ]
    return {"parameters_finite": not non_finite, "non_finite_parameters": non_finite}


def clone_module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def module_state_matches(module: torch.nn.Module, expected: dict[str, torch.Tensor]) -> bool:
    current = module.state_dict()
    return current.keys() == expected.keys() and all(
        torch.equal(current[key].detach().cpu(), expected[key]) for key in current
    )


def write_adapter_config(path: Path, bridge: LayoutAwarePatchMerger) -> None:
    write_json(path, asdict(bridge.adapter.config))


def save_adapter_checkpoint(path: Path, bridge: LayoutAwarePatchMerger, step: int) -> dict[str, Any]:
    report = {"step": step, **adapter_finite_report(bridge.adapter)}
    if not report["parameters_finite"]:
        raise FloatingPointError(
            f"non-finite adapter parameters at checkpoint {step}: "
            f"{report['non_finite_parameters']}"
        )
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in bridge.adapter.state_dict().items()
    }
    save_file(state, path / "adapter.safetensors")
    write_adapter_config(path / "adapter_config.json", bridge)
    reloaded = load_file(str(path / "adapter.safetensors"), device="cpu")
    non_finite_saved = [
        name for name, value in reloaded.items() if not bool(torch.isfinite(value).all())
    ]
    report["checkpoint_finite"] = not non_finite_saved
    report["non_finite_checkpoint_tensors"] = non_finite_saved
    write_json(path / "checkpoint_health.json", report)
    if non_finite_saved:
        raise FloatingPointError(
            f"non-finite serialized checkpoint tensors at step {step}: {non_finite_saved}"
        )
    return report


def load_adapter_checkpoint(path: Path, bridge: LayoutAwarePatchMerger) -> dict[str, Any]:
    config_path = path / "adapter_config.json"
    expected = asdict(bridge.adapter.config)
    if config_path.is_file():
        recorded = json.loads(config_path.read_text(encoding="utf-8"))
        if recorded != expected:
            raise ValueError(
                f"adapter config mismatch for {path}: recorded={recorded}, expected={expected}"
            )
    elif bridge.adapter.config.max_residual_scale is not None:
        raise ValueError(
            "legacy checkpoint has no adapter_config.json; load it with "
            "max_residual_scale=None to preserve its original gate semantics"
        )
    state = load_file(str(path / "adapter.safetensors"), device="cpu")
    non_finite = [name for name, value in state.items() if not bool(torch.isfinite(value).all())]
    if non_finite:
        raise FloatingPointError(f"non-finite checkpoint tensors in {path}: {non_finite}")
    bridge.adapter.load_state_dict(state)
    return state


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def configure_processor(processor: Any, max_pixels: int) -> None:
    size = dict(processor.image_processor.size)
    size["longest_edge"] = max_pixels
    processor.image_processor.size = size


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any, LayoutAwarePatchMerger]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)
    configure_processor(processor, args.max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    bridge = install_layout_adapter(
        model,
        args.mode,
        args.num_queries,
        max_residual_scale=args.residual_scale_cap,
    )
    model.config.use_cache = False
    return model, processor, bridge


def train(
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    bridge: LayoutAwarePatchMerger,
    records: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(bridge.adapter.parameters(), lr=args.learning_rate, weight_decay=0.01)
    loss_weights = LayoutLossConfig(
        box=1.0,
        order=0.5,
        direction=0.5,
        assignment=1.0,
        transport_entropy=0.0,
    )
    rng = random.Random(args.seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    # The GLM-OCR backbone stays in evaluation mode while only the adapter is
    # optimized.  This prevents frozen dropout/statistics from adding noise to
    # the mechanism comparison.
    model.eval()
    bridge.adapter.train()
    running: Counter[str] = Counter()
    started = time.time()
    checkpoint_steps: list[int] = []
    checkpoint_health: list[dict[str, Any]] = []
    for step in range(1, args.max_steps + 1):
        if (step - 1) % len(order) == 0 and step > 1:
            rng.shuffle(order)
        record = records[order[(step - 1) % len(order)]]
        inputs = prepare_training_inputs(processor, record, device)
        bridge.set_grid_thw(inputs["image_grid_thw"])
        learning_rate = learning_rate_at_step(
            step,
            peak_learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(**inputs)
        if outputs.loss is None or bridge.last_output is None or bridge.last_patch_positions is None:
            raise RuntimeError("GLM-OCR forward did not produce OCR loss and layout state")
        if not torch.isfinite(outputs.loss):
            raise FloatingPointError(f"non-finite OCR loss at step {step}")
        auxiliary_loss = outputs.loss.new_zeros(())
        if args.auxiliary_weight > 0.0:
            targets = layout_targets(record, bridge.last_patch_positions, args.num_queries)
            auxiliary_losses = compute_layout_losses(
                bridge.last_output, weights=loss_weights, **targets
            )
            auxiliary_loss = auxiliary_losses["loss"].float()
        if not torch.isfinite(auxiliary_loss):
            raise FloatingPointError(f"non-finite auxiliary loss at step {step}")
        loss = outputs.loss.float() + args.auxiliary_weight * auxiliary_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(bridge.adapter.parameters(), args.max_grad_norm)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        finite_report = adapter_finite_report(bridge.adapter)
        if not finite_report["parameters_finite"]:
            raise FloatingPointError(
                f"non-finite adapter parameters at step {step}: "
                f"{finite_report['non_finite_parameters']}"
            )
        raw_gate = float(bridge.adapter.content_gate.detach())
        effective_scale = float(bridge.adapter.effective_residual_scale().detach())
        metrics = {
            "ocr_loss": float(outputs.loss.detach()),
            "auxiliary_loss": float(auxiliary_loss.detach()),
            "total_loss": float(loss.detach()),
            "gradient_norm": float(gradient_norm.detach()),
            "learning_rate": learning_rate,
            "raw_content_gate": raw_gate,
            "effective_residual_scale": effective_scale,
            "parameters_finite": True,
        }
        for key in ("ocr_loss", "auxiliary_loss", "total_loss", "gradient_norm"):
            running[key] += metrics[key]
        if step == 1 or step % args.log_steps == 0 or step == args.max_steps:
            append_jsonl(
                args.output_dir / "train_metrics.jsonl",
                {"step": step, "page_id": record["page_id"], **metrics},
            )
        if step % args.validation_interval == 0 or step == args.max_steps:
            checkpoint_dir = args.output_dir / f"checkpoint-{step}"
            checkpoint_dir.mkdir(exist_ok=False)
            health = save_adapter_checkpoint(checkpoint_dir, bridge, step)
            health.update(
                {
                    "learning_rate": learning_rate,
                    "raw_content_gate": raw_gate,
                    "effective_residual_scale": effective_scale,
                }
            )
            write_json(checkpoint_dir / "checkpoint_health.json", health)
            checkpoint_health.append(health)
            checkpoint_steps.append(step)
    elapsed = time.time() - started
    state = {key: value.detach().cpu().contiguous() for key, value in bridge.adapter.state_dict().items()}
    save_file(state, args.output_dir / "adapter.safetensors")
    return {
        "steps": args.max_steps,
        "seconds": elapsed,
        "steps_per_second": args.max_steps / max(elapsed, 1e-9),
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_health": checkpoint_health,
        "final_learning_rate": learning_rate_at_step(
            args.max_steps,
            peak_learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            min_lr_ratio=args.min_lr_ratio,
        ),
        "final_raw_content_gate": float(bridge.adapter.content_gate.detach()),
        "final_effective_residual_scale": float(
            bridge.adapter.effective_residual_scale().detach()
        ),
        **{f"mean_{key}": value / args.max_steps for key, value in running.items()},
    }


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    bridge: LayoutAwarePatchMerger,
    validation_records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    device: torch.device,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    model.eval()
    bridge.adapter.eval()
    model.config.use_cache = True
    predictions_path = (output_dir or args.output_dir) / "validation_predictions.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, str]] = []
    box_errors: list[float] = []
    direction_correct = 0
    direction_total = 0
    generation_limit_hits = 0
    started = time.time()
    for record in validation_records:
        inputs = prepare_inference_inputs(processor, record, device)
        bridge.set_grid_thw(inputs["image_grid_thw"])
        prompt_length = inputs["input_ids"].shape[1]
        generated = model.generate(
            **inputs,
            # Keep generation independent of the reference text.  The target
            # is only read after generation for scoring.
            max_new_tokens=args.max_eval_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        prediction = processor.decode(generated[0, prompt_length:], skip_special_tokens=True)
        if generated.shape[1] - prompt_length >= args.max_eval_new_tokens:
            generation_limit_hits += 1
        pairs.append((record["page_text"], prediction))
        if bridge.last_output is None or bridge.last_patch_positions is None:
            raise RuntimeError("generation did not retain layout state")
        targets = layout_targets(record, bridge.last_patch_positions, args.num_queries)
        count = int(targets["query_mask"].sum())
        if count:
            error = (bridge.last_output.boxes[0, :count] - targets["target_boxes"][0, :count]).abs()
            box_errors.append(float(error.mean()))
            predicted_direction = bridge.last_output.direction_logits[0, :count].argmax(dim=-1)
            direction_correct += int(
                (predicted_direction == targets["target_directions"][0, :count]).sum()
            )
            direction_total += count
        append_jsonl(
            predictions_path,
            {"page_id": record["page_id"], "reference": record["page_text"], "prediction": prediction},
        )
    train_counts = Counter(
        character
        for record in train_records
        for character in record["page_text"]
        if not character.isspace()
    )
    metrics = aggregate_ocr_metrics(pairs, train_counts)
    metrics.update(
        {
            "layout_box_mae": sum(box_errors) / max(1, len(box_errors)),
            "layout_direction_accuracy": direction_correct / max(1, direction_total),
            "layout_direction_regions": direction_total,
            "seconds": time.time() - started,
            "generation_max_new_tokens": args.max_eval_new_tokens,
            "generation_limit_hits": generation_limit_hits,
            "generation_limit_hit_rate": generation_limit_hits / max(1, len(validation_records)),
            "raw_content_gate": float(bridge.adapter.content_gate.detach()),
            "effective_residual_scale": float(
                bridge.adapter.effective_residual_scale().detach()
            ),
            **adapter_finite_report(bridge.adapter),
            "test_used_for_selection": False,
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["content_only", "attention", "geometry", "layout_ot"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--residual-scale-cap", type=optional_positive_float, default=0.03)
    parser.add_argument("--auxiliary-weight", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--max-eval-new-tokens", type=int, default=768)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=256)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_only and args.mode != "content_only":
        raise ValueError("--eval-only is reserved for the prompt-only content_only baseline")
    if args.eval_only and args.auxiliary_weight != 0.0:
        raise ValueError("the eval-only content_only baseline requires --auxiliary-weight 0")
    if not args.eval_only and args.auxiliary_weight != 0.2:
        raise ValueError("the stabilization confirmation fixes --auxiliary-weight at 0.2")
    if not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError("--warmup-steps must be non-negative and smaller than --max-steps")
    if not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    metadata = {
        "status": "running",
        "mode": args.mode,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "num_queries": args.num_queries,
        "auxiliary_weight": args.auxiliary_weight,
        "eval_only": args.eval_only,
        "max_eval_new_tokens": args.max_eval_new_tokens,
        "validation_interval": args.validation_interval,
        "optimizer": {
            "name": "AdamW",
            "peak_learning_rate": args.learning_rate,
            "weight_decay": 0.01,
            "max_grad_norm": args.max_grad_norm,
        },
        "scheduler": {
            "name": "linear_warmup_cosine_decay",
            "warmup_steps": args.warmup_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "terminal_learning_rate": learning_rate_at_step(
                args.max_steps,
                peak_learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                max_steps=args.max_steps,
                min_lr_ratio=args.min_lr_ratio,
            ),
        },
        "adapter_config": None,
        "model_path": str(args.model_path.resolve()),
        "protocol": json.loads(args.protocol_file.read_text(encoding="utf-8")),
        "test_used_for_selection": False,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__},
    }
    write_json(args.output_dir / "metadata.json", metadata)
    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        device = torch.device("cuda", 0)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside the Slurm allocation")
        train_records = load_records(args.train_manifest)
        validation_records = load_records(args.validation_manifest)
        if len(train_records) != 128:
            raise ValueError(f"mechanism screen requires 128 train pages, got {len(train_records)}")
        if len(validation_records) != 64:
            raise ValueError(
                f"mechanism screen requires 64 validation pages, got {len(validation_records)}"
            )
        model, processor, bridge = load_model(args, device)
        metadata["adapter_config"] = asdict(bridge.adapter.config)
        metadata["versions"]["transformers"] = __import__("transformers").__version__
        metadata["gpu"] = torch.cuda.get_device_name(0)
        write_json(args.output_dir / "metadata.json", metadata)
        if args.eval_only:
            state_before = clone_module_state(bridge.adapter)
            validation = evaluate(
                args,
                model,
                processor,
                bridge,
                validation_records,
                train_records,
                device,
            )
            unchanged = module_state_matches(bridge.adapter, state_before)
            if not unchanged:
                raise RuntimeError("eval-only baseline modified adapter parameters")
            summary = {
                "status": "complete",
                "mode": args.mode,
                "seed": args.seed,
                "auxiliary_weight": args.auxiliary_weight,
                "eval_only": True,
                "training_updates": 0,
                "parameters_unchanged": True,
                "max_eval_new_tokens": args.max_eval_new_tokens,
                "validation": validation,
                "test_used_for_selection": False,
            }
            write_json(args.output_dir / "summary.json", summary)
            (args.output_dir / "COMPLETED").touch()
            metadata["status"] = "complete"
            write_json(args.output_dir / "metadata.json", metadata)
            print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
            return

        training = train(args, model, processor, bridge, train_records, device)
        checkpoint_steps = training["checkpoint_steps"]
        candidates = []
        for step in checkpoint_steps:
            checkpoint_dir = args.output_dir / f"checkpoint-{step}"
            load_adapter_checkpoint(checkpoint_dir, bridge)
            candidate = evaluate(
                args,
                model,
                processor,
                bridge,
                validation_records,
                train_records,
                device,
                output_dir=args.output_dir / f"validation-{step}",
            )
            health = json.loads(
                (checkpoint_dir / "checkpoint_health.json").read_text(encoding="utf-8")
            )
            candidates.append({"step": step, "checkpoint_health": health, **candidate})
        selected = min(candidates, key=lambda row: (row["cer"], row["step"]))
        selected_checkpoint = args.output_dir / f"checkpoint-{selected['step']}"
        load_adapter_checkpoint(selected_checkpoint, bridge)
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in bridge.adapter.state_dict().items()},
            args.output_dir / "adapter.safetensors",
        )
        write_adapter_config(args.output_dir / "adapter_config.json", bridge)
        shutil.copyfile(
            args.output_dir / f"validation-{selected['step']}" / "validation_predictions.jsonl",
            args.output_dir / "validation_predictions.jsonl",
        )
        (args.output_dir / "selection.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "selection_metric": "validation_cer",
                    "selected_step": selected["step"],
                    "candidates": candidates,
                    "test_used_for_selection": False,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        validation = selected
        summary = {
            "status": "complete",
            "mode": args.mode,
            "seed": args.seed,
            "auxiliary_weight": args.auxiliary_weight,
            "eval_only": False,
            "max_eval_new_tokens": args.max_eval_new_tokens,
            "training": training,
            "validation": validation,
            "selection_candidates": candidates,
            "test_used_for_selection": False,
        }
        write_json(args.output_dir / "summary.json", summary)
        (args.output_dir / "COMPLETED").touch()
        metadata["status"] = "complete"
        write_json(args.output_dir / "metadata.json", metadata)
        print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error_type"] = type(exc).__name__
        metadata["error"] = str(exc)
        write_json(args.output_dir / "metadata.json", metadata)
        (args.output_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
