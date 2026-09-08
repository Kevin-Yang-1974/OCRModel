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

from .config import layout_loss_config
from .data import layout_targets, load_records, prepare_inference_inputs, prepare_training_inputs
from .glm_bridge import LayoutAwarePatchMerger, install_layout_adapter
from .losses import compute_layout_losses, match_layout_targets
from .metrics import aggregate_ocr_metrics


LAYOUT_LOSS_KEYS = (
    "layout_box",
    "layout_order",
    "layout_direction",
    "layout_assignment",
    "transport_entropy",
)


def parse_step_list(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list of non-negative diagnostic steps."""

    if not value.strip():
        return ()
    parsed: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step < 0:
            raise argparse.ArgumentTypeError("diagnostic steps must be non-negative")
        parsed.add(step)
    return tuple(sorted(parsed))


def _dtype_name(value: torch.dtype | None) -> str | None:
    if value is None:
        return None
    return str(value).removeprefix("torch.")


def adapter_dtype_report(bridge: LayoutAwarePatchMerger) -> dict[str, str | None]:
    output = bridge.last_output
    parameter_dtypes = sorted(
        {
            dtype_name
            for parameter in bridge.adapter.parameters()
            if (dtype_name := _dtype_name(parameter.dtype)) is not None
        }
    )
    return {
        "adapter_precision": bridge.adapter_precision,
        "adapter_parameters": ",".join(value for value in parameter_dtypes if value is not None),
        "bridge_input": _dtype_name(bridge.last_input_dtype),
        "adapter_input": _dtype_name(bridge.last_adapter_input_dtype),
        "adapter_output": _dtype_name(bridge.last_adapter_output_dtype),
        "merged_tokens": _dtype_name(output.merged_tokens.dtype if output is not None else None),
        "writeback_tokens": _dtype_name(bridge.last_merged_dtype),
        "boxes": _dtype_name(output.boxes.dtype if output is not None else None),
        "transport": _dtype_name(output.transport.dtype if output is not None and output.transport is not None else None),
    }


def residual_relative_norm(bridge: LayoutAwarePatchMerger) -> float | None:
    if bridge.last_residual is None or bridge.last_visual_tokens is None:
        return None
    residual = bridge.last_residual.float()
    visual = bridge.last_visual_tokens.float()
    return float(residual.norm() / visual.norm().clamp_min(1e-12))


def writeback_residual_relative_norm(bridge: LayoutAwarePatchMerger) -> float | None:
    if bridge.last_writeback_residual is None or bridge.last_visual_tokens is None:
        return None
    residual = bridge.last_writeback_residual.float()
    visual = bridge.last_visual_tokens.float()
    return float(residual.norm() / visual.norm().clamp_min(1e-12))


def transport_diagnostics(
    output: Any, query_mask: torch.Tensor
) -> dict[str, Any]:
    """Summarize raw transport and the normalized query contribution to writeback."""

    transport = output.transport
    if transport is None:
        return {
            "transport_entropy": None,
            "transport_entropy_nats": None,
            "transport_query_mass": None,
            "invalid_query_transport_mass": None,
            "valid_query_transport_mass": None,
            "fusion_query_mass": None,
            "invalid_query_fusion_mass": None,
            "valid_query_fusion_mass": None,
        }

    plan = transport.float().clamp_min(1e-12)
    query_mass = plan.sum(dim=-1)
    probabilities = plan / query_mass.unsqueeze(-1).clamp_min(1e-12)
    entropy_nats = -(probabilities * probabilities.log()).sum(dim=-1)
    token_weights = plan.transpose(1, 2)
    token_weights = token_weights / token_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    fusion_mass = token_weights.sum(dim=1)
    fusion_mass = fusion_mass / fusion_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    mask = query_mask.to(dtype=torch.bool)
    invalid_mass = (fusion_mass * (~mask).to(fusion_mass.dtype)).sum(dim=-1)
    valid_mass = (fusion_mass * mask.to(fusion_mass.dtype)).sum(dim=-1)
    invalid_transport_mass = (query_mass * (~mask).to(query_mass.dtype)).sum(dim=-1)
    valid_transport_mass = (query_mass * mask.to(query_mass.dtype)).sum(dim=-1)
    token_count = max(1, probabilities.shape[-1])
    entropy = entropy_nats / math.log(token_count) if token_count > 1 else entropy_nats * 0.0
    return {
        "transport_entropy": float(entropy.mean().detach()),
        "transport_entropy_nats": float(entropy_nats.mean().detach()),
        "transport_query_mass": query_mass.mean(dim=0).detach().cpu().tolist(),
        "invalid_query_transport_mass": float(invalid_transport_mass.mean().detach()),
        "valid_query_transport_mass": float(valid_transport_mass.mean().detach()),
        "fusion_query_mass": fusion_mass.mean(dim=0).detach().cpu().tolist(),
        "invalid_query_fusion_mass": float(invalid_mass.mean().detach()),
        "valid_query_fusion_mass": float(valid_mass.mean().detach()),
    }


def gradient_norm_for_loss(loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...]) -> float:
    if not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squared = [gradient.float().pow(2).sum() for gradient in gradients if gradient is not None]
    if not squared:
        return 0.0
    return float(torch.stack(squared).sum().sqrt().detach())


def _token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value if item is not None}
    return {int(value)}


def eos_token_ids(model: Any, processor: Any) -> set[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    ids = _token_id_set(getattr(tokenizer, "eos_token_id", None))
    generation_config = getattr(model, "generation_config", None)
    ids.update(_token_id_set(getattr(generation_config, "eos_token_id", None)))
    return ids


def diagnostic_triage(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the first-pass 0/64/128 rules without changing checkpoint selection."""

    by_step = {int(point["step"]): point for point in points}
    required = (0, 64, 128)
    if any(step not in by_step for step in required):
        return {"available": False, "required_steps": list(required)}

    def validation_value(step: int, key: str) -> float | None:
        value = by_step[step]["validation"].get(key)
        if not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    cer0, cer64, cer128 = (validation_value(step, "cer") for step in required)
    invalid64 = validation_value(64, "invalid_query_fusion_mass")
    invalid128 = validation_value(128, "invalid_query_fusion_mass")
    residual64 = validation_value(64, "residual_relative_norm")
    residual128 = validation_value(128, "residual_relative_norm")
    gate64 = validation_value(64, "effective_residual_scale")
    gate128 = validation_value(128, "effective_residual_scale")
    improved = cer0 is not None and cer64 is not None and cer64 < cer0
    degraded = cer64 is not None and cer128 is not None and cer128 > cer64
    invalid_rise = invalid64 is not None and invalid128 is not None and invalid128 > invalid64
    residual_rise = residual64 is not None and residual128 is not None and residual128 > residual64
    small_gate = (
        gate64 is not None
        and gate128 is not None
        and abs(gate64) <= 0.03
        and abs(gate128) <= 0.03
    )
    small_residual = (
        residual64 is not None and residual128 is not None
        and max(abs(residual64), abs(residual128)) <= 0.01
    )

    def dtype_signature(value: Any) -> str:
        items = value if isinstance(value, list) else [value]
        return json.dumps(
            sorted(json.dumps(item, sort_keys=True) for item in items),
            sort_keys=True,
        )

    dtype_mismatch = False
    for step in (64, 128):
        training = by_step[step].get("training") or {}
        validation = by_step[step]["validation"]
        training_loss_dtypes = training.get("loss_dtypes")
        if not isinstance(training_loss_dtypes, dict):
            training_loss_dtypes = {}
        train_signature = json.dumps(
            {
                "adapter": dtype_signature(training.get("adapter_dtypes")),
                "loss": dtype_signature(
                    {
                        key: training_loss_dtypes.get(key)
                        for key in LAYOUT_LOSS_KEYS
                    }
                ),
            },
            sort_keys=True,
        )
        eval_signature = json.dumps(
            {
                "adapter": dtype_signature(validation.get("adapter_dtypes")),
                "loss": dtype_signature(validation.get("layout_loss_dtypes")),
            },
            sort_keys=True,
        )
        dtype_mismatch = dtype_mismatch or train_signature != eval_signature

    rules = {
        "query_mask_or_no_object": bool(improved and degraded and (invalid_rise or residual_rise)),
        "target_slot_or_auxiliary_conflict": bool(degraded and small_gate and small_residual),
        "train_eval_dtype_mismatch": bool(degraded and dtype_mismatch),
        "stable_at_128_extend_to_256": bool(cer64 is not None and cer128 is not None and cer128 <= cer64),
    }
    priority = []
    if rules["query_mask_or_no_object"]:
        priority.append("query_mask/no-object")
    if rules["train_eval_dtype_mismatch"]:
        priority.append("FP32 adapter/transport/loss consistency")
    if rules["target_slot_or_auxiliary_conflict"]:
        priority.append("target-slot matching and auxiliary-loss conflict")
    if rules["stable_at_128_extend_to_256"]:
        priority.append("extend only to step 256")
    if not priority:
        priority.append("inspect the per-point JSON before extending the run")
    return {
        "available": True,
        "cer_delta_0_to_64": None if cer0 is None or cer64 is None else cer64 - cer0,
        "cer_delta_64_to_128": None if cer64 is None or cer128 is None else cer128 - cer64,
        "invalid_query_fusion_mass_rise_64_to_128": invalid_rise,
        "residual_relative_norm_rise_64_to_128": residual_rise,
        "dtype_mismatch": dtype_mismatch,
        "rules": rules,
        "priority": priority,
        "thresholds": {"effective_gate_abs": 0.03, "residual_relative_norm_abs": 0.01},
    }


def optional_positive_float(value: str) -> float | None:
    if value.lower() in {"none", "off"}:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive, 'none', or 'off'")
    return parsed


def optional_positive_int(value: str) -> int | None:
    if value.lower() in {"none", "off"}:
        return None
    parsed = int(value)
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
    report = {
        "step": step,
        "adapter_precision": bridge.adapter_precision,
        **adapter_finite_report(bridge.adapter),
    }
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
        adapter_precision=args.adapter_precision,
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
    loss_weights = layout_loss_config(args.layout_loss_profile)
    lr_schedule_steps = args.lr_schedule_steps or args.max_steps
    rng = random.Random(args.seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    # The GLM-OCR backbone stays in evaluation mode while only the adapter is
    # optimized.  This prevents frozen dropout/statistics from adding noise to
    # the mechanism comparison.
    model.eval()
    bridge.adapter.train()
    model.config.use_cache = False
    adapter_parameters = tuple(bridge.adapter.parameters())
    running: Counter[str] = Counter()
    started = time.time()
    checkpoint_steps: list[int] = []
    checkpoint_health: list[dict[str, Any]] = []
    diagnostic_train: dict[str, dict[str, Any]] = {}
    for step in range(1, args.max_steps + 1):
        if (step - 1) % len(order) == 0 and step > 1:
            rng.shuffle(order)
        record = records[order[(step - 1) % len(order)]]
        inputs = prepare_training_inputs(processor, record, device)
        bridge.set_grid_thw(inputs["image_grid_thw"])
        learning_rate = learning_rate_at_step(
            min(step, lr_schedule_steps),
            peak_learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=lr_schedule_steps,
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
        targets = layout_targets(record, bridge.last_patch_positions, args.num_queries)
        targets = match_layout_targets(
            bridge.last_output,
            targets,
            assignment=args.query_assignment,
        )
        if args.auxiliary_weight > 0.0:
            auxiliary_losses = compute_layout_losses(
                bridge.last_output, weights=loss_weights, **targets
            )
        else:
            zero = outputs.loss.float() * 0.0
            auxiliary_losses = {key: zero for key in (*LAYOUT_LOSS_KEYS, "loss")}
        auxiliary_loss = auxiliary_losses["loss"].float()
        if not torch.isfinite(auxiliary_loss):
            raise FloatingPointError(f"non-finite auxiliary loss at step {step}")
        loss = outputs.loss.float() + args.auxiliary_weight * auxiliary_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        diagnostic = step in args.diagnostic_steps
        gradient_norms: dict[str, float] = {}
        if diagnostic:
            component_losses = {
                "ocr": outputs.loss.float(),
                **{
                    key: auxiliary_losses[key].float()
                    for key in LAYOUT_LOSS_KEYS
                },
                "total": loss,
            }
            gradient_norms = {
                key: gradient_norm_for_loss(value, adapter_parameters)
                for key, value in component_losses.items()
            }
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
        if diagnostic:
            transport = transport_diagnostics(bridge.last_output, targets["query_mask"])
            metrics.update(
                {
                    "step": step,
                    "page_id": record["page_id"],
                    "optimizer_update": True,
                    "loss_components": {
                        "ocr": float(outputs.loss.detach()),
                        **{
                            key: float(auxiliary_losses[key].detach())
                            for key in LAYOUT_LOSS_KEYS
                        },
                    },
                    "loss_dtypes": {
                        "ocr": _dtype_name(outputs.loss.dtype),
                        **{
                            key: _dtype_name(auxiliary_losses[key].dtype)
                            for key in LAYOUT_LOSS_KEYS
                        },
                        "total": _dtype_name(loss.dtype),
                    },
                    "gradient_norms": gradient_norms,
                    "residual_relative_norm": residual_relative_norm(bridge),
                    "writeback_residual_relative_norm": writeback_residual_relative_norm(bridge),
                    "transport": transport,
                    "adapter_dtypes": adapter_dtype_report(bridge),
                    "query_count": int(targets["query_mask"].sum()),
                    "token_count": int(bridge.last_patch_positions.shape[1]),
                }
            )
            diagnostic_train[str(step)] = dict(metrics)
        for key in ("ocr_loss", "auxiliary_loss", "total_loss", "gradient_norm"):
            running[key] += metrics[key]
        if step == 1 or step % args.log_steps == 0 or step == args.max_steps or diagnostic:
            append_jsonl(
                args.output_dir / "train_metrics.jsonl",
                {"step": step, "page_id": record["page_id"], **metrics},
            )
        if (
            step % args.validation_interval == 0
            or step == args.max_steps
            or step in args.diagnostic_steps
        ):
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
        "diagnostic_train": diagnostic_train,
        "lr_schedule_steps": lr_schedule_steps,
        "final_learning_rate": learning_rate_at_step(
            min(args.max_steps, lr_schedule_steps),
            peak_learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=lr_schedule_steps,
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
    generation_lengths: list[int] = []
    generation_eos_hits = 0
    eos_ids = eos_token_ids(model, processor)
    loss_weights = layout_loss_config(args.layout_loss_profile)
    layout_loss_sums = {key: 0.0 for key in LAYOUT_LOSS_KEYS}
    residual_norms: list[float] = []
    writeback_residual_norms: list[float] = []
    transport_entropies: list[float] = []
    transport_entropy_nats: list[float] = []
    transport_query_masses: list[list[float]] = []
    invalid_transport_masses: list[float] = []
    valid_transport_masses: list[float] = []
    fusion_query_masses: list[list[float]] = []
    invalid_query_masses: list[float] = []
    valid_query_masses: list[float] = []
    annotated_query_counts: list[int] = []
    dtype_signatures: set[str] = set()
    layout_loss_dtype_signatures: set[str] = set()
    teacher_forced_ocr_losses: list[float] = []
    teacher_forced_layout_loss_sums = {key: 0.0 for key in LAYOUT_LOSS_KEYS}
    teacher_forced_ocr_dtypes: set[str] = set()
    collect_teacher_forcing = bool(args.diagnostic_steps)
    started = time.time()
    for record in validation_records:
        inputs = prepare_inference_inputs(processor, record, device)
        bridge.set_grid_thw(inputs["image_grid_thw"])
        if collect_teacher_forcing:
            teacher_inputs = prepare_training_inputs(processor, record, device)
            bridge.set_grid_thw(teacher_inputs["image_grid_thw"])
            teacher_outputs = model(**teacher_inputs)
            if teacher_outputs.loss is None or bridge.last_output is None or bridge.last_patch_positions is None:
                raise RuntimeError("validation teacher-forcing forward did not produce OCR/layout state")
            teacher_forced_ocr_losses.append(float(teacher_outputs.loss.detach()))
            teacher_forced_ocr_dtypes.add(_dtype_name(teacher_outputs.loss.dtype) or "unknown")
            teacher_targets = layout_targets(record, bridge.last_patch_positions, args.num_queries)
            teacher_targets = match_layout_targets(
                bridge.last_output,
                teacher_targets,
                assignment=args.query_assignment,
            )
            teacher_layout_losses = compute_layout_losses(
                bridge.last_output,
                weights=loss_weights,
                **teacher_targets,
            )
            for key in LAYOUT_LOSS_KEYS:
                teacher_forced_layout_loss_sums[key] += float(teacher_layout_losses[key].detach())
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
        generated_tokens = generated[0, prompt_length:]
        generation_length = int(generated_tokens.shape[0])
        generation_lengths.append(generation_length)
        eos_hit = bool(eos_ids and any(int(token) in eos_ids for token in generated_tokens.tolist()))
        if eos_hit:
            generation_eos_hits += 1
        prediction = processor.decode(generated_tokens, skip_special_tokens=True)
        generation_limit_hit = generation_length >= args.max_eval_new_tokens
        if generation_limit_hit:
            generation_limit_hits += 1
        pairs.append((record["page_text"], prediction))
        if bridge.last_output is None or bridge.last_patch_positions is None:
            raise RuntimeError("generation did not retain layout state")
        targets = layout_targets(record, bridge.last_patch_positions, args.num_queries)
        targets = match_layout_targets(
            bridge.last_output,
            targets,
            assignment=args.query_assignment,
        )
        annotated_query_counts.append(int(targets["query_mask"].sum()))
        layout_losses = compute_layout_losses(
            bridge.last_output,
            weights=loss_weights,
            **targets,
        )
        for key in LAYOUT_LOSS_KEYS:
            layout_loss_sums[key] += float(layout_losses[key].detach())
        layout_loss_dtype_signatures.add(
            json.dumps(
                {key: _dtype_name(layout_losses[key].dtype) for key in LAYOUT_LOSS_KEYS},
                sort_keys=True,
            )
        )
        relative_norm = residual_relative_norm(bridge)
        if relative_norm is not None:
            residual_norms.append(relative_norm)
        writeback_relative_norm = writeback_residual_relative_norm(bridge)
        if writeback_relative_norm is not None:
            writeback_residual_norms.append(writeback_relative_norm)
        transport = transport_diagnostics(bridge.last_output, targets["query_mask"])
        if transport["transport_entropy"] is not None:
            transport_entropies.append(float(transport["transport_entropy"]))
            transport_entropy_nats.append(float(transport["transport_entropy_nats"]))
            transport_query_masses.append(transport["transport_query_mass"])
            invalid_transport_masses.append(float(transport["invalid_query_transport_mass"]))
            valid_transport_masses.append(float(transport["valid_query_transport_mass"]))
            fusion_query_masses.append(transport["fusion_query_mass"])
            invalid_query_masses.append(float(transport["invalid_query_fusion_mass"]))
            valid_query_masses.append(float(transport["valid_query_fusion_mass"]))
        dtype_signatures.add(json.dumps(adapter_dtype_report(bridge), sort_keys=True))
        count = int(targets["query_mask"].sum())
        if count:
            query_mask = targets["query_mask"][0]
            error = (
                bridge.last_output.boxes[0, query_mask]
                - targets["target_boxes"][0, query_mask]
            ).abs()
            box_errors.append(float(error.mean()))
            predicted_direction = bridge.last_output.direction_logits[0, query_mask].argmax(dim=-1)
            direction_correct += int(
                (predicted_direction == targets["target_directions"][0, query_mask]).sum()
            )
            direction_total += count
        append_jsonl(
            predictions_path,
            {
                "page_id": record["page_id"],
                "reference": record["page_text"],
                "prediction": prediction,
                "generation_length": generation_length,
                "generation_eos_hit": eos_hit if eos_ids else None,
                "generation_limit_hit": generation_limit_hit,
            },
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
            "mean_annotated_queries": sum(annotated_query_counts)
            / max(1, len(annotated_query_counts)),
            "mean_unannotated_queries": args.num_queries
            - sum(annotated_query_counts) / max(1, len(annotated_query_counts)),
            "seconds": time.time() - started,
            "generation_max_new_tokens": args.max_eval_new_tokens,
            "generation_limit_hits": generation_limit_hits,
            "generation_limit_hit_rate": generation_limit_hits / max(1, len(validation_records)),
            "generation_lengths": generation_lengths,
            "generation_mean_new_tokens": sum(generation_lengths) / max(1, len(generation_lengths)),
            "generation_max_new_tokens_observed": max(generation_lengths, default=0),
            "generation_eos_hits": generation_eos_hits,
            "generation_eos_observable": bool(eos_ids),
            "generation_eos_hit_rate": (
                generation_eos_hits / max(1, len(validation_records)) if eos_ids else None
            ),
            "generation_eos_token_ids": sorted(eos_ids),
            "layout_loss_means": {
                key: value / max(1, len(validation_records))
                for key, value in layout_loss_sums.items()
            },
            "transport_entropy_loss": layout_loss_sums["transport_entropy"]
            / max(1, len(validation_records)),
            "teacher_forced_ocr_loss": (
                sum(teacher_forced_ocr_losses) / len(teacher_forced_ocr_losses)
                if teacher_forced_ocr_losses
                else None
            ),
            "teacher_forced_ocr_loss_dtypes": sorted(teacher_forced_ocr_dtypes),
            "teacher_forced_layout_loss_means": (
                {
                    key: value / len(teacher_forced_ocr_losses)
                    for key, value in teacher_forced_layout_loss_sums.items()
                }
                if teacher_forced_ocr_losses
                else None
            ),
            "residual_relative_norm": sum(residual_norms) / max(1, len(residual_norms)),
            "writeback_residual_relative_norm": sum(writeback_residual_norms)
            / max(1, len(writeback_residual_norms)),
            "transport_entropy": sum(transport_entropies) / max(1, len(transport_entropies)),
            "transport_entropy_nats": sum(transport_entropy_nats)
            / max(1, len(transport_entropy_nats)),
            "transport_query_mass": [
                sum(values[index] for values in transport_query_masses)
                / max(1, len(transport_query_masses))
                for index in range(args.num_queries)
            ] if transport_query_masses else None,
            "invalid_query_transport_mass": sum(invalid_transport_masses)
            / max(1, len(invalid_transport_masses)),
            "valid_query_transport_mass": sum(valid_transport_masses)
            / max(1, len(valid_transport_masses)),
            "fusion_query_mass": [
                sum(values[index] for values in fusion_query_masses)
                / max(1, len(fusion_query_masses))
                for index in range(args.num_queries)
            ] if fusion_query_masses else None,
            "invalid_query_fusion_mass": sum(invalid_query_masses)
            / max(1, len(invalid_query_masses)),
            "valid_query_fusion_mass": sum(valid_query_masses)
            / max(1, len(valid_query_masses)),
            "adapter_dtypes": [json.loads(value) for value in sorted(dtype_signatures)],
            "layout_loss_dtypes": [
                json.loads(value) for value in sorted(layout_loss_dtype_signatures)
            ],
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
    parser.add_argument(
        "--lr-schedule-steps",
        type=optional_positive_int,
        default=None,
        help="learning-rate schedule horizon; defaults to --max-steps and may be shorter",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--residual-scale-cap", type=optional_positive_float, default=0.03)
    parser.add_argument("--auxiliary-weight", type=float, default=0.2)
    parser.add_argument(
        "--adapter-precision",
        choices=["mixed_bf16", "fp32"],
        default="mixed_bf16",
        help="precision used inside the pre-merge adapter; the backbone remains BF16",
    )
    parser.add_argument(
        "--layout-loss-profile",
        choices=["full", "ocr_only", "no_assignment", "no_geometry"],
        default="full",
    )
    parser.add_argument(
        "--query-assignment",
        choices=["fixed_order", "hungarian"],
        default="fixed_order",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--max-eval-new-tokens", type=int, default=768)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=256)
    parser.add_argument("--diagnostic-steps", type=parse_step_list, default=())
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
    if args.lr_schedule_steps is not None:
        if args.lr_schedule_steps > args.max_steps:
            raise ValueError("--lr-schedule-steps must not exceed --max-steps")
        if not 0 <= args.warmup_steps < args.lr_schedule_steps:
            raise ValueError("--warmup-steps must be smaller than --lr-schedule-steps")
    if not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if any(step > args.max_steps for step in args.diagnostic_steps):
        raise ValueError("diagnostic steps must not exceed --max-steps")
    if args.validation_interval <= 0:
        raise ValueError("--validation-interval must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    lr_schedule_steps = args.lr_schedule_steps or args.max_steps
    metadata = {
        "status": "running",
        "mode": args.mode,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "lr_schedule_steps": lr_schedule_steps,
        "num_queries": args.num_queries,
        "auxiliary_weight": args.auxiliary_weight,
        "adapter_precision": args.adapter_precision,
        "layout_loss_profile": args.layout_loss_profile,
        "query_assignment": args.query_assignment,
        "eval_only": args.eval_only,
        "max_eval_new_tokens": args.max_eval_new_tokens,
        "validation_interval": args.validation_interval,
        "diagnostic_steps": list(args.diagnostic_steps),
        "optimizer": {
            "name": "AdamW",
            "peak_learning_rate": args.learning_rate,
            "weight_decay": 0.01,
            "max_grad_norm": args.max_grad_norm,
        },
        "scheduler": {
            "name": "linear_warmup_cosine_decay",
            "warmup_steps": args.warmup_steps,
            "schedule_steps": lr_schedule_steps,
            "min_lr_ratio": args.min_lr_ratio,
            "terminal_learning_rate": learning_rate_at_step(
                min(args.max_steps, lr_schedule_steps),
                peak_learning_rate=args.learning_rate,
                warmup_steps=args.warmup_steps,
                max_steps=lr_schedule_steps,
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
                "adapter_precision": args.adapter_precision,
                "layout_loss_profile": args.layout_loss_profile,
                "query_assignment": args.query_assignment,
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

        diagnostic_points: list[dict[str, Any]] = []
        if 0 in args.diagnostic_steps:
            validation0 = evaluate(
                args,
                model,
                processor,
                bridge,
                validation_records,
                train_records,
                device,
                output_dir=args.output_dir / "validation-0",
            )
            diagnostic_points.append(
                {
                    "step": 0,
                    "training": {
                        "step": 0,
                        "optimizer_update": False,
                        "ocr_loss": validation0["teacher_forced_ocr_loss"],
                        "layout_loss_means": validation0["teacher_forced_layout_loss_means"],
                        "gradient_norms": None,
                    },
                    "validation": validation0,
                }
            )

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
            training_diagnostics = training["diagnostic_train"].get(str(step))
            candidate_row = {
                "step": step,
                "checkpoint_health": health,
                "training": training_diagnostics,
                **candidate,
            }
            candidates.append(candidate_row)
            if step in args.diagnostic_steps:
                diagnostic_points.append(
                    {
                        "step": step,
                        "training": training_diagnostics,
                        "validation": candidate,
                    }
                )
        def training_value(point: dict[str, Any], key: str) -> Any:
            training_row = point.get("training") or {}
            if key == "ocr_loss":
                return training_row.get("ocr_loss")
            components = training_row.get("loss_components") or training_row.get(
                "layout_loss_means"
            ) or {}
            return components.get(key)

        def gradient_value(point: dict[str, Any], key: str) -> Any:
            return ((point.get("training") or {}).get("gradient_norms") or {}).get(key)

        if args.diagnostic_steps:
            diagnostic_points.sort(key=lambda row: row["step"])
            identity_point = next(
                (row for row in diagnostic_points if row["step"] == 0),
                None,
            )
            identity_cer = (
                float(identity_point["validation"]["cer"])
                if identity_point is not None
                else None
            )
            trained_points = [row for row in diagnostic_points if row["step"] > 0]
            best_trained_point = (
                min(trained_points, key=lambda row: (row["validation"]["cer"], row["step"]))
                if trained_points
                else None
            )
            best_vs_identity_point = (
                min(diagnostic_points, key=lambda row: (row["validation"]["cer"], row["step"]))
                if identity_point is not None
                else None
            )
            identity_delta_cer = {
                str(row["step"]): (
                    float(row["validation"]["cer"]) - identity_cer
                    if identity_cer is not None
                    else None
                )
                for row in diagnostic_points
            }
            diagnostic_summary = {
                "status": "complete",
                "mode": args.mode,
                "seed": args.seed,
                "adapter_precision": args.adapter_precision,
                "layout_loss_profile": args.layout_loss_profile,
                "query_assignment": args.query_assignment,
                "lr_schedule_steps": training["lr_schedule_steps"],
                "steps": [row["step"] for row in diagnostic_points],
                "identity_baseline_step": 0 if identity_point is not None else None,
                "identity_baseline_cer": identity_cer,
                "identity_delta_cer": identity_delta_cer,
                "best_trained_step": (
                    best_trained_point["step"] if best_trained_point is not None else None
                ),
                "best_vs_identity_step": (
                    best_vs_identity_point["step"]
                    if best_vs_identity_point is not None
                    else None
                ),
                "points": diagnostic_points,
                "curves": {
                    "validation_cer": [
                        {"step": row["step"], "value": row["validation"]["cer"]}
                        for row in diagnostic_points
                    ],
                    "identity_delta_cer": [
                        {"step": row["step"], "value": identity_delta_cer[str(row["step"])]}
                        for row in diagnostic_points
                    ],
                    "exact_page_rate": [
                        {"step": row["step"], "value": row["validation"]["exact_page_rate"]}
                        for row in diagnostic_points
                    ],
                    "generation_limit_hit_rate": [
                        {
                            "step": row["step"],
                            "value": row["validation"]["generation_limit_hit_rate"],
                        }
                        for row in diagnostic_points
                    ],
                    "generation_eos_hit_rate": [
                        {
                            "step": row["step"],
                            "value": row["validation"]["generation_eos_hit_rate"],
                        }
                        for row in diagnostic_points
                    ],
                    "teacher_forced_ocr_loss": [
                        {
                            "step": row["step"],
                            "value": row["validation"]["teacher_forced_ocr_loss"],
                        }
                        for row in diagnostic_points
                    ],
                    "training_ocr_loss": [
                        {"step": row["step"], "value": training_value(row, "ocr_loss")}
                        for row in diagnostic_points
                    ],
                    **{
                        f"training_{key}_loss": [
                            {"step": row["step"], "value": training_value(row, key)}
                            for row in diagnostic_points
                        ]
                        for key in LAYOUT_LOSS_KEYS
                    },
                    **{
                        f"gradient_norm_{key}": [
                            {"step": row["step"], "value": gradient_value(row, key)}
                            for row in diagnostic_points
                        ]
                        for key in ("ocr", *LAYOUT_LOSS_KEYS, "total")
                    },
                    **{
                        f"validation_{key}_loss": [
                            {
                                "step": row["step"],
                                "value": (row["validation"]["layout_loss_means"] or {}).get(key),
                            }
                            for row in diagnostic_points
                        ]
                        for key in LAYOUT_LOSS_KEYS
                    },
                    "raw_content_gate": [
                        {"step": row["step"], "value": row["validation"]["raw_content_gate"]}
                        for row in diagnostic_points
                    ],
                    "effective_residual_scale": [
                        {
                            "step": row["step"],
                            "value": row["validation"]["effective_residual_scale"],
                        }
                        for row in diagnostic_points
                    ],
                    "residual_relative_norm": [
                        {"step": row["step"], "value": row["validation"]["residual_relative_norm"]}
                        for row in diagnostic_points
                    ],
                    "writeback_residual_relative_norm": [
                        {
                            "step": row["step"],
                            "value": row["validation"]["writeback_residual_relative_norm"],
                        }
                        for row in diagnostic_points
                    ],
                    "transport_entropy": [
                        {"step": row["step"], "value": row["validation"]["transport_entropy"]}
                        for row in diagnostic_points
                    ],
                    "transport_query_mass": [
                        {"step": row["step"], "value": row["validation"]["transport_query_mass"]}
                        for row in diagnostic_points
                    ],
                    "fusion_query_mass": [
                        {"step": row["step"], "value": row["validation"]["fusion_query_mass"]}
                        for row in diagnostic_points
                    ],
                    "invalid_query_fusion_mass": [
                        {
                            "step": row["step"],
                            "value": row["validation"]["invalid_query_fusion_mass"],
                        }
                        for row in diagnostic_points
                    ],
                    "invalid_query_transport_mass": [
                        {
                            "step": row["step"],
                            "value": row["validation"]["invalid_query_transport_mass"],
                        }
                        for row in diagnostic_points
                    ],
                },
                "triage": diagnostic_triage(diagnostic_points),
                "test_used_for_selection": False,
            }
            write_json(args.output_dir / "diagnostic_summary.json", diagnostic_summary)
        else:
            diagnostic_summary = None
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
                    "selected_is_better_than_identity": (
                        identity_cer is None or selected["cer"] < identity_cer
                    ) if args.diagnostic_steps else None,
                    "identity_baseline": (
                        {
                            "step": 0,
                            "cer": identity_cer,
                            "exact_page_rate": identity_point["validation"]["exact_page_rate"],
                        }
                        if args.diagnostic_steps and identity_point is not None
                        else None
                    ),
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
            "adapter_precision": args.adapter_precision,
            "layout_loss_profile": args.layout_loss_profile,
            "query_assignment": args.query_assignment,
            "lr_schedule_steps": training["lr_schedule_steps"],
            "eval_only": False,
            "max_eval_new_tokens": args.max_eval_new_tokens,
            "training": training,
            "validation": validation,
            "selection_candidates": candidates,
            "diagnostic_summary": diagnostic_summary,
            "test_used_for_selection": False,
        }
        write_json(args.output_dir / "summary.json", summary)
        (args.output_dir / "COMPLETED").touch()
        metadata["status"] = "complete"
        write_json(args.output_dir / "metadata.json", metadata)
        if args.diagnostic_steps:
            completion = {
                "status": "complete",
                "run_dir": str(args.output_dir),
                "diagnostic_summary": str(args.output_dir / "diagnostic_summary.json"),
                "diagnostic_steps": [row["step"] for row in diagnostic_points],
                "selected_step": selected["step"],
                "selected_validation_cer": selected["cer"],
                "test_used_for_selection": False,
            }
            print(json.dumps(completion, ensure_ascii=False, separators=(",", ":")))
        else:
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
