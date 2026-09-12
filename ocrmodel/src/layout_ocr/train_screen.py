from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import os
import random
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from .config import layout_loss_config
from .data import (
    layout_targets,
    load_records,
    prepare_inference_inputs,
    prepare_training_inputs,
    region_decoder_targets,
    validate_records,
)
from .distributed import (
    DistributedInfo,
    all_finite,
    barrier,
    destroy_distributed,
    initialize_distributed,
    mean_scalar,
    rank_epoch_indices,
    sum_scalar,
    unwrap_module,
    wrap_adapter,
    wrap_model,
)
from .glm_bridge import LayoutAwarePatchMerger, install_layout_adapter
from .lora import (
    decoder_lora_finite_report,
    inject_decoder_lora,
    iter_lora_parameters,
    load_lora_state_dict,
    lora_state_dict,
    set_lora_modules_training,
    trainable_parameter_report,
)
from .losses import compute_layout_losses, match_layout_targets
from .aligned_recovery import CONFIG as ALIGNED_CONFIG, collect_rollout as collect_aligned_rollout, recovery_losses
from .metrics import aggregate_ocr_metrics
from .autoregressive_region import box_iou, compute_region_losses
from .stabilization import (
    ContinuationStopHead,
    RepeatSuppressionConfig,
    continuation_head_loss,
    cycle_escape_losses,
    eos_focus_loss,
    generate_with_loop_recovery,
    generated_cycle_window,
    loop_continuation_diagnostics,
    natural_loop_rollout_loss,
    repeated_cycle_positions,
    repetition_diagnostics,
    unlikelihood_loss,
)


LAYOUT_LOSS_KEYS = (
    "layout_box",
    "layout_order",
    "layout_direction",
    "layout_assignment",
    "transport_entropy",
    "layout_validity",
    "layout_validity_bce",
    "layout_validity_cardinality",
    "layout_validity_ranking",
)
REGION_LOSS_KEYS = (
    "region_pointer",
    "region_bbox",
    "region_giou",
    "region_direction",
    "region_eos",
    "region_coverage",
    "region_duplicate",
    "region_objectness",
    "region_count",
)
TEXT_LOSS_KEYS = ("text_unlikelihood", "text_eos", "natural_loop")


def configure_deterministic_execution() -> dict[str, Any]:
    """Pin CUDA/SDPA choices before the model creates any CUDA handles.

    The mechanism comparison is sensitive to very small changes in the frozen
    backbone activations. ``torch.use_deterministic_algorithms`` alone is not
    sufficient here: PyTorch can still select a non-deterministic fused SDPA
    backend, and cuBLAS needs its workspace contract set before the first CUDA
    handle is created. Keep the report in run metadata so a result cannot be
    mistaken for a reproducible run when the backend was not pinned.
    """

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_float32_matmul_precision("highest")

    sdp_backend = "unavailable"
    if hasattr(torch, "backends") and hasattr(torch.backends, "cuda"):
        # The adapter and the GLM-OCR backbone both use scaled dot-product
        # attention. Math SDP is slower, but is the deterministic reference
        # backend for this small, controlled comparison.
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        sdp_backend = "math"
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False

    # Use strict mode after the backend choices above. If a future dependency
    # introduces another non-deterministic kernel, fail the run instead of
    # silently changing the training trajectory.
    torch.use_deterministic_algorithms(True, warn_only=False)
    return {
        "strict_deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "float32_matmul_precision": "highest",
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "sdp_backend": sdp_backend,
        "flash_sdp": False,
        "memory_efficient_sdp": False,
        "math_sdp": True,
    }


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


def repeated_trigram_rate(text: str) -> float:
    """Return the fraction of character trigrams repeated in a prediction.

    This is an evaluation diagnostic only.  It is deliberately computed on
    decoded text and is never used for training, checkpoint selection, or
    test-time thresholding.
    """

    if len(text) < 3:
        return 0.0
    trigrams = [text[index : index + 3] for index in range(len(text) - 2)]
    return (len(trigrams) - len(set(trigrams))) / len(trigrams)


def _dtype_name(value: torch.dtype | None) -> str | None:
    if value is None:
        return None
    return str(value).removeprefix("torch.")


def adapter_dtype_report(bridge: LayoutAwarePatchMerger) -> dict[str, str | None]:
    output = bridge.last_output
    adapter = unwrap_module(bridge.adapter)
    parameter_dtypes = sorted(
        {
            dtype_name
            for parameter in adapter.parameters()
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
        "validity_logits": _dtype_name(
            output.validity_logits.dtype
            if output is not None and output.validity_logits is not None
            else None
        ),
        "gated_transport": _dtype_name(
            output.gated_transport.dtype
            if output is not None and output.gated_transport is not None
            else None
        ),
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


def _validity_ranking_metrics(
    probabilities: torch.Tensor, query_mask: torch.Tensor
) -> tuple[float | None, float | None]:
    """Return AUROC and average precision without adding an eval dependency."""

    page_aurocs: list[torch.Tensor] = []
    page_aps: list[torch.Tensor] = []
    for page_scores, page_targets in zip(probabilities.float(), query_mask.bool()):
        positive = page_scores[page_targets]
        negative = page_scores[~page_targets]
        if not positive.numel() or not negative.numel():
            continue
        pairwise = positive[:, None] - negative[None, :]
        page_aurocs.append((pairwise.gt(0).float() + 0.5 * pairwise.eq(0).float()).mean())
        thresholds = positive[:, None]
        above = page_scores[None, :] >= thresholds
        true_positive = (above & page_targets[None, :]).sum(dim=-1).float()
        predicted_positive = above.sum(dim=-1).float().clamp_min(1.0)
        page_aps.append((true_positive / predicted_positive).mean())
    if not page_aurocs:
        return None, None
    return (
        float(torch.stack(page_aurocs).mean().detach()),
        float(torch.stack(page_aps).mean().detach()),
    )


def transport_diagnostics(
    output: Any,
    query_mask: torch.Tensor,
    token_owners: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Summarize raw and validity-gated transport without query-sized arrays."""

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
            "gated_transport_total_mass": None,
            "invalid_gated_query_transport_mass": None,
            "valid_gated_query_transport_mass": None,
            "invalid_gated_fusion_mass": None,
            "valid_gated_fusion_mass": None,
            "valid_coverage": None,
            "mean_p_valid": None,
            "mean_p_valid_matched": None,
            "mean_p_valid_no_object": None,
            "validity_p_gap": None,
            "validity_auroc": None,
            "validity_average_precision": None,
            "invalid_gated_context_share": None,
            "valid_coverage_foreground": None,
            "valid_coverage_background": None,
        }

    mask = query_mask.to(dtype=torch.bool)

    def summarize(plan_value: torch.Tensor) -> dict[str, Tensor]:
        plan_value = plan_value.float().clamp_min(1e-12)
        query_mass = plan_value.sum(dim=-1)
        total_mass = query_mass.sum(dim=-1).clamp_min(1e-12)
        normalized_query_mass = query_mass / total_mass.unsqueeze(-1)
        token_weights = plan_value.transpose(1, 2)
        token_weights = token_weights / token_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        fusion_mass = token_weights.sum(dim=1)
        fusion_mass = fusion_mass / fusion_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return {
            "total_mass": total_mass,
            "invalid_transport": (
                normalized_query_mass * (~mask).to(query_mass.dtype)
            ).sum(dim=-1),
            "valid_transport": (
                normalized_query_mass * mask.to(query_mass.dtype)
            ).sum(dim=-1),
            "invalid_fusion": (
                fusion_mass * (~mask).to(fusion_mass.dtype)
            ).sum(dim=-1),
            "valid_fusion": (
                fusion_mass * mask.to(fusion_mass.dtype)
            ).sum(dim=-1),
        }

    raw_plan = transport.float().clamp_min(1e-12)
    raw_query_mass = raw_plan.sum(dim=-1)
    raw_probabilities = raw_plan / raw_query_mass.unsqueeze(-1).clamp_min(1e-12)
    entropy_nats = -(raw_probabilities * raw_probabilities.log()).sum(dim=-1)
    token_count = max(1, raw_probabilities.shape[-1])
    entropy = entropy_nats / math.log(token_count) if token_count > 1 else entropy_nats * 0.0
    raw = summarize(raw_plan)
    gated_plan = output.gated_transport
    gated = summarize(gated_plan) if gated_plan is not None else None
    validity_probs = output.validity_probs
    valid_coverage = output.valid_coverage

    def scalar(value: Tensor | None) -> float | None:
        return None if value is None else float(value.mean().detach())

    if validity_probs is None:
        mean_p_valid = mean_p_matched = mean_p_no_object = None
        validity_auroc = validity_average_precision = None
    else:
        probabilities = validity_probs.float()
        matched = mask.to(probabilities.dtype)
        unmatched = (~mask).to(probabilities.dtype)
        mean_p_valid = scalar(probabilities)
        mean_p_matched = scalar(
            (probabilities * matched).sum(dim=-1) / matched.sum(dim=-1).clamp_min(1)
        )
        mean_p_no_object = scalar(
            (probabilities * unmatched).sum(dim=-1) / unmatched.sum(dim=-1).clamp_min(1)
        )
        validity_p_gap = (
            mean_p_matched - mean_p_no_object
            if mean_p_matched is not None and mean_p_no_object is not None
            else None
        )
        validity_auroc, validity_average_precision = _validity_ranking_metrics(
            probabilities, mask
        )
    if validity_probs is None:
        validity_p_gap = None

    invalid_gated_context_share = None
    if validity_probs is not None and output.validity_gating_mode == "raw_mass":
        raw_token_mass = raw_plan.sum(dim=1)
        raw_token_weights = raw_plan.transpose(1, 2) / raw_token_mass.unsqueeze(-1).clamp_min(
            1e-12
        )
        context_contribution = raw_token_weights * validity_probs.float().unsqueeze(1)
        invalid_contribution = context_contribution * (~mask).to(
            context_contribution.dtype
        ).unsqueeze(1)
        invalid_gated_context_share = float(
            invalid_contribution.sum().detach()
            / context_contribution.sum().detach().clamp_min(1e-12)
        )
    elif gated is not None:
        invalid_gated_context_share = scalar(gated["invalid_fusion"])

    valid_coverage_foreground = valid_coverage_background = None
    if valid_coverage is not None and token_owners is not None:
        foreground = token_owners.to(device=valid_coverage.device) >= 0
        coverage = valid_coverage.float()
        if bool(foreground.any()):
            valid_coverage_foreground = float(coverage[foreground].mean().detach())
        if bool((~foreground).any()):
            valid_coverage_background = float(coverage[~foreground].mean().detach())
    return {
        "transport_entropy": float(entropy.mean().detach()),
        "transport_entropy_nats": float(entropy_nats.mean().detach()),
        # Query-sized arrays were the source of the misleading long rows.
        "transport_query_mass": None,
        "invalid_query_transport_mass": scalar(raw["invalid_transport"]),
        "valid_query_transport_mass": scalar(raw["valid_transport"]),
        "fusion_query_mass": None,
        "invalid_query_fusion_mass": scalar(raw["invalid_fusion"]),
        "valid_query_fusion_mass": scalar(raw["valid_fusion"]),
        "gated_transport_total_mass": (
            scalar(gated["total_mass"]) if gated is not None else None
        ),
        "invalid_gated_query_transport_mass": (
            scalar(gated["invalid_transport"]) if gated is not None else None
        ),
        "valid_gated_query_transport_mass": (
            scalar(gated["valid_transport"]) if gated is not None else None
        ),
        "invalid_gated_fusion_mass": (
            scalar(gated["invalid_fusion"]) if gated is not None else None
        ),
        "valid_gated_fusion_mass": (
            scalar(gated["valid_fusion"]) if gated is not None else None
        ),
        "invalid_gated_context_share": invalid_gated_context_share,
        "valid_coverage": scalar(valid_coverage),
        "valid_coverage_foreground": valid_coverage_foreground,
        "valid_coverage_background": valid_coverage_background,
        "mean_p_valid": mean_p_valid,
        "mean_p_valid_matched": mean_p_matched,
        "mean_p_valid_no_object": mean_p_no_object,
        "validity_p_gap": validity_p_gap,
        "validity_auroc": validity_auroc,
        "validity_average_precision": validity_average_precision,
    }


def average_scalar_diagnostics(values: list[dict[str, Any]]) -> dict[str, Any]:
    """Average scalar transport diagnostics across one optimizer update."""

    if not values:
        return {}
    keys = set().union(*(value.keys() for value in values))
    result: dict[str, Any] = {}
    for key in sorted(keys):
        numeric = [
            float(value[key])
            for value in values
            if isinstance(value.get(key), (int, float))
            and not isinstance(value.get(key), bool)
            and math.isfinite(float(value[key]))
        ]
        result[key] = sum(numeric) / len(numeric) if numeric else None
    return result


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


def validity_assignment_gradient_diagnostics(
    assignment_loss: torch.Tensor,
    validity_logits: torch.Tensor | None,
    query_mask: torch.Tensor,
) -> dict[str, float]:
    """Measure whether gated assignment pushes positive and negative queries."""

    if validity_logits is None or not assignment_loss.requires_grad:
        return {
            "assignment_positive_query_grad": 0.0,
            "assignment_negative_query_grad": 0.0,
        }
    gradient = torch.autograd.grad(
        assignment_loss,
        validity_logits,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if gradient is None:
        return {
            "assignment_positive_query_grad": 0.0,
            "assignment_negative_query_grad": 0.0,
        }
    gradient = gradient.float()
    mask = query_mask.to(device=gradient.device, dtype=torch.bool)
    positive = gradient[mask]
    negative = gradient[~mask]

    def group_norm(values: torch.Tensor) -> float:
        if not values.numel():
            return 0.0
        return float(values.square().sum().sqrt().detach() / values.numel() ** 0.5)

    return {
        "assignment_positive_query_grad": group_norm(positive),
        "assignment_negative_query_grad": group_norm(negative),
    }


def region_generation_metrics(
    region_output: Any,
    record: dict[str, Any],
    *,
    eos_threshold: float = 0.5,
) -> dict[str, float | int | bool | None]:
    """Summarize AR region generation without using labels for decoding."""

    if region_output is None:
        return {
            "region_count": None,
            "region_pointer_reuse_rate": None,
            "region_spatial_duplicate_rate": None,
            "region_eos_hit": None,
            "region_limit_hit": None,
            "region_recall": None,
            "region_bbox_ap50": None,
            "region_bbox_precision50": None,
            "region_bbox_recall50": None,
            "region_reading_order_accuracy": None,
            "region_matched_count": None,
        }
    mask = region_output.selected_mask[0].bool()
    indices = region_output.selected_indices[0][mask]
    boxes = region_output.boxes[0][mask]
    duplicate_pointer = int(indices.numel() - indices.unique().numel())
    pointer_reuse_rate = duplicate_pointer / max(1, int(indices.numel()))
    if boxes.shape[0] > 1:
        overlaps = box_iou(boxes.unsqueeze(0), boxes.unsqueeze(0))[0]
        upper = torch.triu(torch.ones_like(overlaps), diagonal=1).bool()
        spatial_duplicate = int((overlaps > 0.8)[upper].sum())
        pair_count = int(upper.sum())
        spatial_duplicate_rate = spatial_duplicate / max(1, pair_count)
    else:
        spatial_duplicate_rate = 0.0
    eos_hit = bool((region_output.eos_logits[0].sigmoid() >= eos_threshold).any())
    region_limit_hit = not eos_hit
    target_regions = sorted(record["regions"], key=lambda item: int(item["reading_order"]))
    if target_regions and boxes.numel():
        target_boxes = torch.tensor(
            [region["bbox"] for region in target_regions],
            dtype=boxes.dtype,
            device=boxes.device,
        ).unsqueeze(0)
        overlaps = box_iou(boxes.unsqueeze(0), target_boxes)[0]
        target_recall = overlaps.max(dim=0).values >= 0.5
        recall = float(target_recall.float().mean().detach())

        # Greedy one-to-one matching at IoU=0.5 gives a stable page-level
        # detection score without hard-NMS.  Pointer confidence is used only
        # to order predictions for AP; it does not alter decoding.
        scores = region_output.pointer_logits[0][mask].softmax(dim=-1).max(dim=-1).values
        candidate_pairs = [
            (float(overlaps[pred_index, target_index]), pred_index, target_index)
            for pred_index in range(overlaps.shape[0])
            for target_index in range(overlaps.shape[1])
        ]
        matched_pred: set[int] = set()
        matched_target: set[int] = set()
        matches: list[tuple[int, int, float]] = []
        for iou, pred_index, target_index in sorted(candidate_pairs, reverse=True):
            if iou < 0.5 or pred_index in matched_pred or target_index in matched_target:
                continue
            matched_pred.add(pred_index)
            matched_target.add(target_index)
            matches.append((pred_index, target_index, iou))
        ranked = sorted(range(overlaps.shape[0]), key=lambda index: float(scores[index]), reverse=True)
        true_positive = [int(index in matched_pred) for index in ranked]
        if true_positive:
            cumulative = 0
            previous_recall = 0.0
            average_precision = 0.0
            target_count = len(target_regions)
            for rank, is_positive in enumerate(true_positive, start=1):
                cumulative += is_positive
                current_recall = cumulative / target_count
                precision = cumulative / rank
                average_precision += (current_recall - previous_recall) * precision
                previous_recall = current_recall
        else:
            average_precision = 0.0
        precision = len(matched_pred) / max(1, len(boxes))
        matched_order = [target_index for pred_index, target_index, _ in sorted(matches)]
        if len(matched_order) <= 1:
            order_accuracy = 1.0 if matched_order else 0.0
        else:
            pair_total = len(matched_order) * (len(matched_order) - 1) // 2
            pair_correct = sum(
                matched_order[left] < matched_order[right]
                for left in range(len(matched_order))
                for right in range(left + 1, len(matched_order))
            )
            order_accuracy = pair_correct / max(1, pair_total)
    else:
        recall = 0.0 if target_regions else None
        average_precision = 0.0 if target_regions else None
        precision = 0.0 if target_regions else None
        order_accuracy = 0.0 if target_regions else None
        matches = []
    return {
        "region_count": int(indices.numel()),
        "region_pointer_reuse_rate": pointer_reuse_rate,
        "region_spatial_duplicate_rate": spatial_duplicate_rate,
        "region_eos_hit": eos_hit,
        "region_limit_hit": region_limit_hit,
        "region_recall": recall,
        "region_bbox_ap50": average_precision,
        "region_bbox_precision50": precision,
        "region_bbox_recall50": recall,
        "region_reading_order_accuracy": order_accuracy,
        "region_matched_count": len(matches),
    }


def density_bucket_map(records: list[dict[str, Any]]) -> dict[str, str]:
    """Split a fixed validation set into deterministic sparse/normal/dense thirds."""

    ordered = sorted(
        range(len(records)),
        key=lambda index: (len(records[index].get("regions", [])), records[index]["page_id"]),
    )
    labels = ("sparse", "normal", "dense")
    result: dict[str, str] = {}
    for rank, index in enumerate(ordered):
        bucket_index = min(2, (3 * rank) // max(1, len(ordered)))
        result[str(records[index]["page_id"])] = labels[bucket_index]
    return result


def _token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value if item is not None}
    return {int(value)}


def matcher_churn_between_points(points: list[dict[str, Any]]) -> float | None:
    """Compare Hungarian query ownership on the same validation pages."""

    ordered = sorted(points, key=lambda point: int(point["step"]))
    churn_values: list[float] = []
    for previous, current in zip(ordered, ordered[1:]):
        previous_signatures = (previous.get("validation") or {}).get("matcher_signatures")
        current_signatures = (current.get("validation") or {}).get("matcher_signatures")
        if not isinstance(previous_signatures, dict) or not isinstance(current_signatures, dict):
            continue
        common_pages = set(previous_signatures) & set(current_signatures)
        if common_pages:
            changed = sum(
                previous_signatures[page_id] != current_signatures[page_id]
                for page_id in common_pages
            )
            churn_values.append(changed / len(common_pages))
    return sum(churn_values) / len(churn_values) if churn_values else None


def validity_mechanism_acceptance(points: list[dict[str, Any]]) -> dict[str, Any]:
    """Report the query-level gates without using them for checkpoint selection."""

    required_steps = (128, 256)
    by_step = {int(point["step"]): point for point in points}
    checks: dict[str, dict[str, Any]] = {}
    for step in required_steps:
        validation = (by_step.get(step) or {}).get("validation") or {}

        def finite_number(key: str) -> float | None:
            value = validation.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                return None
            return float(value)

        p_gap = finite_number("validity_p_gap")
        auroc = finite_number("validity_auroc")
        no_object = finite_number("mean_p_valid_no_object")
        matched = finite_number("mean_p_valid_matched")
        invalid_share = finite_number("invalid_gated_context_share")
        residual = finite_number("residual_relative_norm")
        checks[str(step)] = {
            "available": all(
                value is not None
                for value in (p_gap, auroc, no_object, matched, invalid_share, residual)
            ),
            "p_gap": p_gap,
            "p_gap_pass": p_gap is not None and p_gap >= 0.10,
            "auroc": auroc,
            "auroc_pass": auroc is not None and auroc >= 0.80,
            "mean_p_valid_no_object": no_object,
            "no_object_pass": no_object is not None and no_object < 0.10,
            "mean_p_valid_matched": matched,
            "matched_pass": matched is not None and matched > 0.40,
            "invalid_gated_context_share": invalid_share,
            "invalid_context_pass": invalid_share is not None and invalid_share < 0.50,
            "residual_relative_norm": residual,
            "residual_pass": residual is not None and residual <= 0.01,
        }
    available = all(check["available"] for check in checks.values())
    passed = available and all(
        all(
            value
            for key, value in check.items()
            if key.endswith("_pass")
        )
        for check in checks.values()
    )
    return {
        "available": available,
        "required_steps": list(required_steps),
        "checks": checks,
        "passed": passed,
    }


def eos_token_ids(model: Any, processor: Any) -> set[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    ids = _token_id_set(getattr(tokenizer, "eos_token_id", None))
    generation_config = getattr(model, "generation_config", None)
    ids.update(_token_id_set(getattr(generation_config, "eos_token_id", None)))
    return ids


def repeat_suppression_config(args: argparse.Namespace) -> RepeatSuppressionConfig:
    return RepeatSuppressionConfig(
        enabled=bool(getattr(args, "text_repeat_suppression", False)),
        unlikelihood_weight=float(getattr(args, "text_ul_weight", 0.1)),
        eos_weight=float(getattr(args, "text_eos_loss_weight", 0.05)),
        recent_window=int(getattr(args, "repeat_recent_window", 96)),
        min_cycle_length=int(getattr(args, "repeat_min_cycle_length", 8)),
        max_cycle_length=int(getattr(args, "repeat_max_cycle_length", 32)),
        cycle_repeats=int(getattr(args, "repeat_cycle_repeats", 3)),
        cycle_penalty=float(getattr(args, "repeat_cycle_penalty", 2.0)),
        force_eos_steps=int(getattr(args, "repeat_force_eos_steps", 16)),
        continuation_escape=bool(getattr(args, "continuation_escape", False)),
        escape_budget=int(getattr(args, "escape_budget", 16)),
        escape_clear_steps=int(getattr(args, "escape_clear_steps", 4)),
        escape_eos_suppression=float(getattr(args, "escape_eos_suppression", 1.0)),
        escape_eos_boost=float(getattr(args, "escape_eos_boost", 0.5)),
    )


def text_repeat_losses(
    args: argparse.Namespace,
    outputs: Any,
    labels: torch.Tensor,
    eos_ids: set[int],
) -> dict[str, torch.Tensor]:
    logits = getattr(outputs, "logits", None)
    if not getattr(args, "text_repeat_suppression", False) or logits is None:
        zero = outputs.loss.float() * 0.0
        return {"text_unlikelihood": zero, "text_eos": zero}
    config = repeat_suppression_config(args)
    return {
        "text_unlikelihood": unlikelihood_loss(
            logits,
            labels,
            min_cycle_length=config.min_cycle_length,
            max_cycle_length=config.max_cycle_length,
            cycle_repeats=config.cycle_repeats,
            recent_window=config.recent_window,
        ),
        "text_eos": eos_focus_loss(logits, labels, eos_ids),
    }


def text_repeat_activation_stats(
    args: argparse.Namespace,
    labels: torch.Tensor,
    eos_ids: set[int],
) -> tuple[float, float]:
    """Return active-token ratios for the optional UL and EOS branches."""

    valid = labels != -100
    valid_count = int(valid.sum().detach().item())
    if valid_count == 0 or not getattr(args, "text_repeat_suppression", False):
        return 0.0, 0.0
    config = repeat_suppression_config(args)
    repeated = repeated_cycle_positions(
        labels,
        min_cycle_length=config.min_cycle_length,
        max_cycle_length=config.max_cycle_length,
        cycle_repeats=config.cycle_repeats,
        recent_window=config.recent_window,
    ) & valid
    eos_mask = torch.zeros_like(valid)
    for eos_id in eos_ids:
        eos_mask |= labels == int(eos_id)
    return (
        float(repeated.sum().detach().item()) / valid_count,
        float((eos_mask & valid).sum().detach().item()) / valid_count,
    )


def natural_loop_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the real-rollout natural prediction loop configuration."""

    if getattr(args, "recovery_mode", "legacy") == "aligned_recovery_v1":
        return {**ALIGNED_CONFIG, "enabled": True, "interval": args.aligned_rollout_interval}

    return {
        "enabled": bool(getattr(args, "natural_loop_loss", False)),
        "weight": float(getattr(args, "natural_loop_weight", 0.05)),
        "recent_window": int(getattr(args, "natural_loop_recent_window", 96)),
        "min_cycle_length": int(getattr(args, "natural_loop_min_cycle_length", 8)),
        "max_cycle_length": int(getattr(args, "natural_loop_max_cycle_length", 32)),
        "cycle_repeats": int(getattr(args, "natural_loop_cycle_repeats", 3)),
        "max_new_tokens": int(getattr(args, "natural_loop_max_new_tokens", 768)),
        "continuation_horizon": int(
            getattr(args, "natural_loop_continuation_horizon", 16)
        ),
        "rollout_no_grad": True,
        "second_forward": True,
        "single_forward": False,
        "teacher_forced_detector": False,
        "synthetic_prefix": False,
        "inference_intervention": False,
    }


def loss_objective_config(args: argparse.Namespace) -> dict[str, Any]:
    """Describe the loss terms that can contribute to a training update."""

    extra_terms: list[str] = []
    if bool(getattr(args, "text_repeat_suppression", False)):
        extra_terms.append("L_text_repeat")
    if bool(getattr(args, "scheduled_sampling", False)):
        extra_terms.append("L_scheduled_prefix")
    if bool(getattr(args, "loop_escape_training", False)):
        extra_terms.append("L_loop_escape")
    if bool(getattr(args, "continuation_head", False)):
        extra_terms.append("L_continuation_head")
    if natural_loop_config(args).get("enabled") is True:
        extra_terms.append("L_natural_loop")
    return {
        "formula": (
            "L_official + auxiliary_weight * L_layout"
            if not extra_terms
            else "L_official + auxiliary_weight * L_layout + extra_terms"
        ),
        "official_loss": "outputs.loss",
        "official_weight": 1.0,
        "layout_weight": float(args.auxiliary_weight),
        "extra_terms": extra_terms,
    }


def _zero_natural_loop_rollout(inputs: dict[str, Any]) -> dict[str, Any]:
    """Create a DDP-safe zero rollout result for pages without a loop."""

    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    shifted_shape = (input_ids.shape[0], input_ids.shape[1] - 1)
    return {
        "prefix_inputs": {**inputs, "input_ids": input_ids.detach().clone()},
        "candidate_mask": torch.zeros(
            shifted_shape, dtype=torch.bool, device=input_ids.device
        ),
        "candidate_token_ids": torch.full(
            shifted_shape, -1, dtype=torch.long, device=input_ids.device
        ),
        "continuation_mask": torch.zeros(
            shifted_shape, dtype=torch.bool, device=input_ids.device
        ),
        "continuation_targets": labels[:, 1:].detach().clone(),
        "active_tokens": 0.0,
        "candidate_tokens": 0.0,
        "continuation_tokens": 0.0,
        "active_pages": 0.0,
        "detected_pages": 0.0,
        "cycle_tokens": 0.0,
        "valid_tokens": float((labels != -100).sum().detach().item()),
        "rollout_tokens": 0.0,
        "cycle_length": 0,
        "cycle_repeats": 0,
        "generation_length": 0,
    }


def collect_natural_loop_rollout(
    model: Any,
    args: argparse.Namespace,
    inputs: dict[str, Any],
    labels: torch.Tensor,
    eos_ids: set[int],
) -> dict[str, Any]:
    """Generate a real prefix and build the second-forward supervision masks.

    Generation is strictly no-grad and plain greedy decoding.  Only the
    resulting token IDs are copied into the second-forward input; the copied
    prefix is detached by construction.  A page without a detected cycle
    still returns full-shape zero masks so every DDP rank executes the same
    second-forward path when the option is enabled.
    """

    if labels.ndim != 2 or labels.shape != inputs["input_ids"].shape:
        raise ValueError("natural-loop labels and input_ids must have the same rank-2 shape")
    result = _zero_natural_loop_rollout(inputs)
    valid_positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    if not valid_positions.numel():
        return result

    config = natural_loop_config(args)
    prompt_length = int(valid_positions[0].item())
    target_capacity = int(valid_positions[-1].item()) - prompt_length + 1
    if target_capacity <= 0:
        return result

    generation_inputs = {
        key: value.detach() if isinstance(value, torch.Tensor) else value
        for key, value in inputs.items()
        if key != "labels"
    }
    # ``prepare_training_inputs`` contains prompt plus the full GT target.
    # Rollout must start from the prompt only; otherwise the supposedly free
    # trajectory would still be teacher-forced by the complete OCR target.
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "token_type_ids"):
        value = generation_inputs.get(key)
        if isinstance(value, torch.Tensor) and value.ndim >= 2:
            generation_inputs[key] = value[:, :prompt_length]
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config["max_new_tokens"],
        "do_sample": False,
        "use_cache": True,
    }
    if eos_ids:
        generation_kwargs["eos_token_id"] = sorted(eos_ids)
    previous_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    try:
        if hasattr(model, "config") and previous_use_cache is not None:
            model.config.use_cache = True
        with torch.no_grad():
            generated = model.generate(**generation_inputs, **generation_kwargs)
    finally:
        if hasattr(model, "config") and previous_use_cache is not None:
            model.config.use_cache = previous_use_cache
    if not isinstance(generated, torch.Tensor):
        generated = getattr(generated, "sequences", None)
    if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
        raise RuntimeError("natural-loop rollout did not return generated token sequences")
    if generated.shape[0] != inputs["input_ids"].shape[0]:
        raise RuntimeError("natural-loop rollout batch size does not match training inputs")

    generated_tokens = generated[0, prompt_length:].detach().to("cpu").tolist()
    result["generation_length"] = int(len(generated_tokens))
    result["rollout_tokens"] = float(len(generated_tokens))
    if not generated_tokens:
        return result
    detection_tokens = list(generated_tokens)
    if eos_ids:
        first_eos = next(
            (index for index, token in enumerate(detection_tokens) if int(token) in eos_ids),
            None,
        )
        if first_eos is not None:
            detection_tokens = detection_tokens[:first_eos]
    cycle = generated_cycle_window(
        detection_tokens,
        min_cycle_length=config["min_cycle_length"],
        max_cycle_length=config["max_cycle_length"],
        cycle_repeats=config["cycle_repeats"],
        recent_window=config["recent_window"],
    )
    if not cycle["detected"]:
        return result

    cycle_end = int(cycle["end"])
    cycle_length = int(cycle["length"])
    cycle_values = [int(value) for value in cycle["cycle"]]
    rollout_end = min(
        len(generated_tokens),
        target_capacity,
        cycle_end + config["continuation_horizon"],
    )
    if cycle_length <= 0 or rollout_end <= cycle_end:
        return result

    input_ids = inputs["input_ids"]
    modified_input_ids = input_ids.detach().clone()
    generated_prefix = torch.tensor(
        generated_tokens[:rollout_end], dtype=input_ids.dtype, device=input_ids.device
    )
    modified_input_ids[:, prompt_length : prompt_length + rollout_end] = generated_prefix
    result["prefix_inputs"] = {**inputs, "input_ids": modified_input_ids}
    result["detected_pages"] = 1.0
    result["cycle_length"] = cycle_length
    result["cycle_repeats"] = int(cycle["repeats"])
    result["cycle_tokens"] = float(cycle_length * int(cycle["repeats"]))

    candidate_mask = result["candidate_mask"]
    candidate_token_ids = result["candidate_token_ids"]
    continuation_mask = result["continuation_mask"]
    for generated_index in range(cycle_end, rollout_end):
        target_index = prompt_length + generated_index
        logit_index = target_index - 1
        if (
            target_index >= labels.shape[1]
            or logit_index < 0
            or logit_index >= input_ids.shape[1] - 1
        ):
            continue
        if int(labels[0, target_index].item()) < 0:
            continue
        continuation_mask[0, logit_index] = True
        expected = cycle_values[(generated_index - cycle_end) % cycle_length]
        if int(generated_tokens[generated_index]) == expected:
            candidate_mask[0, logit_index] = True
            candidate_token_ids[0, logit_index] = expected
    result["candidate_tokens"] = float(candidate_mask.sum().detach())
    result["continuation_tokens"] = float(continuation_mask.sum().detach())
    return result


def natural_loop_loss_for_rollout(
    outputs: Any,
    labels: torch.Tensor,
    rollout: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Compute live-logit loss for a previously collected rollout."""

    logits = getattr(outputs, "logits", None)
    if logits is None:
        zero = outputs.loss.float() * 0.0
        empty = torch.zeros((), device=zero.device, dtype=torch.float32)
        return {
            "loss": zero,
            "unlikelihood": zero,
            "continuation": zero,
            "active_tokens": empty,
            "active_pages": empty,
            "candidate_tokens": empty,
            "continuation_tokens": empty,
        }
    return natural_loop_rollout_loss(
        logits,
        labels,
        rollout["candidate_mask"],
        rollout["candidate_token_ids"],
        rollout["continuation_targets"],
        rollout["continuation_mask"],
    )


def optional_model_loss_components(outputs: Any) -> dict[str, float]:
    """Expose optional model-owned loss components without changing weighting."""

    components: dict[str, float] = {}
    for name in ("mtp_loss", "lm_loss", "language_model_loss", "ce_loss"):
        value = getattr(outputs, name, None)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            components[name] = float(value.detach().float())
    return components


def scheduled_sampling_probability(args: argparse.Namespace, step: int) -> float:
    """Return the bounded mixed-prefix probability for one optimizer step.

    The probability is deliberately zero through the first warm-up window and
    reaches at most ten percent.  This keeps the first forward teacher-forced
    while introducing only a small exposure-bias probe later in the run.
    """

    if not bool(getattr(args, "scheduled_sampling", False)):
        return 0.0
    warmup_steps = int(getattr(args, "scheduled_sampling_warmup_steps", 64))
    ramp_steps = int(getattr(args, "scheduled_sampling_ramp_steps", 64))
    maximum = float(getattr(args, "scheduled_sampling_max_probability", 0.1))
    if step <= warmup_steps:
        return 0.0
    if ramp_steps <= 0 or step >= warmup_steps + ramp_steps:
        return maximum
    return maximum * (step - warmup_steps) / ramp_steps


def token_weighted_loss_stats(outputs: Any, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Compute causal CE numerator and token count without materializing fp32 logits."""

    logits = getattr(outputs, "logits", None)
    if logits is None or logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("outputs.logits and labels must have compatible [batch, sequence] shapes")
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    valid = shift_labels != -100
    count = int(valid.sum().detach().item())
    if count == 0:
        return logits.sum() * 0.0, 0
    selected_logits = shift_logits[valid].float()
    selected_labels = shift_labels[valid].long()
    return F.cross_entropy(selected_logits, selected_labels, reduction="sum"), count


def build_mixed_prefix_inputs(
    inputs: dict[str, Any],
    outputs: Any,
    probability: float,
) -> tuple[dict[str, Any], int, int]:
    """Replace a small fraction of target-prefix tokens with detached argmaxes.

    Prompt/image tensors and the first OCR target token are never replaced.
    The returned counts are diagnostic only and the labels stay identical to
    the teacher-forced batch.
    """

    if probability <= 0.0:
        return dict(inputs), 0, 0
    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    logits = getattr(outputs, "logits", None)
    if logits is None or logits.shape[:2] != input_ids.shape:
        raise ValueError("model logits must align with input_ids for scheduled sampling")
    predicted = logits[:, :-1].detach().argmax(dim=-1)
    mixed_input_ids = input_ids.clone()
    replaced = 0
    eligible_count = 0
    for batch_index in range(input_ids.shape[0]):
        valid_positions = torch.nonzero(labels[batch_index] != -100, as_tuple=False).flatten()
        if not valid_positions.numel():
            continue
        first_target = int(valid_positions[0].item())
        last_target = int(valid_positions[-1].item())
        eligible = labels[batch_index] != -100
        eligible[: first_target + 1] = False
        eligible_positions = torch.nonzero(eligible, as_tuple=False).flatten()
        if eligible_positions.numel() <= 1:
            continue
        # A contiguous prefix corruption is closer to an autoregressive error
        # than independently replacing scattered teacher-forced tokens.
        span = min(8, int(eligible_positions.numel()))
        max_start = min(int(eligible_positions[-1]), last_target - span + 1)
        start = random.randint(int(eligible_positions[0]), max_start)
        choose_length = min(span, last_target - start + 1)
        eligible_count += choose_length
        choose = torch.zeros_like(eligible)
        if choose_length > 0 and random.random() < probability:
            choose[start : start + choose_length] = True
        chosen_positions = torch.nonzero(choose, as_tuple=False).flatten()
        if chosen_positions.numel():
            mixed_input_ids[batch_index, chosen_positions] = predicted[
                batch_index, chosen_positions - 1
            ]
            replaced += int(chosen_positions.numel())
    mixed_inputs = dict(inputs)
    mixed_inputs["input_ids"] = mixed_input_ids
    return mixed_inputs, replaced, eligible_count


def build_loop_escape_inputs(
    inputs: dict[str, Any],
    *,
    cycle_length: int = 8,
    escape_horizon: int = 8,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, int]:
    """Create a same-length prefix containing a random internal three-cycle.

    The labels remain the original OCR target.  Only the decoder prefix is
    corrupted, so the second forward teaches the model to recover the real
    continuation instead of choosing EOS as an escape.  Sampling the cycle
    after a valid prefix matches the arbitrary point at which free-running
    generation can enter a loop.  ``loop_mask`` marks the first continuation
    positions and ``negative_token_ids`` contains the token that would extend
    the synthetic cycle.
    """

    if cycle_length <= 0 or escape_horizon <= 0:
        raise ValueError("loop escape cycle length and horizon must be positive")
    input_ids = inputs["input_ids"]
    labels = inputs["labels"]
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("loop escape inputs and labels must be aligned rank-2 tensors")
    looped_input_ids = input_ids.clone()
    loop_mask = torch.zeros_like(labels, dtype=torch.bool)
    negative_token_ids = torch.full_like(labels, -1)
    active_positions = 0
    for batch_index in range(labels.shape[0]):
        valid_positions = torch.nonzero(labels[batch_index] != -100, as_tuple=False).flatten()
        if valid_positions.numel() < cycle_length * 3 + escape_horizon + 1:
            continue
        first_target = int(valid_positions[0].item())
        last_target = int(valid_positions[-1].item())
        # Sample an internal location so the training corruption matches the
        # arbitrary point at which a free-running decoder can enter a loop.
        min_source = first_target + cycle_length
        # Keep the recovery window before the final EOS target so the
        # non-EOS objective cannot accidentally train on the stop position.
        max_source = last_target - cycle_length * 3 - escape_horizon
        if max_source < min_source:
            continue
        source_start = random.randint(min_source, max_source)
        paste_start = source_start + cycle_length
        paste_end = paste_start + cycle_length * 2
        escape_start = paste_end
        escape_end = min(last_target + 1, escape_start + escape_horizon)
        if escape_end <= escape_start or paste_end >= input_ids.shape[1]:
            continue
        cycle = input_ids[batch_index, source_start : source_start + cycle_length].clone()
        looped_input_ids[batch_index, paste_start : paste_start + cycle_length] = cycle
        looped_input_ids[batch_index, paste_start + cycle_length : paste_end] = cycle
        span = escape_end - escape_start
        loop_mask[batch_index, escape_start:escape_end] = True
        negative_token_ids[batch_index, escape_start:escape_end] = cycle.repeat(
            (span + cycle_length - 1) // cycle_length
        )[:span]
        active_positions += span
    looped_inputs = dict(inputs)
    looped_inputs["input_ids"] = looped_input_ids
    return looped_inputs, loop_mask, negative_token_ids, active_positions


def prompt_target_audit(
    processor: Any,
    prompt_inputs: dict[str, Any],
    training_inputs: dict[str, Any],
    eos_ids: set[int],
) -> dict[str, Any]:
    """Audit the exact prompt/target boundary used by training and generation."""

    prompt_ids = prompt_inputs["input_ids"]
    full_ids = training_inputs["input_ids"]
    labels = training_inputs["labels"]
    prompt_length = int(prompt_ids.shape[1])
    prefix_match = bool(
        full_ids.shape[1] > prompt_length
        and torch.equal(full_ids[:, :prompt_length], prompt_ids)
    )
    target_positions = torch.nonzero(labels[0] != -100, as_tuple=False).flatten()
    target_count = int(target_positions.numel())
    eos_count = int(sum(int(token) in eos_ids for token in labels[0].tolist() if int(token) != -100))
    first_target_ids = (
        labels[0, target_positions[:8]].detach().cpu().tolist()
        if target_positions.numel()
        else []
    )
    try:
        first_target_text = processor.decode(first_target_ids, skip_special_tokens=False)
    except Exception:
        first_target_text = None
    return {
        "prompt_length": prompt_length,
        "full_length": int(full_ids.shape[1]),
        "target_token_count": target_count,
        "eos_label_count": eos_count,
        "prefix_match": prefix_match,
        "first_target_token_ids": first_target_ids,
        "first_target_text": first_target_text,
    }


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


def auxiliary_weight_at_step(
    step: int,
    *,
    start_weight: float,
    end_weight: float,
    ramp_steps: int,
) -> float:
    """Return the optimizer-step auxiliary-loss weight for a linear ramp."""

    if start_weight < 0.0 or end_weight < 0.0 or ramp_steps < 0:
        raise ValueError("auxiliary weights and ramp steps must be non-negative")
    if ramp_steps == 0 or start_weight == end_weight:
        return end_weight
    progress = min(max(step, 0), ramp_steps) / ramp_steps
    return start_weight + (end_weight - start_weight) * progress


def adapter_finite_report(adapter: torch.nn.Module) -> dict[str, Any]:
    adapter = unwrap_module(adapter)
    non_finite = [
        name
        for name, value in adapter.state_dict().items()
        if not bool(torch.isfinite(value).all())
    ]
    return {"parameters_finite": not non_finite, "non_finite_parameters": non_finite}


def clone_module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    module = unwrap_module(module)
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def module_state_matches(module: torch.nn.Module, expected: dict[str, torch.Tensor]) -> bool:
    module = unwrap_module(module)
    current = module.state_dict()
    return current.keys() == expected.keys() and all(
        torch.equal(current[key].detach().cpu(), expected[key]) for key in current
    )


def write_adapter_config(path: Path, bridge: LayoutAwarePatchMerger) -> None:
    write_json(path, asdict(unwrap_module(bridge.adapter).config))


def save_adapter_checkpoint(path: Path, bridge: LayoutAwarePatchMerger, step: int) -> dict[str, Any]:
    adapter = unwrap_module(bridge.adapter)
    report = {
        "step": step,
        "adapter_precision": bridge.adapter_precision,
        **adapter_finite_report(adapter),
    }
    if not report["parameters_finite"]:
        raise FloatingPointError(
            f"non-finite adapter parameters at checkpoint {step}: "
            f"{report['non_finite_parameters']}"
        )
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in adapter.state_dict().items()
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


def save_decoder_lora_checkpoint(path: Path, model: Any, step: int) -> dict[str, Any]:
    """Serialize only decoder LoRA tensors for a checkpoint."""

    state = lora_state_dict(model)
    report = {
        "step": step,
        "decoder_lora": decoder_lora_finite_report(model),
        "tensor_count": len(state),
    }
    if not state:
        return report
    if not report["decoder_lora"]["parameters_finite"]:
        raise FloatingPointError(
            f"non-finite decoder LoRA parameters at checkpoint {step}: "
            f"{report['decoder_lora']['non_finite_parameters']}"
        )
    save_file(state, path / "decoder_lora.safetensors")
    reloaded = load_file(str(path / "decoder_lora.safetensors"), device="cpu")
    non_finite_saved = [
        name for name, value in reloaded.items() if not bool(torch.isfinite(value).all())
    ]
    report["checkpoint_finite"] = not non_finite_saved
    report["non_finite_checkpoint_tensors"] = non_finite_saved
    if non_finite_saved:
        raise FloatingPointError(
            f"non-finite serialized decoder LoRA tensors at step {step}: {non_finite_saved}"
        )
    return report


def load_decoder_lora_checkpoint(path: Path, model: Any) -> dict[str, torch.Tensor]:
    """Reload a LoRA-only checkpoint after the same decoder modules are injected."""

    checkpoint_path = path / "decoder_lora.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"decoder LoRA checkpoint is missing: {checkpoint_path}")
    state = load_file(str(checkpoint_path), device="cpu")
    non_finite = [name for name, value in state.items() if not bool(torch.isfinite(value).all())]
    if non_finite:
        raise FloatingPointError(
            f"non-finite decoder LoRA checkpoint tensors in {checkpoint_path}: {non_finite}"
        )
    load_lora_state_dict(model, state)
    return state


def save_continuation_head_checkpoint(
    path: Path, continuation_head: torch.nn.Module, step: int
) -> dict[str, Any]:
    """Serialize the small loop continuation/stop head with a checkpoint."""

    module = unwrap_module(continuation_head)
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in module.state_dict().items()
    }
    non_finite = [name for name, value in state.items() if not bool(torch.isfinite(value).all())]
    report = {
        "step": step,
        "enabled": True,
        "tensor_count": len(state),
        "parameters_finite": not non_finite,
        "non_finite_parameters": non_finite,
    }
    if non_finite:
        raise FloatingPointError(f"non-finite continuation head at checkpoint {step}: {non_finite}")
    save_file(state, path / "continuation_head.safetensors")
    return report


def load_continuation_head_checkpoint(
    path: Path, continuation_head: torch.nn.Module
) -> dict[str, Tensor]:
    checkpoint_path = path / "continuation_head.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"continuation head checkpoint is missing: {checkpoint_path}")
    state = load_file(str(checkpoint_path), device="cpu")
    module = unwrap_module(continuation_head)
    expected = module.state_dict()
    if set(expected) != set(state):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(
            f"continuation head checkpoint keys do not match: missing={missing[:3]}, "
            f"unexpected={unexpected[:3]}"
        )
    module.load_state_dict(state)
    return state


def load_adapter_checkpoint(path: Path, bridge: LayoutAwarePatchMerger) -> dict[str, Any]:
    adapter = unwrap_module(bridge.adapter)
    config_path = path / "adapter_config.json"
    expected = asdict(adapter.config)
    if config_path.is_file():
        recorded = json.loads(config_path.read_text(encoding="utf-8"))
        # Checkpoints written before gate warm-start support implicitly used
        # the zero-scale identity initialization.
        recorded.setdefault("initial_residual_scale", 0.0)
        # Checkpoints written before validity gating did not serialize these
        # fields; preserve their raw-transport semantics when reloaded.
        recorded.setdefault("use_validity_head", False)
        recorded.setdefault("initial_valid_probability", 0.05)
        # The validity-gating fix is opt-in so old validity checkpoints retain
        # the historical token-wise re-normalization behavior.
        recorded.setdefault("validity_gating_mode", "legacy_normalized")
        recorded.setdefault("validity_use_transport_evidence", False)
        if recorded != expected:
            raise ValueError(
                f"adapter config mismatch for {path}: recorded={recorded}, expected={expected}"
            )
    elif adapter.config.max_residual_scale is not None:
        raise ValueError(
            "legacy checkpoint has no adapter_config.json; load it with "
            "max_residual_scale=None to preserve its original gate semantics"
        )
    state = load_file(str(path / "adapter.safetensors"), device="cpu")
    non_finite = [name for name, value in state.items() if not bool(torch.isfinite(value).all())]
    if non_finite:
        raise FloatingPointError(f"non-finite checkpoint tensors in {path}: {non_finite}")
    adapter.load_state_dict(state)
    return state


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str | None:
    """Return a file digest without loading the whole file into memory."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


def json_compatible(value: Any) -> Any:
    """Convert processor config values such as tuples/enums to JSON values."""

    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def processor_reproducibility_report(
    processor: Any,
    processor_mode: str,
    model_path: Path,
) -> dict[str, Any]:
    """Capture the preprocessing choice that affects frozen vision features."""

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise RuntimeError("AutoProcessor did not expose an image_processor")
    tracked_files = (
        "preprocessor_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "config.json",
    )
    backend = getattr(image_processor, "backend", None)
    backend_resolution = "image_processor.backend" if backend is not None else "not_exposed"
    if (
        backend is None
        and processor_mode == "fast"
        and type(image_processor).__name__.endswith("Fast")
    ):
        # Transformers fast image processors are torchvision-backed.  Some
        # model-specific classes do not expose the backend attribute, so keep
        # the inference explicit in metadata rather than recording ambiguity.
        backend = "torchvision"
        backend_resolution = "fast_processor_class"
    return {
        "requested_mode": processor_mode,
        "requested_use_fast": processor_mode == "fast",
        "processor_class": type(processor).__name__,
        "image_processor_class": type(image_processor).__name__,
        "backend": json_compatible(backend),
        "backend_resolution": backend_resolution,
        "image_processor_size": json_compatible(getattr(image_processor, "size", None)),
        "resource_hashes": {
            name: sha256_file(model_path / name)
            for name in tracked_files
        },
    }


def tensor_fingerprint(value: torch.Tensor) -> dict[str, Any]:
    """Hash tensor bytes and shape while supporting BF16 tensors."""

    detached = value.detach().contiguous().cpu()
    byte_view = detached.view(torch.uint8)
    digest = hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()
    return {
        "dtype": _dtype_name(detached.dtype),
        "shape": list(detached.shape),
        "sha256": digest,
    }


def processor_input_fingerprint(processor: Any, record: dict[str, Any]) -> dict[str, Any]:
    """Fingerprint one fixed-page processor output before model inference."""

    inputs = prepare_inference_inputs(processor, record, torch.device("cpu"))
    tensors = {
        key: tensor_fingerprint(value)
        for key, value in sorted(inputs.items())
        if isinstance(value, torch.Tensor)
    }
    identity = {
        "page_id": record["page_id"],
        "tensors": tensors,
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {**identity, "sha256": digest}


def configure_processor(processor: Any, max_pixels: int) -> None:
    size = dict(processor.image_processor.size)
    size["longest_edge"] = max_pixels
    processor.image_processor.size = size


def load_model(args: argparse.Namespace, device: torch.device) -> tuple[Any, Any, LayoutAwarePatchMerger]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        use_fast=args.processor_mode == "fast",
        local_files_only=True,
    )
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
        initial_residual_scale=args.initial_residual_scale,
        use_validity_head=getattr(args, "use_validity_head", False),
        initial_valid_probability=getattr(args, "initial_valid_probability", 0.05),
        validity_gating_mode=getattr(
            args, "validity_gating_mode", "legacy_normalized"
        ),
        validity_use_transport_evidence=getattr(
            args, "validity_use_transport_evidence", False
        ),
        adapter_precision=args.adapter_precision,
        region_autoregressive=getattr(args, "region_autoregressive", False),
        region_decoder_hidden_size=getattr(args, "region_decoder_hidden_size", 256),
        region_decoder_layers=getattr(args, "region_decoder_layers", 2),
        region_decoder_num_heads=getattr(args, "region_decoder_num_heads", 8),
        region_pointer_mask=getattr(args, "region_pointer_mask", True),
        region_spatial_penalty=getattr(args, "region_spatial_penalty", 4.0),
        region_spatial_iou_threshold=getattr(args, "region_spatial_iou_threshold", 0.8),
    )
    model.config.use_cache = False
    return model, processor, bridge


def train(
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    bridge: LayoutAwarePatchMerger,
    continuation_head: torch.nn.Module | None,
    records: list[dict[str, Any]],
    device: torch.device,
    distributed: DistributedInfo | None = None,
) -> dict[str, Any]:
    distributed = distributed or DistributedInfo(
        strategy="none", rank=0, local_rank=0, world_size=1
    )
    model_module = unwrap_module(model)
    adapter_module = unwrap_module(bridge.adapter)
    gate_parameter = getattr(adapter_module, "content_gate", None)
    adapter_parameters = tuple(adapter_module.parameters())
    lora_parameters = tuple(iter_lora_parameters(model_module))
    lora_parameter_ids = {id(parameter) for parameter in lora_parameters}
    optimizer_parameters = [
        parameter
        for parameter in adapter_parameters
        if parameter is not gate_parameter and id(parameter) not in lora_parameter_ids
    ]
    parameter_groups: list[dict[str, Any]] = []
    if optimizer_parameters:
        parameter_groups.append(
            {"params": optimizer_parameters, "weight_decay": 0.01, "group_name": "adapter"}
        )
    if gate_parameter is not None:
        # A frozen gate is held exactly at its configured warm-start value;
        # decoupled weight decay must not move it while its gradient is zeroed.
        parameter_groups.append(
            {"params": [gate_parameter], "weight_decay": 0.0, "group_name": "content_gate"}
        )
    decoder_learning_rate = getattr(args, "decoder_learning_rate", 1e-6)
    if lora_parameters:
        parameter_groups.append(
            {
                "params": list(lora_parameters),
                "weight_decay": 0.01,
                "lr": decoder_learning_rate,
                "group_name": "decoder_lora",
            }
        )
    continuation_parameters = (
        tuple(continuation_head.parameters()) if continuation_head is not None else ()
    )
    if continuation_parameters:
        parameter_groups.append(
            {
                "params": list(continuation_parameters),
                "weight_decay": 0.01,
                "lr": args.continuation_head_learning_rate,
                "group_name": "continuation_head",
            }
        )
    if not parameter_groups:
        raise RuntimeError("no trainable parameters were found for the optimization run")
    trainable_parameters = tuple(
        parameter for parameter in model_module.parameters() if parameter.requires_grad
    ) + tuple(parameter for parameter in continuation_parameters if parameter.requires_grad)
    parameter_report = trainable_parameter_report(model_module)
    lora_report = decoder_lora_finite_report(model_module)
    optimizer = torch.optim.AdamW(parameter_groups, lr=args.learning_rate)
    loss_weights = layout_loss_config(args.layout_loss_profile)
    lr_schedule_steps = args.lr_schedule_steps or args.max_steps
    accumulation_steps = args.gradient_accumulation_steps
    auxiliary_weight_start = (
        args.auxiliary_weight
        if args.auxiliary_weight_start is None
        else args.auxiliary_weight_start
    )
    rng = random.Random(args.seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    local_epoch_size = (
        math.ceil(
            len(records) / (distributed.world_size * accumulation_steps)
        )
        * accumulation_steps
    )
    steps_per_epoch = local_epoch_size // accumulation_steps
    rank_order: list[int] | None = None

    def record_for_micro(micro_index: int) -> dict[str, Any]:
        nonlocal rank_order
        if distributed.enabled:
            epoch = micro_index // local_epoch_size
            local_step = micro_index % local_epoch_size
            if rank_order is None or local_step == 0:
                rank_order = rank_epoch_indices(
                    len(records),
                    seed=args.seed,
                    epoch=epoch,
                    rank=distributed.rank,
                    world_size=distributed.world_size,
                    batch_size=accumulation_steps,
                )
            assert rank_order is not None
            return records[rank_order[local_step]]
        epoch, offset = divmod(micro_index, len(records))
        if offset == 0 and epoch > 0:
            rng.shuffle(order)
        return records[order[offset]]

    # Keep frozen backbone modules in evaluation mode.  LoRA uses gradients
    # through the frozen decoder activations, but the decoder itself remains in
    # eval mode so the adapter-only/LoRA comparison does not add dropout noise.
    model.eval()
    bridge.adapter.train()
    if continuation_head is not None:
        continuation_head.train()
    set_lora_modules_training(model_module, bool(lora_parameters))
    model_module.config.use_cache = False
    eos_ids = eos_token_ids(model_module, processor)
    region_enabled = bool(getattr(args, "region_autoregressive", False))
    debug_timing = os.environ.get("GLMOCR_DEBUG_TRAIN") == "1"

    def debug(message: str) -> None:
        if debug_timing:
            print(f"[glmocr-train-debug rank={distributed.rank}] {message}", flush=True)

    running: Counter[str] = Counter()
    ocr_ema_16: float | None = None
    token_weighted_sum_total = 0.0
    token_count_total = 0.0
    mixed_token_weighted_sum_total = 0.0
    mixed_token_count_total = 0.0
    started = time.time()
    checkpoint_steps: list[int] = []
    checkpoint_health: list[dict[str, Any]] = []
    diagnostic_train: dict[str, dict[str, Any]] = {}
    for step in range(1, args.max_steps + 1):
        learning_rate = learning_rate_at_step(
            min(step, lr_schedule_steps),
            peak_learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=lr_schedule_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        decoder_step_learning_rate = learning_rate_at_step(
            min(step, lr_schedule_steps),
            peak_learning_rate=decoder_learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=lr_schedule_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        for parameter_group in optimizer.param_groups:
            if parameter_group.get("group_name") == "decoder_lora":
                parameter_group["lr"] = decoder_step_learning_rate
            elif parameter_group.get("group_name") == "continuation_head":
                parameter_group["lr"] = learning_rate_at_step(
                    min(step, lr_schedule_steps),
                    peak_learning_rate=args.continuation_head_learning_rate,
                    warmup_steps=args.warmup_steps,
                    max_steps=lr_schedule_steps,
                    min_lr_ratio=args.min_lr_ratio,
                )
            elif parameter_group.get("group_name") != "content_gate":
                parameter_group["lr"] = learning_rate
        auxiliary_weight = auxiliary_weight_at_step(
            step,
            start_weight=auxiliary_weight_start,
            end_weight=args.auxiliary_weight,
            ramp_steps=args.auxiliary_ramp_steps,
        )
        gate_frozen = gate_parameter is not None and step <= args.gate_freeze_steps
        optimizer.zero_grad(set_to_none=True)
        diagnostic = step in args.diagnostic_steps
        micro_records: list[dict[str, Any]] = []
        micro_sums: Counter[str] = Counter()
        diagnostic_gradient_sums: Counter[str] = Counter()
        micro_transports: list[dict[str, Any]] = []
        matching_costs: list[float] = []
        matched_query_counts: list[int] = []
        token_counts: list[int] = []
        last_outputs = None
        last_targets = None
        last_matching_info = None
        last_auxiliary_losses = None
        last_loss = None
        for accumulation_index in range(accumulation_steps):
            micro_index = (step - 1) * accumulation_steps + accumulation_index
            record = record_for_micro(micro_index)
            micro_records.append(record)
            debug(f"step={step} micro={accumulation_index} record={record['page_id']} prepare")
            inputs = prepare_training_inputs(processor, record, device, eos_ids)
            bridge.set_grid_thw(inputs["image_grid_thw"])
            bridge.set_region_targets(
                region_decoder_targets(record, device, args.num_queries)
                if region_enabled
                else None
            )
            bridge.set_region_decode_controls()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
            debug(f"step={step} micro={accumulation_index} forward_done")
            if outputs.loss is None or bridge.last_output is None or bridge.last_patch_positions is None:
                raise RuntimeError("GLM-OCR forward did not produce OCR loss and layout state")
            if not all_finite(
                bool(torch.isfinite(outputs.loss).all()), distributed, device
            ):
                raise FloatingPointError(f"non-finite OCR loss at step {step}")
            tf_token_loss_sum, tf_token_count = token_weighted_loss_stats(
                outputs, inputs["labels"]
            )
            layout_output = bridge.last_output
            targets = layout_targets(record, bridge.last_patch_positions, args.num_queries)
            targets, matching_info = match_layout_targets(
                bridge.last_output,
                targets,
                assignment=args.query_assignment,
                return_info=True,
            )
            micro_transports.append(
                transport_diagnostics(
                    bridge.last_output,
                    targets["query_mask"],
                    targets["token_owners"],
                )
            )
            matched_query_counts.append(int(targets["query_mask"].sum()))
            token_counts.append(int(bridge.last_patch_positions.shape[1]))
            matching_costs.extend(
                cost
                for page_costs in matching_info["matching_costs"]
                for cost in page_costs
            )
            if auxiliary_weight > 0.0:
                auxiliary_losses = compute_layout_losses(
                    bridge.last_output, weights=loss_weights, **targets
                )
                if region_enabled and bridge.last_output.region_output is not None:
                    region_losses = compute_region_losses(
                        bridge.last_output.region_output,
                        region_decoder_targets(record, device, args.num_queries),
                    )
                    auxiliary_losses.update(region_losses)
                    auxiliary_losses["loss"] = auxiliary_losses["loss"] + region_losses["loss"]
                else:
                    zero = outputs.loss.float() * 0.0
                    auxiliary_losses.update({key: zero for key in REGION_LOSS_KEYS})
            else:
                zero = outputs.loss.float() * 0.0
                auxiliary_losses = {
                    key: zero for key in (*LAYOUT_LOSS_KEYS, *REGION_LOSS_KEYS, "loss")
                }
            auxiliary_loss = auxiliary_losses["loss"].float()
            if not all_finite(
                bool(torch.isfinite(auxiliary_loss).all()), distributed, device
            ):
                raise FloatingPointError(f"non-finite auxiliary loss at step {step}")
            repeat_losses = text_repeat_losses(args, outputs, inputs["labels"], eos_ids)
            repeat_activation_ratio, eos_activation_ratio = text_repeat_activation_stats(
                args, inputs["labels"], eos_ids
            )
            natural_rollout = _zero_natural_loop_rollout(inputs)
            natural_outputs = None
            natural_loop = {
                "loss": outputs.loss.float() * 0.0,
                "unlikelihood": outputs.loss.float() * 0.0,
                "continuation": outputs.loss.float() * 0.0,
                "active_tokens": torch.zeros((), device=device, dtype=torch.float32),
                "active_pages": torch.zeros((), device=device, dtype=torch.float32),
                "candidate_tokens": torch.zeros((), device=device, dtype=torch.float32),
                "continuation_tokens": torch.zeros((), device=device, dtype=torch.float32),
            }
            aligned_mode = getattr(args, "recovery_mode", "legacy") == "aligned_recovery_v1"
            aligned_rollout = None
            if bool(getattr(args, "natural_loop_loss", False)) and (
                not aligned_mode or step % args.aligned_rollout_interval == 0
            ):
                # The rollout must not see GT region targets.  Restore the
                # training bridge state before the differentiable second
                # forward below.
                bridge.set_grid_thw(inputs["image_grid_thw"])
                bridge.set_region_targets(None)
                bridge.set_region_decode_controls()
                if aligned_mode:
                    aligned_rollout = collect_aligned_rollout(model_module, inputs, eos_ids)
                    natural_rollout.update(
                        prefix_inputs=aligned_rollout["prefix_inputs"],
                        detected_pages=float(aligned_rollout["detected"]),
                        rollout_tokens=float(aligned_rollout["rollout_tokens"]),
                        cycle_tokens=float((aligned_rollout["cycle"].get("length") or 0)
                                           * (aligned_rollout["cycle"].get("repeats") or 0)),
                    )
                else:
                    natural_rollout = collect_natural_loop_rollout(
                        model_module, args, inputs, inputs["labels"], eos_ids,
                    )
                bridge.set_grid_thw(inputs["image_grid_thw"])
                bridge.set_region_targets(
                    region_decoder_targets(record, device, args.num_queries)
                    if region_enabled
                    else None
                )
                bridge.set_region_decode_controls()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    natural_outputs = model(**natural_rollout["prefix_inputs"])
                if natural_outputs.logits is None:
                    raise RuntimeError("natural-loop second forward did not produce OCR logits")
                if aligned_mode:
                    supervised = bool(
                        (aligned_rollout["prefix_inputs"]["labels"] != -100).any()
                    )
                    loss_finite = (
                        (not supervised)
                        or (
                            natural_outputs.loss is not None
                            and bool(torch.isfinite(natural_outputs.loss).all())
                        )
                    )
                else:
                    if natural_outputs.loss is None:
                        raise RuntimeError(
                            "natural-loop second forward did not produce OCR loss"
                        )
                    loss_finite = bool(torch.isfinite(natural_outputs.loss).all())
                if not all_finite(loss_finite, distributed, device):
                    raise FloatingPointError(
                        f"non-finite natural-loop OCR loss at step {step}"
                    )
                natural_loop = (
                    recovery_losses(natural_outputs.logits, aligned_rollout, eos_ids)
                    if aligned_mode else natural_loop_loss_for_rollout(
                        natural_outputs, inputs["labels"], natural_rollout,
                    )
                )
            debug(f"step={step} micro={accumulation_index} losses_done")
            repeat_loss = (
                float(getattr(args, "text_ul_weight", 0.1)) * repeat_losses["text_unlikelihood"]
                + float(getattr(args, "text_eos_loss_weight", 0.05)) * repeat_losses["text_eos"]
                if getattr(args, "text_repeat_suppression", False)
                else outputs.loss.float() * 0.0
            )
            scheduled_probability = scheduled_sampling_probability(args, step)
            mixed_prefix_loss = outputs.loss.float() * 0.0
            mixed_token_loss_sum = outputs.loss.float() * 0.0
            mixed_token_count = 0
            replaced_tokens = 0
            eligible_tokens = 0
            mixed_outputs = None
            loop_escape_mask = torch.zeros_like(inputs["labels"], dtype=torch.bool)
            loop_negative_ids = torch.full_like(inputs["labels"], -1)
            loop_escape_active = 0
            escape_losses = {
                "escape": outputs.loss.float() * 0.0,
                "margin": outputs.loss.float() * 0.0,
                "continue": outputs.loss.float() * 0.0,
                "active": outputs.loss.float() * 0.0,
            }
            head_loss = outputs.loss.float() * 0.0
            prefix_inputs = dict(inputs)
            if scheduled_probability > 0.0:
                prefix_inputs, replaced_tokens, eligible_tokens = build_mixed_prefix_inputs(
                    prefix_inputs, outputs, scheduled_probability
                )
            if getattr(args, "loop_escape_training", False):
                prefix_inputs, loop_escape_mask, loop_negative_ids, loop_escape_active = (
                    build_loop_escape_inputs(
                        prefix_inputs,
                        cycle_length=args.loop_escape_cycle_length,
                        escape_horizon=args.loop_escape_horizon,
                    )
                )
            # Every DDP rank must traverse the same second-forward path.  A
            # short page can have no eligible synthetic loop positions while
            # another rank does; gating this forward on the local active
            # count then leaves ranks with different autograd graphs and can
            # hang the first all-reduce.  The zero-active branch contributes
            # a zero escape loss but still keeps the model/head collectives
            # aligned.
            prefix_forward_enabled = scheduled_probability > 0.0 or getattr(
                args, "loop_escape_training", False
            )
            if prefix_forward_enabled:
                # The second forward is OCR-only.  Layout supervision and its
                # matching path remain tied to the first teacher-forced forward.
                bridge.set_grid_thw(inputs["image_grid_thw"])
                bridge.set_region_targets(
                    region_decoder_targets(record, device, args.num_queries)
                    if region_enabled
                    else None
                )
                bridge.set_region_decode_controls()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    mixed_outputs = model(**prefix_inputs)
                if mixed_outputs.loss is None:
                    raise RuntimeError("loop-escape forward did not produce OCR loss")
                if not all_finite(
                    bool(torch.isfinite(mixed_outputs.loss).all()), distributed, device
                ):
                    raise FloatingPointError(f"non-finite loop-escape OCR loss at step {step}")
                mixed_prefix_loss = mixed_outputs.loss.float()
                mixed_token_loss_sum, mixed_token_count = token_weighted_loss_stats(
                    mixed_outputs, prefix_inputs["labels"]
                )
                if getattr(args, "loop_escape_training", False):
                    escape_losses = cycle_escape_losses(
                        mixed_outputs.logits,
                        prefix_inputs["labels"],
                        loop_escape_mask,
                        loop_negative_ids,
                        eos_ids,
                        margin=args.loop_escape_margin,
                    )
            if continuation_head is not None:
                head_loss = continuation_head_loss(
                    continuation_head,
                    outputs.logits,
                    inputs["labels"],
                    eos_ids,
                )
                if mixed_outputs is not None:
                    head_loss = 0.5 * (
                        head_loss
                        + continuation_head_loss(
                            continuation_head,
                            mixed_outputs.logits,
                            prefix_inputs["labels"],
                            eos_ids,
                            loop_mask=loop_escape_mask,
                            negative_token_ids=loop_negative_ids,
                        )
                    )
            ocr_objective_loss = (
                outputs.loss.float()
                + scheduled_probability * (mixed_prefix_loss - outputs.loss.float())
            )
            escape_ramp = min(
                1.0,
                step / max(1, args.loop_escape_ramp_steps),
            )
            loop_loss = escape_ramp * (
                args.loop_escape_weight * escape_losses["escape"]
                + args.loop_escape_margin_weight * escape_losses["margin"]
                + args.loop_continue_weight * escape_losses["continue"]
            )
            natural_loop_weight = (
                float(getattr(args, "natural_loop_weight", 0.05))
                if bool(getattr(args, "natural_loop_loss", False))
                else 0.0
            )
            if aligned_mode:
                natural_loop_weight = min(step / ALIGNED_CONFIG["ramp_steps"], 1.0)
            natural_loop_term = natural_loop_weight * natural_loop["loss"]
            loss = (
                ocr_objective_loss
                + auxiliary_weight * auxiliary_loss
                + repeat_loss
                + loop_loss
                + natural_loop_term
                + args.continuation_head_weight * head_loss
            )
            if not all_finite(bool(torch.isfinite(loss).all()), distributed, device):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if diagnostic:
                component_losses = {
                    "ocr": outputs.loss.float(),
                    "mixed_prefix": mixed_prefix_loss,
                    "ocr_objective": ocr_objective_loss,
                    "loop_escape": escape_losses["escape"],
                    "loop_margin": escape_losses["margin"],
                    "loop_continue": escape_losses["continue"],
                    "natural_loop": natural_loop_term,
                    "continuation_head": head_loss,
                    **{
                        key: auxiliary_losses[key].float()
                        for key in (*LAYOUT_LOSS_KEYS, *REGION_LOSS_KEYS)
                    },
                    **{key: value.float() for key, value in repeat_losses.items()},
                    "total": loss,
                }
                for key, value in component_losses.items():
                    diagnostic_gradient_sums[key] += gradient_norm_for_loss(
                        value, trainable_parameters
                    )
                for key, value in validity_assignment_gradient_diagnostics(
                    auxiliary_losses["layout_assignment"],
                    layout_output.validity_logits,
                    targets["query_mask"],
                ).items():
                    diagnostic_gradient_sums[key] += value
            (loss / accumulation_steps).backward()
            debug(f"step={step} micro={accumulation_index} backward_done")
            micro_sums["ocr_loss"] += float(outputs.loss.detach())
            micro_sums["mixed_prefix_loss"] += float(mixed_prefix_loss.detach())
            micro_sums["ocr_objective_loss"] += float(ocr_objective_loss.detach())
            micro_sums["token_weighted_ocr_numerator"] += float(tf_token_loss_sum.detach())
            micro_sums["token_weighted_ocr_tokens"] += float(tf_token_count)
            micro_sums["mixed_prefix_numerator"] += float(mixed_token_loss_sum.detach())
            micro_sums["mixed_prefix_tokens"] += float(mixed_token_count)
            micro_sums["scheduled_sampling_replaced_tokens"] += float(replaced_tokens)
            micro_sums["scheduled_sampling_eligible_tokens"] += float(eligible_tokens)
            micro_sums["loop_escape_active"] += float(loop_escape_active)
            micro_sums["loop_escape_loss"] += float(escape_losses["escape"].detach())
            micro_sums["loop_margin_loss"] += float(escape_losses["margin"].detach())
            micro_sums["loop_continue_loss"] += float(escape_losses["continue"].detach())
            micro_sums["continuation_head_loss"] += float(head_loss.detach())
            micro_sums["loop_escape_ramp"] += escape_ramp
            micro_sums["auxiliary_loss"] += float(auxiliary_loss.detach())
            micro_sums["text_unlikelihood"] += float(repeat_losses["text_unlikelihood"].detach())
            micro_sums["text_eos"] += float(repeat_losses["text_eos"].detach())
            micro_sums["text_unlikelihood_activation_ratio"] += repeat_activation_ratio
            micro_sums["text_eos_activation_ratio"] += eos_activation_ratio
            micro_sums["official_base_loss"] += float(outputs.loss.detach())
            micro_sums["natural_loop_loss"] += float(natural_loop["loss"].detach())
            if aligned_rollout is not None:
                micro_sums["aligned_rollout_pages"] += 1
                micro_sums["aligned_accepted_pages"] += int(aligned_rollout["accepted"])
                micro_sums["aligned_end_tokens"] += float(natural_loop["end_tokens"])
                micro_sums["aligned_end_loss"] += float(natural_loop["end"].detach())
                if not aligned_rollout["accepted"]:
                    micro_sums["aligned_skip_" + aligned_rollout["reason"]] += 1
            micro_sums["natural_loop_term"] += float(natural_loop_term.detach())
            micro_sums["natural_loop_unlikelihood"] += float(
                natural_loop["unlikelihood"].detach()
            )
            micro_sums["natural_loop_continuation"] += float(
                natural_loop["continuation"].detach()
            )
            micro_sums["natural_loop_active_tokens"] += float(
                natural_loop["active_tokens"]
            )
            micro_sums["natural_loop_candidate_tokens"] += float(
                natural_loop["candidate_tokens"]
            )
            micro_sums["natural_loop_cycle_tokens"] += float(
                natural_rollout["cycle_tokens"]
            )
            micro_sums["natural_loop_continuation_tokens"] += float(
                natural_loop["continuation_tokens"]
            )
            micro_sums["natural_loop_active_pages"] += float(
                natural_loop["active_pages"]
            )
            micro_sums["natural_loop_detected_pages"] += float(
                natural_rollout["detected_pages"]
            )
            micro_sums["natural_loop_rollout_tokens"] += float(
                natural_rollout["rollout_tokens"]
            )
            micro_sums["natural_loop_valid_tokens"] += float(
                natural_rollout["valid_tokens"]
            )
            micro_sums["total_loss"] += float(loss.detach())
            for key in (*LAYOUT_LOSS_KEYS, *REGION_LOSS_KEYS):
                micro_sums[key] += float(auxiliary_losses[key].detach())
            last_outputs = outputs
            last_targets = targets
            last_matching_info = matching_info
            last_auxiliary_losses = auxiliary_losses
            last_loss = loss

        if gate_frozen and gate_parameter is not None and gate_parameter.grad is not None:
            gate_parameter.grad.zero_()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, args.max_grad_norm
        )
        if not all_finite(
            bool(torch.isfinite(gradient_norm).all()), distributed, device
        ):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        optimizer.step()
        debug(f"step={step} optimizer_done")
        finite_report = adapter_finite_report(adapter_module)
        lora_finite_report = decoder_lora_finite_report(model_module)
        continuation_finite = (
            adapter_finite_report(continuation_head)["parameters_finite"]
            if continuation_head is not None
            else True
        )
        parameters_finite = all_finite(
            finite_report["parameters_finite"]
            and lora_finite_report["parameters_finite"]
            and continuation_finite,
            distributed,
            device,
        )
        if not parameters_finite:
            raise FloatingPointError(
                f"non-finite trainable parameters at step {step}: adapter="
                f"{finite_report['non_finite_parameters']}, decoder_lora="
                f"{lora_finite_report['non_finite_parameters']}, continuation_head="
                f"{adapter_finite_report(continuation_head).get('non_finite_parameters', []) if continuation_head is not None else []}"
            )
        raw_gate = float(adapter_module.content_gate.detach())
        effective_scale = float(adapter_module.effective_residual_scale().detach())
        gradient_norms = {
            key: mean_scalar(value / accumulation_steps, distributed, device)
            for key, value in diagnostic_gradient_sums.items()
        }
        log_this_step = (
            step == 1
            or step % args.log_steps == 0
            or step == args.max_steps
            or diagnostic
        )
        global_token_loss_sum = sum_scalar(
            micro_sums["token_weighted_ocr_numerator"], distributed, device
        )
        global_token_count = sum_scalar(
            micro_sums["token_weighted_ocr_tokens"], distributed, device
        )
        token_weighted_ocr_loss = global_token_loss_sum / max(1.0, global_token_count)
        global_mixed_token_loss_sum = sum_scalar(
            micro_sums["mixed_prefix_numerator"], distributed, device
        )
        global_mixed_token_count = sum_scalar(
            micro_sums["mixed_prefix_tokens"], distributed, device
        )
        mixed_prefix_token_weighted_loss = global_mixed_token_loss_sum / max(
            1.0, global_mixed_token_count
        )
        global_natural_active_tokens = sum_scalar(
            micro_sums["natural_loop_active_tokens"], distributed, device
        )
        global_natural_candidate_tokens = sum_scalar(
            micro_sums["natural_loop_candidate_tokens"], distributed, device
        )
        global_natural_continuation_tokens = sum_scalar(
            micro_sums["natural_loop_continuation_tokens"], distributed, device
        )
        global_natural_detected_pages = sum_scalar(
            micro_sums["natural_loop_detected_pages"], distributed, device
        )
        global_natural_rollout_tokens = sum_scalar(
            micro_sums["natural_loop_rollout_tokens"], distributed, device
        )
        global_natural_cycle_tokens = sum_scalar(
            micro_sums["natural_loop_cycle_tokens"], distributed, device
        )
        global_natural_active_pages = sum_scalar(
            micro_sums["natural_loop_active_pages"], distributed, device
        )
        global_natural_valid_tokens = sum_scalar(
            micro_sums["natural_loop_valid_tokens"], distributed, device
        )
        natural_loop_activation_ratio = global_natural_active_tokens / max(
            1.0, global_natural_candidate_tokens
        )
        natural_loop_candidate_mismatch_ratio = global_natural_active_tokens / max(
            1.0, global_natural_candidate_tokens
        )
        token_weighted_sum_total += global_token_loss_sum
        token_count_total += global_token_count
        mixed_token_weighted_sum_total += global_mixed_token_loss_sum
        mixed_token_count_total += global_mixed_token_count
        teacher_forced_step_loss = mean_scalar(
            micro_sums["ocr_loss"] / accumulation_steps, distributed, device
        )
        if ocr_ema_16 is None:
            ocr_ema_16 = teacher_forced_step_loss
        else:
            ocr_ema_16 = (15.0 / 16.0) * ocr_ema_16 + (1.0 / 16.0) * teacher_forced_step_loss
        global_layout_components = (
            {
                key: mean_scalar(
                    micro_sums[key] / accumulation_steps,
                    distributed,
                    device,
                )
                for key in LAYOUT_LOSS_KEYS
                + REGION_LOSS_KEYS
                + TEXT_LOSS_KEYS
            }
            if log_this_step
            else None
        )
        metrics = {
            # ``ocr_loss`` remains the teacher-forced loss for compatibility;
            # the explicit names below prevent a current-batch endpoint from
            # being mistaken for a run-level trend.
            "ocr_loss": teacher_forced_step_loss,
            "last_batch_ocr_loss": teacher_forced_step_loss,
            "official_base_loss": mean_scalar(
                micro_sums["official_base_loss"] / accumulation_steps,
                distributed,
                device,
            ),
            "official_loss_components": optional_model_loss_components(last_outputs)
            if last_outputs is not None and log_this_step
            else {},
            "ema_ocr_loss_16": ocr_ema_16,
            "token_weighted_ocr_loss": token_weighted_ocr_loss,
            "mixed_prefix_token_weighted_loss": mixed_prefix_token_weighted_loss,
            "mixed_prefix_loss": mean_scalar(
                micro_sums["mixed_prefix_loss"] / accumulation_steps,
                distributed,
                device,
            ),
            "ocr_objective_loss": mean_scalar(
                micro_sums["ocr_objective_loss"] / accumulation_steps,
                distributed,
                device,
            ),
            "scheduled_sampling_probability": scheduled_sampling_probability(args, step),
            "scheduled_sampling_replaced_tokens": sum_scalar(
                micro_sums["scheduled_sampling_replaced_tokens"], distributed, device
            ),
            "scheduled_sampling_eligible_tokens": sum_scalar(
                micro_sums["scheduled_sampling_eligible_tokens"], distributed, device
            ),
            "loop_escape_active_tokens": sum_scalar(
                micro_sums["loop_escape_active"], distributed, device
            ),
            "loop_escape_loss": mean_scalar(
                micro_sums["loop_escape_loss"] / accumulation_steps, distributed, device
            ),
            "loop_margin_loss": mean_scalar(
                micro_sums["loop_margin_loss"] / accumulation_steps, distributed, device
            ),
            "loop_continue_loss": mean_scalar(
                micro_sums["loop_continue_loss"] / accumulation_steps, distributed, device
            ),
            "continuation_head_loss": mean_scalar(
                micro_sums["continuation_head_loss"] / accumulation_steps,
                distributed,
                device,
            ),
            "loop_escape_ramp": mean_scalar(
                micro_sums["loop_escape_ramp"] / accumulation_steps, distributed, device
            ),
            "auxiliary_loss": mean_scalar(
                micro_sums["auxiliary_loss"] / accumulation_steps, distributed, device
            ),
            "text_unlikelihood": mean_scalar(
                micro_sums["text_unlikelihood"] / accumulation_steps, distributed, device
            ),
            "text_eos": mean_scalar(
                micro_sums["text_eos"] / accumulation_steps, distributed, device
            ),
            "text_unlikelihood_activation_ratio": mean_scalar(
                micro_sums["text_unlikelihood_activation_ratio"] / accumulation_steps,
                distributed,
                device,
            ),
            "text_eos_activation_ratio": mean_scalar(
                micro_sums["text_eos_activation_ratio"] / accumulation_steps,
                distributed,
                device,
            ),
            "natural_loop_loss": mean_scalar(
                micro_sums["natural_loop_loss"] / accumulation_steps,
                distributed,
                device,
            ),
            "natural_loop_weighted_loss": mean_scalar(
                micro_sums["natural_loop_term"] / accumulation_steps,
                distributed,
                device,
            ),
            "natural_loop_unlikelihood": mean_scalar(
                micro_sums["natural_loop_unlikelihood"] / accumulation_steps,
                distributed,
                device,
            ),
            "natural_loop_continuation": mean_scalar(
                micro_sums["natural_loop_continuation"] / accumulation_steps,
                distributed,
                device,
            ),
            "natural_loop_active_tokens": global_natural_active_tokens,
            "natural_loop_candidate_tokens": global_natural_candidate_tokens,
            "natural_loop_continuation_tokens": global_natural_continuation_tokens,
            "natural_loop_active_pages": global_natural_active_pages,
            "natural_loop_detected_pages": global_natural_detected_pages,
            "natural_loop_rollout_tokens": global_natural_rollout_tokens,
            "natural_loop_valid_tokens": global_natural_valid_tokens,
            "natural_loop_activation_ratio": natural_loop_activation_ratio,
            "natural_loop_candidate_mismatch_ratio": natural_loop_candidate_mismatch_ratio,
            "total_loss": mean_scalar(micro_sums["total_loss"] / accumulation_steps, distributed, device),
            "gradient_norm": mean_scalar(float(gradient_norm.detach()), distributed, device),
            "learning_rate": learning_rate,
            "decoder_learning_rate": decoder_step_learning_rate,
            "auxiliary_weight": auxiliary_weight,
            "gate_frozen": gate_frozen,
            "gradient_accumulation_steps": accumulation_steps,
            "raw_content_gate": raw_gate,
            "effective_residual_scale": effective_scale,
            "parameters_finite": parameters_finite,
            "trainable_parameter_count": parameter_report["trainable_parameters"],
            "adapter_trainable_parameter_count": parameter_report[
                "adapter_trainable_parameters"
            ],
            "decoder_lora_trainable_parameter_count": parameter_report[
                "decoder_lora_trainable_parameters"
            ],
            "decoder_lora": lora_report,
            "layout_loss_components": global_layout_components,
        }
        if log_this_step and distributed.is_main:
            assert last_outputs is not None
            assert last_targets is not None
            assert last_auxiliary_losses is not None
            assert last_loss is not None
            transport = average_scalar_diagnostics(micro_transports)
            metrics.update(
                {
                    "step": step,
                    "page_id": record["page_id"],
                    "page_ids": [item["page_id"] for item in micro_records],
                    "optimizer_update": True,
                    "loss_components": {
                        "ocr": metrics["ocr_loss"],
                        "official_base": metrics["official_base_loss"],
                        "mixed_prefix": metrics["mixed_prefix_loss"],
                        "ocr_objective": metrics["ocr_objective_loss"],
                        **(global_layout_components or {}),
                        # These values were already reduced above on every
                        # rank.  Never launch an extra rank-0-only collective
                        # while building the JSON row.
                        "text_unlikelihood": metrics["text_unlikelihood"],
                        "text_eos": metrics["text_eos"],
                        "text_unlikelihood_activation_ratio": metrics[
                            "text_unlikelihood_activation_ratio"
                        ],
                        "text_eos_activation_ratio": metrics["text_eos_activation_ratio"],
                        "natural_loop": metrics["natural_loop_weighted_loss"],
                        "natural_loop_raw": metrics["natural_loop_loss"],
                        "natural_loop_unlikelihood": metrics["natural_loop_unlikelihood"],
                        "natural_loop_continuation": metrics["natural_loop_continuation"],
                        "loop_escape": metrics["loop_escape_loss"],
                        "loop_margin": metrics["loop_margin_loss"],
                        "loop_continue": metrics["loop_continue_loss"],
                        "continuation_head": metrics["continuation_head_loss"],
                        "total": metrics["total_loss"],
                    },
                    "loss_component_scope": "global_accumulated_micro_batches",
                    "ocr_loss_scope": "teacher_forced_current_optimizer_batch",
                    "official_base_loss_scope": "teacher_forced_current_optimizer_batch",
                    "token_weighted_ocr_loss_scope": "all_valid_ocr_tokens_in_current_optimizer_batch",
                    "loss_dtypes": {
                        "ocr": _dtype_name(last_outputs.loss.dtype),
                        "mixed_prefix": _dtype_name(
                            (mixed_outputs.loss if mixed_outputs is not None else last_outputs.loss).dtype
                        ),
                        **{
                            key: _dtype_name(last_auxiliary_losses[key].dtype)
                            for key in (*LAYOUT_LOSS_KEYS, *REGION_LOSS_KEYS)
                        },
                        "text_unlikelihood": _dtype_name(
                            repeat_losses["text_unlikelihood"].dtype
                        ),
                        "text_eos": _dtype_name(repeat_losses["text_eos"].dtype),
                        "natural_loop": _dtype_name(natural_loop["loss"].dtype),
                        "total": _dtype_name(last_loss.dtype),
                    },
                    "gradient_norms": gradient_norms,
                    "residual_relative_norm": residual_relative_norm(bridge),
                    "writeback_residual_relative_norm": writeback_residual_relative_norm(bridge),
                    "transport": transport,
                    "transport_scope": "accumulated_micro_batches",
                    "transport_micro_batch_count": len(micro_transports),
                    "adapter_dtypes": adapter_dtype_report(bridge),
                    "query_count": sum(matched_query_counts) / max(1, len(matched_query_counts)),
                    "token_count": sum(token_counts) / max(1, len(token_counts)),
                    "matcher": {
                        "matched_query_count": sum(matched_query_counts)
                        / max(1, len(matched_query_counts)),
                        "mean_cost": sum(matching_costs) / max(1, len(matching_costs))
                        if matching_costs
                        else None,
                    },
                }
            )
            if diagnostic:
                diagnostic_train[str(step)] = dict(metrics)
        if getattr(args, "recovery_mode", "legacy") == "aligned_recovery_v1":
            for key in ("aligned_rollout_pages", "aligned_accepted_pages", "aligned_end_tokens",
                        "aligned_end_loss", "aligned_skip_no_cycle", "aligned_skip_ambiguous_endpoint",
                        "aligned_skip_alignment_error", "aligned_skip_weak_tail",
                        "aligned_skip_legal_or_ambiguous_repeat", "aligned_skip_unsafe_end",
                        "aligned_skip_context_capacity"):
                metrics[key] = (mean_scalar(micro_sums[key], distributed, device)
                                if key.endswith("loss") else sum_scalar(micro_sums[key], distributed, device))
                running[key] += metrics[key]
            metrics["aligned_detection_rate"] = metrics["natural_loop_detected_pages"] / max(1, metrics["aligned_rollout_pages"])
            metrics["aligned_acceptance_rate"] = metrics["aligned_accepted_pages"] / max(1, metrics["aligned_rollout_pages"])
            # Log every sampled step, not only every 16th optimizer update.
            log_this_step = log_this_step or step % args.aligned_rollout_interval == 0
        for key in (
            "ocr_loss",
            "official_base_loss",
            "token_weighted_ocr_loss",
            "mixed_prefix_token_weighted_loss",
            "mixed_prefix_loss",
            "ocr_objective_loss",
            "ema_ocr_loss_16",
            "loop_escape_loss",
            "loop_margin_loss",
            "loop_continue_loss",
            "continuation_head_loss",
            "auxiliary_loss",
            "total_loss",
            "gradient_norm",
            "auxiliary_weight",
            "text_unlikelihood",
            "text_eos",
            "text_unlikelihood_activation_ratio",
            "text_eos_activation_ratio",
            "natural_loop_loss",
            "natural_loop_weighted_loss",
            "natural_loop_unlikelihood",
            "natural_loop_continuation",
            "natural_loop_active_tokens",
            "natural_loop_candidate_tokens",
            "natural_loop_continuation_tokens",
            "natural_loop_active_pages",
            "natural_loop_detected_pages",
            "natural_loop_rollout_tokens",
            "natural_loop_activation_ratio",
            "natural_loop_candidate_mismatch_ratio",
        ):
            if key in metrics:
                running[key] += metrics[key]
        if distributed.is_main and log_this_step:
            append_jsonl(
                args.output_dir / "train_metrics.jsonl",
                {"step": step, "page_id": record["page_id"], **metrics},
            )
        # Rank 0 performs the compact JSON/diagnostic write above.  Keep the
        # next page forward from overtaking it; otherwise whole-page token
        # imbalance can make one rank enter DDP's next broadcast while rank 0
        # is still finishing the current-step reductions.
        barrier(distributed)
        if (
            step == args.max_steps
            or (
                not getattr(args, "no_validation", False)
                and (
                    step % args.validation_interval == 0
                    or step in args.diagnostic_steps
                )
            )
        ):
            if distributed.is_main:
                checkpoint_dir = args.output_dir / f"checkpoint-{step}"
                checkpoint_dir.mkdir(exist_ok=False)
                health = save_adapter_checkpoint(checkpoint_dir, bridge, step)
                health["decoder_lora"] = save_decoder_lora_checkpoint(
                    checkpoint_dir, model_module, step
                )
                if continuation_head is not None:
                    health["continuation_head"] = save_continuation_head_checkpoint(
                        checkpoint_dir, continuation_head, step
                    )
                health.update(
                    {
                        "learning_rate": learning_rate,
                        "decoder_learning_rate": decoder_step_learning_rate,
                        "raw_content_gate": raw_gate,
                        "effective_residual_scale": effective_scale,
                    }
                )
                write_json(checkpoint_dir / "checkpoint_health.json", health)
                checkpoint_health.append(health)
            checkpoint_steps.append(step)
            barrier(distributed)
    elapsed = time.time() - started
    if distributed.is_main:
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in adapter_module.state_dict().items()
        }
        save_file(state, args.output_dir / "adapter.safetensors")
        lora_state = lora_state_dict(model_module)
        if lora_state:
            save_file(lora_state, args.output_dir / "decoder_lora.safetensors")
        if continuation_head is not None:
            save_continuation_head_checkpoint(args.output_dir, continuation_head, args.max_steps)
    barrier(distributed)
    return {
        "steps": args.max_steps,
        "seconds": elapsed,
        "steps_per_second": args.max_steps / max(elapsed, 1e-9),
        "checkpoint_steps": checkpoint_steps,
        "checkpoint_health": checkpoint_health,
        "diagnostic_train": diagnostic_train,
        "lr_schedule_steps": lr_schedule_steps,
        "auxiliary_weight_start": auxiliary_weight_start,
        "auxiliary_weight_end": args.auxiliary_weight,
        "auxiliary_weight_ramp_steps": args.auxiliary_ramp_steps,
        "gate_freeze_steps": args.gate_freeze_steps,
        "gradient_accumulation_steps": accumulation_steps,
        "initial_residual_scale": args.initial_residual_scale,
        "decoder_adaptation": getattr(args, "decoder_adaptation", "frozen"),
        "decoder_learning_rate": decoder_learning_rate,
        "official_base_loss": "outputs.loss",
        "loss_objective": loss_objective_config(args),
        "natural_loop": natural_loop_config(args),
        "scheduled_sampling": {
            "enabled": bool(getattr(args, "scheduled_sampling", False)),
            "warmup_steps": int(getattr(args, "scheduled_sampling_warmup_steps", 64)),
            "ramp_steps": int(getattr(args, "scheduled_sampling_ramp_steps", 64)),
            "max_probability": float(
                getattr(args, "scheduled_sampling_max_probability", 0.1)
            ),
        },
        "loop_escape": {
            "enabled": bool(getattr(args, "loop_escape_training", False)),
            "cycle_length": int(getattr(args, "loop_escape_cycle_length", 8)),
            "horizon": int(getattr(args, "loop_escape_horizon", 8)),
            "ramp_steps": int(getattr(args, "loop_escape_ramp_steps", 128)),
            "weight": float(getattr(args, "loop_escape_weight", 0.1)),
            "margin_weight": float(getattr(args, "loop_escape_margin_weight", 0.05)),
            "continue_weight": float(getattr(args, "loop_continue_weight", 0.05)),
        },
        "continuation_head": {
            "enabled": continuation_head is not None,
            "weight": float(getattr(args, "continuation_head_weight", 0.05)),
            "learning_rate": float(
                getattr(args, "continuation_head_learning_rate", 5e-4)
            ),
        },
        "token_weighted_ocr_loss": token_weighted_sum_total / max(1.0, token_count_total),
        "mixed_prefix_token_weighted_loss": mixed_token_weighted_sum_total
        / max(1.0, mixed_token_count_total),
        "final_decoder_learning_rate": decoder_step_learning_rate,
        "trainable_parameter_report": parameter_report,
        "decoder_lora": decoder_lora_finite_report(model_module),
        "continuation_head_parameters_finite": (
            adapter_finite_report(continuation_head)["parameters_finite"]
            if continuation_head is not None
            else True
        ),
        "final_learning_rate": learning_rate_at_step(
            min(args.max_steps, lr_schedule_steps),
            peak_learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            max_steps=lr_schedule_steps,
            min_lr_ratio=args.min_lr_ratio,
        ),
        "final_raw_content_gate": float(adapter_module.content_gate.detach()),
        "final_effective_residual_scale": float(
            adapter_module.effective_residual_scale().detach()
        ),
        "steps_per_epoch": steps_per_epoch,
        "world_size": distributed.world_size,
        "global_batch_size": distributed.world_size * args.per_device_batch_size,
        "effective_global_batch_size": (
            distributed.world_size * args.per_device_batch_size * accumulation_steps
        ),
        **{f"mean_{key}": value / args.max_steps for key, value in running.items()},
    }


@torch.inference_mode()
def evaluate(
    args: argparse.Namespace,
    model: Any,
    processor: Any,
    bridge: LayoutAwarePatchMerger,
    continuation_head: torch.nn.Module | None,
    validation_records: list[dict[str, Any]],
    train_records: list[dict[str, Any]],
    device: torch.device,
    output_dir: Path | None = None,
    split_name: str = "validation",
) -> dict[str, Any]:
    model_module = unwrap_module(model)
    model_module.eval()
    bridge.adapter.eval()
    if continuation_head is not None:
        continuation_head.eval()
    model_module.config.use_cache = True
    predictions_path = (output_dir or args.output_dir) / f"{split_name}_predictions.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    pairs: list[tuple[str, str]] = []
    box_errors: list[float] = []
    direction_correct = 0
    direction_total = 0
    generation_limit_hits = 0
    generation_lengths: list[int] = []
    generation_eos_hits = 0
    repeated_trigram_rates: list[float] = []
    loop_detected_pages = 0
    loop_escape_successes = 0
    loop_post_eos = 0
    loop_early_eos = 0
    eos_ids = eos_token_ids(model_module, processor)
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
    gated_transport_total_masses: list[float] = []
    invalid_gated_transport_masses: list[float] = []
    valid_gated_transport_masses: list[float] = []
    invalid_gated_fusion_masses: list[float] = []
    valid_gated_fusion_masses: list[float] = []
    valid_coverages: list[float] = []
    mean_p_valid_values: list[float] = []
    mean_p_valid_matched_values: list[float] = []
    mean_p_valid_no_object_values: list[float] = []
    validity_p_gaps: list[float] = []
    validity_aurocs: list[float] = []
    validity_average_precisions: list[float] = []
    invalid_gated_context_shares: list[float] = []
    valid_coverage_foreground_values: list[float] = []
    valid_coverage_background_values: list[float] = []
    annotated_query_counts: list[int] = []
    matcher_signatures: dict[str, str] = {}
    matcher_costs: list[float] = []
    dtype_signatures: set[str] = set()
    layout_loss_dtype_signatures: set[str] = set()
    teacher_forced_ocr_losses: list[float] = []
    teacher_forced_layout_loss_sums = {key: 0.0 for key in LAYOUT_LOSS_KEYS}
    teacher_forced_ocr_dtypes: set[str] = set()
    teacher_forced_target_tokens = 0
    teacher_forced_eos_labels = 0
    teacher_forced_eos_pages = 0
    prompt_prefix_mismatches = 0
    prompt_lengths: list[int] = []
    full_lengths: list[int] = []
    first_target_audit: dict[str, Any] | None = None
    collect_teacher_forcing = bool(args.diagnostic_steps) or bool(args.audit_prompt_prefix)
    region_enabled = bool(getattr(args, "region_autoregressive", False))
    repeat_config = repeat_suppression_config(args)
    if bool(getattr(args, "eval_disable_repeat_guard", False)):
        repeat_config = replace(repeat_config, enabled=False)
    region_counts: list[int] = []
    region_pointer_reuse_rates: list[float] = []
    region_spatial_duplicate_rates: list[float] = []
    region_eos_hits = 0
    region_limit_hits = 0
    region_recalls: list[float] = []
    region_bbox_aps: list[float] = []
    region_bbox_precisions: list[float] = []
    region_bbox_recalls: list[float] = []
    region_order_accuracies: list[float] = []
    density_buckets = density_bucket_map(validation_records)
    stratified_rows: dict[str, list[tuple[str, str]]] = {
        "sparse": [],
        "normal": [],
        "dense": [],
    }
    stratified_region_rows: dict[str, list[dict[str, Any]]] = {
        "sparse": [],
        "normal": [],
        "dense": [],
    }
    started = time.time()
    for record in validation_records:
        inputs = prepare_inference_inputs(processor, record, device)
        bridge.set_grid_thw(inputs["image_grid_thw"])
        bridge.set_region_decode_controls(
            pointer_mask=(getattr(args, "region_pointer_mask", True) if region_enabled else None),
            spatial_penalty=(
                bool(getattr(args, "region_spatial_penalty", 0.0) > 0.0)
                if region_enabled
                else None
            ),
        )
        if collect_teacher_forcing:
            bridge.set_region_targets(
                region_decoder_targets(record, device, args.num_queries)
                if region_enabled
                else None
            )
            teacher_inputs = prepare_training_inputs(processor, record, device, eos_ids)
            audit = prompt_target_audit(processor, inputs, teacher_inputs, eos_ids)
            teacher_forced_target_tokens += int(audit["target_token_count"])
            teacher_forced_eos_labels += int(audit["eos_label_count"])
            teacher_forced_eos_pages += int(audit["eos_label_count"] > 0)
            prompt_prefix_mismatches += int(not audit["prefix_match"])
            prompt_lengths.append(int(audit["prompt_length"]))
            full_lengths.append(int(audit["full_length"]))
            if first_target_audit is None:
                first_target_audit = {
                    "page_id": record["page_id"],
                    "first_target_token_ids": audit["first_target_token_ids"],
                    "first_target_text": audit["first_target_text"],
                }
            bridge.set_grid_thw(teacher_inputs["image_grid_thw"])
            teacher_outputs = model_module(**teacher_inputs)
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
        bridge.set_region_targets(None)
        prompt_length = inputs["input_ids"].shape[1]
        generated = generate_with_loop_recovery(
            model_module,
            inputs,
            prompt_length=prompt_length,
            eos_token_ids=eos_ids,
            config=repeat_config,
            max_new_tokens=args.max_eval_new_tokens,
            mode=args.generation_mode,
            continuation_head=(
                unwrap_module(continuation_head)
                if continuation_head is not None
                else None
            ),
        )
        generated_tokens = generated[0, prompt_length:]
        generation_length = int(generated_tokens.shape[0])
        generation_lengths.append(generation_length)
        eos_hit = bool(eos_ids and any(int(token) in eos_ids for token in generated_tokens.tolist()))
        if eos_hit:
            generation_eos_hits += 1
        prediction = processor.decode(generated_tokens, skip_special_tokens=True)
        prediction_repeated_trigram_rate = repeated_trigram_rate(prediction)
        loop_continuation = loop_continuation_diagnostics(
            generated_tokens.tolist(),
            eos_ids,
            recent_window=repeat_config.recent_window,
            min_cycle_length=repeat_config.min_cycle_length,
            max_cycle_length=repeat_config.max_cycle_length,
            cycle_repeats=repeat_config.cycle_repeats,
        )
        loop_detected_pages += int(bool(loop_continuation["loop_detected"]))
        loop_escape_successes += int(
            bool(loop_continuation["loop_detected"])
            and int(loop_continuation["continued_non_cycle_tokens"]) >= 4
        )
        loop_post_eos += int(bool(loop_continuation["post_loop_eos"]))
        loop_early_eos += int(bool(loop_continuation["post_loop_early_eos"]))
        prediction_repeat_diagnostics = repetition_diagnostics(
            prediction,
            recent_window=repeat_config.recent_window,
            min_cycle_length=repeat_config.min_cycle_length,
            max_cycle_length=repeat_config.max_cycle_length,
            cycle_repeats=repeat_config.cycle_repeats,
        )
        repeated_trigram_rates.append(prediction_repeated_trigram_rate)
        generation_limit_hit = generation_length >= args.max_eval_new_tokens
        if generation_limit_hit:
            generation_limit_hits += 1
        pairs.append((record["page_text"], prediction))
        bucket = density_buckets[str(record["page_id"])]
        stratified_rows[bucket].append((record["page_text"], prediction))
        if bridge.last_output is None or bridge.last_patch_positions is None:
            raise RuntimeError("generation did not retain layout state")
        region_metrics = region_generation_metrics(
            bridge.last_output.region_output, record
        )
        if region_metrics["region_count"] is not None:
            region_counts.append(int(region_metrics["region_count"]))
            region_pointer_reuse_rates.append(float(region_metrics["region_pointer_reuse_rate"]))
            region_spatial_duplicate_rates.append(
                float(region_metrics["region_spatial_duplicate_rate"])
            )
            region_eos_hits += int(bool(region_metrics["region_eos_hit"]))
            region_limit_hits += int(bool(region_metrics["region_limit_hit"]))
            if region_metrics["region_recall"] is not None:
                region_recalls.append(float(region_metrics["region_recall"]))
            for metric_name, values in (
                ("region_bbox_ap50", region_bbox_aps),
                ("region_bbox_precision50", region_bbox_precisions),
                ("region_bbox_recall50", region_bbox_recalls),
                ("region_reading_order_accuracy", region_order_accuracies),
            ):
                value = region_metrics.get(metric_name)
                if value is not None:
                    values.append(float(value))
            stratified_region_rows[bucket].append(region_metrics)
        targets = layout_targets(record, bridge.last_patch_positions, args.num_queries)
        targets, matching_info = match_layout_targets(
            bridge.last_output,
            targets,
            assignment=args.query_assignment,
            return_info=True,
        )
        matched_pairs = matching_info["matched_pairs"]
        matcher_signature_payload = json.dumps(
            matched_pairs,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        matcher_signatures[record["page_id"]] = hashlib.sha256(
            matcher_signature_payload
        ).hexdigest()
        matcher_cost_values = [
            cost for page_costs in matching_info["matching_costs"] for cost in page_costs
        ]
        matcher_costs.extend(matcher_cost_values)
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
        transport = transport_diagnostics(
            bridge.last_output,
            targets["query_mask"],
            targets["token_owners"],
        )
        if transport["transport_entropy"] is not None:
            transport_entropies.append(float(transport["transport_entropy"]))
            transport_entropy_nats.append(float(transport["transport_entropy_nats"]))
            if transport["transport_query_mass"] is not None:
                transport_query_masses.append(transport["transport_query_mass"])
            invalid_transport_masses.append(float(transport["invalid_query_transport_mass"]))
            valid_transport_masses.append(float(transport["valid_query_transport_mass"]))
            if transport["fusion_query_mass"] is not None:
                fusion_query_masses.append(transport["fusion_query_mass"])
            invalid_query_masses.append(float(transport["invalid_query_fusion_mass"]))
            valid_query_masses.append(float(transport["valid_query_fusion_mass"]))
            for key, values in (
                ("gated_transport_total_mass", gated_transport_total_masses),
                ("invalid_gated_query_transport_mass", invalid_gated_transport_masses),
                ("valid_gated_query_transport_mass", valid_gated_transport_masses),
                ("invalid_gated_fusion_mass", invalid_gated_fusion_masses),
                ("valid_gated_fusion_mass", valid_gated_fusion_masses),
                ("valid_coverage", valid_coverages),
                ("mean_p_valid", mean_p_valid_values),
                ("mean_p_valid_matched", mean_p_valid_matched_values),
                ("mean_p_valid_no_object", mean_p_valid_no_object_values),
                ("validity_p_gap", validity_p_gaps),
                ("validity_auroc", validity_aurocs),
                ("validity_average_precision", validity_average_precisions),
                ("invalid_gated_context_share", invalid_gated_context_shares),
                ("valid_coverage_foreground", valid_coverage_foreground_values),
                ("valid_coverage_background", valid_coverage_background_values),
            ):
                if transport[key] is not None:
                    values.append(float(transport[key]))
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
                "density_bucket": bucket,
                "reference": record["page_text"],
                "prediction": prediction,
                "generation_length": generation_length,
                "generation_eos_hit": eos_hit if eos_ids else None,
                "generation_limit_hit": generation_limit_hit,
                "repeated_trigram_rate": prediction_repeated_trigram_rate,
                "repeated_cycle_detected": prediction_repeat_diagnostics[
                    "repeated_cycle_detected"
                ],
                "repeated_cycle_rate": prediction_repeat_diagnostics["repeated_cycle_rate"],
                "loop_continuation": loop_continuation,
                "region_metrics": region_metrics,
            },
        )
    train_counts = Counter(
        character
        for record in train_records
        for character in record["page_text"]
        if not character.isspace()
    )
    metrics = aggregate_ocr_metrics(pairs, train_counts)

    def mean_region_metric(rows: list[dict[str, Any]], key: str) -> float | None:
        values = [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), (int, float))
        ]
        return sum(values) / len(values) if values else None

    stratified_metrics: dict[str, Any] = {}
    for bucket in ("sparse", "normal", "dense"):
        region_rows = stratified_region_rows[bucket]
        stratified_metrics[bucket] = {
            "pages": len(stratified_rows[bucket]),
            "ocr": aggregate_ocr_metrics(stratified_rows[bucket], train_counts),
            "region": {
                key: mean_region_metric(region_rows, key)
                for key in (
                    "region_count",
                    "region_pointer_reuse_rate",
                    "region_spatial_duplicate_rate",
                    "region_bbox_ap50",
                    "region_bbox_precision50",
                    "region_bbox_recall50",
                    "region_reading_order_accuracy",
                    "region_eos_hit",
                    "region_limit_hit",
                    "region_recall",
                )
            },
        }
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
            "repeated_trigram_rate": sum(repeated_trigram_rates)
            / max(1, len(repeated_trigram_rates)),
            "repeated_cycle_page_rate": sum(
                bool(
                    repetition_diagnostics(
                        prediction,
                        recent_window=repeat_config.recent_window,
                        min_cycle_length=repeat_config.min_cycle_length,
                        max_cycle_length=repeat_config.max_cycle_length,
                        cycle_repeats=repeat_config.cycle_repeats,
                    )["repeated_cycle_detected"]
                )
                for _, prediction in pairs
            ) / max(1, len(pairs)),
            "repeated_cycle_rate": sum(
                float(
                    repetition_diagnostics(
                        prediction,
                        recent_window=repeat_config.recent_window,
                        min_cycle_length=repeat_config.min_cycle_length,
                        max_cycle_length=repeat_config.max_cycle_length,
                        cycle_repeats=repeat_config.cycle_repeats,
                    )["repeated_cycle_rate"]
                )
                for _, prediction in pairs
            ) / max(1, len(pairs)),
            "loop_detected_page_rate": loop_detected_pages / max(1, len(pairs)),
            "loop_escape_success_rate": loop_escape_successes / max(1, loop_detected_pages),
            "loop_post_eos_rate": loop_post_eos / max(1, loop_detected_pages),
            "loop_early_eos_rate": loop_early_eos / max(1, loop_detected_pages),
            "region_autoregressive": region_enabled,
            "region_count_mean": sum(region_counts) / max(1, len(region_counts))
            if region_counts
            else None,
            "region_pointer_reuse_rate": sum(region_pointer_reuse_rates)
            / max(1, len(region_pointer_reuse_rates))
            if region_pointer_reuse_rates
            else None,
            "region_spatial_duplicate_rate": sum(region_spatial_duplicate_rates)
            / max(1, len(region_spatial_duplicate_rates))
            if region_spatial_duplicate_rates
            else None,
            "region_eos_hit_rate": region_eos_hits / max(1, len(region_counts))
            if region_counts
            else None,
            "region_limit_hit_rate": region_limit_hits / max(1, len(region_counts))
            if region_counts
            else None,
            "region_recall": sum(region_recalls) / max(1, len(region_recalls))
            if region_recalls
            else None,
            "region_bbox_ap50": sum(region_bbox_aps) / max(1, len(region_bbox_aps))
            if region_bbox_aps
            else None,
            "region_bbox_precision50": sum(region_bbox_precisions)
            / max(1, len(region_bbox_precisions))
            if region_bbox_precisions
            else None,
            "region_bbox_recall50": sum(region_bbox_recalls)
            / max(1, len(region_bbox_recalls))
            if region_bbox_recalls
            else None,
            "region_reading_order_accuracy": sum(region_order_accuracies)
            / max(1, len(region_order_accuracies))
            if region_order_accuracies
            else None,
            "density_bucket_counts": {
                bucket: len(rows) for bucket, rows in stratified_rows.items()
            },
            "stratified": stratified_metrics,
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
            "teacher_forced_validation_loss": (
                sum(teacher_forced_ocr_losses) / len(teacher_forced_ocr_losses)
                if teacher_forced_ocr_losses
                else None
            ),
            "teacher_forced_ocr_loss_dtypes": sorted(teacher_forced_ocr_dtypes),
            "prompt_target_audit": {
                "collected": collect_teacher_forcing,
                "pages": len(teacher_forced_ocr_losses),
                "prompt_prefix_mismatches": prompt_prefix_mismatches,
                "prompt_prefix_match_rate": (
                    1.0 - prompt_prefix_mismatches / max(1, len(teacher_forced_ocr_losses))
                    if teacher_forced_ocr_losses
                    else None
                ),
                "target_token_count": teacher_forced_target_tokens,
                "eos_label_count": teacher_forced_eos_labels,
                "eos_label_page_rate": (
                    teacher_forced_eos_pages / max(1, len(teacher_forced_ocr_losses))
                    if teacher_forced_ocr_losses
                    else None
                ),
                "prompt_length_mean": (
                    sum(prompt_lengths) / max(1, len(prompt_lengths)) if prompt_lengths else None
                ),
                "full_length_mean": (
                    sum(full_lengths) / max(1, len(full_lengths)) if full_lengths else None
                ),
                "first_target": first_target_audit,
            },
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
            "transport_query_mass": (
                [
                    sum(values[index] for values in transport_query_masses)
                    / max(1, len(transport_query_masses))
                    for index in range(args.num_queries)
                ]
                if transport_query_masses
                else None
            ),
            "invalid_query_transport_mass": sum(invalid_transport_masses)
            / max(1, len(invalid_transport_masses)),
            "valid_query_transport_mass": sum(valid_transport_masses)
            / max(1, len(valid_transport_masses)),
            "fusion_query_mass": (
                [
                    sum(values[index] for values in fusion_query_masses)
                    / max(1, len(fusion_query_masses))
                    for index in range(args.num_queries)
                ]
                if fusion_query_masses
                else None
            ),
            "invalid_query_fusion_mass": sum(invalid_query_masses)
            / max(1, len(invalid_query_masses)),
            "valid_query_fusion_mass": sum(valid_query_masses)
            / max(1, len(valid_query_masses)),
            "gated_transport_total_mass": sum(gated_transport_total_masses)
            / max(1, len(gated_transport_total_masses))
            if gated_transport_total_masses
            else None,
            "invalid_gated_query_transport_mass": sum(invalid_gated_transport_masses)
            / max(1, len(invalid_gated_transport_masses))
            if invalid_gated_transport_masses
            else None,
            "valid_gated_query_transport_mass": sum(valid_gated_transport_masses)
            / max(1, len(valid_gated_transport_masses))
            if valid_gated_transport_masses
            else None,
            "invalid_gated_fusion_mass": sum(invalid_gated_fusion_masses)
            / max(1, len(invalid_gated_fusion_masses))
            if invalid_gated_fusion_masses
            else None,
            "valid_gated_fusion_mass": sum(valid_gated_fusion_masses)
            / max(1, len(valid_gated_fusion_masses))
            if valid_gated_fusion_masses
            else None,
            "valid_coverage": sum(valid_coverages) / max(1, len(valid_coverages))
            if valid_coverages
            else None,
            "mean_p_valid": sum(mean_p_valid_values) / max(1, len(mean_p_valid_values))
            if mean_p_valid_values
            else None,
            "mean_p_valid_matched": sum(mean_p_valid_matched_values)
            / max(1, len(mean_p_valid_matched_values))
            if mean_p_valid_matched_values
            else None,
            "mean_p_valid_no_object": sum(mean_p_valid_no_object_values)
            / max(1, len(mean_p_valid_no_object_values))
            if mean_p_valid_no_object_values
            else None,
            "validity_p_gap": sum(validity_p_gaps) / max(1, len(validity_p_gaps))
            if validity_p_gaps
            else None,
            "validity_auroc": sum(validity_aurocs) / max(1, len(validity_aurocs))
            if validity_aurocs
            else None,
            "validity_average_precision": sum(validity_average_precisions)
            / max(1, len(validity_average_precisions))
            if validity_average_precisions
            else None,
            "invalid_gated_context_share": sum(invalid_gated_context_shares)
            / max(1, len(invalid_gated_context_shares))
            if invalid_gated_context_shares
            else None,
            "valid_coverage_foreground": sum(valid_coverage_foreground_values)
            / max(1, len(valid_coverage_foreground_values))
            if valid_coverage_foreground_values
            else None,
            "valid_coverage_background": sum(valid_coverage_background_values)
            / max(1, len(valid_coverage_background_values))
            if valid_coverage_background_values
            else None,
            "matcher_mean_cost": sum(matcher_costs) / max(1, len(matcher_costs))
            if matcher_costs
            else None,
            "matcher_signatures": matcher_signatures,
            "adapter_dtypes": [json.loads(value) for value in sorted(dtype_signatures)],
            "layout_loss_dtypes": [
                json.loads(value) for value in sorted(layout_loss_dtype_signatures)
            ],
            "raw_content_gate": float(unwrap_module(bridge.adapter).content_gate.detach()),
            "effective_residual_scale": float(
                unwrap_module(bridge.adapter).effective_residual_scale().detach()
            ),
            **adapter_finite_report(bridge.adapter),
            "decoder_lora": decoder_lora_finite_report(model_module),
            "test_used_for_selection": False,
        }
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-mode", choices=["legacy", "aligned_recovery_v1"], default="legacy")
    parser.add_argument("--aligned-rollout-interval", type=int, default=4)
    parser.add_argument("--mode", choices=["content_only", "attention", "geometry", "layout_ot"], required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--protocol-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--distributed-strategy",
        choices=["none", "ddp"],
        default="none",
        help="use torchrun DDP with one whole page per rank when set to ddp",
    )
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--experiment-label",
        default="",
        help="human-readable attribution group label stored in run metadata",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument(
        "--decoder-adaptation",
        choices=["frozen", "lora"],
        default="frozen",
        help="keep the GLM decoder frozen or train selected decoder projections with LoRA",
    )
    parser.add_argument("--decoder-lora-rank", type=int, default=8)
    parser.add_argument("--decoder-lora-alpha", type=float, default=8.0)
    parser.add_argument("--decoder-lora-dropout", type=float, default=0.0)
    parser.add_argument("--decoder-learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=64)
    parser.add_argument(
        "--lr-schedule-steps",
        type=optional_positive_int,
        default=None,
        help="learning-rate schedule horizon; defaults to --max-steps and may be shorter",
    )
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--residual-scale-cap", type=optional_positive_float, default=0.03)
    parser.add_argument("--initial-residual-scale", type=float, default=0.0)
    parser.add_argument("--auxiliary-weight", type=float, default=0.2)
    parser.add_argument(
        "--auxiliary-weight-start",
        type=float,
        default=None,
        help="initial auxiliary-loss weight; defaults to --auxiliary-weight",
    )
    parser.add_argument(
        "--auxiliary-ramp-steps",
        type=int,
        default=0,
        help="optimizer steps used to ramp auxiliary weight to --auxiliary-weight",
    )
    parser.add_argument(
        "--gate-freeze-steps",
        type=int,
        default=0,
        help="keep the non-zero residual gate fixed for this many optimizer steps",
    )
    parser.add_argument(
        "--adapter-precision",
        choices=["mixed_bf16", "fp32"],
        default="mixed_bf16",
        help="precision used inside the pre-merge adapter; the backbone remains BF16",
    )
    parser.add_argument(
        "--layout-loss-profile",
        choices=[
            "full",
            "ocr_only",
            "no_assignment",
            "no_assignment_validity",
            "validity_assignment",
            "no_geometry",
        ],
        default="full",
    )
    parser.add_argument(
        "--query-assignment",
        choices=["fixed_order", "hungarian"],
        default="fixed_order",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--use-validity-head",
        action="store_true",
        help="predict valid/no-object probabilities and gate query fusion",
    )
    parser.add_argument(
        "--initial-valid-probability",
        type=float,
        default=None,
        help="valid query prior; defaults to 0.066 for validity_assignment and 0.05 otherwise",
    )
    parser.add_argument(
        "--validity-gating-mode",
        choices=["legacy_normalized", "raw_mass"],
        default="legacy_normalized",
        help="validity fusion semantics; legacy_normalized preserves old checkpoints",
    )
    parser.add_argument(
        "--validity-use-transport-evidence",
        action="store_true",
        help="include detached raw-transport visual evidence in the validity head",
    )
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument(
        "--processor-mode",
        choices=["fast", "slow"],
        default="fast",
        help="explicitly select the Hugging Face image processor implementation",
    )
    parser.add_argument("--max-eval-new-tokens", type=int, default=768)
    parser.add_argument(
        "--text-repeat-suppression",
        action="store_true",
        help="enable cycle-specific unlikelihood/EOS loss and inference guard",
    )
    parser.add_argument("--text-ul-weight", type=float, default=0.1)
    parser.add_argument("--text-eos-loss-weight", type=float, default=0.05)
    parser.add_argument("--repeat-recent-window", type=int, default=96)
    parser.add_argument("--repeat-min-cycle-length", type=int, default=8)
    parser.add_argument("--repeat-max-cycle-length", type=int, default=32)
    parser.add_argument("--repeat-cycle-repeats", type=int, default=3)
    parser.add_argument("--repeat-cycle-penalty", type=float, default=2.0)
    parser.add_argument("--repeat-force-eos-steps", type=int, default=16)
    parser.add_argument(
        "--natural-loop-loss",
        action="store_true",
        help="observe a no-grad greedy rollout and train on its detached loop prefix",
    )
    parser.add_argument("--natural-loop-weight", type=float, default=0.05)
    parser.add_argument("--natural-loop-recent-window", type=int, default=96)
    parser.add_argument("--natural-loop-min-cycle-length", type=int, default=8)
    parser.add_argument("--natural-loop-max-cycle-length", type=int, default=32)
    parser.add_argument("--natural-loop-cycle-repeats", type=int, default=3)
    parser.add_argument("--natural-loop-max-new-tokens", type=int, default=768)
    parser.add_argument("--natural-loop-continuation-horizon", type=int, default=16)
    parser.add_argument(
        "--continuation-escape",
        action="store_true",
        help="use soft loop escape and continuation-biased EOS adjustment instead of forced EOS",
    )
    parser.add_argument("--escape-budget", type=int, default=16)
    parser.add_argument("--escape-clear-steps", type=int, default=4)
    parser.add_argument("--escape-eos-suppression", type=float, default=1.0)
    parser.add_argument("--escape-eos-boost", type=float, default=0.5)
    parser.add_argument(
        "--scheduled-sampling",
        action="store_true",
        help="mix a bounded fraction of detached argmax OCR prefixes after warm-up",
    )
    parser.add_argument("--scheduled-sampling-warmup-steps", type=int, default=64)
    parser.add_argument("--scheduled-sampling-ramp-steps", type=int, default=64)
    parser.add_argument("--scheduled-sampling-max-probability", type=float, default=0.1)
    parser.add_argument(
        "--loop-escape-training",
        action="store_true",
        help="train on synthetic repeated prefixes and real continuations",
    )
    parser.add_argument("--loop-escape-cycle-length", type=int, default=8)
    parser.add_argument("--loop-escape-horizon", type=int, default=8)
    parser.add_argument("--loop-escape-ramp-steps", type=int, default=128)
    parser.add_argument("--loop-escape-weight", type=float, default=0.1)
    parser.add_argument("--loop-escape-margin", type=float, default=0.5)
    parser.add_argument("--loop-escape-margin-weight", type=float, default=0.05)
    parser.add_argument("--loop-continue-weight", type=float, default=0.05)
    parser.add_argument(
        "--continuation-head",
        action="store_true",
        help="train and use the lightweight loop-risk/stop calibration head",
    )
    parser.add_argument("--continuation-head-hidden-size", type=int, default=32)
    parser.add_argument("--continuation-head-weight", type=float, default=0.05)
    parser.add_argument("--continuation-head-learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--generation-mode",
        choices=["plain", "loop_recovery"],
        default="loop_recovery",
        help="validation decoding mode; loop_recovery uses the stateful cycle controller",
    )
    parser.add_argument(
        "--eval-disable-repeat-guard",
        action="store_true",
        help="disable the inference-only cycle logits processor for checkpoint audit",
    )
    parser.add_argument(
        "--audit-prompt-prefix",
        action="store_true",
        help="collect prompt/target token-boundary diagnostics during validation",
    )
    parser.add_argument(
        "--eval-checkpoint-dir",
        type=Path,
        default=None,
        help="load adapter/decoder LoRA tensors from this checkpoint for eval-only audit",
    )
    parser.add_argument(
        "--init-checkpoint-dir",
        type=Path,
        default=None,
        help="initialize a training run from an adapter/decoder LoRA checkpoint",
    )
    parser.add_argument(
        "--region-autoregressive",
        action="store_true",
        help="use the 512-query autoregressive region decoder",
    )
    parser.add_argument("--region-decoder-hidden-size", type=int, default=256)
    parser.add_argument("--region-decoder-layers", type=int, default=2)
    parser.add_argument("--region-decoder-num-heads", type=int, default=8)
    parser.add_argument(
        "--region-pointer-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--region-spatial-penalty", type=float, default=4.0)
    parser.add_argument("--region-spatial-iou-threshold", type=float, default=0.8)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=256)
    parser.add_argument("--diagnostic-steps", type=parse_step_list, default=())
    parser.add_argument(
        "--skip-selection",
        action="store_true",
        help="evaluate all saved checkpoints without writing selection.json",
    )
    parser.add_argument(
        "--defer-validation",
        action="store_true",
        help="finish training after checkpoint writes and defer validation/selection to an external evaluator",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="skip validation evaluation and mark the fixed final checkpoint for direct test",
    )
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.recovery_mode == "aligned_recovery_v1":
        if args.aligned_rollout_interval < 1:
            raise ValueError("aligned rollout interval must be positive")
        if any((args.scheduled_sampling, args.loop_escape_training, args.continuation_head,
                args.text_repeat_suppression)):
            raise ValueError("aligned_recovery_v1 cannot mix legacy recovery objectives")
        args.natural_loop_loss = True
    if args.layout_loss_profile == "validity_assignment":
        # This profile is intentionally self-contained: it cannot silently
        # fall back to the old normalized gate or a query-only validity head.
        args.use_validity_head = True
        args.validity_gating_mode = "raw_mass"
        args.validity_use_transport_evidence = True
    if args.initial_valid_probability is None:
        args.initial_valid_probability = (
            0.066 if args.layout_loss_profile == "validity_assignment" else 0.05
        )
    if args.auxiliary_weight_start is None:
        args.auxiliary_weight_start = args.auxiliary_weight
    if args.eval_checkpoint_dir is not None and not args.eval_only:
        raise ValueError("--eval-checkpoint-dir is only valid with --eval-only")
    if args.init_checkpoint_dir is not None and args.eval_only:
        raise ValueError("--init-checkpoint-dir is only valid for training runs")
    if args.init_checkpoint_dir is not None and args.eval_checkpoint_dir is not None:
        raise ValueError("--init-checkpoint-dir and --eval-checkpoint-dir are mutually exclusive")
    if args.eval_only and args.eval_checkpoint_dir is None:
        if args.mode != "content_only":
            raise ValueError(
                "checkpoint-free --eval-only is reserved for the prompt-only content_only baseline"
            )
        if args.decoder_adaptation != "frozen":
            raise ValueError("checkpoint-free --eval-only baseline must keep the decoder frozen")
        if args.auxiliary_weight != 0.0 or args.auxiliary_weight_start != 0.0:
            raise ValueError("checkpoint-free eval-only baseline requires --auxiliary-weight 0")
    if args.eval_only and args.scheduled_sampling:
        raise ValueError("scheduled sampling is a training-only option")
    if args.no_validation and args.eval_only:
        raise ValueError("--no-validation is only valid for training runs")
    if args.no_validation and args.diagnostic_steps:
        raise ValueError("--no-validation cannot be combined with diagnostic validation steps")
    if args.defer_validation and args.eval_only:
        raise ValueError("--defer-validation is only valid for training runs")
    if args.defer_validation and args.no_validation:
        raise ValueError("--defer-validation cannot be combined with --no-validation")
    if args.defer_validation and args.skip_selection:
        raise ValueError("--defer-validation cannot be combined with --skip-selection")
    if args.auxiliary_weight < 0.0 or args.auxiliary_weight_start < 0.0:
        raise ValueError("auxiliary weights must be non-negative")
    if args.auxiliary_ramp_steps < 0:
        raise ValueError("--auxiliary-ramp-steps must be non-negative")
    if args.gate_freeze_steps < 0:
        raise ValueError("--gate-freeze-steps must be non-negative")
    if not 0 <= args.warmup_steps < args.max_steps:
        raise ValueError("--warmup-steps must be non-negative and smaller than --max-steps")
    if args.lr_schedule_steps is not None:
        if args.lr_schedule_steps > args.max_steps:
            raise ValueError("--lr-schedule-steps must not exceed --max-steps")
        if not 0 <= args.warmup_steps < args.lr_schedule_steps:
            raise ValueError("--warmup-steps must be smaller than --lr-schedule-steps")
    if not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("--min-lr-ratio must be in [0, 1]")
    if not -1.0 < args.initial_residual_scale < 1.0:
        raise ValueError("--initial-residual-scale must be strictly between -1 and 1")
    if not 0.0 < args.initial_valid_probability < 1.0:
        raise ValueError("--initial-valid-probability must be strictly between 0 and 1")
    if (
        args.residual_scale_cap is not None
        and abs(args.initial_residual_scale) > args.residual_scale_cap
    ):
        raise ValueError("--initial-residual-scale must not exceed --residual-scale-cap")
    if any(step > args.max_steps for step in args.diagnostic_steps):
        raise ValueError("diagnostic steps must not exceed --max-steps")
    if args.auxiliary_ramp_steps > args.max_steps:
        raise ValueError("--auxiliary-ramp-steps must not exceed --max-steps")
    if args.gate_freeze_steps > args.max_steps:
        raise ValueError("--gate-freeze-steps must not exceed --max-steps")
    if args.validation_interval <= 0:
        raise ValueError("--validation-interval must be positive")
    if args.per_device_batch_size != 1:
        raise ValueError("the whole-page GLMOCR path currently requires --per-device-batch-size 1")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be positive")
    if args.decoder_lora_rank <= 0:
        raise ValueError("--decoder-lora-rank must be positive")
    if args.decoder_lora_alpha <= 0 or args.decoder_learning_rate <= 0:
        raise ValueError("decoder LoRA alpha and learning rate must be positive")
    if not 0.0 <= args.decoder_lora_dropout < 1.0:
        raise ValueError("--decoder-lora-dropout must be in [0, 1)")
    if args.max_eval_new_tokens <= 0:
        raise ValueError("--max-eval-new-tokens must be positive")
    if args.text_ul_weight < 0 or args.text_eos_loss_weight < 0:
        raise ValueError("text repetition loss weights must be non-negative")
    if args.repeat_recent_window <= 0 or args.repeat_min_cycle_length <= 0:
        raise ValueError("repeat detection windows must be positive")
    if args.repeat_max_cycle_length < args.repeat_min_cycle_length:
        raise ValueError("repeat max cycle length must not be smaller than min")
    if args.repeat_cycle_repeats < 2 or args.repeat_cycle_penalty < 0:
        raise ValueError("invalid repeat cycle configuration")
    if args.repeat_force_eos_steps < 0:
        raise ValueError("repeat force EOS steps must be non-negative")
    if args.natural_loop_weight < 0:
        raise ValueError("natural loop loss weight must be non-negative")
    if args.natural_loop_recent_window <= 0:
        raise ValueError("natural loop recent window must be positive")
    if (
        args.natural_loop_min_cycle_length <= 0
        or args.natural_loop_max_cycle_length < args.natural_loop_min_cycle_length
    ):
        raise ValueError("invalid natural loop cycle length range")
    if args.natural_loop_cycle_repeats < 2:
        raise ValueError("natural loop cycle repeats must be at least two")
    if args.natural_loop_max_new_tokens <= 0 or args.natural_loop_continuation_horizon <= 0:
        raise ValueError("natural loop rollout limits must be positive")
    if args.escape_budget <= 0 or args.escape_clear_steps <= 0:
        raise ValueError("escape budget and clear steps must be positive")
    if args.escape_eos_suppression < 0 or args.escape_eos_boost < 0:
        raise ValueError("escape EOS adjustments must be non-negative")
    if args.scheduled_sampling_warmup_steps < 0 or args.scheduled_sampling_ramp_steps < 0:
        raise ValueError("scheduled-sampling warmup and ramp steps must be non-negative")
    if not 0.0 <= args.scheduled_sampling_max_probability <= 0.1:
        raise ValueError("scheduled-sampling max probability must be in [0, 0.1]")
    if args.loop_escape_cycle_length <= 0 or args.loop_escape_horizon <= 0:
        raise ValueError("loop escape cycle length and horizon must be positive")
    if args.loop_escape_ramp_steps <= 0:
        raise ValueError("loop escape ramp steps must be positive")
    if (
        args.loop_escape_weight < 0
        or args.loop_escape_margin < 0
        or args.loop_escape_margin_weight < 0
        or args.loop_continue_weight < 0
    ):
        raise ValueError("loop escape loss weights and margin must be non-negative")
    if args.continuation_head_hidden_size <= 0 or args.continuation_head_weight < 0:
        raise ValueError("continuation head size and weight must be valid")
    if args.continuation_head_learning_rate <= 0:
        raise ValueError("continuation head learning rate must be positive")
    if args.region_autoregressive and args.num_queries != 512:
        raise ValueError("--region-autoregressive requires --num-queries 512")
    if args.region_decoder_hidden_size <= 0 or args.region_decoder_layers <= 0:
        raise ValueError("region decoder size must be positive")
    if args.region_decoder_hidden_size % args.region_decoder_num_heads:
        raise ValueError("region decoder hidden size must be divisible by head count")
    if args.region_spatial_penalty < 0 or not 0.0 < args.region_spatial_iou_threshold <= 1.0:
        raise ValueError("invalid region spatial duplicate configuration")
    if args.log_steps <= 0:
        raise ValueError("--log-steps must be positive")
    if args.layout_loss_profile in {"no_assignment_validity", "validity_assignment"} and not args.use_validity_head:
        raise ValueError(
            f"--layout-loss-profile {args.layout_loss_profile} requires --use-validity-head"
        )
    if args.layout_loss_profile == "validity_assignment" and args.query_assignment != "hungarian":
        raise ValueError("--layout-loss-profile validity_assignment requires --query-assignment hungarian")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.eval_checkpoint_dir is not None:
        if not args.eval_checkpoint_dir.is_dir():
            raise FileNotFoundError(f"evaluation checkpoint directory is missing: {args.eval_checkpoint_dir}")
        if not (args.eval_checkpoint_dir / "adapter.safetensors").is_file():
            raise FileNotFoundError(
                f"evaluation adapter checkpoint is missing: {args.eval_checkpoint_dir / 'adapter.safetensors'}"
            )
    reproducibility = configure_deterministic_execution()
    distributed = initialize_distributed(args.distributed_strategy)
    if args.eval_only and distributed.enabled:
        raise ValueError("--eval-only must run without DDP")
    if distributed.is_main:
        args.output_dir.mkdir(parents=True)
    barrier(distributed)
    lr_schedule_steps = args.lr_schedule_steps or args.max_steps
    protocol_metadata = json.loads(args.protocol_file.read_text(encoding="utf-8"))
    if args.defer_validation and protocol_metadata.get("test_manifest_read") is not False:
        raise ValueError("--defer-validation requires a protocol with test_manifest_read=false")
    metadata = {
        "status": "running",
        "mode": args.mode,
        "experiment_label": args.experiment_label,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "lr_schedule_steps": lr_schedule_steps,
        "num_queries": args.num_queries,
        "auxiliary_weight": args.auxiliary_weight,
        "auxiliary_weight_start": args.auxiliary_weight_start,
        "auxiliary_ramp_steps": args.auxiliary_ramp_steps,
        "gate_freeze_steps": args.gate_freeze_steps,
        "use_validity_head": args.use_validity_head,
        "initial_valid_probability": args.initial_valid_probability,
        "validity_gating_mode": args.validity_gating_mode,
        "validity_use_transport_evidence": args.validity_use_transport_evidence,
        "adapter_precision": args.adapter_precision,
        "decoder_adaptation": args.decoder_adaptation,
        "decoder_lora_config": {
            "rank": args.decoder_lora_rank,
            "alpha": args.decoder_lora_alpha,
            "dropout": args.decoder_lora_dropout,
            "learning_rate": args.decoder_learning_rate,
        },
        "layout_loss_profile": args.layout_loss_profile,
        "query_assignment": args.query_assignment,
        "processor_mode": args.processor_mode,
        "eval_only": args.eval_only,
        "max_eval_new_tokens": args.max_eval_new_tokens,
        "generation_mode": args.generation_mode,
        "validation_interval": args.validation_interval,
        "diagnostic_steps": list(args.diagnostic_steps),
        "skip_selection": args.skip_selection,
        "defer_validation": args.defer_validation,
        "no_validation": args.no_validation,
        "selection_pending": args.defer_validation,
        "validation_evaluated": False,
        "selection_performed": False,
        "text_repeat_suppression": args.text_repeat_suppression,
        "text_repeat_config": asdict(repeat_suppression_config(args)),
        "loss_objective": loss_objective_config(args),
        "natural_loop_loss": natural_loop_config(args),
        "eval_disable_repeat_guard": args.eval_disable_repeat_guard,
        "audit_prompt_prefix": args.audit_prompt_prefix,
        "eval_checkpoint_dir": str(args.eval_checkpoint_dir) if args.eval_checkpoint_dir else None,
        "init_checkpoint_dir": str(args.init_checkpoint_dir) if args.init_checkpoint_dir else None,
        "loop_escape_training": args.loop_escape_training,
        "loop_escape_config": {
            "cycle_length": args.loop_escape_cycle_length,
            "horizon": args.loop_escape_horizon,
            "ramp_steps": args.loop_escape_ramp_steps,
            "weight": args.loop_escape_weight,
            "margin": args.loop_escape_margin,
            "margin_weight": args.loop_escape_margin_weight,
            "continue_weight": args.loop_continue_weight,
        },
        "continuation_head": {
            "enabled": args.continuation_head,
            "hidden_size": args.continuation_head_hidden_size,
            "weight": args.continuation_head_weight,
            "learning_rate": args.continuation_head_learning_rate,
        },
        "scheduled_sampling": {
            "enabled": args.scheduled_sampling,
            "warmup_steps": args.scheduled_sampling_warmup_steps,
            "ramp_steps": args.scheduled_sampling_ramp_steps,
            "max_probability": args.scheduled_sampling_max_probability,
        },
        "region_autoregressive": args.region_autoregressive,
        "region_decoder_config": {
            "hidden_size": args.region_decoder_hidden_size,
            "layers": args.region_decoder_layers,
            "num_heads": args.region_decoder_num_heads,
            "pointer_mask": args.region_pointer_mask,
            "spatial_penalty": args.region_spatial_penalty,
            "spatial_iou_threshold": args.region_spatial_iou_threshold,
        },
        "distributed_strategy": args.distributed_strategy,
        "rank": distributed.rank,
        "local_rank": distributed.local_rank,
        "world_size": distributed.world_size,
        "per_device_batch_size": args.per_device_batch_size,
        "global_batch_size": distributed.world_size * args.per_device_batch_size,
        "effective_global_batch_size": (
            distributed.world_size
            * args.per_device_batch_size
            * args.gradient_accumulation_steps
        ),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "optimizer": {
            "name": "AdamW",
            "peak_learning_rate": args.learning_rate,
            "decoder_peak_learning_rate": args.decoder_learning_rate,
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
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "recovery_code_sha256": sha256_file(Path(__file__).with_name("aligned_recovery.py")),
        "protocol": protocol_metadata,
        "test_manifest_read": bool(protocol_metadata.get("test_manifest_read", False)),
        "test_used_for_selection": False,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__},
    }
    if distributed.is_main:
        write_json(args.output_dir / "metadata.json", metadata)
    try:
        metadata["reproducibility"] = reproducibility
        if distributed.is_main:
            write_json(args.output_dir / "metadata.json", metadata)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside the Slurm allocation")
        device = torch.device("cuda", distributed.local_rank)
        torch.cuda.set_device(device)
        train_records = load_records(args.train_manifest)
        validation_records = load_records(args.validation_manifest)
        validate_records(train_records, split="train", num_queries=args.num_queries)
        validate_records(validation_records, split="validation", num_queries=args.num_queries)
        model, processor, bridge = load_model(args, device)
        continuation_head = (
            ContinuationStopHead(hidden_size=args.continuation_head_hidden_size).to(device)
            if args.continuation_head
            else None
        )
        if args.decoder_adaptation == "lora":
            decoder_lora_config = inject_decoder_lora(
                model,
                rank=args.decoder_lora_rank,
                alpha=args.decoder_lora_alpha,
                dropout=args.decoder_lora_dropout,
            )
        else:
            decoder_lora_config = {
                "enabled": False,
                "rank": args.decoder_lora_rank,
                "alpha": args.decoder_lora_alpha,
                "dropout": args.decoder_lora_dropout,
                "target_count": 0,
                "targets": [],
            }
        if distributed.enabled:
            if args.decoder_adaptation == "lora":
                model = wrap_model(model, distributed)
            else:
                bridge.adapter = wrap_adapter(bridge.adapter, distributed)  # type: ignore[assignment]
            if continuation_head is not None:
                continuation_head = wrap_model(continuation_head, distributed)
        if args.eval_checkpoint_dir is not None:
            load_adapter_checkpoint(args.eval_checkpoint_dir, bridge)
            if args.decoder_adaptation == "lora":
                load_decoder_lora_checkpoint(args.eval_checkpoint_dir, model)
            if continuation_head is not None:
                load_continuation_head_checkpoint(args.eval_checkpoint_dir, continuation_head)
        if args.init_checkpoint_dir is not None:
            if not args.init_checkpoint_dir.is_dir():
                raise FileNotFoundError(
                    f"training initialization checkpoint directory is missing: {args.init_checkpoint_dir}"
                )
            load_adapter_checkpoint(args.init_checkpoint_dir, bridge)
            if args.decoder_adaptation == "lora":
                load_decoder_lora_checkpoint(args.init_checkpoint_dir, model)
            if continuation_head is not None:
                load_continuation_head_checkpoint(args.init_checkpoint_dir, continuation_head)
        adapter = unwrap_module(bridge.adapter)
        metadata["adapter_config"] = asdict(adapter.config)
        metadata["decoder_lora_config"] = decoder_lora_config
        metadata["continuation_head_config"] = {
            "enabled": continuation_head is not None,
            "hidden_size": args.continuation_head_hidden_size,
            "weight": args.continuation_head_weight,
            "learning_rate": args.continuation_head_learning_rate,
        }
        metadata["trainable_parameter_report"] = trainable_parameter_report(model)
        metadata["decoder_lora_finite"] = decoder_lora_finite_report(model)
        metadata["versions"].update(
            {
                "transformers": __import__("transformers").__version__,
                "torchvision": package_version("torchvision"),
                "pillow": package_version("Pillow"),
            }
        )
        metadata["processor"] = processor_reproducibility_report(
            processor,
            args.processor_mode,
            args.model_path.resolve(),
        )
        metadata["processor_input_fingerprint"] = processor_input_fingerprint(
            processor,
            validation_records[0],
        )
        metadata["gpu"] = torch.cuda.get_device_name(distributed.local_rank)
        if distributed.is_main:
            write_json(args.output_dir / "metadata.json", metadata)
        if args.eval_only:
            state_before = clone_module_state(bridge.adapter)
            validation = evaluate(
                args,
                model,
                processor,
                bridge,
                continuation_head,
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
                "experiment_label": args.experiment_label,
                "seed": args.seed,
                "auxiliary_weight": args.auxiliary_weight,
                "adapter_precision": args.adapter_precision,
                "layout_loss_profile": args.layout_loss_profile,
                "query_assignment": args.query_assignment,
                "decoder_adaptation": args.decoder_adaptation,
                "decoder_lora_config": decoder_lora_config,
                "decoder_lora_loaded": args.decoder_adaptation == "lora",
                "decoder_lora_finite": metadata["decoder_lora_finite"],
                "trainable_parameter_report": trainable_parameter_report(model),
                "eval_only": True,
                "training_updates": 0,
                "parameters_unchanged": True,
                "eval_checkpoint_dir": (
                    str(args.eval_checkpoint_dir) if args.eval_checkpoint_dir is not None else None
                ),
                "repeat_guard_enabled": bool(
                    repeat_suppression_config(args).enabled and not args.eval_disable_repeat_guard
                ),
                "natural_loop_config": natural_loop_config(args),
                "max_eval_new_tokens": args.max_eval_new_tokens,
                "validation": validation,
                "test_manifest_read": metadata["test_manifest_read"],
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
            # Validation is rank-0-only, so all ranks must rendezvous before
            # and after it.  Otherwise non-main ranks enter the first DDP
            # backward while rank 0 is still evaluating, which deadlocks NCCL.
            barrier(distributed)
            if distributed.is_main:
                write_json(args.output_dir / "phase.json", {"phase": "initial_validation", "status": "running"})
                validation0 = evaluate(
                    args,
                    model,
                    processor,
                    bridge,
                    continuation_head,
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
            barrier(distributed)

        if distributed.is_main:
            write_json(args.output_dir / "phase.json", {"phase": "training", "status": "running"})
        training = train(
            args,
            model,
            processor,
            bridge,
            continuation_head,
            train_records,
            device,
            distributed=distributed,
        )
        # DDP ranks must not independently reload checkpoints or evaluate the
        # validation split.  Rank 0 owns selection and all run-level artifacts;
        # the other ranks wait until rank 0 has finished the training barrier.
        barrier(distributed)
        if not distributed.is_main:
            return
        checkpoint_steps = training["checkpoint_steps"]
        if args.defer_validation:
            if not checkpoint_steps:
                raise RuntimeError("deferred-validation training did not save a checkpoint")
            write_adapter_config(args.output_dir / "adapter_config.json", bridge)
            if continuation_head is not None:
                save_continuation_head_checkpoint(args.output_dir, continuation_head, max(checkpoint_steps))
            summary = {
                "status": "complete",
                "mode": args.mode,
                "experiment_label": args.experiment_label,
                "seed": args.seed,
                "auxiliary_weight": args.auxiliary_weight,
                "adapter_precision": args.adapter_precision,
                "layout_loss_profile": args.layout_loss_profile,
                "query_assignment": args.query_assignment,
                "decoder_adaptation": args.decoder_adaptation,
                "decoder_lora_config": metadata["decoder_lora_config"],
                "natural_loop_config": natural_loop_config(args),
                "trainable_parameter_report": metadata["trainable_parameter_report"],
                "lr_schedule_steps": training["lr_schedule_steps"],
                "eval_only": False,
                "skip_selection": False,
                "defer_validation": True,
                "no_validation": False,
                "selection_pending": True,
                "validation_evaluated": False,
                "selection_performed": False,
                "max_eval_new_tokens": args.max_eval_new_tokens,
                "training": training,
                "validation": None,
                "validation_candidates": [],
                "selection_candidates": [],
                "selection": None,
                "test_manifest_read": metadata["test_manifest_read"],
                "test_used_for_selection": False,
            }
            write_json(args.output_dir / "summary.json", summary)
            (args.output_dir / "COMPLETED").touch()
            metadata["status"] = "complete"
            metadata["selection_pending"] = True
            metadata["validation_evaluated"] = False
            metadata["selection_performed"] = False
            write_json(args.output_dir / "metadata.json", metadata)
            print(json.dumps({
                "status": "complete",
                "run_dir": str(args.output_dir),
                "checkpoint_steps": checkpoint_steps,
                "selection_pending": True,
                "validation_evaluated": False,
                "selection_performed": False,
                "test_used_for_selection": False,
            }, ensure_ascii=False, separators=(",", ":")))
            return
        if args.no_validation:
            if not checkpoint_steps:
                raise RuntimeError("no-validation training did not save a final checkpoint")
            final_step = max(checkpoint_steps)
            final_checkpoint = args.output_dir / f"checkpoint-{final_step}"
            if not (final_checkpoint / "adapter.safetensors").is_file():
                raise FileNotFoundError(final_checkpoint / "adapter.safetensors")
            fixed_selection = {
                "status": "complete",
                "selection_metric": "fixed_final_step_no_validation",
                "selected_step": final_step,
                "selection_performed": False,
                "validation_evaluated": False,
                "candidates": [],
                "seeds": [args.seed],
                "seed_runs": {
                    str(args.seed): {
                        "run_dir": str(args.output_dir),
                        "selected_step": final_step,
                    }
                },
                "test_manifest_read": metadata["test_manifest_read"],
                "test_used_for_selection": False,
            }
            write_json(args.output_dir / "selection.json", fixed_selection)
            summary = {
                "status": "complete",
                "mode": args.mode,
                "experiment_label": args.experiment_label,
                "seed": args.seed,
                "auxiliary_weight": args.auxiliary_weight,
                "adapter_precision": args.adapter_precision,
                "layout_loss_profile": args.layout_loss_profile,
                "query_assignment": args.query_assignment,
                "decoder_adaptation": args.decoder_adaptation,
                "decoder_lora_config": metadata["decoder_lora_config"],
                "natural_loop_config": natural_loop_config(args),
                "trainable_parameter_report": metadata["trainable_parameter_report"],
                "lr_schedule_steps": training["lr_schedule_steps"],
                "eval_only": False,
                "skip_selection": False,
                "no_validation": True,
                "max_eval_new_tokens": args.max_eval_new_tokens,
                "training": training,
                "validation": None,
                "validation_candidates": [],
                "selection_candidates": [],
                "selection": fixed_selection,
                "test_manifest_read": metadata["test_manifest_read"],
                "test_used_for_selection": False,
            }
            write_json(args.output_dir / "summary.json", summary)
            (args.output_dir / "COMPLETED").touch()
            metadata["status"] = "complete"
            write_json(args.output_dir / "metadata.json", metadata)
            print(json.dumps({
                "status": "complete",
                "run_dir": str(args.output_dir),
                "final_step": final_step,
                "validation_evaluated": False,
                "selection_performed": False,
                "test_used_for_selection": False,
            }, ensure_ascii=False, separators=(",", ":")))
            return
        candidates = []
        for step in checkpoint_steps:
            write_json(args.output_dir / "phase.json", {"phase": "validation", "step": step, "status": "running"})
            checkpoint_dir = args.output_dir / f"checkpoint-{step}"
            load_adapter_checkpoint(checkpoint_dir, bridge)
            if args.decoder_adaptation == "lora":
                load_decoder_lora_checkpoint(checkpoint_dir, model)
            if continuation_head is not None:
                load_continuation_head_checkpoint(checkpoint_dir, continuation_head)
            candidate = evaluate(
                args,
                model,
                processor,
                bridge,
                continuation_head,
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
            if key in {
                "ocr_loss",
                "last_batch_ocr_loss",
                "ema_ocr_loss_16",
                "token_weighted_ocr_loss",
                "mixed_prefix_token_weighted_loss",
                "mixed_prefix_loss",
                "ocr_objective_loss",
                "official_base_loss",
                "text_unlikelihood_activation_ratio",
                "text_eos_activation_ratio",
                "natural_loop_loss",
                "natural_loop_weighted_loss",
                "natural_loop_activation_ratio",
                "natural_loop_candidate_mismatch_ratio",
            }:
                return training_row.get(key)
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
                "matcher_churn": matcher_churn_between_points(diagnostic_points),
                "validity_mechanism_acceptance": validity_mechanism_acceptance(
                    diagnostic_points
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
                    "repeated_trigram_rate": [
                        {
                            "step": row["step"],
                            "value": row["validation"]["repeated_trigram_rate"],
                        }
                        for row in diagnostic_points
                    ],
                    "repeated_cycle_page_rate": [
                        {
                            "step": row["step"],
                            "value": row["validation"].get("repeated_cycle_page_rate"),
                        }
                        for row in diagnostic_points
                    ],
                    "region_pointer_reuse_rate": [
                        {
                            "step": row["step"],
                            "value": row["validation"].get("region_pointer_reuse_rate"),
                        }
                        for row in diagnostic_points
                    ],
                    "region_spatial_duplicate_rate": [
                        {
                            "step": row["step"],
                            "value": row["validation"].get("region_spatial_duplicate_rate"),
                        }
                        for row in diagnostic_points
                    ],
                    "region_eos_hit_rate": [
                        {
                            "step": row["step"],
                            "value": row["validation"].get("region_eos_hit_rate"),
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
                    "training_last_batch_ocr_loss": [
                        {
                            "step": row["step"],
                            "value": training_value(row, "last_batch_ocr_loss"),
                        }
                        for row in diagnostic_points
                    ],
                    "training_ema_ocr_loss_16": [
                        {
                            "step": row["step"],
                            "value": training_value(row, "ema_ocr_loss_16"),
                        }
                        for row in diagnostic_points
                    ],
                    "training_token_weighted_ocr_loss": [
                        {
                            "step": row["step"],
                            "value": training_value(row, "token_weighted_ocr_loss"),
                        }
                        for row in diagnostic_points
                    ],
                    "training_mixed_prefix_loss": [
                        {
                            "step": row["step"],
                            "value": training_value(row, "mixed_prefix_loss"),
                        }
                        for row in diagnostic_points
                    ],
                    "training_ocr_objective_loss": [
                        {
                            "step": row["step"],
                            "value": training_value(row, "ocr_objective_loss"),
                        }
                        for row in diagnostic_points
                    ],
                    "training_official_base_loss": [
                        {
                            "step": row["step"],
                            "value": training_value(row, "official_base_loss"),
                        }
                        for row in diagnostic_points
                    ],
                    "training_natural_loop_loss": [
                        {
                            "step": row["step"],
                            "value": training_value(row, "natural_loop_loss"),
                        }
                        for row in diagnostic_points
                    ],
                    "training_natural_loop_weighted_loss": [
                        {
                            "step": row["step"],
                            "value": training_value(row, "natural_loop_weighted_loss"),
                        }
                        for row in diagnostic_points
                    ],
                    "training_natural_loop_activation_ratio": [
                        {
                            "step": row["step"],
                            "value": training_value(row, "natural_loop_activation_ratio"),
                        }
                        for row in diagnostic_points
                    ],
                    "training_natural_loop_candidate_mismatch_ratio": [
                        {
                            "step": row["step"],
                            "value": training_value(
                                row, "natural_loop_candidate_mismatch_ratio"
                            ),
                        }
                        for row in diagnostic_points
                    ],
                    "training_text_unlikelihood_activation_ratio": [
                        {
                            "step": row["step"],
                            "value": training_value(
                                row, "text_unlikelihood_activation_ratio"
                            ),
                        }
                        for row in diagnostic_points
                    ],
                    "training_text_eos_activation_ratio": [
                        {
                            "step": row["step"],
                            "value": training_value(row, "text_eos_activation_ratio"),
                        }
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
                        for key in ("ocr", "mixed_prefix", "ocr_objective", *LAYOUT_LOSS_KEYS, "total")
                    },
                    "assignment_positive_query_grad": [
                        {
                            "step": row["step"],
                            "value": gradient_value(
                                row, "assignment_positive_query_grad"
                            ),
                        }
                        for row in diagnostic_points
                    ],
                    "assignment_negative_query_grad": [
                        {
                            "step": row["step"],
                            "value": gradient_value(
                                row, "assignment_negative_query_grad"
                            ),
                        }
                        for row in diagnostic_points
                    ],
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
                    "validity_p_gap": [
                        {"step": row["step"], "value": row["validation"]["validity_p_gap"]}
                        for row in diagnostic_points
                    ],
                    "validity_auroc": [
                        {"step": row["step"], "value": row["validation"]["validity_auroc"]}
                        for row in diagnostic_points
                    ],
                    "validity_average_precision": [
                        {
                            "step": row["step"],
                            "value": row["validation"]["validity_average_precision"],
                        }
                        for row in diagnostic_points
                    ],
                    "invalid_gated_context_share": [
                        {
                            "step": row["step"],
                            "value": row["validation"]["invalid_gated_context_share"],
                        }
                        for row in diagnostic_points
                    ],
                    "valid_coverage_foreground": [
                        {
                            "step": row["step"],
                            "value": row["validation"]["valid_coverage_foreground"],
                        }
                        for row in diagnostic_points
                    ],
                    "valid_coverage_background": [
                        {
                            "step": row["step"],
                            "value": row["validation"]["valid_coverage_background"],
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
        if args.skip_selection:
            final_candidate = max(candidates, key=lambda row: (row["step"],))
            final_checkpoint = args.output_dir / f"checkpoint-{final_candidate['step']}"
            load_adapter_checkpoint(final_checkpoint, bridge)
            if args.decoder_adaptation == "lora":
                load_decoder_lora_checkpoint(final_checkpoint, model)
            if continuation_head is not None:
                load_continuation_head_checkpoint(final_checkpoint, continuation_head)
            save_file(
                {
                    key: value.detach().cpu().contiguous()
                    for key, value in adapter.state_dict().items()
                },
                args.output_dir / "adapter.safetensors",
            )
            final_lora_state = lora_state_dict(model)
            if final_lora_state:
                save_file(final_lora_state, args.output_dir / "decoder_lora.safetensors")
            if continuation_head is not None:
                save_continuation_head_checkpoint(
                    args.output_dir, continuation_head, final_candidate["step"]
                )
            write_adapter_config(args.output_dir / "adapter_config.json", bridge)
            summary = {
                "status": "complete",
                "mode": args.mode,
                "experiment_label": args.experiment_label,
                "seed": args.seed,
                "auxiliary_weight": args.auxiliary_weight,
                "adapter_precision": args.adapter_precision,
                "layout_loss_profile": args.layout_loss_profile,
                "query_assignment": args.query_assignment,
                "decoder_adaptation": args.decoder_adaptation,
                "decoder_lora_config": metadata["decoder_lora_config"],
                "natural_loop_config": natural_loop_config(args),
                "trainable_parameter_report": metadata["trainable_parameter_report"],
                "lr_schedule_steps": training["lr_schedule_steps"],
                "eval_only": False,
                "skip_selection": True,
                "max_eval_new_tokens": args.max_eval_new_tokens,
                "training": training,
                "validation": final_candidate,
                "validation_candidates": candidates,
                "diagnostic_summary": diagnostic_summary,
                "test_manifest_read": metadata["test_manifest_read"],
                "test_used_for_selection": False,
            }
            write_json(args.output_dir / "summary.json", summary)
            (args.output_dir / "COMPLETED").touch()
            metadata["status"] = "complete"
            write_json(args.output_dir / "metadata.json", metadata)
            completion = {
                "status": "complete",
                "run_dir": str(args.output_dir),
                "diagnostic_summary": (
                    str(args.output_dir / "diagnostic_summary.json")
                    if args.diagnostic_steps
                    else None
                ),
                "evaluated_steps": [row["step"] for row in candidates],
                "final_step": final_candidate["step"],
                "selection_performed": False,
                "test_used_for_selection": False,
            }
            print(json.dumps(completion, ensure_ascii=False, separators=(",", ":")))
            return
        selected = min(candidates, key=lambda row: (row["cer"], row["step"]))
        selected_checkpoint = args.output_dir / f"checkpoint-{selected['step']}"
        load_adapter_checkpoint(selected_checkpoint, bridge)
        if args.decoder_adaptation == "lora":
            load_decoder_lora_checkpoint(selected_checkpoint, model)
        if continuation_head is not None:
            load_continuation_head_checkpoint(selected_checkpoint, continuation_head)
        save_file(
            {key: value.detach().cpu().contiguous() for key, value in adapter.state_dict().items()},
            args.output_dir / "adapter.safetensors",
        )
        final_lora_state = lora_state_dict(model)
        if final_lora_state:
            save_file(final_lora_state, args.output_dir / "decoder_lora.safetensors")
        if continuation_head is not None:
            save_continuation_head_checkpoint(args.output_dir, continuation_head, selected["step"])
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
                        selected["cer"] < identity_cer
                        if identity_cer is not None
                        else None
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
                    "test_manifest_read": metadata["test_manifest_read"],
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
            "experiment_label": args.experiment_label,
            "seed": args.seed,
            "auxiliary_weight": args.auxiliary_weight,
            "adapter_precision": args.adapter_precision,
            "layout_loss_profile": args.layout_loss_profile,
            "query_assignment": args.query_assignment,
            "decoder_adaptation": args.decoder_adaptation,
            "decoder_lora_config": metadata["decoder_lora_config"],
            "natural_loop_config": natural_loop_config(args),
            "trainable_parameter_report": metadata["trainable_parameter_report"],
            "lr_schedule_steps": training["lr_schedule_steps"],
            "eval_only": False,
            "max_eval_new_tokens": args.max_eval_new_tokens,
            "training": training,
            "validation": validation,
            "selection_candidates": candidates,
            "diagnostic_summary": diagnostic_summary,
            "test_manifest_read": metadata["test_manifest_read"],
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
        if distributed.is_main:
            metadata["status"] = "failed"
            metadata["error_type"] = type(exc).__name__
            metadata["error"] = str(exc)
            write_json(args.output_dir / "metadata.json", metadata)
            (args.output_dir / "error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
        raise
    finally:
        destroy_distributed(distributed)


if __name__ == "__main__":
    main()
