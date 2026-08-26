#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
from torch import Tensor
from transformers import AutoTokenizer

from pvld_diagnostic_protocol import (
    BUCKET_NAMES,
    duplicate_diagnostics,
    mean_or_none,
    record_region_count,
    select_bucket_indices,
)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded PVLD failure-mode diagnostics.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-image-root", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--validation-image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--pages-per-bucket", type=positive_int, default=1)
    parser.add_argument("--layout-token-limit", type=positive_int, default=512)
    parser.add_argument("--layout-record-limit", type=positive_int, default=128)
    parser.add_argument("--ocr-pages", type=positive_int, default=2)
    parser.add_argument("--ocr-max-new-tokens", type=positive_int, default=512)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def add_project_paths(project_root: Path) -> None:
    for path in (project_root, project_root / "scripts"):
        value = str(path.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)


def move_images(images: Sequence[tuple[Tensor, Tensor]], device: torch.device) -> list[tuple[Tensor, Tensor]]:
    return [(image.to(device), image_high.to(device)) for image, image_high in images]


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "images":
            moved[key] = move_images(value, device)
        elif isinstance(value, Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def selected_indices(dataset: Any, pages_per_bucket: int) -> tuple[list[int], dict[str, list[int]]]:
    by_bucket = select_bucket_indices(dataset.records, pages_per_bucket)
    missing = [name for name in BUCKET_NAMES if not by_bucket[name]]
    if missing:
        raise RuntimeError(f"Dataset lacks diagnostic buckets: {missing}")
    flattened = [index for name in BUCKET_NAMES for index in by_bucket[name]]
    return flattened, by_bucket


def extract_high_resolution_features(model: Any, image_high: Tensor) -> Tensor:
    vision_tower = model.get_model().vision_tower_high
    image_high = image_high.to(dtype=next(vision_tower.parameters()).dtype)
    with torch.no_grad():
        feature = vision_tower(image_high)
    return feature.flatten(2).permute(0, 2, 1)


def tensor_mean_or_none(values: Tensor) -> float | None:
    return float(values.mean().item()) if values.numel() else None


def aligned_bbox_iou(predicted: Tensor, target: Tensor) -> Tensor:
    intersection_x0 = torch.maximum(predicted[..., 0], target[..., 0])
    intersection_y0 = torch.maximum(predicted[..., 1], target[..., 1])
    intersection_x1 = torch.minimum(predicted[..., 2], target[..., 2])
    intersection_y1 = torch.minimum(predicted[..., 3], target[..., 3])
    intersection = (
        (intersection_x1 - intersection_x0).clamp_min(0)
        * (intersection_y1 - intersection_y0).clamp_min(0)
    )
    predicted_area = (
        (predicted[..., 2] - predicted[..., 0]).clamp_min(0)
        * (predicted[..., 3] - predicted[..., 1]).clamp_min(0)
    )
    target_area = (
        (target[..., 2] - target[..., 0]).clamp_min(0)
        * (target[..., 3] - target[..., 1]).clamp_min(0)
    )
    return intersection / (predicted_area + target_area - intersection).clamp_min(1e-7)


def oracle_and_free_page(
    model: Any,
    item: dict[str, Any],
    *,
    split: str,
    bucket: str,
    device: torch.device,
    layout_token_limit: int,
    layout_record_limit: int,
) -> dict[str, Any]:
    adapter = model.get_model().variable_layout_adapter
    image_high = item["image_high"][0].unsqueeze(0).to(device)
    high_resolution = extract_high_resolution_features(model, image_high)
    visual_tokens = model.get_model().mm_projector_vary(high_resolution)

    def one_batch(name: str) -> Tensor:
        return item[name].unsqueeze(0).to(device)

    with torch.no_grad():
        oracle = adapter(
            visual_tokens,
            high_resolution,
            layout_input_ids=one_batch("layout_input_ids"),
            layout_attention_mask=one_batch("layout_attention_mask"),
            layout_region_positions=one_batch("layout_region_positions"),
            layout_record_mask=one_batch("layout_record_mask"),
            layout_bbox_targets=one_batch("layout_bbox_targets"),
            layout_type_targets=one_batch("layout_type_targets"),
            layout_direction_targets=one_batch("layout_direction_targets"),
            layout_count_targets=one_batch("layout_count_targets"),
        )
        original_token_limit = adapter.max_layout_tokens
        original_record_limit = adapter.max_layout_records
        adapter.max_layout_tokens = layout_token_limit
        adapter.max_layout_records = layout_record_limit
        try:
            free = adapter(visual_tokens, high_resolution, generate_layout=True)
        finally:
            adapter.max_layout_tokens = original_token_limit
            adapter.max_layout_records = original_record_limit

    vocabulary = adapter.vocabulary
    gold_ids = one_batch("layout_input_ids")
    logits = oracle.decoder_output.logits[:, :-1].float()
    targets = gold_ids[:, 1:]
    boundary = targets.eq(vocabulary.region_id) | targets.eq(vocabulary.eos_id)
    target_probability = logits.softmax(dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    boundary_accuracy = logits.argmax(dim=-1)[boundary].eq(targets[boundary]).float().mean()
    region_boundary = targets.eq(vocabulary.region_id)
    eos_boundary = targets.eq(vocabulary.eos_id)

    record_count = int(item["layout_record_mask"].sum().item())
    oracle_boxes_tensor = oracle.record_output.bbox[0, :record_count].float()
    target_boxes = item["layout_bbox_targets"][:record_count].to(device).float()
    oracle_boxes = oracle_boxes_tensor.cpu().tolist()
    oracle_duplicate = duplicate_diagnostics(oracle_boxes)
    oracle_bbox_iou = (
        float(aligned_bbox_iou(oracle_boxes_tensor, target_boxes).mean().item())
        if record_count else None
    )

    free_mask = free.record_mask[0].bool()
    free_boxes = free.record_output.bbox[0][free_mask].float().cpu().tolist()
    free_duplicate = duplicate_diagnostics(free_boxes)
    generated = free.decoder_output
    return {
        "split": split,
        "bucket": bucket,
        "page_id": item.get("page_id", "dataset_item"),
        "ground_truth_regions": record_count,
        "oracle_prefix": {
            "boundary_accuracy": float(boundary_accuracy.item()),
            "region_probability_mean": tensor_mean_or_none(
                target_probability[region_boundary]
            ),
            "eos_probability": tensor_mean_or_none(target_probability[eos_boundary]),
            "bbox_mean_iou": oracle_bbox_iou,
            "count_head_prediction": float(oracle.record_output.count[0].float().item()),
            "count_head_absolute_error": abs(
                float(oracle.record_output.count[0].float().item()) - record_count
            ),
            **oracle_duplicate,
        },
        "free_prefix": {
            "generated_regions": int(generated.num_generated_regions[0].item()),
            "generated_eos": bool(generated.generated_eos[0].item()),
            "truncated_by_layout_tokens": bool(
                generated.truncated_by_max_layout_tokens[0].item()
            ),
            "stopped_by_layout_records": bool(
                generated.stopped_by_max_layout_records[0].item()
            ),
            "layout_tokens": int(generated.num_layout_tokens[0].item()),
            **free_duplicate,
        },
    }


def summarize_oracle_free(pages: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pages": len(pages),
        "oracle_boundary_accuracy": mean_or_none(
            [page["oracle_prefix"]["boundary_accuracy"] for page in pages]
        ),
        "oracle_eos_probability": mean_or_none(
            [page["oracle_prefix"]["eos_probability"] for page in pages]
        ),
        "oracle_bbox_mean_iou": mean_or_none(
            [page["oracle_prefix"]["bbox_mean_iou"] for page in pages]
        ),
        "oracle_duplicate_rate": mean_or_none(
            [page["oracle_prefix"]["duplicate_after_first_rate"] for page in pages]
        ),
        "oracle_count_head_mae": mean_or_none(
            [page["oracle_prefix"]["count_head_absolute_error"] for page in pages]
        ),
        "free_generated_regions_mean": mean_or_none(
            [page["free_prefix"]["generated_regions"] for page in pages]
        ),
        "ground_truth_regions_mean": mean_or_none(
            [page["ground_truth_regions"] for page in pages]
        ),
        "free_duplicate_rate": mean_or_none(
            [page["free_prefix"]["duplicate_after_first_rate"] for page in pages]
        ),
        "free_eos_rate": mean_or_none(
            [float(page["free_prefix"]["generated_eos"]) for page in pages]
        ),
        "free_token_cap_rate": mean_or_none(
            [float(page["free_prefix"]["truncated_by_layout_tokens"]) for page in pages]
        ),
    }


def parameter_groups(model: Any) -> dict[str, list[Tensor]]:
    base = model.get_model()
    adapter = base.variable_layout_adapter
    writeback_modules = [adapter.visual_routing, adapter.writeback_output]
    groups = {
        "layout_evidence": list(adapter.decoder.prompt_attention.parameters()),
        "causal_decoder": [
            parameter
            for name, parameter in adapter.decoder.named_parameters()
            if not name.startswith("prompt_attention.")
        ],
        "record_heads": list(adapter.record_heads.parameters()),
        "visual_writeback": [parameter for module in writeback_modules for parameter in module.parameters()]
        + [adapter.residual_gate],
        "projector": list(base.mm_projector_vary.parameters()),
    }
    for name, values in groups.items():
        unique: dict[int, Tensor] = {}
        for value in values:
            unique[id(value)] = value
        groups[name] = list(unique.values())
    return groups


def gradient_audit(model: Any, batch: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    model.requires_grad_(False)
    groups = parameter_groups(model)
    parameters: list[Tensor] = []
    parameter_index: dict[int, int] = {}
    group_indices: dict[str, list[int]] = {}
    for name, values in groups.items():
        indices: list[int] = []
        for parameter in values:
            index = parameter_index.get(id(parameter))
            if index is None:
                index = len(parameters)
                parameter_index[id(parameter)] = index
                parameters.append(parameter)
                parameter.requires_grad_(True)
            indices.append(index)
        group_indices[name] = indices

    with torch.autocast(device_type="cuda", dtype=dtype):
        outputs = model(**batch, use_cache=False, return_dict=True)
    config = model.config
    losses = {
        "ocr": outputs.ocr_loss * float(config.ocr_loss_weight),
        "sequence": outputs.layout_sequence_loss * float(config.layout_loss_weight),
        "bbox_l1": outputs.layout_bbox_l1_loss
        * float(config.layout_bbox_loss_weight)
        * float(config.layout_loss_weight),
        "bbox_giou": outputs.layout_bbox_giou_loss
        * float(config.layout_bbox_giou_loss_weight)
        * float(config.layout_loss_weight),
        "type": outputs.layout_type_loss
        * float(config.layout_type_loss_weight)
        * float(config.layout_loss_weight),
        "direction": outputs.layout_direction_loss
        * float(config.layout_direction_loss_weight)
        * float(config.layout_loss_weight),
        "count": outputs.layout_count_loss
        * float(config.layout_count_loss_weight)
        * float(config.layout_loss_weight),
    }
    gradients: dict[str, list[Tensor | None]] = {}
    names = list(losses)
    for position, name in enumerate(names):
        values = torch.autograd.grad(
            losses[name],
            parameters,
            retain_graph=position + 1 < len(names),
            allow_unused=True,
        )
        gradients[name] = [value.detach().float().cpu() if value is not None else None for value in values]

    def group_norm(loss_name: str, group_name: str) -> float:
        squared = 0.0
        for index in group_indices[group_name]:
            value = gradients[loss_name][index]
            if value is not None:
                squared += float(value.square().sum().item())
        return math.sqrt(squared)

    def cosine(first: str, second: str, group_name: str) -> float | None:
        dot = first_norm = second_norm = 0.0
        for index in group_indices[group_name]:
            left = gradients[first][index]
            right = gradients[second][index]
            if left is None or right is None:
                continue
            dot += float((left * right).sum().item())
            first_norm += float(left.square().sum().item())
            second_norm += float(right.square().sum().item())
        if not first_norm or not second_norm:
            return None
        return dot / math.sqrt(first_norm * second_norm)

    report = {
        "weighted_loss_values": {
            name: float(value.detach().float().item()) for name, value in losses.items()
        },
        "gradient_norms": {
            name: {group: group_norm(name, group) for group in groups}
            for name in losses
        },
        "ocr_cosine_on_layout_evidence": {
            name: cosine("ocr", name, "layout_evidence")
            for name in losses
            if name != "ocr"
        },
        "optimizer_step_executed": False,
    }
    model.requires_grad_(False)
    return report


@contextmanager
def gate_value(adapter: Any, value: float | None) -> Iterator[None]:
    original = adapter.residual_gate.detach().clone()
    if value is not None:
        adapter.residual_gate.data.fill_(value)
    try:
        yield
    finally:
        adapter.residual_gate.data.copy_(original)


@contextmanager
def evidence_override(prompt_attention: Any, evidence: Sequence[Tensor] | None) -> Iterator[None]:
    if evidence is None:
        yield
        return
    original = prompt_attention.forward
    counter = 0

    def replacement(
        visual_tokens: Tensor,
        visual_padding_mask: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        nonlocal counter
        selected = evidence[counter].to(device=visual_tokens.device, dtype=visual_tokens.dtype)
        counter += 1
        return selected, None

    prompt_attention.forward = replacement
    try:
        yield
    finally:
        prompt_attention.forward = original


def precompute_evidence(model: Any, items: Sequence[dict[str, Any]], device: torch.device) -> list[Tensor]:
    prompt_attention = model.get_model().variable_layout_adapter.decoder.prompt_attention
    evidence: list[Tensor] = []
    for item in items:
        image_high = item["image_high"][0].unsqueeze(0).to(device)
        high_resolution = extract_high_resolution_features(model, image_high)
        with torch.no_grad():
            value, _ = prompt_attention(high_resolution, return_attention=False)
        evidence.append(value.detach())
    return evidence


def generation_kwargs(
    model: Any,
    tokenizer: Any,
    stop_string: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "no_repeat_ngram_size": 20,
    }
    stop_ids = tokenizer(stop_string, add_special_tokens=False).input_ids
    if len(stop_ids) == 1:
        eos_ids = {int(stop_ids[0])}
        configured = model.generation_config.eos_token_id
        if configured is not None:
            eos_ids.update(
                int(value)
                for value in (configured if isinstance(configured, list) else [configured])
            )
        kwargs["eos_token_id"] = sorted(eos_ids)
    return kwargs


def routing_condition(
    model: Any,
    tokenizer: Any,
    full_batch: dict[str, Any],
    prompt_batch: dict[str, Any],
    references: Sequence[str],
    stop_string: str,
    condition: str,
    swapped_evidence: Sequence[Tensor] | None,
    dtype: torch.dtype,
    max_new_tokens: int,
    ocr_accumulator_type: Any,
) -> dict[str, Any]:
    adapter = model.get_model().variable_layout_adapter
    gate = 0.0 if condition == "alpha_zero" else None
    evidence = swapped_evidence if condition == "shuffled_evidence" else None
    with gate_value(adapter, gate), evidence_override(adapter.decoder.prompt_attention, evidence):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            output = model(
                input_ids=full_batch["input_ids"],
                attention_mask=full_batch["attention_mask"],
                labels=full_batch["labels"],
                images=full_batch["images"],
                use_cache=False,
                return_dict=True,
            )
        teacher_forced_nll = float(output.ocr_loss.detach().float().item())
    with gate_value(adapter, gate), evidence_override(adapter.decoder.prompt_attention, evidence):
        started = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=dtype):
            generated = model.generate(
                prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                images=prompt_batch["images"],
                **generation_kwargs(model, tokenizer, stop_string, max_new_tokens),
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    accumulator = ocr_accumulator_type()
    prefix = prompt_batch["input_ids"].shape[1]
    lengths: list[int] = []
    for row, reference in enumerate(references):
        decoded = tokenizer.decode(generated[row, prefix:], skip_special_tokens=True).strip()
        if stop_string and decoded.endswith(stop_string):
            decoded = decoded[: -len(stop_string)].strip()
        accumulator.add_page(reference, decoded)
        lengths.append(len(decoded))
    return {
        "teacher_forced_ocr_nll": teacher_forced_nll,
        "generation": accumulator.summary(),
        "predicted_text_lengths": lengths,
        "seconds": seconds,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    project_root = args.project_root.resolve()
    add_project_paths(project_root)

    from GOT.model.GOT_ocr_2_0 import GOTQwenForCausalLM
    from GOT.model.plug.blip_process import BlipImageEvalProcessor
    from GOT.utils.utils import disable_torch_init
    from layout_page_dataset import (
        LayoutPageConversationDataset,
        LayoutPageDataCollator,
        LayoutPageValidationCollator,
        LayoutPageValidationDataset,
    )
    from layout_validation_metrics import OCRValidationAccumulator
    from local_tokenizer import load_local_tokenizer, tokenizer_candidates

    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    disable_torch_init()
    tokenizer, tokenizer_path = load_local_tokenizer(
        AutoTokenizer,
        tokenizer_candidates(args.tokenizer.resolve(), None),
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    tokenizer.model_max_length = 2048
    model = GOTQwenForCausalLM.from_pretrained(
        args.model.resolve(),
        low_cpu_mem_usage=True,
        use_safetensors=True,
        pad_token_id=151643,
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device=device, dtype=dtype).eval()
    if model.get_model().variable_layout_adapter is None:
        raise RuntimeError("The diagnostic checkpoint is not PVLD.")

    processor = BlipImageEvalProcessor(image_size=1024)
    multimodal = {
        "sep_image_conv_front": False,
        "image_token_len": 256,
        "image_aspect_ratio": "square",
        "use_im_start_end": True,
        "image_processor": processor,
        "image_processor_high": processor,
        "box_limit": 0,
    }
    dataset_kwargs = {
        "datasets": "layout-page-jsonl",
        "tokenizer": tokenizer,
        "multimodal_cfg": multimodal,
        "max_regions": 512,
        "max_records": 0,
        "supervise_ocr": True,
        "layout_target_mode": "pvld",
        "max_layout_tokens": int(model.config.max_layout_tokens),
        "max_layout_records": int(model.config.max_layout_records),
    }
    train_dataset = LayoutPageConversationDataset(
        manifest=args.train_manifest,
        image_root=args.train_image_root,
        split="train",
        **dataset_kwargs,
    )
    validation_dataset = LayoutPageConversationDataset(
        manifest=args.validation_manifest,
        image_root=args.validation_image_root,
        split="validation",
        **dataset_kwargs,
    )
    train_indices, train_buckets = selected_indices(train_dataset, args.pages_per_bucket)
    validation_indices, validation_buckets = selected_indices(
        validation_dataset, args.pages_per_bucket
    )

    oracle_free_pages: list[dict[str, Any]] = []
    for split, dataset, buckets in (
        ("train", train_dataset, train_buckets),
        ("validation", validation_dataset, validation_buckets),
    ):
        for bucket in BUCKET_NAMES:
            for index in buckets[bucket]:
                item = dataset[index]
                item["page_id"] = dataset.records[index]["page_id"]
                oracle_free_pages.append(
                    oracle_and_free_page(
                        model,
                        item,
                        split=split,
                        bucket=bucket,
                        device=device,
                        layout_token_limit=args.layout_token_limit,
                        layout_record_limit=args.layout_record_limit,
                    )
                )

    gradient_index = validation_buckets["17-32"][0]
    tokenizer.padding_side = "right"
    gradient_item = validation_dataset[gradient_index]
    gradient_batch = LayoutPageDataCollator(tokenizer)([gradient_item])
    gradient_batch = move_batch(gradient_batch, device)
    gradients = gradient_audit(model, gradient_batch, dtype)

    ocr_candidates = [
        *validation_buckets["0-8"],
        *validation_buckets[">32"],
        *validation_buckets["9-16"],
        *validation_buckets["17-32"],
    ]
    if args.ocr_pages > len(ocr_candidates):
        raise RuntimeError(
            f"--ocr-pages={args.ocr_pages} exceeds the {len(ocr_candidates)} "
            "selected validation diagnostic pages."
        )
    ocr_indices = ocr_candidates[: args.ocr_pages]
    full_items = [validation_dataset[index] for index in ocr_indices]
    tokenizer.padding_side = "right"
    full_batch = move_batch(LayoutPageDataCollator(tokenizer)(full_items), device)
    prompt_dataset = LayoutPageValidationDataset(
        tokenizer=tokenizer,
        datasets="layout-page-jsonl",
        multimodal_cfg=multimodal,
        manifest=args.validation_manifest,
        image_root=args.validation_image_root,
        split="validation",
        max_regions=512,
        max_records=0,
    )
    prompt_items = [prompt_dataset[index] for index in ocr_indices]
    tokenizer.padding_side = "left"
    prompt_batch = move_batch(LayoutPageValidationCollator(tokenizer)(prompt_items), device)
    evidence = precompute_evidence(model, full_items, device)
    swapped = list(reversed(evidence))
    stop_strings = {item["stop_string"] for item in prompt_items}
    if len(stop_strings) != 1:
        raise RuntimeError("OCR diagnostic pages must share one stop string.")
    stop_string = next(iter(stop_strings))
    references = [validation_dataset.records[index]["page_text"] for index in ocr_indices]
    routing = {}
    for condition in ("normal", "alpha_zero", "shuffled_evidence"):
        routing[condition] = routing_condition(
            model,
            tokenizer,
            full_batch,
            prompt_batch,
            references,
            stop_string,
            condition,
            swapped,
            dtype,
            args.ocr_max_new_tokens,
            OCRValidationAccumulator,
        )

    torch.cuda.synchronize()
    summary = {
        "status": "completed",
        "label": args.label,
        "model": str(args.model.resolve()),
        "tokenizer": str(tokenizer_path),
        "protocol": {
            "splits_read": ["train", "validation"],
            "test_read": False,
            "optimizer_steps": 0,
            "checkpoint_written": False,
            "whole_page_image_only": True,
            "metadata_as_inference_input": False,
            "layout_token_limit": args.layout_token_limit,
            "layout_record_limit": args.layout_record_limit,
            "selected_train_indices": train_indices,
            "selected_validation_indices": validation_indices,
            "ocr_validation_indices": ocr_indices,
        },
        "oracle_vs_free": {
            "summary": summarize_oracle_free(oracle_free_pages),
            "by_split": {
                split: summarize_oracle_free(
                    [page for page in oracle_free_pages if page["split"] == split]
                )
                for split in ("train", "validation")
            },
            "pages": oracle_free_pages,
        },
        "gradient_audit": gradients,
        "ocr_routing_ablation": {
            "pages": [
                {
                    "page_id": validation_dataset.records[index]["page_id"],
                    "regions": record_region_count(validation_dataset.records[index]),
                }
                for index in ocr_indices
            ],
            "residual_gate": float(
                model.get_model().variable_layout_adapter.residual_gate.detach().float().item()
            ),
            "conditions": routing,
        },
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    write_json(output, summary)
    print(
        compact(
            {
                "event": "pvld_bounded_diagnostics_completed",
                "label": args.label,
                "output": str(output),
                "oracle_vs_free": summary["oracle_vs_free"]["summary"],
                "routing": {
                    name: {
                        "nll": result["teacher_forced_ocr_nll"],
                        "cer": result["generation"]["page_cer"],
                    }
                    for name, result in routing.items()
                },
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
