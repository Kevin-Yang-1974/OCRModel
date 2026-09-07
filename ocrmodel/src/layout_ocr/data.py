from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"empty manifest: {path}")
    return records


def _messages(target: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Text Recognition:"},
            ],
        }
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def prepare_training_inputs(
    processor: Any, record: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    image_path = Path(record["image_path"])
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        prompt = processor.apply_chat_template(
            _messages(), tokenize=False, add_generation_prompt=True
        )
        full = processor.apply_chat_template(
            _messages(record["page_text"]), tokenize=False, add_generation_prompt=False
        )
        prompt_inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        inputs = processor(text=[full], images=[image], return_tensors="pt")

    prompt_ids = prompt_inputs["input_ids"]
    full_ids = inputs["input_ids"]
    prompt_length = prompt_ids.shape[1]
    if full_ids.shape[1] <= prompt_length or not torch.equal(full_ids[:, :prompt_length], prompt_ids):
        raise RuntimeError("assistant target does not extend the GLM-OCR prompt token prefix")
    labels = full_ids.clone()
    labels[:, :prompt_length] = -100
    inputs["labels"] = labels
    inputs.pop("token_type_ids", None)
    return _to_device(dict(inputs), device)


def prepare_inference_inputs(
    processor: Any, record: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    image_path = Path(record["image_path"])
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        prompt = processor.apply_chat_template(
            _messages(), tokenize=False, add_generation_prompt=True
        )
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
    inputs.pop("token_type_ids", None)
    return _to_device(dict(inputs), device)


def layout_targets(
    record: dict[str, Any], positions: Tensor, num_queries: int
) -> dict[str, Tensor]:
    regions = sorted(record["regions"], key=lambda item: int(item["reading_order"]))
    if len(regions) > num_queries:
        raise ValueError(f"{record['page_id']} has {len(regions)} regions for {num_queries} queries")
    device = positions.device
    boxes = torch.zeros(1, num_queries, 4, dtype=torch.float32, device=device)
    orders = torch.zeros(1, num_queries, dtype=torch.float32, device=device)
    directions = torch.zeros(1, num_queries, dtype=torch.long, device=device)
    mask = torch.zeros(1, num_queries, dtype=torch.bool, device=device)
    direction_ids = {"vertical_rtl": 0, "horizontal_ltr": 1, "unknown": 2}
    for index, region in enumerate(regions):
        boxes[0, index] = torch.tensor(region["bbox"], dtype=torch.float32, device=device)
        orders[0, index] = index / max(1, len(regions) - 1)
        directions[0, index] = direction_ids.get(region.get("writing_direction", "unknown"), 2)
        mask[0, index] = True

    owners = torch.full((1, positions.shape[1]), -1, dtype=torch.long, device=device)
    if regions:
        centers = positions[0]
        valid_boxes = boxes[0, : len(regions)]
        inside = (
            (centers[:, None, 0] >= valid_boxes[None, :, 0])
            & (centers[:, None, 0] <= valid_boxes[None, :, 2])
            & (centers[:, None, 1] >= valid_boxes[None, :, 1])
            & (centers[:, None, 1] <= valid_boxes[None, :, 3])
        )
        has_owner = inside.any(dim=-1)
        owners[0, has_owner] = inside[has_owner].float().argmax(dim=-1)
    return {
        "target_boxes": boxes,
        "target_orders": orders,
        "target_directions": directions,
        "query_mask": mask,
        "token_owners": owners,
    }
