from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from torch import Tensor


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise ValueError(f"empty manifest: {path}")
    page_ids: set[str] = set()
    for record in records:
        page_id = str(record.get("page_id", ""))
        if not page_id or page_id in page_ids:
            raise ValueError(f"duplicate or missing page_id in manifest: {path}")
        page_ids.add(page_id)
        image = record.get("image_path") or record.get("image")
        if not image:
            raise ValueError(f"record {page_id} has no image or image_path: {path}")
        image_path = Path(str(image))
        if not image_path.is_absolute():
            image_path = path.parent / image_path
        record["image_path"] = str(image_path.resolve())
        if not isinstance(record.get("page_text"), str) or not record["page_text"]:
            raise ValueError(f"record {page_id} has no page_text: {path}")
    return records


def validate_records(
    records: list[dict[str, Any]], *, split: str | None = None, num_queries: int | None = None
) -> None:
    """Validate the data contracts that affect training correctness."""

    for record in records:
        page_id = record["page_id"]
        if split is not None:
            record_split = record.get("split", record.get("official_split"))
            if record_split != split:
                raise ValueError(
                    f"record {page_id} has split {record_split!r}, expected {split!r}"
                )
        image_path = Path(record["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"image for {page_id} does not exist: {image_path}")
        regions = record.get("regions")
        if not isinstance(regions, list) or not regions:
            raise ValueError(f"record {page_id} has no regions")
        if num_queries is not None and len(regions) > num_queries:
            raise ValueError(
                f"record {page_id} has {len(regions)} regions, exceeding num_queries={num_queries}"
            )


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


def _token_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value if item is not None}
    return {int(value)}


def _processor_eos_ids(processor: Any) -> set[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    return _token_id_set(getattr(tokenizer, "eos_token_id", None))


def append_eos_label_token(
    inputs: dict[str, Any], eos_token_ids: Iterable[int]
) -> dict[str, Any]:
    """Append exactly one model EOS token to a tokenized assistant target."""

    eos_ids = sorted({int(token_id) for token_id in eos_token_ids})
    if not eos_ids:
        return inputs
    input_ids = inputs["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("GLM-OCR training inputs must contain one page at a time")
    if int(input_ids[0, -1].item()) in eos_ids:
        return inputs
    eos = torch.tensor([[eos_ids[0]],], dtype=input_ids.dtype, device=input_ids.device)
    inputs["input_ids"] = torch.cat((input_ids, eos), dim=1)
    attention_mask = inputs.get("attention_mask")
    if isinstance(attention_mask, Tensor):
        ones = torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        inputs["attention_mask"] = torch.cat((attention_mask, ones), dim=1)
    # GLM-OCR's processor returns ``mm_token_type_ids`` rather than the
    # legacy ``token_type_ids`` key.  The model indexes this tensor with the
    # attention mask, so an appended EOS must extend it as well.  EOS belongs
    # to the text stream and therefore reuses the final text-stream type.
    mm_token_type_ids = inputs.get("mm_token_type_ids")
    if isinstance(mm_token_type_ids, Tensor):
        tail = mm_token_type_ids[:, -1:].clone()
        inputs["mm_token_type_ids"] = torch.cat((mm_token_type_ids, tail), dim=1)
    return inputs


def prepare_training_inputs(
    processor: Any,
    record: dict[str, Any],
    device: torch.device,
    eos_token_ids: Iterable[int] | None = None,
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

    inputs = append_eos_label_token(
        dict(inputs), _processor_eos_ids(processor) if eos_token_ids is None else eos_token_ids
    )

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


def region_decoder_targets(
    record: dict[str, Any], device: torch.device, max_regions: int = 512
) -> dict[str, Tensor]:
    """Build ordered region labels for the autoregressive candidate decoder."""

    regions = sorted(record["regions"], key=lambda item: int(item["reading_order"]))
    if len(regions) > max_regions:
        raise ValueError(
            f"{record['page_id']} has {len(regions)} regions for max_regions={max_regions}"
        )
    boxes = torch.zeros(1, max_regions, 4, dtype=torch.float32, device=device)
    directions = torch.zeros(1, max_regions, dtype=torch.long, device=device)
    mask = torch.zeros(1, max_regions, dtype=torch.bool, device=device)
    direction_ids = {"vertical_rtl": 0, "horizontal_ltr": 1, "unknown": 2}
    for index, region in enumerate(regions):
        boxes[0, index] = torch.tensor(region["bbox"], dtype=torch.float32, device=device)
        directions[0, index] = direction_ids.get(
            region.get("writing_direction", "unknown"), 2
        )
        mask[0, index] = True
    return {
        "target_boxes": boxes,
        "target_directions": directions,
        "query_mask": mask,
    }
