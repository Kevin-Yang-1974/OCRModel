from __future__ import annotations

import copy
import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from GOT.data import DataCollatorForSupervisedDataset
from GOT.utils import conversation as conversation_lib
from GOT.utils.constants import IGNORE_INDEX
from linelevel_dataset import LineLevelConversationDataset


DIRECTION_TO_INDEX = {
    "horizontal_ltr": 0,
    "horizontal_rtl": 1,
    "vertical_rtl": 2,
    "vertical_ltr": 3,
    "unknown": 4,
}
LAYOUT_ANNOTATION_STATUSES = {"complete", "partial", "none"}


def load_manifest_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"Layout manifest is empty: {path}")
    parsed = None
    if path.suffix.lower() != ".jsonl":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
    if parsed is not None:
        if isinstance(parsed, dict):
            records = parsed.get("records")
        else:
            records = parsed
    else:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    if not isinstance(records, list) or not records:
        raise ValueError(f"Layout manifest must contain a non-empty record list: {path}")
    if any(not isinstance(record, dict) for record in records):
        raise TypeError(f"Every layout manifest record must be a JSON object: {path}")
    return records


def resolve_image_path(image_root: Path, value: Any, context: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.image must be a non-empty relative path.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{context}.image is unsafe: {value!r}.")
    resolved = image_root / Path(*relative.parts)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def validate_bbox(value: Any, context: str) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(not isinstance(item, (int, float)) for item in value)
    ):
        raise ValueError(f"{context}.bbox must contain four numbers.")
    x0, y0, x1, y1 = (float(item) for item in value)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"{context}.bbox must be normalized xyxy, got {value}.")
    return x0, y0, x1, y1


class LayoutPageConversationDataset(LineLevelConversationDataset):
    """Whole-page GOT2 OCR samples with optional layout-query supervision."""

    def __init__(
        self,
        datasets: str,
        tokenizer: Any,
        multimodal_cfg: dict[str, Any],
        manifest: Path,
        image_root: Path | None,
        split: str,
        max_regions: int,
        max_records: int = 0,
        supervise_ocr: bool = True,
    ) -> None:
        Dataset.__init__(self)
        if datasets != "layout-page-jsonl":
            raise ValueError(
                f"This entry point only accepts --datasets layout-page-jsonl, got {datasets!r}."
            )
        if not split:
            raise ValueError("split must be non-empty.")
        if max_regions < 1:
            raise ValueError("max_regions must be positive.")
        if max_records < 0:
            raise ValueError("max_records cannot be negative.")

        self.tokenizer = tokenizer
        self.multimodal_cfg = multimodal_cfg
        self.manifest = manifest.resolve()
        self.image_root = (image_root or self.manifest.parent).resolve()
        self.split = split
        self.max_regions = max_regions
        self.supervise_ocr = supervise_ocr
        conversation_lib.default_conversation = conversation_lib.conv_templates["mpt"]
        if not self.image_root.is_dir():
            raise FileNotFoundError(self.image_root)

        all_records = load_manifest_records(self.manifest)
        seen_pages: set[str] = set()
        group_splits: dict[str, str] = {}
        content_splits: dict[str, str] = {}
        selected: list[dict[str, Any]] = []
        for record_index, record in enumerate(all_records):
            context = f"{self.manifest.name}[{record_index}]"
            page_id = record.get("page_id")
            record_split = record.get("split")
            if not isinstance(page_id, str) or not page_id:
                raise ValueError(f"{context}.page_id must be a non-empty string.")
            if page_id in seen_pages:
                raise ValueError(f"Duplicate page_id in manifest: {page_id!r}.")
            seen_pages.add(page_id)
            if not isinstance(record_split, str) or not record_split:
                raise ValueError(f"{context}.split must be a non-empty string.")
            if record.get("input_level") != "page":
                raise ValueError(f"{context}.input_level must be 'page'.")
            annotation_status = record.get("layout_annotation_status", "none")
            if annotation_status not in LAYOUT_ANNOTATION_STATUSES:
                raise ValueError(
                    f"{context}.layout_annotation_status must be one of "
                    f"{sorted(LAYOUT_ANNOTATION_STATUSES)}."
                )

            image_path = resolve_image_path(self.image_root, record.get("image"), context)
            with Image.open(image_path) as image:
                if image.width < 1 or image.height < 1:
                    raise ValueError(f"{context} has an empty image: {image_path}")

            page_text = record.get("page_text")
            if not isinstance(page_text, str) or not page_text:
                raise ValueError(f"{context}.page_text must be a non-empty string.")
            conversations = record.get("conversations")
            expected_conversations = [
                {"from": "human", "value": "<image>\nOCR: "},
                {"from": "gpt", "value": page_text},
            ]
            if conversations != expected_conversations:
                raise ValueError(f"{context}.conversations does not match page_text.")

            regions = record.get("regions", [])
            if not isinstance(regions, list):
                raise TypeError(f"{context}.regions must be a list.")
            if len(regions) > max_regions:
                raise ValueError(
                    f"{context} has {len(regions)} regions, exceeding max_regions={max_regions}."
                )
            if annotation_status == "complete" and not regions:
                raise ValueError(f"{context} has complete layout status but no regions.")
            if annotation_status == "none" and regions:
                raise ValueError(f"{context} has layout status none but declares regions.")

            expected_orders = list(range(len(regions)))
            actual_orders = [
                region.get("reading_order") if isinstance(region, dict) else None
                for region in regions
            ]
            if actual_orders != expected_orders:
                raise ValueError(
                    f"{context} regions must be stored in contiguous reading order."
                )
            for region_index, region in enumerate(regions):
                region_context = f"{context}.regions[{region_index}]"
                if not isinstance(region, dict):
                    raise TypeError(f"{region_context} must be a JSON object.")
                validate_bbox(region.get("bbox"), region_context)
                direction = region.get("writing_direction", "unknown")
                if direction not in DIRECTION_TO_INDEX:
                    raise ValueError(
                        f"{region_context}.writing_direction is unsupported: {direction!r}."
                    )
                content_id = region.get("content_id")
                source_group_id = region.get("source_group_id")
                if not isinstance(content_id, str) or not content_id:
                    raise ValueError(f"{region_context}.content_id must be non-empty.")
                if not isinstance(source_group_id, str) or not source_group_id:
                    raise ValueError(f"{region_context}.source_group_id must be non-empty.")
                previous_content_split = content_splits.setdefault(content_id, record_split)
                previous_group_split = group_splits.setdefault(source_group_id, record_split)
                if previous_content_split != record_split:
                    raise ValueError(
                        f"content_id occurs in multiple splits: {content_id!r}."
                    )
                if previous_group_split != record_split:
                    raise ValueError(
                        f"source_group_id occurs in multiple splits: {source_group_id!r}."
                    )

            if record_split == split:
                selected.append(record)

        if max_records:
            selected = selected[:max_records]
        if not selected:
            raise RuntimeError(f"No layout page records selected for split={split!r}.")
        self.records = selected
        logging.warning(
            "Loading %d whole-page layout conversations from %s "
            "(split=%s, max_regions=%d, max_records=%d)",
            len(self.records),
            self.manifest,
            split,
            max_regions,
            max_records,
        )

    def _layout_targets(self, record: dict[str, Any]) -> dict[str, torch.Tensor]:
        boxes = torch.zeros((self.max_regions, 4), dtype=torch.float32)
        bbox_mask = torch.zeros(self.max_regions, dtype=torch.bool)
        object_targets = torch.zeros(self.max_regions, dtype=torch.float32)
        object_mask = torch.zeros(self.max_regions, dtype=torch.bool)
        direction_targets = torch.full(
            (self.max_regions,),
            fill_value=-100,
            dtype=torch.long,
        )

        annotation_status = record.get("layout_annotation_status", "none")
        regions = record.get("regions", [])
        if annotation_status == "complete":
            object_mask[:] = True
        elif annotation_status == "partial":
            object_mask[: len(regions)] = True

        for query_index, region in enumerate(regions):
            object_targets[query_index] = 1.0
            boxes[query_index] = torch.tensor(
                validate_bbox(region.get("bbox"), f"regions[{query_index}]"),
                dtype=torch.float32,
            )
            bbox_mask[query_index] = True
            direction = region.get("writing_direction", "unknown")
            direction_targets[query_index] = DIRECTION_TO_INDEX[direction]

        return {
            "layout_bbox_targets": boxes,
            "layout_bbox_mask": bbox_mask,
            "layout_object_targets": object_targets,
            "layout_object_mask": object_mask,
            "layout_direction_targets": direction_targets,
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = copy.deepcopy(self.records[index])
        image_path = resolve_image_path(self.image_root, record["image"], record["page_id"])
        with Image.open(image_path) as source:
            image = self.image_processor(source.convert("RGB"))

        conversations = self.multimodal_processor([record["conversations"]])
        tokens = self.token_processor(conversations, str(image_path))
        labels = tokens["labels"][0]
        if not self.supervise_ocr:
            labels = torch.full_like(labels, fill_value=IGNORE_INDEX)
        item = {
            "input_ids": tokens["input_ids"][0],
            "labels": labels,
            "image": [image],
            "image_high": [image],
        }
        item.update(self._layout_targets(record))
        return item


class LayoutPageValidationDataset(LayoutPageConversationDataset):
    """Prompt-only whole-page samples for generation and layout evaluation."""

    def __init__(
        self,
        datasets: str,
        tokenizer: Any,
        multimodal_cfg: dict[str, Any],
        manifest: Path,
        image_root: Path | None,
        split: str,
        max_regions: int,
        max_records: int = 0,
    ) -> None:
        super().__init__(
            datasets=datasets,
            tokenizer=tokenizer,
            multimodal_cfg=multimodal_cfg,
            manifest=manifest,
            image_root=image_root,
            split=split,
            max_regions=max_regions,
            max_records=max_records,
            supervise_ocr=False,
        )

    def _prompt_input_ids(self, record: dict[str, Any]) -> tuple[torch.Tensor, str, str]:
        human_message = copy.deepcopy(record["conversations"][0])
        processed = self.multimodal_processor([[human_message]])[0][0]["value"]
        conversation = conversation_lib.default_conversation.copy()
        conversation.messages = []
        conversation.append_message(conversation.roles[0], processed)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()
        input_ids = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=False,
            truncation=False,
        ).input_ids[0]
        if input_ids.numel() > self.tokenizer.model_max_length:
            raise RuntimeError(
                f"Validation prompt for {record['page_id']} has {input_ids.numel()} tokens, "
                f"exceeding model_max_length={self.tokenizer.model_max_length}."
            )
        stop_string = (
            conversation.sep
            if conversation.sep_style != conversation_lib.SeparatorStyle.TWO
            else conversation.sep2
        )
        return input_ids, prompt, stop_string

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = copy.deepcopy(self.records[index])
        image_path = resolve_image_path(self.image_root, record["image"], record["page_id"])
        with Image.open(image_path) as source:
            image = self.image_processor(source.convert("RGB"))
        input_ids, prompt, stop_string = self._prompt_input_ids(record)
        item: dict[str, Any] = {
            "input_ids": input_ids,
            "image": [image],
            "image_high": [image],
            "page_id": record["page_id"],
            "page_text": record["page_text"],
            "regions": record.get("regions", []),
            "layout_annotation_status": record.get("layout_annotation_status", "none"),
            "image_path": str(image_path),
            "prompt": prompt,
            "stop_string": stop_string,
        }
        item.update(self._layout_targets(record))
        return item


class LayoutPageDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.base = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
        batch = self.base(instances)
        for key in (
            "layout_bbox_targets",
            "layout_bbox_mask",
            "layout_object_targets",
            "layout_object_mask",
            "layout_direction_targets",
        ):
            batch[key] = torch.stack([instance[key] for instance in instances])
        return batch


class InterleavedLayoutDataset(Dataset):
    """Deterministic primary:replay schedule for small-domain adaptation."""

    def __init__(
        self,
        primary: Dataset[Any],
        replay: Dataset[Any],
        *,
        primary_per_replay: int = 3,
    ) -> None:
        if len(primary) < 1 or len(replay) < 1:
            raise ValueError("primary and replay datasets must be non-empty.")
        if primary_per_replay < 1:
            raise ValueError("primary_per_replay must be positive.")
        self.primary = primary
        self.replay = replay
        self.primary_per_replay = primary_per_replay
        self.period = primary_per_replay + 1

    def __len__(self) -> int:
        return len(self.primary) + max(1, len(self.primary) // self.primary_per_replay)

    def __getitem__(self, index: int) -> Any:
        if index % self.period == self.primary_per_replay:
            replay_index = (index // self.period) % len(self.replay)
            return self.replay[replay_index]
        primary_index = (index - index // self.period) % len(self.primary)
        return self.primary[primary_index]


class LayoutPageValidationCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not instances:
            raise ValueError("Validation collator received an empty batch.")
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [instance["input_ids"] for instance in instances],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        images = [torch.stack(instance["image"]) for instance in instances]
        images_high = [torch.stack(instance["image_high"]) for instance in instances]
        batch: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "images": list(zip(images, images_high)),
        }
        for key in (
            "layout_bbox_targets",
            "layout_bbox_mask",
            "layout_object_targets",
            "layout_object_mask",
            "layout_direction_targets",
        ):
            batch[key] = torch.stack([instance[key] for instance in instances])
        for key in (
            "page_id",
            "page_text",
            "regions",
            "layout_annotation_status",
            "image_path",
            "prompt",
            "stop_string",
        ):
            batch[key] = [instance[key] for instance in instances]
        return batch


def make_layout_page_data_module(
    tokenizer: Any,
    data_args: Any,
    manifest: Path,
    image_root: Path | None,
    split: str,
    max_regions: int,
    max_records: int = 0,
    supervise_ocr: bool = True,
    replay_manifest: Path | None = None,
    replay_image_root: Path | None = None,
    replay_split: str | None = None,
    replay_max_records: int = 0,
    primary_per_replay: int = 3,
) -> dict[str, Any]:
    if data_args.conversation_version != "mpt":
        raise ValueError("The whole-page layout entry point requires conversation_version=mpt.")
    train_dataset = LayoutPageConversationDataset(
        tokenizer=tokenizer,
        datasets=data_args.datasets,
        multimodal_cfg={
            "sep_image_conv_front": data_args.sep_image_conv_front,
            "image_token_len": data_args.image_token_len,
            "image_aspect_ratio": data_args.image_aspect_ratio,
            "use_im_start_end": data_args.use_im_start_end,
            "image_processor": data_args.image_processor,
            "image_processor_high": data_args.image_processor_high,
            "box_limit": data_args.box_limit,
        },
        manifest=manifest,
        image_root=image_root,
        split=split,
        max_regions=max_regions,
        max_records=max_records,
        supervise_ocr=supervise_ocr,
    )
    if replay_manifest is not None:
        if primary_per_replay < 1:
            raise ValueError("primary_per_replay must be positive.")
        replay_dataset = LayoutPageConversationDataset(
            tokenizer=tokenizer,
            datasets=data_args.datasets,
            multimodal_cfg={
                "sep_image_conv_front": data_args.sep_image_conv_front,
                "image_token_len": data_args.image_token_len,
                "image_aspect_ratio": data_args.image_aspect_ratio,
                "use_im_start_end": data_args.use_im_start_end,
                "image_processor": data_args.image_processor,
                "image_processor_high": data_args.image_processor_high,
                "box_limit": data_args.box_limit,
            },
            manifest=replay_manifest,
            image_root=replay_image_root,
            split=replay_split or split,
            max_regions=max_regions,
            max_records=replay_max_records,
            supervise_ocr=supervise_ocr,
        )
        train_dataset = InterleavedLayoutDataset(
            train_dataset,
            replay_dataset,
            primary_per_replay=primary_per_replay,
        )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": LayoutPageDataCollator(tokenizer=tokenizer),
    }


def make_layout_page_validation_data_module(
    tokenizer: Any,
    manifest: Path,
    image_root: Path | None,
    split: str,
    max_regions: int,
    image_processor: Any,
    image_token_len: int = 256,
    max_records: int = 0,
) -> dict[str, Any]:
    eval_dataset = LayoutPageValidationDataset(
        tokenizer=tokenizer,
        datasets="layout-page-jsonl",
        multimodal_cfg={
            "sep_image_conv_front": False,
            "image_token_len": image_token_len,
            "image_aspect_ratio": "square",
            "use_im_start_end": True,
            "image_processor": image_processor,
            "image_processor_high": image_processor,
            "box_limit": 0,
        },
        manifest=manifest,
        image_root=image_root,
        split=split,
        max_regions=max_regions,
        max_records=max_records,
    )
    return {
        "eval_dataset": eval_dataset,
        "data_collator": LayoutPageValidationCollator(tokenizer=tokenizer),
    }
