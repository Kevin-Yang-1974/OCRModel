from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset

from GOT.data import DataCollatorForSupervisedDataset
from GOT.utils import conversation as conversation_lib
from GOT.utils.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IGNORE_INDEX,
)


class LineLevelConversationDataset(Dataset):
    """Local GOT dataset restricted to one text/symbol line per image."""

    def __init__(
        self,
        datasets: str,
        tokenizer: Any,
        multimodal_cfg: dict[str, Any],
        annotations: Path,
        image_root: Path,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.multimodal_cfg = multimodal_cfg
        conversation_lib.default_conversation = conversation_lib.conv_templates["mpt"]

        if datasets != "linelevel-json":
            raise ValueError(
                f"This entry point only accepts --datasets linelevel-json, got {datasets!r}."
            )

        self.annotations = annotations.resolve()
        self.image_root = image_root.resolve()
        if not self.annotations.is_file():
            raise FileNotFoundError(f"Annotations do not exist: {self.annotations}")
        if not self.image_root.is_dir():
            raise FileNotFoundError(f"Image root does not exist: {self.image_root}")

        records = json.loads(self.annotations.read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise ValueError("Annotations must be a non-empty JSON list.")

        self.records: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(f"Record {index} is not a JSON object.")
            if record.get("input_level") != "line":
                raise ValueError(
                    f"Record {index} is not explicitly marked input_level='line'."
                )

            image_value = record.get("image")
            if not isinstance(image_value, str) or not image_value:
                raise ValueError(f"Record {index} has no relative image path.")
            relative_image = Path(image_value)
            if relative_image.is_absolute() or ".." in relative_image.parts:
                raise ValueError(f"Record {index} image must stay under image_root: {image_value}")
            image_path = (self.image_root / relative_image).resolve()
            try:
                image_path.relative_to(self.image_root)
            except ValueError as exc:
                raise ValueError(f"Record {index} escapes image_root: {image_value}") from exc
            if not image_path.is_file():
                raise FileNotFoundError(f"Record {index} image does not exist: {image_path}")
            with Image.open(image_path) as image:
                if image.width < 1 or image.height < 1:
                    raise ValueError(f"Record {index} has an empty image: {image_path}")

            conversations = record.get("conversations")
            if not isinstance(conversations, list) or len(conversations) != 2:
                raise ValueError(f"Record {index} must contain one human/GPT conversation pair.")
            human, assistant = conversations
            if not isinstance(human, dict) or not isinstance(assistant, dict):
                raise TypeError(f"Record {index} conversations must be JSON objects.")
            if human.get("from") != "human" or "<image>" not in str(human.get("value", "")):
                raise ValueError(f"Record {index} has an invalid image prompt.")
            target = str(assistant.get("value", ""))
            if assistant.get("from") != "gpt" or not target.strip():
                raise ValueError(f"Record {index} has an invalid transcription target.")
            if "\n" in target or "\r" in target:
                raise ValueError(
                    f"Record {index} target contains a line break; split it into line-level samples."
                )
            self.records.append(record)

        logging.warning(
            "Loading %d validated line-level conversations from %s",
            len(self.records),
            self.annotations,
        )

    def __len__(self) -> int:
        return len(self.records)

    def image_processor(self, image: Image.Image) -> torch.Tensor:
        return self.multimodal_cfg["image_processor_high"](image)

    def multimodal_processor(self, sources: list[list[dict[str, str]]]) -> list[list[dict[str, str]]]:
        for source in sources:
            if self.multimodal_cfg["sep_image_conv_front"]:
                if DEFAULT_IMAGE_TOKEN not in source[0]["value"]:
                    raise ValueError("The human prompt does not contain <image>.")
                source[0]["value"] = source[0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                source[0]["value"] = (
                    DEFAULT_IMAGE_TOKEN
                    + conversation_lib.default_conversation.sep
                    + conversation_lib.default_conversation.roles[0]
                    + ": "
                    + source[0]["value"]
                )

            replace_token = DEFAULT_IM_START_TOKEN + (
                DEFAULT_IMAGE_PATCH_TOKEN * self.multimodal_cfg["image_token_len"]
            ) + DEFAULT_IM_END_TOKEN
            for sentence in source:
                sentence["value"] = str(sentence["value"]).replace(
                    DEFAULT_IMAGE_TOKEN,
                    replace_token,
                )
        return sources

    def token_processor(
        self,
        sources: list[list[dict[str, str]]],
        image_name: str,
    ) -> dict[str, torch.Tensor]:
        conv = conversation_lib.default_conversation.copy()
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations: list[str] = []
        for source in sources:
            if roles[source[0]["from"]] != conv.roles[0]:
                source = source[1:]
            conv.messages = []
            for message_index, sentence in enumerate(source):
                role = roles[sentence["from"]]
                if role != conv.roles[message_index % 2]:
                    raise ValueError(f"Conversation role order is invalid for {image_name}.")
                conv.append_message(role, sentence["value"])
            conversations.append(conv.get_prompt())

        input_ids = self.tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        targets = input_ids.clone()
        sep = conv.sep + conv.roles[1]
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(self.tokenizer.pad_token_id).sum())
            rounds = conversation.split(conv.sep)
            grouped_rounds = [conv.sep.join(rounds[:3])]
            for round_index in range(3, len(rounds), 2):
                grouped_rounds.append(conv.sep.join(rounds[round_index : round_index + 2]))

            current_length = 0
            target[:current_length] = IGNORE_INDEX
            for grouped_round in grouped_rounds:
                if not grouped_round:
                    break
                parts = grouped_round.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_length = len(self.tokenizer(grouped_round).input_ids) + len(
                    self.tokenizer(conv.sep).input_ids
                )
                instruction_length = len(self.tokenizer(parts[0]).input_ids)
                target[current_length : current_length + instruction_length] = IGNORE_INDEX
                current_length += round_length
            target[current_length:] = IGNORE_INDEX

            if current_length < self.tokenizer.model_max_length and current_length != total_len:
                target[:] = IGNORE_INDEX
                raise RuntimeError(
                    f"Tokenization mismatch for {image_name}: {current_length} vs {total_len}."
                )
        return {"input_ids": input_ids, "labels": targets}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = copy.deepcopy(self.records[index])
        image_path = (self.image_root / record["image"]).resolve()
        with Image.open(image_path) as source:
            image = self.image_processor(source.convert("RGB"))

        conversations = self.multimodal_processor([record["conversations"]])
        tokens = self.token_processor(conversations, str(image_path))
        return {
            "input_ids": tokens["input_ids"][0],
            "labels": tokens["labels"][0],
            "image": [image],
            "image_high": [image],
        }


def make_linelevel_data_module(
    tokenizer: Any,
    data_args: Any,
    annotations: Path,
    image_root: Path,
) -> dict[str, Any]:
    if data_args.conversation_version != "mpt":
        raise ValueError("The line-level GOT entry point currently requires conversation_version=mpt.")

    train_dataset = LineLevelConversationDataset(
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
        annotations=annotations,
        image_root=image_root,
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": DataCollatorForSupervisedDataset(tokenizer=tokenizer),
    }
