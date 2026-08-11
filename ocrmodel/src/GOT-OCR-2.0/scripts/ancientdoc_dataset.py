from __future__ import annotations

import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from GOT.data import DataCollatorForSupervisedDataset
from GOT.utils import conversation as conversation_lib

from linelevel_dataset import LineLevelConversationDataset


def parse_split_ids(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        try:
            split_ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid AncientDoc split list: {value!r}") from exc
    else:
        split_ids = tuple(int(part) for part in value)
    if not split_ids or len(set(split_ids)) != len(split_ids):
        raise ValueError(f"AncientDoc split IDs must be non-empty and unique: {split_ids}")
    if any(split_id not in range(1, 6) for split_id in split_ids):
        raise ValueError(f"AncientDoc split IDs must be between 1 and 5: {split_ids}")
    return split_ids


class AncientDocPageDataset(LineLevelConversationDataset):
    """Read the reference AncientDoc page labels without modifying shared data."""

    def __init__(
        self,
        datasets: str,
        tokenizer: Any,
        multimodal_cfg: dict[str, Any],
        data_root: Path,
        split_ids: Sequence[int],
        record_selection: str = "all",
        max_records: int = 0,
    ) -> None:
        if datasets != "ancientdoc-page":
            raise ValueError(
                f"This entry point only accepts --datasets ancientdoc-page, got {datasets!r}."
            )
        if record_selection not in {"all", "first", "longest"}:
            raise ValueError(
                "--record_selection must be one of: all, first, longest."
            )
        if max_records < 0:
            raise ValueError("--max_train_records cannot be negative.")

        self.tokenizer = tokenizer
        self.multimodal_cfg = multimodal_cfg
        self.image_root = data_root.resolve()
        self.split_ids = parse_split_ids(split_ids)
        self.label_paths = [
            self.image_root / f"label_for_got_split{split_id}.json"
            for split_id in self.split_ids
        ]
        conversation_lib.default_conversation = conversation_lib.conv_templates["mpt"]
        if not self.image_root.is_dir():
            raise FileNotFoundError(self.image_root)

        records_with_split: list[tuple[int, dict[str, Any]]] = []
        split_counts: dict[int, int] = {}
        seen_images: set[str] = set()
        for split_id, label_path in zip(self.split_ids, self.label_paths):
            if not label_path.is_file():
                raise FileNotFoundError(label_path)
            records = json.loads(label_path.read_text(encoding="utf-8"))
            if not isinstance(records, list) or not records:
                raise ValueError(f"Expected a non-empty JSON list: {label_path}")
            split_counts[split_id] = len(records)
            for index, record in enumerate(records):
                if not isinstance(record, dict):
                    raise TypeError(f"{label_path.name}[{index}] is not an object.")
                image_value = record.get("image")
                if not isinstance(image_value, str) or not image_value:
                    raise ValueError(f"{label_path.name}[{index}] has no image path.")
                relative = PurePosixPath(image_value)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"{label_path.name}[{index}] has an unsafe image path: {image_value}"
                    )
                image_path = (self.image_root / Path(*relative.parts)).resolve()
                try:
                    image_path.relative_to(self.image_root)
                except ValueError as exc:
                    raise ValueError(
                        f"{label_path.name}[{index}] escapes data root: {image_value}"
                    ) from exc
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                if image_value in seen_images:
                    raise ValueError(f"Image occurs in multiple selected splits: {image_value}")
                seen_images.add(image_value)

                conversations = record.get("conversations")
                if not isinstance(conversations, list) or len(conversations) != 2:
                    raise ValueError(
                        f"{label_path.name}[{index}] must contain one human/GPT pair."
                    )
                human, assistant = conversations
                if not isinstance(human, dict) or not isinstance(assistant, dict):
                    raise TypeError(
                        f"{label_path.name}[{index}] conversations must be JSON objects."
                    )
                if human.get("from") != "human" or human.get("value") != "<image>\nOCR: ":
                    raise ValueError(f"{label_path.name}[{index}] has an unexpected prompt.")
                target = assistant.get("value")
                if assistant.get("from") != "gpt" or not isinstance(target, str) or not target.strip():
                    raise ValueError(f"{label_path.name}[{index}] has an invalid target.")
                records_with_split.append((split_id, record))

        if record_selection == "longest":
            def target_token_length(item: tuple[int, dict[str, Any]]) -> int:
                target = item[1]["conversations"][1]["value"]
                return len(tokenizer(target, add_special_tokens=False).input_ids)

            records_with_split.sort(
                key=target_token_length,
                reverse=True,
            )
        if max_records:
            records_with_split = records_with_split[:max_records]
        if not records_with_split:
            raise RuntimeError("AncientDoc selection produced no records.")

        self.records = [record for _, record in records_with_split]
        self.record_splits = [split_id for split_id, _ in records_with_split]
        self.split_counts = split_counts
        logging.warning(
            "Loading %d AncientDoc page conversations from splits %s "
            "(selection=%s, max_records=%d)",
            len(self.records),
            ",".join(map(str, self.split_ids)),
            record_selection,
            max_records,
        )


def make_ancientdoc_data_module(
    tokenizer: Any,
    data_args: Any,
    data_root: Path,
    split_ids: Sequence[int],
    record_selection: str = "all",
    max_records: int = 0,
) -> dict[str, Any]:
    if data_args.conversation_version != "mpt":
        raise ValueError("The AncientDoc GOT entry point requires conversation_version=mpt.")
    train_dataset = AncientDocPageDataset(
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
        data_root=data_root,
        split_ids=split_ids,
        record_selection=record_selection,
        max_records=max_records,
    )
    return {
        "train_dataset": train_dataset,
        "eval_dataset": None,
        "data_collator": DataCollatorForSupervisedDataset(tokenizer=tokenizer),
    }
