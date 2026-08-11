from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from PIL import Image, ImageFile
from transformers import AutoTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the read-only AncientDoc full labels and five reference splits."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-max-lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096, 8192],
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_numbers(values: Iterable[float | int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot describe an empty sequence.")

    def percentile(fraction: float) -> float | int:
        index = round((len(ordered) - 1) * fraction)
        return ordered[index]

    return {
        "min": ordered[0],
        "p05": percentile(0.05),
        "median": statistics.median(ordered),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON list: {path}")
    return records


def validate_record(
    record: dict[str, Any],
    index: int,
    label_name: str,
    data_root: Path,
) -> tuple[str, str, str]:
    if not isinstance(record, dict):
        raise TypeError(f"{label_name}[{index}] is not an object.")
    image_value = record.get("image")
    if not isinstance(image_value, str) or not image_value:
        raise ValueError(f"{label_name}[{index}] has no image path.")
    relative = PurePosixPath(image_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label_name}[{index}] has an unsafe image path: {image_value}")
    image_path = (data_root / Path(*relative.parts)).resolve()
    try:
        image_path.relative_to(data_root)
    except ValueError as exc:
        raise ValueError(f"{label_name}[{index}] escapes data root: {image_value}") from exc

    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError(f"{label_name}[{index}] must have one human/GPT pair.")
    human, assistant = conversations
    if human.get("from") != "human" or assistant.get("from") != "gpt":
        raise ValueError(f"{label_name}[{index}] has invalid roles.")
    prompt = human.get("value")
    target = assistant.get("value")
    if not isinstance(prompt, str) or "<image>" not in prompt:
        raise ValueError(f"{label_name}[{index}] has an invalid image prompt.")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"{label_name}[{index}] has an empty target.")
    return image_value, prompt, target


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    source_model = args.source_model.resolve()
    output = args.output.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not (source_model / "model.safetensors").is_file():
        raise FileNotFoundError(source_model / "model.safetensors")
    limits = sorted(set(args.model_max_lengths))
    if not limits or limits[0] < 1:
        raise ValueError("--model-max-lengths must contain positive integers.")

    full_path = data_root / "label_for_got.json"
    split_paths = [data_root / f"label_for_got_split{index}.json" for index in range(1, 6)]
    full_records = load_records(full_path)
    split_records = [load_records(path) for path in split_paths]

    full_map: dict[str, str] = {}
    full_prompts: list[str] = []
    full_targets: list[str] = []
    prompt_counts: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    books: Counter[str] = Counter()
    for index, record in enumerate(full_records):
        image_value, prompt, target = validate_record(record, index, full_path.name, data_root)
        if image_value in full_map:
            raise ValueError(f"Duplicate image in full labels: {image_value}")
        full_map[image_value] = target
        full_prompts.append(prompt)
        full_targets.append(target)
        prompt_counts[prompt] += 1
        parts = PurePosixPath(image_value).parts
        if len(parts) < 4 or parts[0] != "imgs":
            raise ValueError(f"Unexpected AncientDoc image layout: {image_value}")
        categories[parts[1]] += 1
        books[parts[2]] += 1

    split_maps: list[dict[str, str]] = []
    split_books: list[set[str]] = []
    for split_index, records in enumerate(split_records, start=1):
        current: dict[str, str] = {}
        current_books: set[str] = set()
        for index, record in enumerate(records):
            image_value, _, target = validate_record(
                record,
                index,
                split_paths[split_index - 1].name,
                data_root,
            )
            if image_value in current:
                raise ValueError(f"Duplicate image in split {split_index}: {image_value}")
            current[image_value] = target
            current_books.add(PurePosixPath(image_value).parts[2])
        split_maps.append(current)
        split_books.append(current_books)

    split_path_sets = [set(mapping) for mapping in split_maps]
    union_paths = set().union(*split_path_sets)
    pairwise_overlap = [
        [len(split_path_sets[left] & split_path_sets[right]) for right in range(5)]
        for left in range(5)
    ]
    content_mismatches = sum(
        1
        for mapping in split_maps
        for image_value, target in mapping.items()
        if full_map.get(image_value) != target
    )
    train_books = set().union(*split_books[:4])
    eval_books = split_books[4]

    missing_images: list[str] = []
    corrupt_images: list[str] = []
    widths: list[int] = []
    heights: list[int] = []
    aspect_ratios: list[float] = []
    modes: Counter[str] = Counter()
    for image_value in full_map:
        image_path = data_root / Path(*PurePosixPath(image_value).parts)
        if not image_path.is_file():
            missing_images.append(image_value)
            continue
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode
        except Exception as exc:  # noqa: BLE001 - report every unreadable shared asset.
            corrupt_images.append(f"{image_value}: {type(exc).__name__}: {exc}")
            continue
        widths.append(width)
        heights.append(height)
        aspect_ratios.append(width / height)
        modes[mode] += 1

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from GOT.utils import conversation as conversation_lib
    from GOT.utils.constants import (
        DEFAULT_IMAGE_PATCH_TOKEN,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_END_TOKEN,
        DEFAULT_IM_START_TOKEN,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        source_model,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    replacement = (
        DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * 256 + DEFAULT_IM_END_TOKEN
    )
    prompts: list[str] = []
    for human_prompt, target in zip(full_prompts, full_targets):
        conversation = conversation_lib.conv_templates["mpt"].copy()
        conversation.append_message(
            conversation.roles[0],
            human_prompt.replace(DEFAULT_IMAGE_TOKEN, replacement),
        )
        conversation.append_message(conversation.roles[1], target)
        prompts.append(conversation.get_prompt())

    prompt_token_ids = tokenizer(prompts, padding=False, truncation=False).input_ids
    target_token_ids = tokenizer(full_targets, padding=False, truncation=False).input_ids
    prompt_token_lengths = [len(token_ids) for token_ids in prompt_token_ids]
    target_token_lengths = [len(token_ids) for token_ids in target_token_ids]

    report = {
        "status": "ancientdoc_audit_ok",
        "data_root": str(data_root),
        "source_model": str(source_model),
        "labels": {
            "full": {
                "path": str(full_path),
                "sha256": sha256_file(full_path),
                "records": len(full_records),
            },
            "splits": [
                {
                    "split": index,
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "records": len(records),
                }
                for index, (path, records) in enumerate(
                    zip(split_paths, split_records), start=1
                )
            ],
            "union_records": len(union_paths),
            "union_matches_full": union_paths == set(full_map),
            "pairwise_image_overlap": pairwise_overlap,
            "content_mismatches_against_full": content_mismatches,
        },
        "grouping": {
            "categories": len(categories),
            "category_counts": dict(categories.most_common()),
            "books": len(books),
            "book_counts_top20": books.most_common(20),
            "books_per_split": [len(values) for values in split_books],
            "train_1_4_books": len(train_books),
            "eval_5_books": len(eval_books),
            "train_eval_book_overlap": len(train_books & eval_books),
            "train_eval_book_overlap_examples": sorted(train_books & eval_books)[:30],
        },
        "images": {
            "expected": len(full_map),
            "readable": len(widths),
            "missing": missing_images,
            "corrupt": corrupt_images,
            "width": describe_numbers(widths),
            "height": describe_numbers(heights),
            "aspect_ratio": describe_numbers(aspect_ratios),
            "modes": dict(modes),
        },
        "targets": {
            "codepoints": describe_numbers(map(len, full_targets)),
            "target_tokens": describe_numbers(target_token_lengths),
            "full_prompt_tokens": describe_numbers(prompt_token_lengths),
            "records_with_actual_newline": sum("\n" in target for target in full_targets),
            "records_with_literal_backslash_n": sum(
                "\\n" in target for target in full_targets
            ),
            "literal_backslash_n_count": sum(
                target.count("\\n") for target in full_targets
            ),
            "prompt_variants": dict(prompt_counts),
            "over_model_max_length": {
                str(limit): sum(length > limit for length in prompt_token_lengths)
                for limit in limits
            },
        },
        "reference_protocol": {
            "train_splits": [1, 2, 3, 4],
            "eval_split": 5,
            "train_records": sum(len(records) for records in split_records[:4]),
            "eval_records": len(split_records[4]),
            "input_level": "page",
        },
    }
    if missing_images or corrupt_images:
        raise RuntimeError(
            f"AncientDoc has missing/corrupt images: {len(missing_images)}/{len(corrupt_images)}"
        )
    if not report["labels"]["union_matches_full"] or content_mismatches:
        raise RuntimeError("AncientDoc splits do not exactly reconstruct the full labels.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("ANCIENTDOC_AUDIT_OK")
    print(f"records={len(full_records)}")
    print(f"train_records={report['reference_protocol']['train_records']}")
    print(f"eval_records={report['reference_protocol']['eval_records']}")
    print(f"train_eval_book_overlap={report['grouping']['train_eval_book_overlap']}")
    print(f"max_full_prompt_tokens={max(prompt_token_lengths)}")
    print(f"over_8192={report['targets']['over_model_max_length'].get('8192')}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
