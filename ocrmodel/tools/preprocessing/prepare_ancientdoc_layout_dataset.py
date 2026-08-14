#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PROMPT = "<image>\nOCR: "
DEFAULT_TRAIN_SPLITS = (1, 2, 3)
DEFAULT_VALIDATION_SPLITS = (4,)
DEFAULT_TEST_SPLITS = (5,)


def parse_split_ids(value: str) -> tuple[int, ...]:
    try:
        split_ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid split list: {value!r}") from exc
    if not split_ids or len(set(split_ids)) != len(split_ids):
        raise argparse.ArgumentTypeError(f"split IDs must be non-empty and unique: {value!r}")
    if any(split_id not in range(1, 6) for split_id in split_ids):
        raise argparse.ArgumentTypeError("AncientDoc split IDs must be between 1 and 5")
    return split_ids


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def safe_relative_path(value: Any, *, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.image must be a non-empty relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context}.image is unsafe: {value!r}.")
    return path


def image_path_for(data_root: Path, image_value: str) -> Path:
    relative = safe_relative_path(image_value, context="record")
    return data_root / Path(*relative.parts)


def text_from_record(record: dict[str, Any], *, context: str) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ValueError(f"{context}.conversations must contain one human/GPT pair.")
    human, assistant = conversations
    if not isinstance(human, dict) or human.get("from") != "human" or human.get("value") != PROMPT:
        raise ValueError(f"{context} has an unexpected prompt.")
    if not isinstance(assistant, dict) or assistant.get("from") != "gpt":
        raise ValueError(f"{context} has an invalid assistant message.")
    target = assistant.get("value")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"{context} has an empty OCR target.")
    return target


def iter_split_records(data_root: Path, split_ids: Iterable[int]) -> Iterable[tuple[int, int, dict[str, Any]]]:
    for split_id in split_ids:
        path = data_root / f"label_for_got_split{split_id}.json"
        records = load_json(path)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Expected a non-empty record list: {path}")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(f"{path.name}[{index}] must be a JSON object.")
            yield split_id, index, record


def copy_or_symlink(source: Path, target: Path, *, symlink_images: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if symlink_images:
        target.symlink_to(source)
    else:
        shutil.copy2(source, target)


def build_records(
    *,
    data_root: Path,
    output_root: Path,
    split_name: str,
    split_ids: tuple[int, ...],
    max_records: int,
    symlink_images: bool,
) -> list[dict[str, Any]]:
    output_split = output_root / split_name
    image_output_root = output_split / "images"
    manifest_records: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for split_id, index, source_record in iter_split_records(data_root, split_ids):
        if max_records and len(manifest_records) >= max_records:
            break
        context = f"split{split_id}[{index}]"
        image_value = source_record.get("image")
        relative = safe_relative_path(image_value, context=context)
        source_image = (data_root / Path(*relative.parts)).resolve()
        if not source_image.is_file():
            raise FileNotFoundError(source_image)
        image_key = str(relative)
        if image_key in seen_images:
            raise ValueError(f"Image repeats within selected {split_name} splits: {image_key}")
        seen_images.add(image_key)
        page_text = text_from_record(source_record, context=context)
        page_stem = "_".join(relative.with_suffix("").parts)
        page_id = f"ancientdoc_split{split_id}_{index:06d}_{page_stem}"
        suffix = source_image.suffix.lower() or ".jpg"
        output_image_relative = PurePosixPath("images") / f"{page_id}{suffix}"
        output_image = output_split / Path(*output_image_relative.parts)
        copy_or_symlink(source_image, output_image, symlink_images=symlink_images)
        manifest_records.append(
            {
                "schema_version": 2,
                "input_level": "page",
                "layout_source": "real_ancientdoc",
                "layout_annotation_status": "none",
                "bbox_format": "xyxy_normalized",
                "page_id": page_id,
                "split": split_name,
                "tier": "real-ancientdoc",
                "source_group_id": f"ancientdoc_reference_split{split_id}",
                "source_group_ids": [f"ancientdoc_reference_split{split_id}"],
                "content_id": f"ancientdoc_reference_split{split_id}_{index:06d}",
                "image": str(output_image_relative),
                "original_image": image_key,
                "page_text": page_text,
                "page_text_separator": "",
                "regions": [],
                "conversations": [
                    {"from": "human", "value": PROMPT},
                    {"from": "gpt", "value": page_text},
                ],
            }
        )
    if not manifest_records:
        raise RuntimeError(f"No records selected for {split_name}.")
    return manifest_records


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert read-only AncientDoc GOT labels into layout-page-jsonl OCR-only manifests."
    )
    parser.add_argument("--ancientdoc-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-splits", type=parse_split_ids, default=DEFAULT_TRAIN_SPLITS)
    parser.add_argument("--validation-splits", type=parse_split_ids, default=DEFAULT_VALIDATION_SPLITS)
    parser.add_argument("--test-splits", type=parse_split_ids, default=DEFAULT_TEST_SPLITS)
    parser.add_argument("--max-train-records", type=int, default=0)
    parser.add_argument("--max-validation-records", type=int, default=0)
    parser.add_argument("--max-test-records", type=int, default=0)
    parser.add_argument("--symlink-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.ancientdoc_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output dataset already exists: {output_root}")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    split_specs = {
        "train": (args.train_splits, args.max_train_records),
        "validation": (args.validation_splits, args.max_validation_records),
        "test": (args.test_splits, args.max_test_records),
    }
    summary = {
        "status": "ok",
        "event": "ancientdoc_layout_dataset_prepared",
        "source": str(data_root),
        "output_root": str(output_root),
        "layout_annotation_status": "none",
        "splits": {},
    }
    seen_source_groups: dict[str, str] = {}
    for split_name, (split_ids, max_records) in split_specs.items():
        records = build_records(
            data_root=data_root,
            output_root=output_root,
            split_name=split_name,
            split_ids=split_ids,
            max_records=max_records,
            symlink_images=args.symlink_images,
        )
        write_manifest(output_root / split_name / "manifest.jsonl", records)
        for record in records:
            group = record["source_group_id"]
            previous = seen_source_groups.setdefault(group, split_name)
            if previous != split_name:
                raise RuntimeError(f"source_group_id crosses splits: {group}")
        summary["splits"][split_name] = {
            "source_split_ids": list(split_ids),
            "records": len(records),
            "manifest": str(output_root / split_name / "manifest.jsonl"),
        }
    write_json(output_root / "split_audit.json", summary)
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
