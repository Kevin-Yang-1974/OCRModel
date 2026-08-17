#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


PROMPT = "<image>\nOCR: "
ALL_SOURCE_SPLITS = (1, 2, 3, 4, 5)
TARGET_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SourcePage:
    split_id: int
    index: int
    record: dict[str, Any]
    image_relative: PurePosixPath
    image_absolute: Path
    category: str
    book: str
    book_key: str
    page_text: str


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def parse_split_ids(value: str) -> tuple[int, ...]:
    try:
        split_ids = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid split list: {value!r}") from exc
    if not split_ids or len(set(split_ids)) != len(split_ids):
        raise argparse.ArgumentTypeError(f"split IDs must be non-empty and unique: {value!r}")
    if any(split_id not in ALL_SOURCE_SPLITS for split_id in split_ids):
        raise argparse.ArgumentTypeError("AncientDoc split IDs must be between 1 and 5")
    return split_ids


def ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("ratio must be in (0, 1)")
    return parsed


def safe_relative_path(value: Any, *, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.image must be a non-empty relative path.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{context}.image is unsafe: {value!r}")
    return path


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


def parse_category_book(relative: PurePosixPath) -> tuple[str, str]:
    parts = relative.parts
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    if len(parts) >= 2:
        return parts[-2], parts[-1].rsplit(".", 1)[0]
    return "unknown", relative.stem or "unknown"


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_group_id(book_key: str) -> str:
    return f"ancientdoc_book_{stable_digest(book_key)[:20]}"


def iter_source_pages(data_root: Path, split_ids: Iterable[int]) -> Iterable[SourcePage]:
    seen_images: set[str] = set()
    for split_id in split_ids:
        path = data_root / f"label_for_got_split{split_id}.json"
        records = load_json(path)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Expected a non-empty record list: {path}")
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise TypeError(f"{path.name}[{index}] must be a JSON object.")
            context = f"split{split_id}[{index}]"
            relative = safe_relative_path(record.get("image"), context=context)
            image_absolute = (data_root / Path(*relative.parts)).resolve()
            if not image_absolute.is_file():
                raise FileNotFoundError(image_absolute)
            image_key = str(relative)
            if image_key in seen_images:
                raise ValueError(f"Image repeats across selected source splits: {image_key}")
            seen_images.add(image_key)
            category, book = parse_category_book(relative)
            yield SourcePage(
                split_id=split_id,
                index=index,
                record=record,
                image_relative=relative,
                image_absolute=image_absolute,
                category=category,
                book=book,
                book_key=f"{category}/{book}",
                page_text=text_from_record(record, context=context),
            )


def assign_groups(
    pages: list[SourcePage],
    *,
    seed: int,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> dict[str, str]:
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-9:
        raise ValueError(
            f"train/validation/test ratios must sum to 1.0, got {ratio_sum:.12f}"
        )
    group_sizes = Counter(page.book_key for page in pages)
    total = sum(group_sizes.values())
    targets = {
        "train": total * train_ratio,
        "validation": total * validation_ratio,
        "test": total * test_ratio,
    }
    current = {split: 0 for split in TARGET_SPLITS}
    assignments: dict[str, str] = {}

    ordered_groups = sorted(
        group_sizes,
        key=lambda key: (
            -group_sizes[key],
            stable_digest(f"{seed}:{key}"),
            key,
        ),
    )
    split_order = sorted(
        TARGET_SPLITS,
        key=lambda split: stable_digest(f"{seed}:split:{split}"),
    )
    for group in ordered_groups:
        size = group_sizes[group]

        def score(split: str) -> tuple[float, str]:
            # Weighted list scheduling: equal projected fill ratios correspond
            # to page counts in the requested train/validation/test ratio.
            projected_fill = (current[split] + size) / targets[split]
            return projected_fill, stable_digest(f"{seed}:{group}:{split}")

        best_split = min(split_order, key=score)
        assignments[group] = best_split
        current[best_split] += size

    def objective(counts: dict[str, int]) -> tuple[float, float, float]:
        errors = [
            abs(counts[split] - targets[split]) / total
            for split in TARGET_SPLITS
        ]
        return max(errors), sum(error * error for error in errors), sum(errors)

    # Greedy weighted scheduling is already close to the requested ratios.
    # Deterministic single-group moves and pair swaps remove residual errors
    # caused by indivisible book groups without ever splitting a book.
    while True:
        current_objective = objective(current)
        best_change: tuple[tuple[float, float, float], tuple[Any, ...]] | None = None

        for group in sorted(assignments):
            source = assignments[group]
            size = group_sizes[group]
            if current[source] == size:
                continue
            for destination in split_order:
                if destination == source:
                    continue
                candidate = dict(current)
                candidate[source] -= size
                candidate[destination] += size
                candidate_objective = objective(candidate)
                change = ("move", group, source, destination)
                if candidate_objective < current_objective and (
                    best_change is None
                    or (candidate_objective, change) < best_change
                ):
                    best_change = candidate_objective, change

        groups = sorted(assignments)
        for left_index, left in enumerate(groups):
            left_split = assignments[left]
            left_size = group_sizes[left]
            for right in groups[left_index + 1 :]:
                right_split = assignments[right]
                if left_split == right_split:
                    continue
                right_size = group_sizes[right]
                candidate = dict(current)
                candidate[left_split] += right_size - left_size
                candidate[right_split] += left_size - right_size
                candidate_objective = objective(candidate)
                change = ("swap", left, right, left_split, right_split)
                if candidate_objective < current_objective and (
                    best_change is None
                    or (candidate_objective, change) < best_change
                ):
                    best_change = candidate_objective, change

        if best_change is None:
            break
        _, change = best_change
        if change[0] == "move":
            _, group, source, destination = change
            size = group_sizes[group]
            assignments[group] = destination
            current[source] -= size
            current[destination] += size
        else:
            _, left, right, left_split, right_split = change
            left_size = group_sizes[left]
            right_size = group_sizes[right]
            assignments[left] = right_split
            assignments[right] = left_split
            current[left_split] += right_size - left_size
            current[right_split] += left_size - right_size

    if any(current[split] == 0 for split in TARGET_SPLITS):
        raise RuntimeError(f"At least one target split is empty: {current}")
    return assignments


def copy_or_symlink(source: Path, target: Path, *, symlink_images: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if symlink_images:
        target.symlink_to(source)
    else:
        shutil.copy2(source, target)


def manifest_record(page: SourcePage, split: str, output_root: Path, *, symlink_images: bool) -> dict[str, Any]:
    page_stem = "_".join(page.image_relative.with_suffix("").parts)
    page_id = f"ancientdoc_srcsplit{page.split_id}_{page.index:06d}_{page_stem}"
    suffix = page.image_absolute.suffix.lower() or ".jpg"
    image_relative = PurePosixPath("images") / f"{page_id}{suffix}"
    image_target = output_root / split / Path(*image_relative.parts)
    copy_or_symlink(page.image_absolute, image_target, symlink_images=symlink_images)
    group_id = source_group_id(page.book_key)
    return {
        "schema_version": 2,
        "input_level": "page",
        "layout_source": "real_ancientdoc",
        "layout_annotation_status": "none",
        "bbox_format": "xyxy_normalized",
        "page_id": page_id,
        "split": split,
        "tier": "real-ancientdoc",
        "source_group_id": group_id,
        "source_group_ids": [group_id],
        "content_id": f"ancientdoc_srcsplit{page.split_id}_{page.index:06d}",
        "group_isolation_key": page.book_key,
        "category": page.category,
        "book": page.book,
        "image": str(image_relative),
        "original_image": str(page.image_relative),
        "original_split_id": page.split_id,
        "original_split_index": page.index,
        "page_text": page.page_text,
        "page_text_separator": "",
        "regions": [],
        "conversations": [
            {"from": "human", "value": PROMPT},
            {"from": "gpt", "value": page.page_text},
        ],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert AncientDoc GOT labels into layout-page-jsonl manifests with "
            "category/book group-isolated train/validation/test splits."
        )
    )
    parser.add_argument("--ancientdoc-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-splits", type=parse_split_ids, default=ALL_SOURCE_SPLITS)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--train-ratio", type=ratio, default=0.6)
    parser.add_argument("--validation-ratio", type=ratio, default=0.2)
    parser.add_argument("--test-ratio", type=ratio, default=0.2)
    parser.add_argument(
        "--max-ratio-deviation",
        type=ratio,
        default=0.03,
        help=(
            "Maximum absolute difference between requested and actual split ratio; "
            "formal preparation fails when any split exceeds it."
        ),
    )
    parser.add_argument("--symlink-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    data_root = args.ancientdoc_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output dataset already exists: {output_root}")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    pages = list(iter_source_pages(data_root, args.source_splits))
    if not pages:
        raise RuntimeError("No AncientDoc records were loaded.")
    assignments = assign_groups(
        pages,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
    )

    assigned_page_counts = Counter(assignments[page.book_key] for page in pages)
    requested_ratios = {
        "train": args.train_ratio,
        "validation": args.validation_ratio,
        "test": args.test_ratio,
    }
    actual_ratios = {
        split: assigned_page_counts[split] / len(pages)
        for split in TARGET_SPLITS
    }
    ratio_deviations = {
        split: actual_ratios[split] - requested_ratios[split]
        for split in TARGET_SPLITS
    }
    if any(
        abs(ratio_deviations[split]) > args.max_ratio_deviation
        for split in TARGET_SPLITS
    ):
        raise RuntimeError(
            "Book-isolated allocation is outside --max-ratio-deviation: "
            f"requested={requested_ratios}, actual={actual_ratios}, "
            f"deviation={ratio_deviations}, limit={args.max_ratio_deviation}"
        )

    records_by_split = {split: [] for split in TARGET_SPLITS}
    group_split_counts: dict[str, Counter[str]] = {}
    for page in pages:
        split = assignments[page.book_key]
        records_by_split[split].append(
            manifest_record(page, split, output_root, symlink_images=args.symlink_images)
        )
        group_split_counts.setdefault(page.book_key, Counter())[split] += 1

    crossing_groups = {
        group: dict(counter)
        for group, counter in group_split_counts.items()
        if len(counter) > 1
    }
    if crossing_groups:
        raise RuntimeError(f"Internal group assignment crossed splits: {crossing_groups}")

    summary = {
        "status": "ok",
        "event": "ancientdoc_group_isolated_dataset_prepared",
        "source": str(data_root),
        "output_root": str(output_root),
        "source_split_ids": list(args.source_splits),
        "seed": args.seed,
        "split_strategy": "category/book group isolation",
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "allocation": {
            "algorithm": "weighted_lpt_with_deterministic_move_swap_refinement",
            "max_ratio_deviation": args.max_ratio_deviation,
            "actual_ratios": actual_ratios,
            "ratio_deviations": ratio_deviations,
            "max_absolute_ratio_deviation": max(
                abs(value) for value in ratio_deviations.values()
            ),
        },
        "layout_annotation_status": "none",
        "total_records": len(pages),
        "total_groups": len(assignments),
        "splits": {},
    }
    for split in TARGET_SPLITS:
        records = sorted(records_by_split[split], key=lambda record: str(record["page_id"]))
        write_manifest(output_root / split / "manifest.jsonl", records)
        groups = sorted({str(record["group_isolation_key"]) for record in records})
        original_split_counts = Counter(str(record["original_split_id"]) for record in records)
        summary["splits"][split] = {
            "records": len(records),
            "groups": len(groups),
            "manifest": str(output_root / split / "manifest.jsonl"),
            "original_split_counts": dict(sorted(original_split_counts.items())),
            "sample_groups": groups[:20],
        }
    write_json(output_root / "split_audit.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    summary = build_dataset(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
