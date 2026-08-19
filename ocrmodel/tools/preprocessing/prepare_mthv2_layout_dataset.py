#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from PIL import Image


PROMPT = "<image>\nOCR: "
SUBSETS = ("MTH1000", "MTH1200", "TKH")
OUTPUT_SPLITS = ("train", "validation", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True)
class SourcePage:
    subset: str
    stem: str
    image: Path
    official_split: str

    @property
    def key(self) -> str:
        return f"{self.subset}/{self.stem}"


def stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def parse_ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("ratio must be in (0, 1)")
    return parsed


def official_key(value: str) -> str:
    normalized = PurePosixPath(value.strip().replace("\\", "/"))
    parts = normalized.parts
    try:
        image_index = max(index for index, part in enumerate(parts) if part == "img")
    except ValueError as exc:
        raise ValueError(f"Official split entry has no img component: {value!r}") from exc
    if image_index < 1 or image_index + 1 >= len(parts):
        raise ValueError(f"Official split entry is incomplete: {value!r}")
    subset = parts[image_index - 1]
    if subset not in SUBSETS:
        raise ValueError(f"Unsupported MTHv2 subset in split entry: {value!r}")
    return f"{subset}/{PurePosixPath(parts[image_index + 1]).stem}"


def load_official_split(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    keys = {
        official_key(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    if not keys:
        raise ValueError(f"Official split list is empty: {path}")
    return keys


def find_images(raw_root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for subset in SUBSETS:
        image_root = raw_root / subset / "img"
        if not image_root.is_dir():
            raise FileNotFoundError(image_root)
        for image in sorted(image_root.iterdir()):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            key = f"{subset}/{image.stem}"
            if key in images:
                raise ValueError(f"Duplicate MTHv2 image stem: {key}")
            images[key] = image
    if not images:
        raise RuntimeError(f"No MTHv2 images found below {raw_root}")
    return images


def build_source_pages(raw_root: Path, train_list: Path, test_list: Path) -> list[SourcePage]:
    images = find_images(raw_root)
    official_train = load_official_split(train_list)
    official_test = load_official_split(test_list)
    overlap = official_train & official_test
    if overlap:
        raise ValueError(f"Official train/test overlap: {sorted(overlap)[:5]}")
    listed = official_train | official_test
    missing_images = listed - set(images)
    unlisted_images = set(images) - listed
    if missing_images or unlisted_images:
        raise ValueError(
            "Official split coverage differs from extracted images: "
            f"missing_images={sorted(missing_images)[:5]}, "
            f"unlisted_images={sorted(unlisted_images)[:5]}"
        )
    pages = []
    for key, image in sorted(images.items()):
        subset, stem = key.split("/", maxsplit=1)
        pages.append(
            SourcePage(
                subset=subset,
                stem=stem,
                image=image,
                official_split="train" if key in official_train else "test",
            )
        )
    return pages


def assign_output_splits(
    pages: Sequence[SourcePage], *, validation_ratio: float, seed: int
) -> dict[str, str]:
    assignments = {page.key: "test" for page in pages if page.official_split == "test"}
    by_subset: dict[str, list[SourcePage]] = defaultdict(list)
    for page in pages:
        if page.official_split == "train":
            by_subset[page.subset].append(page)
    for subset in SUBSETS:
        candidates = sorted(
            by_subset[subset],
            key=lambda page: (stable_digest(f"{seed}:{page.key}"), page.key),
        )
        validation_count = max(1, int(round(len(candidates) * validation_ratio)))
        validation_keys = {page.key for page in candidates[:validation_count]}
        for page in candidates:
            assignments[page.key] = "validation" if page.key in validation_keys else "train"
    if set(assignments) != {page.key for page in pages}:
        raise RuntimeError("Not every MTHv2 page received an output split.")
    return assignments


def label_path(raw_root: Path, page: SourcePage, kind: str) -> Path:
    path = raw_root / page.subset / kind / f"{page.stem}.txt"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def parse_number(value: str, *, context: str) -> float:
    try:
        parsed = float(value.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid number in {context}: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite number in {context}: {value!r}")
    return parsed


def parse_textlines(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 9:
            raise ValueError(f"Expected transcription plus 8 coordinates at {path}:{line_number}")
        text = ",".join(parts[:-8])
        coordinates = [
            parse_number(value, context=f"{path}:{line_number}") for value in parts[-8:]
        ]
        polygon = [[coordinates[index], coordinates[index + 1]] for index in range(0, 8, 2)]
        records.append({"text": text, "polygon_px": polygon})
    if not records:
        raise ValueError(f"Text-line annotation is empty: {path}")
    return records


def parse_characters(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Expected character plus 4 coordinates at {path}:{line_number}")
        character = " ".join(parts[:-4])
        bbox = [parse_number(value, context=f"{path}:{line_number}") for value in parts[-4:]]
        records.append({"character": character, "bbox_xyxy_px": bbox})
    return records


def parse_boundary_lines(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) < 4:
            raise ValueError(f"Expected 4 coordinates at {path}:{line_number}")
        values = [parse_number(value, context=f"{path}:{line_number}") for value in parts[-4:]]
        records.append({"start_px": values[:2], "end_px": values[2:]})
    return records


def polygon_bbox(
    polygon: Sequence[Sequence[float]], *, width: int, height: int
) -> tuple[list[float], list[float], bool]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    raw = [min(xs), min(ys), max(xs), max(ys)]
    clipped = [
        min(max(raw[0], 0.0), float(width)),
        min(max(raw[1], 0.0), float(height)),
        min(max(raw[2], 0.0), float(width)),
        min(max(raw[3], 0.0), float(height)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"Degenerate text-line polygon after clipping: {polygon!r}")
    normalized = [
        clipped[0] / width,
        clipped[1] / height,
        clipped[2] / width,
        clipped[3] / height,
    ]
    return clipped, normalized, clipped != raw


def infer_writing_direction(bbox_px: Sequence[float]) -> str:
    width = bbox_px[2] - bbox_px[0]
    height = bbox_px[3] - bbox_px[1]
    if height >= width * 1.2:
        return "vertical_rtl"
    if width >= height * 1.2:
        return "horizontal_ltr"
    return "unknown"


def materialize_image(source: Path, target: Path, *, copy_image: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    if copy_image:
        shutil.copy2(source, target)
    else:
        target.symlink_to(source)


def convert_page(
    *,
    raw_root: Path,
    output_root: Path,
    page: SourcePage,
    output_split: str,
    copy_images: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with Image.open(page.image) as image:
        width, height = image.size
    textline_path = label_path(raw_root, page, "label_textline")
    character_path = label_path(raw_root, page, "label_char")
    boundary_path = label_path(raw_root, page, "label_table")
    textlines = parse_textlines(textline_path)
    characters = parse_characters(character_path)
    boundary_lines = parse_boundary_lines(boundary_path)

    page_id = f"mthv2_{page.subset.lower()}_{page.stem}"
    page_group_id = f"mthv2_page_{stable_digest(page.key)[:20]}"
    image_relative = PurePosixPath("images") / f"{page.subset}__{page.image.name}"
    materialize_image(
        page.image.resolve(),
        output_root / output_split / Path(*image_relative.parts),
        copy_image=copy_images,
    )

    regions = []
    clipped_regions = 0
    for reading_order, textline in enumerate(textlines):
        bbox_px, bbox, clipped = polygon_bbox(
            textline["polygon_px"], width=width, height=height
        )
        clipped_regions += int(clipped)
        regions.append(
            {
                "region_id": f"{page_id}_line_{reading_order:03d}",
                "content_id": f"{page_id}_line_{reading_order:03d}",
                "source_group_id": page_group_id,
                "source_kind": "mthv2_textline",
                "layout_level": "textline",
                "text": textline["text"],
                "polygon_px": textline["polygon_px"],
                "bbox_px": [round(value, 4) for value in bbox_px],
                "bbox": [round(value, 8) for value in bbox],
                "reading_order": reading_order,
                "writing_direction": infer_writing_direction(bbox_px),
                "writing_direction_source": "bbox_aspect_ratio_inferred",
                "valid": True,
            }
        )
    page_text = "".join(region["text"] for region in regions)
    if not page_text:
        raise ValueError(f"Page OCR target is empty: {page.key}")

    annotation_relative = PurePosixPath("annotations") / f"{page.subset}__{page.stem}.json"
    annotation_target = output_root / output_split / Path(*annotation_relative.parts)
    original_image = str(PurePosixPath(page.subset) / "img" / page.image.name)
    annotation_payload = {
        "schema_version": 1,
        "dataset": "MTHv2",
        "page_id": page_id,
        "subset": page.subset,
        "original_image": original_image,
        "page_size": [width, height],
        "official_split": page.official_split,
        "output_split": output_split,
        "textlines": regions,
        "characters": characters,
        "boundary_lines": boundary_lines,
        "source_annotations": {
            "textline": str(PurePosixPath(page.subset) / "label_textline" / textline_path.name),
            "character": str(PurePosixPath(page.subset) / "label_char" / character_path.name),
            "boundary": str(PurePosixPath(page.subset) / "label_table" / boundary_path.name),
        },
    }
    write_json(annotation_target, annotation_payload)

    manifest = {
        "schema_version": 2,
        "input_level": "page",
        "layout_source": "real_mthv2_official",
        "layout_annotation_status": "complete",
        "layout_level": "textline",
        "bbox_format": "xyxy_normalized",
        "page_id": page_id,
        "split": output_split,
        "official_split": page.official_split,
        "tier": "real-mthv2",
        "subset": page.subset,
        "source_group_id": page_group_id,
        "source_group_ids": [page_group_id],
        "group_isolation_status": "unavailable_official_random_page_split",
        "content_id": page_id,
        "image": str(image_relative),
        "original_image": original_image,
        "annotation_file": str(annotation_relative),
        "page_size": [width, height],
        "page_text": page_text,
        "page_text_separator": "",
        "regions": regions,
        "conversations": [
            {"from": "human", "value": PROMPT},
            {"from": "gpt", "value": page_text},
        ],
    }
    return manifest, {
        "textlines": len(regions),
        "characters": len(characters),
        "boundary_lines": len(boundary_lines),
        "clipped_textline_boxes": clipped_regions,
        "directions": Counter(region["writing_direction"] for region in regions),
    }


def percentile(values: Sequence[int], fraction: float) -> int:
    ordered = sorted(values)
    index = int(math.ceil(fraction * len(ordered))) - 1
    return ordered[min(max(index, 0), len(ordered) - 1)]


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    raw_root = args.raw_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not raw_root.is_dir():
        raise FileNotFoundError(raw_root)
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(f"Output dataset already exists: {output_root}")
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    try:
        pages = build_source_pages(raw_root, args.train_list, args.test_list)
        assignments = assign_output_splits(
            pages, validation_ratio=args.validation_ratio, seed=args.seed
        )
        manifests: dict[str, list[dict[str, Any]]] = {
            split: [] for split in OUTPUT_SPLITS
        }
        split_subset_counts = {split: Counter() for split in OUTPUT_SPLITS}
        region_counts: list[int] = []
        direction_counts: Counter[str] = Counter()
        totals = Counter()

        for page in pages:
            output_split = assignments[page.key]
            manifest, stats = convert_page(
                raw_root=raw_root,
                output_root=output_root,
                page=page,
                output_split=output_split,
                copy_images=args.copy_images,
            )
            manifests[output_split].append(manifest)
            split_subset_counts[output_split][page.subset] += 1
            region_counts.append(stats["textlines"])
            totals.update(
                {
                    "textlines": stats["textlines"],
                    "characters": stats["characters"],
                    "boundary_lines": stats["boundary_lines"],
                    "clipped_textline_boxes": stats["clipped_textline_boxes"],
                }
            )
            direction_counts.update(stats["directions"])
        for split in OUTPUT_SPLITS:
            manifests[split].sort(key=lambda record: record["page_id"])
            write_jsonl(output_root / split / "manifest.jsonl", manifests[split])

        official_counts = Counter(page.official_split for page in pages)
        summary = {
            "status": "ok",
            "event": "mthv2_layout_dataset_prepared",
            "source": str(raw_root),
            "output_root": str(output_root),
            "total_pages": len(pages),
            "official_split_counts": dict(sorted(official_counts.items())),
            "validation_derivation": {
                "source": "official_train_only",
                "method": "deterministic_sha256_stratified_by_subset",
                "seed": args.seed,
                "requested_ratio_within_official_train": args.validation_ratio,
            },
            "splits": {
                split: {
                    "pages": len(manifests[split]),
                    "subsets": dict(sorted(split_subset_counts[split].items())),
                    "manifest": str(output_root / split / "manifest.jsonl"),
                }
                for split in OUTPUT_SPLITS
            },
            "annotation_totals": dict(sorted(totals.items())),
            "textline_regions_per_page": {
                "min": min(region_counts),
                "p50": percentile(region_counts, 0.50),
                "p90": percentile(region_counts, 0.90),
                "p95": percentile(region_counts, 0.95),
                "p99": percentile(region_counts, 0.99),
                "max": max(region_counts),
                "pages_over_16": sum(count > 16 for count in region_counts),
                "pages_over_32": sum(count > 32 for count in region_counts),
            },
            "writing_direction_counts": dict(sorted(direction_counts.items())),
            "layout_level": "textline",
            "layout_annotation_status": "complete",
            "image_storage": (
                "copied_from_raw" if args.copy_images else "absolute_symlink_to_raw"
            ),
            "limitations": [
                "The official split is random at page level and provides no book/version grouping metadata.",
                "Writing direction is inferred from text-line bbox aspect ratio, not directly annotated.",
                "label_table is preserved as boundary line segments and is not treated as table semantics.",
                "Text-line regions are not truncated to the current 16-query VLQA configuration.",
            ],
        }
        write_json(output_root / "conversion_summary.json", summary)
        write_json(
            output_root / "split_audit.json",
            {
                "status": "ok",
                "event": "mthv2_split_and_conversion_audit_ok",
                "total_records": len(pages),
                "official_train_test_overlap": 0,
                "official_split_coverage": len(pages),
                "output_split_counts": {
                    split: len(manifests[split]) for split in OUTPUT_SPLITS
                },
                "missing_images": 0,
                "missing_annotation_files": 0,
                "layout_level": "textline",
                "group_isolation_status": "unavailable_official_random_page_split",
            },
        )
        return summary
    except Exception:
        if output_root.exists():
            shutil.rmtree(output_root)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert MTHv2 text-line, character, and boundary-line annotations "
            "into whole-page layout manifests plus structured sidecars."
        )
    )
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--test-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=parse_ratio, default=0.1)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--copy-images", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    summary = build_dataset(parse_args(argv))
    print(
        json.dumps(
            {
                "status": summary["status"],
                "event": summary["event"],
                "output_root": summary["output_root"],
                "total_pages": summary["total_pages"],
                "splits": {
                    split: details["pages"]
                    for split, details in summary["splits"].items()
                },
                "regions": summary["textline_regions_per_page"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
