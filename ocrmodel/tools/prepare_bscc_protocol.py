#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


def load_source_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for split in ("train", "validation", "test"):
        manifest = root / split / "manifest.jsonl"
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                page_id = record["page_id"]
                if page_id in seen:
                    raise ValueError(f"duplicate page_id across source manifests: {page_id}")
                seen.add(page_id)
                image_path = (manifest.parent / record["image"]).resolve()
                copied = dict(record)
                copied["image_path"] = str(image_path)
                records.append(copied)
    return records


def derive_version_group(record: dict[str, Any]) -> str:
    subset = str(record["subset"])
    stem = PurePosixPath(str(record["original_image"])).stem
    volume = re.match(r"(?i)(.*?V\d+)P", stem)
    if volume:
        return f"{subset}:{volume.group(1).upper()}"
    if stem.isdigit():
        return f"{subset}:UNVERSIONED_NUMERIC"
    prefix = re.match(r"(.+?)[_-]?\d+$", stem)
    return f"{subset}:{(prefix.group(1) if prefix else stem).upper()}"


def image_dhash(path: Path, size: int = 16) -> int:
    with Image.open(path) as source:
        gray = source.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
    value = 0
    for row in range(size):
        offset = row * (size + 1)
        for column in range(size):
            value = (value << 1) | (pixels[offset + column] > pixels[offset + column + 1])
    return value


def stratified_sample(records: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subset[str(record["subset"])].append(record)
    for values in by_subset.values():
        rng.shuffle(values)
    chosen: list[dict[str, Any]] = []
    subsets = sorted(by_subset)
    while len(chosen) < count:
        progressed = False
        for subset in subsets:
            if by_subset[subset] and len(chosen) < count:
                chosen.append(by_subset[subset].pop())
                progressed = True
        if not progressed:
            raise ValueError(f"requested {count} pages but only found {len(chosen)}")
    return sorted(chosen, key=lambda record: record["page_id"])


def assign_groups(
    records: list[dict[str, Any]], validation_pages: int, test_pages: int, rng: random.Random
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[derive_version_group(record)].append(record)
    keys = sorted(grouped)
    rng.shuffle(keys)
    assignments: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    target = "validation"
    for key in keys:
        if target == "validation" and len(assignments[target]) >= validation_pages:
            target = "test"
        if target == "test" and len(assignments[target]) >= test_pages:
            target = "train"
        assignments[target].extend(grouped[key])
    return assignments


def audit_near_duplicates(splits: dict[str, list[dict[str, Any]]], threshold: int) -> None:
    hashes: list[tuple[str, str, int]] = []
    for split, records in splits.items():
        for record in records:
            hashes.append((split, record["page_id"], image_dhash(Path(record["image_path"]))))
    for index, (split_a, page_a, hash_a) in enumerate(hashes):
        for split_b, page_b, hash_b in hashes[index + 1 :]:
            if split_a != split_b and (hash_a ^ hash_b).bit_count() <= threshold:
                raise ValueError(
                    f"near-duplicate pages cross splits: {page_a}/{split_a}, {page_b}/{split_b}"
                )


def write_manifest(path: Path, records: list[dict[str, Any]], split: str) -> str:
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            copied = dict(record)
            copied["original_split"] = copied["split"]
            copied["split"] = split
            copied["source_group"] = derive_version_group(copied)
            copied["duplicate_group"] = copied["page_id"]
            line = json.dumps(copied, ensure_ascii=False, separators=(",", ":")) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-pages", type=int, default=128)
    parser.add_argument("--validation-pages", type=int, default=64)
    parser.add_argument("--test-pages", type=int, default=64)
    parser.add_argument("--max-regions", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dhash-threshold", type=int, default=4)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)

    source = load_source_records(args.source_root)
    eligible = [
        record
        for record in source
        if 0 < len(record.get("regions", [])) <= args.max_regions and record.get("page_text")
    ]
    rng = random.Random(args.seed)
    pools = assign_groups(eligible, args.validation_pages, args.test_pages, rng)
    selected = {
        "train": stratified_sample(pools["train"], args.train_pages, rng),
        "validation": stratified_sample(pools["validation"], args.validation_pages, rng),
        "test": stratified_sample(pools["test"], args.test_pages, rng),
    }
    audit_near_duplicates(selected, args.dhash_threshold)
    source_groups = {
        split: {derive_version_group(record) for record in records}
        for split, records in selected.items()
    }
    if any(
        source_groups[left] & source_groups[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise RuntimeError("version-group isolation failed")

    args.output_root.mkdir(parents=True)
    manifest_hashes = {
        split: write_manifest(args.output_root / f"{split}.jsonl", records, split)
        for split, records in selected.items()
    }
    train_counts = Counter(
        character
        for record in selected["train"]
        for character in record["page_text"]
        if not character.isspace()
    )
    validation_counts = Counter(
        character
        for record in selected["validation"]
        for character in record["page_text"]
        if not character.isspace()
    )
    summary = {
        "status": "ok",
        "protocol": "glm_ocr_r1_r2_group_isolated_128_seed42_v1",
        "seed": args.seed,
        "source_root": str(args.source_root.resolve()),
        "eligible_pages": len(eligible),
        "max_regions": args.max_regions,
        "split_pages": {split: len(records) for split, records in selected.items()},
        "source_group_counts": {split: len(groups) for split, groups in source_groups.items()},
        "source_group_overlap": 0,
        "dhash_threshold": args.dhash_threshold,
        "near_duplicate_cross_split_pairs": 0,
        "manifest_sha256": manifest_hashes,
        "r2": {
            f"k{k}_train_characters": sum(count <= k for count in train_counts.values())
            for k in (1, 3, 5)
        },
        "r2_validation_support": {
            f"k{k}_reference_occurrences": sum(
                validation_counts[char] for char, count in train_counts.items() if count <= k
            )
            for k in (1, 3, 5)
        },
        "test_used_for_selection": False,
    }
    (args.output_root / "protocol.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
