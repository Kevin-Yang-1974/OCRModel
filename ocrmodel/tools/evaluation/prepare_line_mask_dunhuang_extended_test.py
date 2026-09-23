#!/usr/bin/env python3
"""Build an isolated 59-labeled + new-image Dunhuang test extension.

The original Dunhuang/local-gazetteer split is left untouched. Newly shared
images have region-coordinate annotations but no OCR transcription, so their
records are included for prediction and generation diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    ids = [str(row.get("page_id", "")) for row in rows]
    if any(not page_id for page_id in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"missing or duplicate page_id in {path}")
    return rows


def row_groups(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in ("source_group", "source_group_id", "duplicate_group"):
        value = row.get(key)
        if isinstance(value, str) and value:
            values.add(value)
    for key in ("source_group_ids",):
        value = row.get(key)
        if isinstance(value, str) and value:
            values.add(value)
        elif isinstance(value, list):
            values.update(str(item) for item in value if item)
    return values


def absolute_image_path(row: dict[str, Any], dataset_root: Path) -> Path:
    image = row.get("image_path") or row.get("image")
    if not image:
        raise ValueError(f"record {row.get('page_id')} has no image path")
    path = Path(str(image))
    if not path.is_absolute():
        path = dataset_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing test image: {path}")
    return path


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--new-sample-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-new-pages", type=int, default=77)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    test_manifest = args.test_manifest.resolve()
    sample_root = args.new_sample_root.resolve()
    images_dir = sample_root / "img"
    annotations_dir = sample_root / "work"
    if not images_dir.is_dir() or not annotations_dir.is_dir():
        raise FileNotFoundError("new sample package must contain img/ and work/ folders")

    train_manifest = args.train_manifest.resolve()
    validation_manifest = args.validation_manifest.resolve()
    if not train_manifest.is_file() or not validation_manifest.is_file():
        raise FileNotFoundError("expected train and validation manifests beside test split")
    original_rows = read_jsonl(test_manifest)
    train_rows = read_jsonl(train_manifest)
    validation_rows = read_jsonl(validation_manifest)
    for row in original_rows:
        if row.get("split", row.get("official_split")) != "test":
            raise ValueError(f"non-test row found in original test manifest: {row['page_id']}")
        if not isinstance(row.get("page_text"), str) or not row["page_text"]:
            raise ValueError(f"original test row has no reference text: {row['page_id']}")

    image_files = sorted(images_dir.glob("*.jpg"), key=lambda path: path.name.lower())
    annotation_files = sorted(annotations_dir.glob("*.RGN"), key=lambda path: path.name.lower())
    if len(image_files) != args.expected_new_pages:
        raise ValueError(f"expected {args.expected_new_pages} new JPGs, found {len(image_files)}")
    image_stems = {path.stem for path in image_files}
    annotation_stems = {path.stem for path in annotation_files}
    if len(annotation_files) != args.expected_new_pages or image_stems != annotation_stems:
        raise ValueError("new JPG/RGN basenames do not form complete one-to-one pairs")

    prior_rows = train_rows + validation_rows + original_rows
    prior_ids = {str(row["page_id"]) for row in prior_rows}
    prior_groups = set().union(*(row_groups(row) for row in prior_rows))
    prior_basenames = {
        Path(str(row.get("image") or row.get("image_path") or "")).name.lower()
        for row in prior_rows
    }
    new_rows: list[dict[str, Any]] = []
    basename_overlaps: list[str] = []
    group_overlaps: list[dict[str, Any]] = []
    for image_path in image_files:
        stem = image_path.stem
        match = re.match(r"^(BD\d+)R\d+_", stem, flags=re.IGNORECASE)
        if not match:
            raise ValueError(f"cannot derive source group from filename: {image_path.name}")
        group = f"dunhuang_{match.group(1).upper()}"
        annotation_path = annotations_dir / f"{stem}.RGN"
        page_id = f"dunhuang_new77_{stem}"
        if page_id in prior_ids:
            raise ValueError(f"new page_id already occurs in an existing split: {page_id}")
        if image_path.name.lower() in prior_basenames:
            basename_overlaps.append(image_path.name)
        overlap = sorted({group} & prior_groups)
        if overlap:
            group_overlaps.append({"page_id": page_id, "source_group": group, "matches": overlap})
        new_rows.append(
            {
                "schema_version": "dunhuang_extended_test_v1",
                "page_id": page_id,
                "split": "test",
                "official_split": "test",
                "domain": "dunhuang",
                "source_group": group,
                "image_path": str(image_path.resolve()),
                "annotation_path": str(annotation_path.resolve()),
                "annotation_type": "binary RGN region coordinates; no OCR transcription",
                "page_text": None,
                "reference_available": False,
                "sample_source": "2026.08.20敦煌古籍样本-新增77",
            }
        )

    if basename_overlaps:
        raise ValueError(f"new sample image basenames overlap existing splits: {basename_overlaps}")
    expanded_rows: list[dict[str, Any]] = []
    for original in original_rows:
        row = dict(original)
        row["image_path"] = str(absolute_image_path(row, dataset_root))
        row["reference_available"] = True
        row["sample_source"] = "existing_dunhuang_local_gazetteer_test"
        expanded_rows.append(row)
    expanded_rows.extend(new_rows)
    page_ids = [str(row["page_id"]) for row in expanded_rows]
    if len(page_ids) != len(set(page_ids)):
        raise ValueError("expanded test manifest contains duplicate page IDs")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    labeled_out = output_root / "labeled-test-manifest.jsonl"
    unscored_out = output_root / "new77-unscored-manifest.jsonl"
    expanded_out = output_root / "expanded-test-manifest.jsonl"
    write_jsonl(labeled_out, expanded_rows[: len(original_rows)])
    write_jsonl(unscored_out, new_rows)
    write_jsonl(expanded_out, expanded_rows)

    domains = Counter(str(row.get("domain", "unknown")) for row in original_rows)
    manifest_stats = {
        "original_test_pages": len(original_rows),
        "new_unscored_pages": len(new_rows),
        "expanded_test_pages": len(expanded_rows),
        "original_test_domain_counts": dict(sorted(domains.items())),
        "train_pages": len(train_rows),
        "validation_pages": len(validation_rows),
    }
    protocol = {
        "status": "prepared_locked_test_extension",
        "dataset": "dunhuang_local_gazetteer_q32_v1_plus_new77",
        "protocol": "glm_ocr_dunhuang_local_gazetteer_group_isolated_v1_extended_test",
        "seed": 42,
        "num_queries": 32,
        "generation": {
            "input": "whole-page image plus fixed Text Recognition prompt only",
            "prompt": "Text Recognition:",
            "max_new_tokens": 1536,
            "do_sample": False,
            "precision": "bfloat16",
            "attention_backend": "sdpa",
        },
        "split_pages": {
            "historical_labeled_test": len(original_rows),
            "new_unscored_dunhuang_images": len(new_rows),
            "expanded_test_total": len(expanded_rows),
        },
        "manifest_stats": manifest_stats,
        "source_manifests": {
            "train_sha256": sha256(train_manifest),
            "validation_sha256": sha256(validation_manifest),
            "original_test_sha256": sha256(test_manifest),
        },
        "derived_manifests": {
            "labeled_test_sha256": sha256(labeled_out),
            "new77_unscored_sha256": sha256(unscored_out),
            "expanded_test_sha256": sha256(expanded_out),
        },
        "new_sample_source": {
            "folder": "2026.08.20敦煌古籍样本-新增77/20260820",
            "image_pages": len(image_files),
            "paired_rgn_annotations": len(annotation_files),
            "ocr_transcriptions": 0,
            "rgn_used_as_ocr_reference": False,
        },
        "source_group_overlap_with_existing_train_validation_test": group_overlaps,
        "test_manifest_read": True,
        "test_used_for_selection": False,
        "canonical_dataset_split_modified": False,
        "notes": [
            "The 77 new pages are included for inference and generation diagnostics.",
            "They have no OCR transcript; CER/I-D-S is computed only on the 59 reference-labeled original test pages.",
            "The original canonical train/validation/test manifests are unchanged.",
        ],
    }
    write_json(output_root / "protocol.json", protocol)
    print(json.dumps({"status": "prepared", **manifest_stats,
                      "new_group_overlap_count": len(group_overlaps),
                      "expanded_test_sha256": protocol["derived_manifests"]["expanded_test_sha256"]},
                     ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
