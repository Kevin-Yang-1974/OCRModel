#!/usr/bin/env python3
"""Audit AncientDoc train/validation/test split leakage.

This script is read-only and CPU-only. It audits converted AncientDoc
layout-page manifests and reports source/book overlap, exact image duplicates,
exact normalized-text duplicates, and optional perceptual-image hash overlap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence


SPLITS = ("train", "validation", "test")


class AncientDocAuditFailure(RuntimeError):
    pass


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise AncientDocAuditFailure(f"Expected JSON object at {path}:{line_number}")
            records.append(payload)
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_text(value: str) -> str:
    return "".join(value.split())


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit AncientDoc converted manifests for cross-split source, image, "
            "and text leakage."
        )
    )
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        help="Manifest path. Repeat for train/validation/test when --dataset-root is not used.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-image-hash", action="store_true")
    parser.add_argument("--enable-perceptual-hash", action="store_true")
    parser.add_argument("--perceptual-hash-threshold", type=nonnegative_int, default=4)
    parser.add_argument("--max-examples", type=positive_int, default=30)
    return parser.parse_args(argv)


def resolve_manifests(args: argparse.Namespace) -> list[Path]:
    if args.dataset_root is not None and args.manifest:
        raise AncientDocAuditFailure("Pass --dataset-root or --manifest, not both.")
    if args.dataset_root is not None:
        root = args.dataset_root.expanduser().resolve()
        paths = [root / split / "manifest.jsonl" for split in SPLITS]
    elif args.manifest:
        paths = [path.expanduser().resolve() for path in args.manifest]
    else:
        raise AncientDocAuditFailure("Pass --dataset-root or at least one --manifest.")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise AncientDocAuditFailure(f"Missing manifest(s): {missing}")
    return paths


def safe_relative(value: Any, context: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise AncientDocAuditFailure(f"{context} must be a non-empty relative path.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise AncientDocAuditFailure(f"{context} is unsafe: {value!r}")
    return relative


def resolve_image(record: dict[str, Any], manifest: Path, context: str) -> Path:
    relative = safe_relative(record.get("image"), f"{context}.image")
    image = manifest.parent / Path(*relative.parts)
    if not image.is_file():
        raise AncientDocAuditFailure(f"{context}.image does not exist: {image}")
    return image


def page_text(record: dict[str, Any]) -> str:
    value = record.get("page_text")
    if isinstance(value, str):
        return value
    conversations = record.get("conversations")
    if isinstance(conversations, list):
        for message in reversed(conversations):
            if isinstance(message, dict) and message.get("from") == "gpt":
                text = message.get("value")
                if isinstance(text, str):
                    return text
    return ""


def split_from_manifest(manifest: Path, record: dict[str, Any]) -> str:
    value = record.get("split")
    if isinstance(value, str) and value:
        return value
    parent = manifest.parent.name
    return parent if parent else "unknown"


def parse_original_image(record: dict[str, Any], page_id: str) -> dict[str, Any]:
    image = str(record.get("original_image") or record.get("image") or "")
    parts = PurePosixPath(image).parts if image else ()
    category = "unknown"
    book = "unknown"
    if len(parts) >= 3:
        category = parts[-3]
        book = parts[-2]
    elif len(parts) >= 2:
        category = parts[-2]
        book = parts[-1].rsplit(".", 1)[0]
    page_number = None
    for candidate in reversed(parts or (page_id,)):
        match = re.search(r"(?:page|p|页)[_\-]?(\d+)", candidate, flags=re.IGNORECASE)
        if match:
            page_number = int(match.group(1))
            break
    return {
        "original_image": image,
        "category": category,
        "book": book,
        "book_key": f"{category}/{book}",
        "page_number": page_number,
        "book_page_key": (
            f"{category}/{book}/page_{page_number}" if page_number is not None else None
        ),
    }


def average_hash(path: Path) -> int:
    try:
        from PIL import Image
    except ImportError as exc:
        raise AncientDocAuditFailure(
            "--enable-perceptual-hash requires Pillow to be installed."
        ) from exc
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((8, 8))
        values = list(grayscale.getdata())
    mean = sum(values) / len(values)
    result = 0
    for index, value in enumerate(values):
        if value >= mean:
            result |= 1 << index
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def record_summary(
    *,
    manifest: Path,
    record: dict[str, Any],
    index: int,
    skip_image_hash: bool,
    enable_perceptual_hash: bool,
) -> dict[str, Any]:
    page_id = record.get("page_id")
    if not isinstance(page_id, str) or not page_id:
        raise AncientDocAuditFailure(f"{manifest}[{index}].page_id must be non-empty.")
    split = split_from_manifest(manifest, record)
    image_path = resolve_image(record, manifest, f"{manifest.name}[{index}]")
    text = page_text(record)
    normalized = normalize_text(text)
    metadata = parse_original_image(record, page_id)
    return {
        "page_id": page_id,
        "split": split,
        "manifest": str(manifest),
        "image": str(image_path),
        "source_group_id": str(record.get("source_group_id", "")),
        "content_id": str(record.get("content_id", "")),
        "reference_characters": len(text),
        "normalized_text_characters": len(normalized),
        "normalized_text_sha256": sha256_text(normalized),
        "image_sha256": None if skip_image_hash else sha256_file(image_path),
        "perceptual_hash": average_hash(image_path) if enable_perceptual_hash else None,
        **metadata,
    }


def split_set(items: Iterable[dict[str, Any]]) -> set[str]:
    return {str(item["split"]) for item in items}


def example_records(items: list[dict[str, Any]], max_examples: int) -> list[dict[str, Any]]:
    fields = (
        "split",
        "page_id",
        "category",
        "book",
        "page_number",
        "original_image",
        "reference_characters",
        "image_sha256",
        "normalized_text_sha256",
    )
    examples = []
    for item in sorted(items, key=lambda row: (str(row["split"]), str(row["page_id"])))[:max_examples]:
        examples.append({field: item.get(field) for field in fields})
    return examples


def cross_split_overlaps(
    pages: list[dict[str, Any]],
    key: str,
    *,
    max_examples: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for page in pages:
        value = page.get(key)
        if value is None or value == "":
            continue
        grouped[str(value)].append(page)
    overlapping = {
        value: items for value, items in grouped.items() if len(split_set(items)) > 1
    }
    examples = []
    for value, items in sorted(
        overlapping.items(),
        key=lambda pair: (-len(split_set(pair[1])), -len(pair[1]), pair[0]),
    )[:max_examples]:
        examples.append(
            {
                "key": value,
                "splits": sorted(split_set(items)),
                "pages": len(items),
                "examples": example_records(items, max_examples=4),
            }
        )
    return {
        "key": key,
        "unique_values": len(grouped),
        "cross_split_value_count": len(overlapping),
        "cross_split_page_count": sum(len(items) for items in overlapping.values()),
        "examples": examples,
    }


def perceptual_hash_overlaps(
    pages: list[dict[str, Any]],
    *,
    threshold: int,
    max_examples: int,
) -> dict[str, Any]:
    hashed = [page for page in pages if isinstance(page.get("perceptual_hash"), int)]
    pairs = []
    for left_index, left in enumerate(hashed):
        for right in hashed[left_index + 1 :]:
            if left["split"] == right["split"]:
                continue
            distance = hamming_distance(int(left["perceptual_hash"]), int(right["perceptual_hash"]))
            if distance <= threshold:
                pairs.append(
                    {
                        "distance": distance,
                        "left": example_records([left], 1)[0],
                        "right": example_records([right], 1)[0],
                    }
                )
    pairs.sort(key=lambda item: (int(item["distance"]), str(item["left"]["page_id"])))
    return {
        "enabled": True,
        "threshold": threshold,
        "cross_split_pair_count": len(pairs),
        "examples": pairs[:max_examples],
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifests = resolve_manifests(args)
    pages: list[dict[str, Any]] = []
    seen_page_ids: dict[str, str] = {}
    for manifest in manifests:
        records = read_jsonl(manifest)
        for index, record in enumerate(records):
            summary = record_summary(
                manifest=manifest,
                record=record,
                index=index,
                skip_image_hash=args.skip_image_hash,
                enable_perceptual_hash=args.enable_perceptual_hash,
            )
            previous = seen_page_ids.setdefault(summary["page_id"], summary["split"])
            if previous != summary["split"]:
                raise AncientDocAuditFailure(
                    f"page_id occurs in multiple splits: {summary['page_id']}"
                )
            pages.append(summary)
    split_counts = Counter(str(page["split"]) for page in pages)
    category_counts = Counter(str(page["category"]) for page in pages)
    checks = {
        "book_key": cross_split_overlaps(pages, "book_key", max_examples=args.max_examples),
        "book": cross_split_overlaps(pages, "book", max_examples=args.max_examples),
        "book_page_key": cross_split_overlaps(
            pages, "book_page_key", max_examples=args.max_examples
        ),
        "original_image": cross_split_overlaps(
            pages, "original_image", max_examples=args.max_examples
        ),
        "normalized_text_sha256": cross_split_overlaps(
            pages, "normalized_text_sha256", max_examples=args.max_examples
        ),
        "image_sha256": (
            {"enabled": False, "reason": "--skip-image-hash"}
            if args.skip_image_hash
            else cross_split_overlaps(pages, "image_sha256", max_examples=args.max_examples)
        ),
        "perceptual_hash": (
            perceptual_hash_overlaps(
                pages,
                threshold=args.perceptual_hash_threshold,
                max_examples=args.max_examples,
            )
            if args.enable_perceptual_hash
            else {"enabled": False, "reason": "pass --enable-perceptual-hash"}
        ),
    }
    high_risk_keys = ("book_key", "book_page_key", "original_image", "normalized_text_sha256")
    high_risk_count = sum(
        int(checks[key].get("cross_split_value_count", 0))
        for key in high_risk_keys
        if isinstance(checks[key], dict)
    )
    image_duplicate_count = (
        int(checks["image_sha256"].get("cross_split_value_count", 0))
        if isinstance(checks["image_sha256"], dict)
        else 0
    )
    status = "ok"
    if high_risk_count or image_duplicate_count:
        status = "leakage_found"
    elif int(checks["book"].get("cross_split_value_count", 0)):
        status = "source_overlap_found"

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            args.dataset_root.expanduser().resolve() / "audit" / "ancientdoc_split_leakage"
            if args.dataset_root is not None
            else manifests[0].parent / "audit" / "ancientdoc_split_leakage"
        )
    )
    report = {
        "schema_version": 1,
        "status": status,
        "manifests": [str(path) for path in manifests],
        "page_count": len(pages),
        "split_counts": dict(sorted(split_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "image_hash_checked": not args.skip_image_hash,
        "perceptual_hash_checked": bool(args.enable_perceptual_hash),
        "checks": checks,
        "recommendation": (
            "Rebuild source-isolated splits before using AncientDoc as a few-shot "
            "generalization benchmark."
            if status != "ok"
            else "No cross-split leakage was detected by the enabled exact checks."
        ),
    }
    write_json(output_dir / "split_leakage_audit.json", report)
    write_markdown(output_dir / "split_leakage_audit.md", report)
    return {
        "event": "ancientdoc_split_leakage_audit_completed",
        "status": status,
        "page_count": len(pages),
        "split_counts": report["split_counts"],
        "book_key_cross_split": checks["book_key"]["cross_split_value_count"],
        "book_page_key_cross_split": checks["book_page_key"]["cross_split_value_count"],
        "original_image_cross_split": checks["original_image"]["cross_split_value_count"],
        "normalized_text_cross_split": checks["normalized_text_sha256"][
            "cross_split_value_count"
        ],
        "image_sha256_cross_split": (
            checks["image_sha256"].get("cross_split_value_count")
            if isinstance(checks["image_sha256"], dict)
            else None
        ),
        "output_dir": str(output_dir),
        "summary": str(output_dir / "split_leakage_audit.json"),
        "report": str(output_dir / "split_leakage_audit.md"),
    }


def markdown_table(rows: list[dict[str, Any]], fields: Sequence[str]) -> str:
    if not rows:
        return "_No examples._\n"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(str(row.get(field, "")).replace("|", "\\|") for field in fields)
            + " |"
        )
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# AncientDoc split leakage audit",
        "",
        "## Overview",
        "",
        "```json",
        compact_json(
            {
                "status": report["status"],
                "page_count": report["page_count"],
                "split_counts": report["split_counts"],
                "image_hash_checked": report["image_hash_checked"],
                "perceptual_hash_checked": report["perceptual_hash_checked"],
            }
        ),
        "```",
        "",
        "## Cross-split checks",
        "",
    ]
    rows = []
    for key, check in report["checks"].items():
        if not isinstance(check, dict):
            continue
        rows.append(
            {
                "check": key,
                "enabled": check.get("enabled", True),
                "cross_split_value_count": check.get("cross_split_value_count", ""),
                "cross_split_page_count": check.get("cross_split_page_count", ""),
                "cross_split_pair_count": check.get("cross_split_pair_count", ""),
                "reason": check.get("reason", ""),
            }
        )
    lines.append(
        markdown_table(
            rows,
            (
                "check",
                "enabled",
                "cross_split_value_count",
                "cross_split_page_count",
                "cross_split_pair_count",
                "reason",
            ),
        )
    )
    for key, check in report["checks"].items():
        if not isinstance(check, dict):
            continue
        examples = check.get("examples")
        if not examples:
            continue
        lines.extend(
            [
                "",
                f"## Examples: {key}",
                "",
                "```json",
                json.dumps(examples, ensure_ascii=False, indent=2, allow_nan=False),
                "```",
            ]
        )
    lines.extend(["", "## Recommendation", "", str(report["recommendation"])])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = audit(parse_args(argv))
    except AncientDocAuditFailure as exc:
        print(compact_json({"event": "ancientdoc_split_leakage_audit_failed", "error": str(exc)}))
        return 1
    print(compact_json(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
