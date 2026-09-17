"""AncientDoc split5 manifest preparation and access helpers.

The historical AncientDoc test protocol is the book-isolated ``split5``
label file.  The images remain on the shared read-only dataset; only a
portable personal manifest is written under the experiment workspace.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


DATASET_ID = "AncientDoc"
SOURCE_SPLIT = "split5"
EVALUATION_SPLIT = "test"
EXPECTED_COUNT = 516


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _reference_text(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("AncientDoc record has no conversations list")
    for conversation in conversations:
        if isinstance(conversation, dict) and conversation.get("from") == "gpt":
            value = conversation.get("value")
            if isinstance(value, str):
                return value
    raise ValueError("AncientDoc record has no GPT OCR reference")


def _relative_image(value: Any) -> Path:
    image = Path(str(value).replace("\\", "/"))
    if image.is_absolute() or ".." in image.parts:
        raise ValueError(f"AncientDoc image must be relative: {value!r}")
    return image


def build_manifest(
    label_json: Path,
    image_root: Path,
    output: Path,
    *,
    source_split: str = SOURCE_SPLIT,
) -> dict[str, Any]:
    if source_split != SOURCE_SPLIT:
        raise ValueError(f"Unsupported AncientDoc source split: {source_split!r}")
    labels = _read_json(label_json)
    if not isinstance(labels, list) or len(labels) != EXPECTED_COUNT:
        raise ValueError(
            f"AncientDoc {source_split} must contain {EXPECTED_COUNT} records; "
            f"got {len(labels) if isinstance(labels, list) else type(labels).__name__}"
        )

    image_root = image_root.expanduser().resolve()
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(labels):
        if not isinstance(source, dict) or source.get("image") is None:
            raise ValueError(f"AncientDoc record {index} has no image")
        image = _relative_image(source["image"])
        image_path = (image_root / image).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        parts = image.parts
        category = parts[1] if len(parts) >= 2 and parts[0] == "imgs" else "unknown"
        book = parts[2] if len(parts) >= 3 and parts[0] == "imgs" else "unknown"
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "source_split": source_split,
                "split": EVALUATION_SPLIT,
                "page_id": f"ancientdoc_{source_split}_{index:04d}",
                "image": image.as_posix(),
                "page_text": _reference_text(source),
                "domain": DATASET_ID,
                "category": category,
                "book": book,
            }
        )

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "dataset_id": DATASET_ID,
        "source_split": source_split,
        "split": EVALUATION_SPLIT,
        "count": len(rows),
        "manifest": str(output),
        "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "source_label": str(label_json.expanduser().resolve()),
        "source_label_sha256": hashlib.sha256(label_json.read_bytes()).hexdigest(),
        "image_root": str(image_root),
    }


def manifest_contract(manifest: Path, *, split: str = EVALUATION_SPLIT) -> dict[str, Any]:
    rows = _read_jsonl(manifest)
    if len(rows) != EXPECTED_COUNT:
        raise ValueError(f"AncientDoc manifest has {len(rows)} rows, expected {EXPECTED_COUNT}")
    domains: dict[str, int] = {}
    for row in rows:
        if row.get("dataset_id") != DATASET_ID or row.get("split") != split:
            raise ValueError("AncientDoc manifest dataset or split mismatch")
        page_id = row.get("page_id")
        image = row.get("image")
        if not page_id or not image:
            raise ValueError("AncientDoc manifest row is missing page_id or image")
        domain = str(row.get("domain", "unknown"))
        domains[domain] = domains.get(domain, 0) + 1
    return {
        "dataset_id": DATASET_ID,
        "source_split": SOURCE_SPLIT,
        "split": split,
        "expected_count": EXPECTED_COUNT,
        "manifest_count": len(rows),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "domain_counts": domains,
        "test_used": True,
        "input_granularity": "whole_page_image",
        "model_inputs": ["whole_page_image", "ocr_prompt"],
    }


def iter_manifest(
    manifest: Path,
    image_root: Path,
    *,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    image_root = image_root.expanduser().resolve()
    seen = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("dataset_id") != DATASET_ID or record.get("split") != EVALUATION_SPLIT:
                raise ValueError(f"AncientDoc manifest mismatch at line {line_number}")
            image = _relative_image(record.get("image"))
            image_path = (image_root / image).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            copied = dict(record)
            copied["image"] = image.as_posix()
            copied["image_path"] = str(image_path)
            copied["protocol"] = {
                "model_inputs": ["whole_page_image", "ocr_prompt"],
                "bbox_as_input": False,
                "direction_as_input": False,
                "reading_order_as_input": False,
            }
            yield copied
            seen += 1
            if limit is not None and seen >= limit:
                return

