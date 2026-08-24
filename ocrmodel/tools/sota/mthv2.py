"""Safe MTHv2 manifest access for SOTA smoke and future evaluation."""

from pathlib import Path
import json
from typing import Any, Iterator


DATASET_ID = "mthv2_layout_page_v1"
EXPECTED_COUNTS = {"train": 2159, "validation": 240, "test": 800}


def iter_manifest(manifest: Path, image_root: Path, *, split: str, limit: int | None = None, allow_test: bool = False) -> Iterator[dict[str, Any]]:
    if split not in EXPECTED_COUNTS:
        raise ValueError(f"Unsupported MTHv2 split: {split!r}")
    if split == "test" and not allow_test:
        raise PermissionError("MTHv2 test is locked for the current SOTA smoke phase.")
    seen = 0
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != split:
                raise ValueError(f"Manifest split mismatch at line {line_number}: {record.get('split')!r}")
            if record.get("input_level") != "page":
                raise ValueError("SOTA protocol requires page-level whole-page records.")
            if record.get("image") is None or record.get("page_id") is None:
                raise ValueError(f"Missing page_id/image at line {line_number}")
            if any(key in record for key in ("oracle_chunk", "chunk_index", "source_region_indices")):
                raise ValueError("Oracle chunks are forbidden in the MTHv2 whole-page SOTA protocol.")
            image_path = (image_root / Path(*str(record["image"]).replace("\\", "/").split("/"))).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            copied = dict(record)
            copied["image_path"] = str(image_path)
            copied["protocol"] = {"model_inputs": ["whole_page_image", "ocr_prompt"], "bbox_as_input": False, "direction_as_input": False, "reading_order_as_input": False}
            yield copied
            seen += 1
            if limit is not None and seen >= limit:
                return


def manifest_contract(manifest: Path, split: str) -> dict[str, Any]:
    return {"dataset_id": DATASET_ID, "split": split, "expected_count": EXPECTED_COUNTS[split], "test_used": False, "input_granularity": "whole_page_image", "model_inputs": ["whole_page_image", "ocr_prompt"]}
