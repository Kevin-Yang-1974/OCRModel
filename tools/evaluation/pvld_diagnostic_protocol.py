from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


BUCKET_NAMES = ("0-8", "9-16", "17-32", ">32")


def region_count_bucket(count: int) -> str:
    if count <= 8:
        return "0-8"
    if count <= 16:
        return "9-16"
    if count <= 32:
        return "17-32"
    return ">32"


def record_region_count(record: Mapping[str, Any]) -> int:
    regions = record.get("layout_regions", record.get("regions", ()))
    return len(regions)


def select_bucket_indices(
    records: Sequence[Mapping[str, Any]],
    pages_per_bucket: int,
) -> dict[str, list[int]]:
    selected = {name: [] for name in BUCKET_NAMES}
    for index, record in enumerate(records):
        bucket = region_count_bucket(record_region_count(record))
        if len(selected[bucket]) < pages_per_bucket:
            selected[bucket].append(index)
        if all(len(indices) == pages_per_bucket for indices in selected.values()):
            break
    return selected


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def duplicate_diagnostics(
    boxes: Sequence[Sequence[float]],
    threshold: float = 0.9,
) -> dict[str, float | int | None]:
    duplicate_flags: list[bool] = []
    first_duplicate: int | None = None
    for index, box in enumerate(boxes):
        duplicate = any(box_iou(box, previous) >= threshold for previous in boxes[:index])
        duplicate_flags.append(duplicate)
        if duplicate and first_duplicate is None:
            first_duplicate = index
    duplicate_count = sum(duplicate_flags)
    return {
        "regions": len(boxes),
        "duplicate_after_first_count": duplicate_count,
        "duplicate_after_first_rate": duplicate_count / len(boxes) if boxes else 0.0,
        "first_duplicate_region_index": first_duplicate,
    }


def mean_or_none(values: Sequence[float | int | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None

