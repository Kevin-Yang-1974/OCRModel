"""Deterministic validation sampling and conservative character alignment for line-mask v3."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SOURCE_GROUP_FIELDS = (
    "source_group_id",
    "source_group",
    "source_id",
    "book_id",
    "version_id",
    "collection_id",
    "archive_id",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_safetensors(model_path: Path, model_revision: str) -> dict[str, Any]:
    """Fingerprint the original model weight shards and reject adapter-only files."""
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"raw model snapshot does not exist: {root}")
    files = sorted(path for path in root.rglob("*.safetensors") if path.is_file())
    if not files:
        raise FileNotFoundError(f"no safetensors model weights found under: {root}")
    adapter_files = [path for path in files if "adapter" in path.name.lower()]
    if adapter_files:
        raise ValueError(f"raw model snapshot contains adapter weights: {adapter_files[:3]}")

    entries = {path.relative_to(root).as_posix(): sha256_file(path) for path in files}
    aggregate = hashlib.sha256()
    for relative, file_digest in entries.items():
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(bytes([0]))
        aggregate.update(file_digest.encode("ascii"))
        aggregate.update(bytes([10]))
    return {
        "model_path": str(root),
        "model_revision": model_revision,
        "model_weights_sha256": aggregate.hexdigest(),
        "safetensors": entries,
        "decoder_lora_loaded": False,
    }


def fingerprint_decoder_lora(checkpoint_path: Path, model_revision: str) -> dict[str, Any]:
    """Fingerprint the frozen rank-8 decoder LoRA paired with the v2 mask head."""
    root = Path(checkpoint_path).resolve()
    weights = root / "decoder_lora.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"decoder LoRA weights do not exist: {weights}")
    return {
        "checkpoint_path": str(root),
        "model_revision": model_revision,
        "decoder_lora_sha256": sha256_file(weights),
        "decoder_lora_file": weights.name,
        "rank": 8,
        "alpha": 8.0,
        "dropout": 0.0,
        "decoder_lora_loaded": True,
    }


def source_group(record: dict[str, Any], field: str | None = None) -> tuple[str, str]:
    """Return a source group and the manifest field that supplied it."""

    fields = (field,) if field else SOURCE_GROUP_FIELDS
    for name in fields:
        if name and record.get(name) not in (None, ""):
            return str(record[name]), name
    raise ValueError(
        f"page {record.get('page_id')!r} has no source-group field; "
        f"tried {fields!r}; pass --source-group-field if the manifest uses another name"
    )


def _quartile_ranks(values: Sequence[int]) -> list[int]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    result = [0] * len(values)
    for rank, index in enumerate(order):
        result[index] = min(3, rank * 4 // max(1, len(values)))
    return result


def _largest_remainder_quotas(buckets: dict[Any, Sequence[int]], count: int, seed: int) -> dict[Any, int]:
    total = sum(len(indices) for indices in buckets.values())
    exact = {key: count * len(indices) / total for key, indices in buckets.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remainder = count - sum(quotas.values())
    ranked = sorted(
        buckets,
        key=lambda key: (
            -(exact[key] - quotas[key]),
            hashlib.sha256(f"{seed}|quota|{key}".encode()).hexdigest(),
        ),
    )
    for key in ranked[:remainder]:
        quotas[key] += 1
    return quotas


def layout_region_count(record: dict[str, Any]) -> int:
    regions = record.get("regions")
    if regions is not None:
        return len(regions)
    return len({
        str(character.get("line_index"))
        for character in record.get("characters", [])
        if isinstance(character, dict) and character.get("line_index") is not None
    })


def stratified_sample(
    records: Sequence[dict[str, Any]],
    count: int,
    seed: int = 42,
    source_group_field: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic, approximately proportional source/length/layout sample."""

    if count < 1 or count > len(records):
        raise ValueError(f"sample size {count} must be in [1, {len(records)}]")
    lengths = [len("".join(str(record["page_text"]).split())) for record in records]
    region_counts = [layout_region_count(record) for record in records]
    length_bins = _quartile_ranks(lengths)
    region_bins = _quartile_ranks(region_counts)
    source_values = [source_group(record, source_group_field) for record in records]

    buckets: dict[tuple[str, int, int], list[int]] = defaultdict(list)
    for index, (group, _) in enumerate(source_values):
        buckets[(group, length_bins[index], region_bins[index])].append(index)

    if count < len(set(region_bins)):
        raise ValueError("sample size is too small to cover every available layout-density quartile")
    density_buckets: dict[int, list[int]] = defaultdict(list)
    for index, density_bin in enumerate(region_bins):
        density_buckets[density_bin].append(index)
    density_quotas = _largest_remainder_quotas(density_buckets, count, seed)
    chosen: list[int] = []
    for density_bin in sorted(density_buckets):
        density_members = density_buckets[density_bin]
        nested_buckets: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index in density_members:
            group, _ = source_values[index]
            nested_buckets[(group, length_bins[index])].append(index)
        nested_quotas = _largest_remainder_quotas(
            nested_buckets, density_quotas[density_bin], seed + density_bin
        )
        for key in sorted(nested_buckets):
            members = list(nested_buckets[key])
            local_seed = f"{seed}|{key[0]}|{key[1]}|{density_bin}".encode()
            bucket_seed = int.from_bytes(hashlib.sha256(local_seed).digest()[:8], "big")
            random.Random(bucket_seed).shuffle(members)
            chosen.extend(members[: nested_quotas[key]])
    if len(chosen) != count or len(set(chosen)) != count:
        raise RuntimeError("stratified allocation did not produce the requested unique sample")

    selected = [records[index] for index in chosen]
    selected.sort(key=lambda record: str(record["page_id"]))
    selected_ids = {str(record["page_id"]) for record in selected}
    trace_page_ids = sorted(
        selected_ids,
        key=lambda page_id: hashlib.sha256(f"{seed}|trace|{page_id}".encode()).hexdigest(),
    )[:3]
    selected_bucket_counts = Counter(
        (source_values[index][0], length_bins[index], region_bins[index]) for index in chosen
    )
    bucket_report = [
        {
            "source_group": key[0],
            "text_length_quartile": key[1],
            "region_count_quartile": key[2],
            "available_pages": len(indices),
            "selected_pages": selected_bucket_counts[key],
        }
        for key, indices in sorted(buckets.items())
    ]
    return selected, {
        "seed": seed,
        "requested_pages": count,
        "selected_pages": len(selected),
        "source_group_fields": sorted({field for _, field in source_values}),
        "stratification": [
            "source_group",
            "text_length_quartile_within_layout_density_quartile",
            "layout_density_quartile",
        ],
        "layout_density_quartile_counts": {
            str(density_bin): sum(value == density_bin for value in region_bins)
            for density_bin in sorted(density_buckets)
        },
        "selected_layout_density_quartile_counts": {
            str(density_bin): sum(region_bins[index] == density_bin for index in chosen)
            for density_bin in sorted(density_buckets)
        },
        "layout_region_count_range_by_quartile": {
            str(density_bin): [
                min(region_counts[index] for index in density_buckets[density_bin]),
                max(region_counts[index] for index in density_buckets[density_bin]),
            ]
            for density_bin in sorted(density_buckets)
        },
        "selected_layout_region_count_range_by_quartile": {
            str(density_bin): [
                min(region_counts[index] for index in chosen if region_bins[index] == density_bin),
                max(region_counts[index] for index in chosen if region_bins[index] == density_bin),
            ]
            for density_bin in sorted(density_buckets)
        },
        "selected_layout_region_count_range": [
            min(region_counts[index] for index in chosen),
            max(region_counts[index] for index in chosen),
        ],
        "selected_pages_above_sparse24_region_cap": sum(
            region_counts[index] > 24 for index in chosen
        ),
        "layout_density_quartile_coverage_complete": (
            {region_bins[index] for index in chosen} == set(region_bins)
        ),
        "bucket_counts": bucket_report,
        "full_mask_trace_page_ids": trace_page_ids,
        "selected_page_ids": [str(record["page_id"]) for record in selected],
    }


def align_nonspace(reference: str, prediction: str) -> dict[str, Any]:
    """Align whitespace-stripped text and retain ambiguity across optimal edit paths.

    ``prediction_to_reference`` is one deterministic optimal alignment. A prediction
    position is marked ambiguous when more than one reference position can match it
    along an optimal path, which makes repeated-character uncertainty explicit.
    """

    ref_positions = [index for index, char in enumerate(reference) if not char.isspace()]
    pred_positions = [index for index, char in enumerate(prediction) if not char.isspace()]
    ref = [reference[index] for index in ref_positions]
    pred = [prediction[index] for index in pred_positions]
    n, m = len(ref), len(pred)

    forward = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        forward[i][0] = i
    for j in range(1, m + 1):
        forward[0][j] = j
    moves = [bytearray(m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        moves[0][j] = 2  # insertion
    for i in range(1, n + 1):
        moves[i][0] = 1  # deletion
        for j in range(1, m + 1):
            best, move = forward[i - 1][j - 1] + (ref[i - 1] != pred[j - 1]), 0
            deletion = forward[i - 1][j] + 1
            insertion = forward[i][j - 1] + 1
            if deletion < best:
                best, move = deletion, 1
            if insertion < best:
                best, move = insertion, 2
            forward[i][j], moves[i][j] = best, move

    # One backward row at a time is enough to find every exact diagonal that can
    # occur on an optimal path; avoid a second full dynamic-programming matrix.
    candidates: list[set[int]] = [set() for _ in range(m)]
    next_row = [m - j for j in range(m + 1)]
    total = forward[n][m]
    for i in range(n, -1, -1):
        current = [0] * (m + 1)
        current[m] = n - i
        for j in range(m - 1, -1, -1):
            if i == n:
                current[j] = m - j
            else:
                diagonal = next_row[j + 1] + (ref[i] != pred[j])
                current[j] = min(diagonal, next_row[j] + 1, current[j + 1] + 1)
                if ref[i] == pred[j] and forward[i][j] + next_row[j + 1] == total:
                    candidates[j].add(i)
        next_row = current

    pred_to_ref: list[int | None] = [None] * m
    aligned_ref: list[int | None] = [None] * m
    operations = ["I"] * m
    i, j = n, m
    while i or j:
        move = moves[i][j]
        if i and j and move == 0:
            aligned_ref[j - 1] = i - 1
            pred_to_ref[j - 1] = i - 1 if ref[i - 1] == pred[j - 1] else None
            operations[j - 1] = "M" if ref[i - 1] == pred[j - 1] else "S"
            i -= 1
            j -= 1
        elif i and (not j or move == 1):
            i -= 1
        else:
            operations[j - 1] = "I"
            j -= 1

    return {
        "edit_distance": total,
        "reference_positions": ref_positions,
        "prediction_positions": pred_positions,
        "prediction_to_reference": pred_to_ref,
        "prediction_aligned_reference_position": aligned_ref,
        "operations": operations,
        "candidate_reference_positions": [sorted(values) for values in candidates],
        "ambiguous": [len(values) > 1 for values in candidates],
    }
