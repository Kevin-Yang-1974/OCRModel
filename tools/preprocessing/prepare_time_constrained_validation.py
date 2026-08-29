#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_VERSION = "time_constrained_freeze_strategy_v1"
VARIANT = "original_pvld_freeze_strategy"
TIERS = ("s3-ancient-hard", "s4-mixed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def region_bucket(count: int) -> str:
    if count <= 8:
        return "001-008"
    if count <= 16:
        return "009-016"
    if count <= 32:
        return "017-032"
    if count <= 64:
        return "033-064"
    if count <= 128:
        return "065-128"
    return "129+"


def complexity_score(record: dict[str, Any]) -> float:
    generator = record.get("generator") or {}
    operations = (record.get("degradation") or {}).get("operations") or {}
    directions = {
        str(region.get("writing_direction", "unknown"))
        for region in record.get("regions", [])
    }
    return (
        math.log2(len(record.get("regions", [])) + 1.0)
        + 0.30 * float(generator.get("column_count", 0) or 0)
        + 0.30 * float(generator.get("row_count", 0) or 0)
        + 0.75 * max(0, len(directions) - 1)
        + 0.80 * float(operations.get("occlusion_count", 0) or 0)
        + 0.50 * float(operations.get("texture_strength", 0) or 0)
        + 0.08 * float(operations.get("gaussian_noise_sigma", 0) or 0)
        + 0.50 * float(operations.get("gaussian_blur_radius", 0) or 0)
    )


def complexity_levels(records: Iterable[dict[str, Any]]) -> dict[str, str]:
    scored = sorted(
        ((complexity_score(record), str(record["page_id"])) for record in records)
    )
    result: dict[str, str] = {}
    total = len(scored)
    for index, (_, page_id) in enumerate(scored):
        result[page_id] = ("low", "medium", "high")[min(2, index * 3 // total)]
    return result


def stable_key(seed: int, page_id: str) -> str:
    return hashlib.sha256(f"{seed}:{page_id}".encode("utf-8")).hexdigest()


def select_stratified(
    records: list[dict[str, Any]], *, count: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    levels = complexity_levels(records)
    strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        page_id = str(record["page_id"])
        strata[(region_bucket(len(record.get("regions", []))), levels[page_id])].append(record)
    for values in strata.values():
        values.sort(key=lambda record: stable_key(seed, str(record["page_id"])))
    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in strata}
    ordered_keys = sorted(strata)
    while len(selected) < count:
        progressed = False
        for key in ordered_keys:
            offset = offsets[key]
            if offset >= len(strata[key]):
                continue
            selected.append(strata[key][offset])
            offsets[key] += 1
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise RuntimeError(f"Only {len(selected)} pages are available for requested count {count}.")
    return selected, levels


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lock the 400-page S3/S4 validation manifest for the freeze-strategy run."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--pages-per-tier", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.pages_per_tier != 200:
        raise ValueError("This protocol requires exactly 200 pages per S3/S4 tier.")
    if args.output_manifest.exists() or args.metadata.exists():
        raise FileExistsError("Validation lock output already exists; use a new run directory.")

    source = args.source_manifest.resolve()
    records = read_jsonl(source)
    by_tier: dict[str, list[dict[str, Any]]] = {tier: [] for tier in TIERS}
    page_ids: set[str] = set()
    for record in records:
        if record.get("split") != "validation" or record.get("input_level") != "page":
            raise ValueError("Source manifest must contain validation whole-page records only.")
        page_id = str(record.get("page_id", ""))
        if not page_id or page_id in page_ids:
            raise ValueError(f"Missing or duplicate page_id: {page_id!r}")
        page_ids.add(page_id)
        tier = str(record.get("tier", ""))
        if tier in by_tier:
            by_tier[tier].append(record)

    selected: list[dict[str, Any]] = []
    selected_levels: dict[str, str] = {}
    for tier_index, tier in enumerate(TIERS):
        if len(by_tier[tier]) < args.pages_per_tier:
            raise ValueError(f"Tier {tier} has only {len(by_tier[tier])} validation pages.")
        tier_selected, levels = select_stratified(
            by_tier[tier], count=args.pages_per_tier, seed=args.seed + tier_index
        )
        selected.extend(tier_selected)
        selected_levels.update(levels)
    selected.sort(key=lambda record: (str(record["tier"]), str(record["page_id"])))
    if len(selected) != 400 or len({str(record["page_id"]) for record in selected}) != 400:
        raise RuntimeError("Locked validation selection is not exactly 400 unique pages.")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in selected
        ),
        encoding="utf-8",
    )
    tier_counts = Counter(str(record["tier"]) for record in selected)
    region_counts = Counter(
        f"{record['tier']}:{region_bucket(len(record.get('regions', [])))}"
        for record in selected
    )
    complexity_counts = Counter(
        f"{record['tier']}:{selected_levels[str(record['page_id'])]}"
        for record in selected
    )
    payload = {
        "status": "locked",
        "protocol_version": PROTOCOL_VERSION,
        "variant": VARIANT,
        "selection_split": "validation",
        "validation_page_count": 400,
        "pages_per_tier": 200,
        "test_used_for_selection": False,
        "source_manifest": str(source),
        "source_manifest_sha256": sha256(source),
        "validation_manifest": str(args.output_manifest.resolve()),
        "validation_manifest_sha256": sha256(args.output_manifest),
        "seed": args.seed,
        "selection_rule": (
            "Within each S3/S4 tier, deterministic round-robin over "
            "region-count bucket x within-tier complexity tertile."
        ),
        "complexity_score": (
            "log2(region_count+1)+columns+rows+direction_mix+occlusion+texture+noise+blur"
        ),
        "tier_counts": dict(sorted(tier_counts.items())),
        "region_count_stratum_counts": dict(sorted(region_counts.items())),
        "complexity_stratum_counts": dict(sorted(complexity_counts.items())),
    }
    write_json(args.metadata, payload)
    print(json.dumps({"event": "time_constrained_validation_locked", **payload}, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
