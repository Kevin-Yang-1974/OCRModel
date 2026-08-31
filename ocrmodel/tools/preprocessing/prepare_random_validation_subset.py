#!/usr/bin/env python3
"""Lock a seed-keyed random subset of a whole-page validation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_random_key(seed: int, page_id: str) -> str:
    return hashlib.sha256(f"{seed}:{page_id}".encode("utf-8")).hexdigest()


def read_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def build_lock(
    source: Path,
    output: Path,
    metadata: Path,
    *,
    page_count: int,
    seed: int,
) -> dict[str, Any]:
    source = source.resolve()
    records = read_records(source)
    if len(records) < page_count:
        raise ValueError(f"Source manifest has only {len(records)} pages; requested {page_count}.")

    page_ids: set[str] = set()
    for record in records:
        if record.get("split") != "validation" or record.get("input_level") != "page":
            raise ValueError("Source manifest must contain validation whole-page records only.")
        page_id = str(record.get("page_id", ""))
        if not page_id or page_id in page_ids:
            raise ValueError(f"Missing or duplicate page_id: {page_id!r}")
        page_ids.add(page_id)

    selected = sorted(
        records,
        key=lambda record: stable_random_key(seed, str(record["page_id"])),
    )[:page_count]
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in selected
        ),
        encoding="utf-8",
    )
    tier_counts = Counter(str(record.get("tier", "unknown")) for record in selected)
    payload = {
        "status": "locked",
        "protocol_version": "bscc_seeded_validation_400_v1",
        "selection_split": "validation",
        "selection_rule": "Take the lowest SHA-256(seed:page_id) keys without replacement.",
        "seed": seed,
        "validation_page_count": page_count,
        "test_used_for_selection": False,
        "source_manifest": str(source),
        "source_manifest_sha256": sha256(source),
        "validation_manifest": str(output.resolve()),
        "validation_manifest_sha256": sha256(output),
        "tier_counts": dict(sorted(tier_counts.items())),
    }
    metadata.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def verify_existing(
    source: Path,
    output: Path,
    metadata: Path,
    *,
    page_count: int,
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    checks = {
        "status": payload.get("status") == "locked",
        "seed": payload.get("seed") == seed,
        "page_count": payload.get("validation_page_count") == page_count,
        "source_hash": payload.get("source_manifest_sha256") == sha256(source.resolve()),
        "output_hash": payload.get("validation_manifest_sha256") == sha256(output.resolve()),
        "test_isolation": payload.get("test_used_for_selection") is False,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Existing validation lock does not match: {failed}")
    if len(read_records(output)) != page_count:
        raise ValueError("Existing validation subset page count does not match its lock.")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--page-count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    if args.page_count < 1:
        raise ValueError("--page-count must be positive.")
    existing = (args.output_manifest.exists(), args.metadata.exists())
    if any(existing):
        if not all(existing) or not args.reuse_existing:
            raise FileExistsError("Validation manifest and lock must be created or reused together.")
        payload = verify_existing(
            args.source_manifest,
            args.output_manifest,
            args.metadata,
            page_count=args.page_count,
            seed=args.seed,
        )
        event = "random_validation_lock_reused"
    else:
        payload = build_lock(
            args.source_manifest,
            args.output_manifest,
            args.metadata,
            page_count=args.page_count,
            seed=args.seed,
        )
        event = "random_validation_locked"
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
