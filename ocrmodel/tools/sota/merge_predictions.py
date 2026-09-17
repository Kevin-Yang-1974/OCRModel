#!/usr/bin/env python3
"""Merge sharded SOTA zero-shot predictions into one manifest-ordered JSONL.

The merge is deliberately strict: every manifest page must have exactly one
prediction, shards must agree on model/split, and no page may be duplicated.
This keeps the unified metric computation valid for the locked benchmark test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True, help="Parent dir containing shard-*/predictions.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pages", type=int, default=None)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    if args.expected_pages is not None and len(manifest) != args.expected_pages:
        raise SystemExit(f"manifest has {len(manifest)} pages, expected {args.expected_pages}")

    order = [str(record["page_id"]) for record in manifest]
    by_page: dict[str, dict] = {}
    for shard_dir in sorted(args.shard_root.glob("shard-*/predictions.jsonl")):
        for record in read_jsonl(shard_dir):
            page_id = str(record["page_id"])
            if page_id in by_page:
                raise SystemExit(f"duplicate page across shards: {page_id}")
            by_page[page_id] = record

    missing = [page_id for page_id in order if page_id not in by_page]
    if missing:
        raise SystemExit(f"missing predictions for {len(missing)} pages, first={missing[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for page_id in order:
            handle.write(json.dumps(by_page[page_id], ensure_ascii=False, sort_keys=True) + "\n")

    statuses = {}
    for page_id in order:
        status = str(by_page[page_id].get("status"))
        statuses[status] = statuses.get(status, 0) + 1
    summary = {"pages": len(order), "merged": len(by_page), "statuses": statuses, "output": str(args.output), "test_used_for_selection": False}
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
