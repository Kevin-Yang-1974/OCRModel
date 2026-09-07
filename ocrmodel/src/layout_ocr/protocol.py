from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PageRecord:
    page_id: str
    split: str
    source_group: str
    duplicate_group: str

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "PageRecord":
        return cls(**{field: str(value[field]) for field in cls.__dataclass_fields__})


def read_manifest(path: Path) -> list[PageRecord]:
    with path.open(encoding="utf-8") as handle:
        return [PageRecord.from_dict(json.loads(line)) for line in handle if line.strip()]


def audit_split_isolation(records: Iterable[PageRecord]) -> list[str]:
    records = list(records)
    errors: list[str] = []
    page_ids: set[str] = set()
    for record in records:
        if record.split not in {"train", "validation", "test"}:
            errors.append(f"page {record.page_id}: unsupported split {record.split}")
        if record.page_id in page_ids:
            errors.append(f"duplicate page_id: {record.page_id}")
        page_ids.add(record.page_id)

    for field in ("source_group", "duplicate_group"):
        memberships: dict[str, set[str]] = defaultdict(set)
        for record in records:
            memberships[getattr(record, field)].add(record.split)
        for group, splits in sorted(memberships.items()):
            if len(splits) > 1:
                errors.append(f"{field} {group} crosses splits: {','.join(sorted(splits))}")
    return errors


def select_mechanism_screen(
    records: Iterable[PageRecord], pages: int = 128, seed: int = 42
) -> list[PageRecord]:
    candidates = [record for record in records if record.split == "train"]
    if len(candidates) < pages:
        raise ValueError(f"requested {pages} train pages, found {len(candidates)}")
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, pages), key=lambda record: record.page_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit R1/R2 manifest split isolation")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--screen-pages", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = read_manifest(args.manifest)
    errors = audit_split_isolation(records)
    result: dict[str, object] = {"records": len(records), "errors": errors}
    if not errors:
        selected = select_mechanism_screen(records, args.screen_pages, args.seed)
        result["mechanism_screen"] = {
            "pages": len(selected),
            "seed": args.seed,
            "page_ids": [record.page_id for record in selected],
        }
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
