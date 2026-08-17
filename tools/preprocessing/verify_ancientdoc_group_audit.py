#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


REQUIRED_CHECKS = (
    "book_key",
    "book_page_key",
    "original_image",
    "normalized_text_sha256",
    "image_sha256",
)


def validate_audit(path: Path) -> dict[str, int]:
    if not path.is_file():
        raise ValueError(f"group-isolated leakage audit is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise ValueError(
            f"group-isolated leakage audit status is {payload.get('status')!r}: {path}"
        )

    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise ValueError(f"group-isolated leakage audit has no checks object: {path}")

    counts: dict[str, int] = {}
    for name in REQUIRED_CHECKS:
        check = checks.get(name)
        if not isinstance(check, dict) or "cross_split_value_count" not in check:
            raise ValueError(f"group-isolated leakage audit is missing checks.{name}: {path}")
        try:
            count = int(check["cross_split_value_count"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"checks.{name}.cross_split_value_count is not an integer: {path}"
            ) from exc
        if count != 0:
            raise ValueError(
                f"checks.{name}.cross_split_value_count must be 0, got {count}: {path}"
            )
        counts[name] = count
    return counts


def validate_split_ratios(path: Path, max_ratio_deviation: float) -> dict[str, float]:
    if not path.is_file():
        raise ValueError(f"group-isolated split audit is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise ValueError(f"group-isolated split audit status is not ok: {path}")
    requested = payload.get("ratios")
    allocation = payload.get("allocation")
    actual = allocation.get("actual_ratios") if isinstance(allocation, dict) else None
    if not isinstance(requested, dict) or not isinstance(actual, dict):
        raise ValueError(f"split audit does not contain requested/actual ratios: {path}")
    deviations: dict[str, float] = {}
    for split in ("train", "validation", "test"):
        try:
            deviations[split] = float(actual[split]) - float(requested[split])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid ratio metadata for {split}: {path}") from exc
        if abs(deviations[split]) > max_ratio_deviation:
            raise ValueError(
                f"{split} ratio deviation {deviations[split]:.6f} exceeds "
                f"{max_ratio_deviation:.6f}: {path}"
            )
    return deviations


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the persisted AncientDoc group-isolated leakage audit."
    )
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("--split-audit", type=Path)
    parser.add_argument("--max-ratio-deviation", type=float, default=0.03)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        counts = validate_audit(args.audit_json.expanduser().resolve())
        deviations = (
            validate_split_ratios(
                args.split_audit.expanduser().resolve(),
                args.max_ratio_deviation,
            )
            if args.split_audit is not None
            else None
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 66
    print(
        json.dumps(
            {
                "event": "ancientdoc_group_audit_verified",
                "audit": str(args.audit_json.expanduser().resolve()),
                "cross_split_counts": counts,
                "ratio_deviations": deviations,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
