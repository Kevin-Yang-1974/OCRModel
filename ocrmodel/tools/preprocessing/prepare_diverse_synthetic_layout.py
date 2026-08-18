#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence


TIERS = ("s0-html-text", "s1-html-crop", "s2-hard")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Prepare audited train/validation/test ancient-photo-diverse layouts."
    )
    parser.add_argument("--content-manifest", type=Path, required=True)
    parser.add_argument("--content-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "config" / "synthetic_layout.ancient_photo_diverse_v1.json",
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--train-pages-per-tier", type=positive_int, default=8000)
    parser.add_argument("--validation-pages-per-tier", type=positive_int, default=1000)
    parser.add_argument("--test-pages-per-tier", type=positive_int, default=1000)
    parser.add_argument("--tier", choices=TIERS, action="append", default=[])
    browser = parser.add_mutually_exclusive_group()
    browser.add_argument("--browser-channel")
    browser.add_argument("--chromium-executable", type=Path)
    parser.add_argument("--expected-browser-version")
    parser.add_argument("--expected-browser-sha256")
    parser.add_argument("--font-file", type=Path, action="append", default=[])
    parser.add_argument("--expected-font-sha256", action="append", default=[])
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--progress-every", type=positive_int, default=1000)
    return parser.parse_args(argv)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if len(set(args.tier)) != len(args.tier):
        raise ValueError("--tier values must not repeat")
    if len(args.font_file) != len(args.expected_font_sha256):
        raise ValueError("Each --font-file requires one --expected-font-sha256")
    content_manifest = args.content_manifest.expanduser().resolve()
    content_root = args.content_root.expanduser().resolve() if args.content_root else None
    config = args.config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if not content_manifest.is_file() or not config.is_file():
        raise FileNotFoundError("content manifest or generator config is missing")
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    output_root.mkdir(parents=True)

    tiers = args.tier or list(TIERS)
    pages = {
        "train": args.train_pages_per_tier,
        "validation": args.validation_pages_per_tier,
        "test": args.test_pages_per_tier,
    }
    script_dir = Path(__file__).resolve().parent
    generator = script_dir / "generate_synthetic_layout.py"
    for split, split_pages in pages.items():
        command = [
            sys.executable,
            str(generator),
            "--content-manifest",
            str(content_manifest),
            "--output-dir",
            str(output_root / split),
            "--split",
            split,
            "--num-pages",
            str(split_pages),
            "--seed",
            str(args.seed),
            "--config",
            str(config),
            "--progress-every",
            str(args.progress_every),
        ]
        if content_root is not None:
            command.extend(("--content-root", str(content_root)))
        for tier in tiers:
            command.extend(("--tier", tier))
        if args.plan_only:
            command.append("--plan-only")
        else:
            if args.browser_channel:
                command.extend(("--browser-channel", args.browser_channel))
            if args.chromium_executable:
                command.extend(("--chromium-executable", str(args.chromium_executable)))
            for name in ("expected_browser_version", "expected_browser_sha256"):
                value = getattr(args, name)
                if value:
                    command.extend((f"--{name.replace('_', '-')}", value))
            for path, digest in zip(args.font_file, args.expected_font_sha256):
                command.extend(("--font-file", str(path), "--expected-font-sha256", digest))
        subprocess.run(command, check=True)

    audit_summary = output_root / "audit_summary.json"
    if not args.plan_only:
        audit_command = [sys.executable, str(script_dir / "audit_synthetic_layout.py")]
        for split in pages:
            audit_command.extend(
                ("--manifest", str(output_root / split / "manifest.jsonl"))
            )
        audit_command.extend(("--summary-json", str(audit_summary)))
        subprocess.run(audit_command, check=True, stdout=subprocess.DEVNULL)

    protocol = {
        "status": "plan_only" if args.plan_only else "ready",
        "protocol": "ancient_photo_diverse_v1",
        "dataset_root": str(output_root),
        "content_manifest": str(content_manifest),
        "content_manifest_sha256": sha256(content_manifest),
        "generator_config": str(config),
        "generator_config_sha256": sha256(config),
        "seed": args.seed,
        "tiers": tiers,
        "pages_per_tier": pages,
        "total_pages": len(tiers) * sum(pages.values()),
        "input_level": "whole_page_image",
        "layout_metadata_as_model_input": False,
        "formal_manifest_emitted": not args.plan_only,
        "audit_summary": str(audit_summary) if not args.plan_only else None,
    }
    protocol_path = output_root / "dataset_protocol.json"
    write_json(protocol_path, protocol)
    print(
        json.dumps(
            {
                "event": "diverse_synthetic_dataset_prepared",
                "protocol": str(protocol_path),
                "status": protocol["status"],
                "total_pages": protocol["total_pages"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "event": "diverse_synthetic_dataset_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:800],
                },
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        raise SystemExit(1)
