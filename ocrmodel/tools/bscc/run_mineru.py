#!/usr/bin/env python3
"""BSCC compatibility entry point with the official MinerU SFT CLI shape.

The public MinerU tutorial invokes ``python run_mineru.py sft --config ...``.
The tutorial's ``mineru_ext`` archive is not publicly tracked, while the
MinerU2.5-Pro-2605 model is a standard Qwen2-VL checkpoint.  This forwarder
keeps the official invocation contract and delegates to the official
MS-SWIFT CLI installed in the isolated BSCC environment.
"""

from __future__ import annotations

import os
import sys


def _normalize_official_config_arg() -> None:
    """Translate MinerU's ``--config PATH`` to MS-SWIFT's YAML argv form."""
    argv = sys.argv[1:]
    if not argv or argv[0] != "sft":
        return

    sft_args = argv[1:]
    config_path = None
    remaining = []
    index = 0
    while index < len(sft_args):
        item = sft_args[index]
        if item == "--config":
            if index + 1 >= len(sft_args):
                raise SystemExit("--config requires a YAML path")
            config_path = sft_args[index + 1]
            index += 2
            continue
        if item.startswith("--config="):
            config_path = item.split("=", 1)[1]
            index += 1
            continue
        remaining.append(item)
        index += 1

    if config_path is not None:
        sys.argv[:] = [sys.argv[0], "sft", config_path, *remaining]


def main() -> None:
    vendor_root = os.environ.get("MINERU_MS_SWIFT_ROOT")
    if vendor_root and vendor_root not in sys.path:
        sys.path.insert(0, vendor_root)

    if len(sys.argv) < 2 or sys.argv[1] != "sft":
        raise SystemExit("BSCC MinerU compatibility entry supports: run_mineru.py sft --config <path>")

    os.environ.setdefault("MINERU_COMPAT_MODE", "bscc_ms_swift_forwarder")
    _normalize_official_config_arg()
    from swift.cli.main import cli_main

    cli_main()


if __name__ == "__main__":
    main()
