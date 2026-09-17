#!/usr/bin/env python3
"""Build the personal AncientDoc split5 evaluation manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.sota.ancientdoc import build_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.label_json, args.image_root, args.output)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

