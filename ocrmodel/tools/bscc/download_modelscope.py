#!/usr/bin/env python3
"""Download a ModelScope model repository without the ``modelscope`` package.

Compute nodes are offline and the BSCC login node reaches ModelScope much
faster than the Hugging Face mirror, so the external-SOTA weights are staged
with plain HTTP.  The script resumes partial files, verifies sizes, records
sha256 digests and never overwrites a complete file.

Usage:
    python3 download_modelscope.py --repo PaddlePaddle/PaddleOCR-VL-1.6 \
        --target /path/to/model_dir [--revision master] [--exclude .gitattributes]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path


API = "https://www.modelscope.cn/api/v1/models/{repo}/repo/files?Revision={revision}&Recursive=True"
RESOLVE = "https://www.modelscope.cn/models/{repo}/resolve/{revision}/{path}"


def list_files(repo: str, revision: str) -> list[dict]:
    url = API.format(repo=repo, revision=revision)
    with urllib.request.urlopen(url, timeout=60) as response:
        payload = json.load(response)
    if payload.get("Code") != 200:
        raise SystemExit(f"modelscope api error for {repo}: {payload.get('Code')} {payload.get('Message')}")
    files = (payload.get("Data") or {}).get("Files") or []
    return [entry for entry in files if entry.get("Type") == "blob" or int(entry.get("Size") or 0) > 0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(repo: str, revision: str, path: str, size: int, target: Path) -> dict:
    destination = target / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == size:
        return {"path": path, "size": size, "status": "present", "sha256": sha256(destination)}
    url = RESOLVE.format(repo=repo, revision=revision, path=path)
    existing = destination.stat().st_size if destination.is_file() else 0
    request = urllib.request.Request(url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    mode = "ab" if existing else "wb"
    with urllib.request.urlopen(request, timeout=120) as response, destination.open(mode) as handle:
        while True:
            chunk = response.read(1 << 22)
            if not chunk:
                break
            handle.write(chunk)
    if destination.stat().st_size != size:
        raise SystemExit(f"size mismatch for {path}: {destination.stat().st_size} != {size}")
    return {"path": path, "size": size, "status": "downloaded", "sha256": sha256(destination)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    args.target.mkdir(parents=True, exist_ok=True)
    entries = list_files(args.repo, args.revision)
    results = []
    for entry in entries:
        path = str(entry["Path"])
        if any(token in path for token in args.exclude):
            continue
        size = int(entry["Size"])
        result = download(args.repo, args.revision, path, size, args.target)
        result["file"] = path
        results.append(result)
        print(json.dumps({"event": "modelscope_file", "repo": args.repo, **result}, ensure_ascii=False), flush=True)

    manifest = {"repo": args.repo, "revision": args.revision, "target": str(args.target), "files": results}
    (args.target / "_modelscope_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "modelscope_download_complete", "repo": args.repo, "files": len(results)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
