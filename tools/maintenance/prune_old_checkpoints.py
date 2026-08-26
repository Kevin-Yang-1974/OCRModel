#!/usr/bin/env python3
"""Safely report, and optionally remove, old checkpoints from named runs.

The command is intentionally narrow: a run id must be named explicitly and
the default mode is dry-run. Selection files, final models, source models and
running runs are protected before any candidate is considered.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-id", action="append", required=True)
    parser.add_argument("--keep-latest", type=int, default=2)
    parser.add_argument("--apply", action="store_true", help="Delete only listed candidates.")
    return parser.parse_args()


def load_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def path_values(value: object) -> Iterable[str]:
    if isinstance(value, dict):
        for child in value.values():
            yield from path_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from path_values(child)
    elif isinstance(value, str):
        yield value


def resolved_path(raw: str, run_root: Path) -> Path | None:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = run_root / candidate
    try:
        return candidate.resolve()
    except OSError:
        return None


def checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def directory_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                pass
    return total


def is_running(run_root: Path) -> bool:
    status = load_json(run_root / "metadata" / "status.txt")
    if isinstance(status, dict):
        return status.get("status") in {"running", "training", "evaluating"}
    text = (run_root / "metadata" / "status.txt").read_text(encoding="utf-8", errors="ignore") if (run_root / "metadata" / "status.txt").is_file() else ""
    normalized = text.replace(" ", "")
    return '"status":"running"' in normalized or '"stage_status":"running"' in normalized


def protected_paths(run_root: Path) -> set[Path]:
    protected: set[Path] = set()
    for path in run_root.rglob("selection.json"):
        payload = load_json(path)
        if payload is None:
            continue
        for raw in path_values(payload):
            candidate = resolved_path(raw, run_root)
            if candidate is not None:
                protected.add(candidate)
    for name in ("summary.json", "layout_training_metrics.json"):
        payload = load_json(run_root / name)
        if payload is not None:
            for raw in path_values(payload):
                candidate = resolved_path(raw, run_root)
                if candidate is not None:
                    protected.add(candidate)
    for path in (run_root / "metadata" / "status.txt", run_root / "PVLD_TRAINING_FINISHED"):
        if path.exists():
            protected.add(path.resolve())
    return protected


def under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def main() -> int:
    args = parse_args()
    if args.keep_latest < 0:
        raise ValueError("--keep-latest must be non-negative")
    runs_root = args.runs_root.expanduser().resolve()
    if not runs_root.is_dir():
        raise FileNotFoundError(f"runs root is not a directory: {runs_root}")
    report: dict[str, object] = {"runs_root": str(runs_root), "apply": args.apply, "runs": []}
    for run_id in args.run_id:
        if Path(run_id).name != run_id or run_id in {"", ".", ".."}:
            raise ValueError(f"run id must be a single directory name: {run_id!r}")
        run_root = (runs_root / run_id).resolve()
        if not under(run_root, runs_root) or not run_root.is_dir():
            raise FileNotFoundError(f"named run is missing: {run_root}")
        checkpoints = sorted(
            (path for path in run_root.rglob("checkpoint-*") if path.is_dir()),
            key=lambda path: (checkpoint_step(path), path.stat().st_mtime),
        )
        protected = protected_paths(run_root)
        keep = set(checkpoints[-args.keep_latest:] if args.keep_latest else [])
        candidates = []
        for checkpoint in checkpoints:
            resolved = checkpoint.resolve()
            is_protected = any(
                item == resolved or under(item, resolved) or under(resolved, item)
                for item in protected
            )
            if checkpoint in keep or is_protected:
                continue
            candidates.append(checkpoint)
        entry = {
            "run_id": run_id,
            "status_running": is_running(run_root),
            "checkpoint_count": len(checkpoints),
            "protected_count": len(protected),
            "kept": [str(path) for path in sorted(keep)],
            "candidates": [str(path) for path in candidates],
            "candidate_bytes": sum(directory_bytes(path) for path in candidates),
        }
        if entry["status_running"]:
            entry["candidates"] = []
            entry["blocked_reason"] = "run status is active"
        if args.apply and not entry["status_running"]:
            removed = []
            removed_bytes = 0
            for checkpoint in candidates:
                removed_bytes += directory_bytes(checkpoint)
                shutil.rmtree(checkpoint)
                removed.append(str(checkpoint))
            entry["removed"] = removed
            entry["removed_bytes"] = removed_bytes
        report["runs"].append(entry)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
