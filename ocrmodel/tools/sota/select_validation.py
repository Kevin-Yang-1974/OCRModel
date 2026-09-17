"""Validation-only selection contract for future SOTA runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    records = []
    with args.predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("status") not in {"ok", "failed", "blocked"}:
                    raise ValueError("Invalid prediction status in validation output")
                records.append(record)
    if not records:
        raise ValueError("Validation selection requires at least one prediction record")
    payload = {"purpose": "sota_mthv2_validation_selection", "selection_split": "validation", "test_used_for_selection": False, "model": args.model, "checkpoint": args.checkpoint, "prompt_locked": True, "postprocess_locked": True, "threshold_locked": True, "prediction_count": len(records), "selection_rule": ["page_cer", "whitespace_stripped_page_cer", "earlier_step"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"event": "sota_validation_selection_created", "selection": str(args.output), "test_used_for_selection": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
