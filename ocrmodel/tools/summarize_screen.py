#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    rows = []
    for mode in ("content_only", "attention", "geometry", "layout_ot"):
        path = args.run_root / mode / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        validation = summary["validation"]
        rows.append(
            {
                "mode": mode,
                "cer": validation["cer"],
                "exact_page_rate": validation["exact_page_rate"],
                "r2_k1_recall": validation["r2_k1_recall"],
                "r2_k3_recall": validation["r2_k3_recall"],
                "r2_k5_recall": validation["r2_k5_recall"],
                "layout_box_mae": validation["layout_box_mae"],
                "layout_direction_accuracy": validation["layout_direction_accuracy"],
            }
        )
    best = min(rows, key=lambda row: (row["cer"], -row["exact_page_rate"]))
    output = {
        "status": "complete",
        "selection_metric": "validation_cer",
        "selected_mode": best["mode"],
        "results": rows,
        "test_used_for_selection": False,
    }
    target = args.run_root / "selection.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
