"""Drop the predicted line boxes that duplicate a neighbour, before the tracker picks one.

§10.2 found that lowering the export threshold is free for the static arm -- a spurious box only
re-biases tokens that were mostly going to be biased anyway -- and *not* free for the tracked arm,
which selects one box, so every extra box is a candidate it can select wrongly. The missed-line
attribution says where those candidates come from: of the false positives at threshold 0.2, 28.5%
sit on an already-matched line at IoU >= 0.5 and another 47% overlap one at IoU 0.1-0.5.

The detector's own post-processing already runs NMS, at torchvision's default 0.6 for this
architecture, so boxes duplicating a neighbour at 0.5-0.6 survive it. Tightening the score
threshold instead is the wrong lever: §11 swept it and the curve is flat except where recall
collapses, because the score distribution is confident-or-absent.

This is a pure function of an exported file, so it costs nothing to try and can be checked before
it is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predicted", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="boxes overlapping a kept box above this are dropped; the audit's matching bar",
    )
    return parser.parse_args(argv)


def area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def iou(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0]))) * max(
        0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1]))
    )
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def keep_boxes(boxes: list[list[float]], scores: list[float], threshold: float) -> list[int]:
    """Greedy highest-score-first, the same rule the detector's own NMS uses.

    Reading order is preserved in the output file, but the suppression has to run in score order:
    suppressing by reading order would keep whichever box happened to come first rather than the
    one the detector was most sure of.
    """

    order = sorted(range(len(boxes)), key=lambda index: -scores[index])
    kept: list[int] = []
    for index in order:
        if all(iou(boxes[index], boxes[other]) < threshold for other in kept):
            kept.append(index)
    return sorted(kept)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows: list[dict[str, Any]] = []
    removed = 0
    before = 0
    with args.predicted.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            boxes = [list(box) for box in record["boxes"]]
            scores = [float(score) for score in record.get("scores") or [1.0] * len(boxes)]
            before += len(boxes)
            kept = keep_boxes(boxes, scores, args.iou)
            removed += len(boxes) - len(kept)
            record["boxes"] = [boxes[index] for index in kept]
            if record.get("scores") is not None:
                record["scores"] = [scores[index] for index in kept]
            record["detected"] = len(kept)
            record["deduped_from"] = len(boxes)
            rows.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    kept_per_page = [len(record["boxes"]) for record in rows]
    print(f"pages {len(rows)}  boxes {before} -> {before - removed}  "
          f"removed {removed} ({removed / before:.1%})  "
          f"median per page {sorted(kept_per_page)[len(kept_per_page) // 2]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
