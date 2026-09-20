"""Split the line detector's missed lines into causes, not just sizes.

`docs/LAYOUT_LINE_DETECTOR_AND_PREDMAP_RESULT.md` §5.2 measured that the missed MTHv2 lines are
the same size as the matched ones (median height 796px vs 782px), which rules out resolution and
small-object detection and leaves the question it names and does not answer: is the detector not
seeing the column, or seeing it and putting the box in the wrong place? The two have different
fixes -- one is capacity or scale, the other is regression or the box prior -- so a single recall
figure cannot choose between them.

This reads the exported predictions rather than the model, so it costs nothing to rerun and does
not depend on the GPU being free. The matching is the audit's own greedy IoU-0.5 rule, so the
"missed" set is exactly the one the audit reports.

A missed ground-truth line is placed in one of four buckets, by priority:

``absorbed``    its best overlap is with a box that was already matched to a *different* line, so
                two columns share one detection. A merge that the IoU-0.5 merge count misses
                because the second column falls below the bar.
``box_off``     a free box overlaps it above 0.1 but below 0.5. The column was found; the box is
                displaced or badly sized. This is the regression/prior failure.
``coarse``      a free box covers at least half its area but scores a low IoU because that box is
                much larger. Same column, box spanning more than it should.
``unseen``      nothing overlaps it. The failure is upstream of placement.

False positives are attributed the same way, because the tracked arm selects *one* box where the
static arm biases the union: a spurious box is a candidate the tracker can pick wrongly, so where
they sit decides whether precision matters here or not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

MATCH_IOU = 0.5
OVERLAP_FLOOR = 0.1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path, help="ground-truth line index")
    parser.add_argument("--predicted", required=True, type=Path, help="exported prediction jsonl")
    parser.add_argument("--iou", type=float, default=MATCH_IOU)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def area(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = (float(value) for value in box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in a)
    bx1, by1, bx2, by2 = (float(value) for value in b)
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def iou(a: Iterable[float], b: Iterable[float]) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def centre_inside(point_box: Iterable[float], container: Iterable[float]) -> bool:
    x1, y1, x2, y2 = (float(value) for value in point_box)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bx1, by1, bx2, by2 = (float(value) for value in container)
    return bx1 <= cx <= bx2 and by1 <= cy <= by2


def normalised_gt(record: dict[str, Any]) -> list[list[float]]:
    width, height = float(record["width"]), float(record["height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"{record['page_id']}: page size must be positive")
    return [
        [float(box[0]) / width, float(box[1]) / height, float(box[2]) / width, float(box[3]) / height]
        for box in record["boxes"]
    ]


def match_page(
    gt: list[list[float]], pred: list[list[float]], iou_threshold: float
) -> tuple[list[int | None], set[int]]:
    """Greedy best-IoU matching, ground truth in order -- the audit's own rule."""

    assignment: list[int | None] = [None] * len(gt)
    taken: set[int] = set()
    for gt_index, gt_box in enumerate(gt):
        best, best_pred = 0.0, -1
        for pred_index, pred_box in enumerate(pred):
            if pred_index in taken:
                continue
            value = iou(gt_box, pred_box)
            if value > best:
                best, best_pred = value, pred_index
        if best >= iou_threshold:
            taken.add(best_pred)
            assignment[gt_index] = best_pred
    return assignment, taken


def classify_missed(
    gt_box: list[float],
    pred: list[list[float]],
    taken: set[int],
    iou_threshold: float,
    page_size: tuple[float, float],
) -> dict[str, Any]:
    """Which bucket a missed line falls in, with the numbers that put it there.

    Boxes stay normalized for the geometry; the sizes come back in pixels because they are
    compared against §5.2's table, which is in pixels.
    """

    best_any, best_any_index = 0.0, -1
    best_free, best_free_index = 0.0, -1
    coverage_free = 0.0
    centred = False
    gt_area = area(gt_box)
    for pred_index, pred_box in enumerate(pred):
        value = iou(gt_box, pred_box)
        if value > best_any:
            best_any, best_any_index = value, pred_index
        if pred_index in taken:
            continue
        if value > best_free:
            best_free, best_free_index = value, pred_index
        if gt_area > 0:
            coverage_free = max(coverage_free, intersection(gt_box, pred_box) / gt_area)
        centred = centred or centre_inside(gt_box, pred_box)

    if best_any >= OVERLAP_FLOOR and best_any_index in taken:
        # A box claimed by another line overlaps this one.  Counted apart from box_off because
        # the fix is a merge guard, not a better regressed box.
        bucket = "absorbed"
    elif best_free >= OVERLAP_FLOOR:
        bucket = "box_off"
    elif coverage_free >= 0.5:
        bucket = "coarse"
    else:
        bucket = "unseen"
    return {
        "bucket": bucket,
        "best_any_iou": best_any,
        "best_free_iou": best_free,
        "coverage_free": coverage_free,
        "centre_in_some_box": centred,
        "width": (float(gt_box[2]) - float(gt_box[0])) * page_size[0],
        "height": (float(gt_box[3]) - float(gt_box[1])) * page_size[1],
    }


def classify_extra(
    pred_box: list[float],
    gt: list[list[float]],
    assignment: list[int | None],
    page_size: tuple[float, float],
) -> dict[str, Any]:
    """Where a false positive sits: on top of a matched line, or on empty space."""

    best, best_index = 0.0, -1
    for gt_index, gt_box in enumerate(gt):
        value = iou(pred_box, gt_box)
        if value > best:
            best, best_index = value, gt_index
    if best_index >= 0 and best >= OVERLAP_FLOOR:
        bucket = "duplicate_of_matched" if assignment[best_index] is not None else "over_unmatched"
    else:
        bucket = "on_background"
    return {
        "bucket": bucket,
        "best_iou": best,
        "width": (float(pred_box[2]) - float(pred_box[0])) * page_size[0],
        "height": (float(pred_box[3]) - float(pred_box[1])) * page_size[1],
    }


def quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return {"p10": pick(0.10), "p50": pick(0.50), "p90": pick(0.90)}


def summarise(rows: list[dict[str, Any]], buckets: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {"total": len(rows)}
    for bucket in buckets:
        chunk = [row for row in rows if row["bucket"] == bucket]
        out[bucket] = {
            "count": len(chunk),
            "share": len(chunk) / len(rows) if rows else None,
            "height": quantiles([row["height"] for row in chunk]),
            "width": quantiles([row["width"] for row in chunk]),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    index = {record["page_id"]: record for record in load_jsonl(args.index)}
    predicted = {record["page_id"]: record for record in load_jsonl(args.predicted)}
    shared = [page_id for page_id in index if page_id in predicted]
    if not shared:
        raise SystemExit("index and predictions share no page_id")

    missed_rows: list[dict[str, Any]] = []
    extra_rows: list[dict[str, Any]] = []
    totals = {"gt": 0, "pred": 0, "matched": 0}
    for page_id in shared:
        record = index[page_id]
        page_size = (float(record["width"]), float(record["height"]))
        gt = normalised_gt(record)
        pred = [list(box) for box in predicted[page_id]["boxes"]]
        assignment, taken = match_page(gt, pred, args.iou)
        totals["gt"] += len(gt)
        totals["pred"] += len(pred)
        totals["matched"] += sum(1 for entry in assignment if entry is not None)
        for gt_index, gt_box in enumerate(gt):
            if assignment[gt_index] is None:
                row = classify_missed(gt_box, pred, taken, args.iou, page_size)
                row["page_id"] = page_id
                missed_rows.append(row)
        for pred_index, pred_box in enumerate(pred):
            if pred_index not in taken:
                row = classify_extra(pred_box, gt, assignment, page_size)
                row["page_id"] = page_id
                extra_rows.append(row)

    report = {
        "index": str(args.index),
        "predicted": str(args.predicted),
        "pages": len(shared),
        "iou_threshold": args.iou,
        "totals": totals,
        "recall": totals["matched"] / totals["gt"] if totals["gt"] else 0.0,
        "precision": totals["matched"] / totals["pred"] if totals["pred"] else 0.0,
        "missed": summarise(missed_rows, ("absorbed", "box_off", "coarse", "unseen")),
        "extra": summarise(extra_rows, ("duplicate_of_matched", "over_unmatched", "on_background")),
        "missed_pages": sorted({row["page_id"] for row in missed_rows}),
    }

    print(f"pages {report['pages']}  gt {totals['gt']}  pred {totals['pred']}  "
          f"matched {totals['matched']}  recall {report['recall']:.4f}  "
          f"precision {report['precision']:.4f}")
    print()
    print("漏行归因（按优先级，先命中先归类）:")
    print(f"{'bucket':>10} {'count':>6} {'share':>8} {'h_p50':>8} {'w_p50':>8}")
    for bucket in ("absorbed", "box_off", "coarse", "unseen"):
        entry = report["missed"][bucket]
        height = (entry["height"] or {}).get("p50")
        width = (entry["width"] or {}).get("p50")
        print(f"{bucket:>10} {entry['count']:6d} "
              f"{(entry['share'] or 0.0):8.4f} "
              f"{(height if height is not None else -1):8.1f} "
              f"{(width if width is not None else -1):8.1f}")
    print()
    print("虚警框归因:")
    for bucket in ("duplicate_of_matched", "over_unmatched", "on_background"):
        entry = report["extra"][bucket]
        print(f"{bucket:>20} {entry['count']:6d}  share {(entry['share'] or 0.0):.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
