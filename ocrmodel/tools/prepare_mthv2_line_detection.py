"""Index MTHv2 pages for line detection: page image in, line boxes out.

The layout branch's own prediction quality is why this exists. The recorded routing gain came
from biasing a *true* line box; to turn that into something deployable the boxes have to come
from an image. So a detector is trained on the split that is allowed to train on, and this
builds its index.

## What is and is not copied

Only an index is written. The page images already live on disk and are referenced by path, so
nothing is duplicated -- the volume is at 98% and a copy of 1324 full-page scans would not fit.

## Which splits

``train`` and ``validation`` only. The test split is refused: rule 10 of AGENTS.md locks it,
and a detector trained or tuned on it would make every downstream number on it meaningless.
The refusal is in the code rather than in a note, because the cost of forgetting is the whole
locked-test protocol.

## The boxes

``regions`` are MTHv2's own textlines, which is the level the routing bias needs. Each carries
``bbox_px`` (absolute xyxy), ``reading_order`` and ``writing_direction``. Degenerate boxes are
dropped and counted: a 1-pixel sliver is an annotation artifact, and letting it into training
teaches the detector to emit slivers.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path
from typing import Any

ALLOWED_SPLITS = ("train", "validation")
# Below this, a box is an annotation artifact rather than a line.
MIN_SIDE_PX = 4.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", required=True, type=Path,
                        help="the converted MTHv2 dir holding <split>/manifest.char.jsonl")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--min-side-px", type=float, default=MIN_SIDE_PX)
    return parser.parse_args(argv)


def page_rows(
    record: dict[str, Any], min_side: float
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    """One page's detection row, plus what had to be dropped to get it.

    Kept as a function of the record alone so it can be tested without touching an image or a
    filesystem: the filtering rules are where a quiet mistake would show up as a detector that
    simply never learns thin columns.
    """

    stats = {"kept": 0, "degenerate": 0, "invalid": 0, "no_box_field": 0}
    regions = record.get("regions") or []
    size = record.get("page_size")
    if not regions or not size:
        return None, stats
    boxes: list[list[float]] = []
    orders: list[int] = []
    directions: list[str] = []
    for region in regions:
        if not region.get("valid", True):
            stats["invalid"] += 1
            continue
        box = region.get("bbox_px")
        if not box:
            stats["no_box_field"] += 1
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        if x2 - x1 < min_side or y2 - y1 < min_side:
            stats["degenerate"] += 1
            continue
        boxes.append([x1, y1, x2, y2])
        orders.append(int(region.get("reading_order", -1)))
        directions.append(region.get("writing_direction", "unknown"))
        stats["kept"] += 1
    if not boxes:
        return None, stats
    return {
        "page_id": record["page_id"],
        "image_path": record["image_path"],
        "width": float(size[0]),
        "height": float(size[1]),
        "boxes": boxes,
        "reading_order": orders,
        "writing_direction": directions,
    }, stats


def _quantiles(values: list[float], points=(0.05, 0.5, 0.95)) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        f"p{int(point * 100)}": ordered[min(len(ordered) - 1, int(point * len(ordered)))]
        for point in points
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    unknown = [split for split in args.splits if split not in ALLOWED_SPLITS]
    if unknown:
        raise SystemExit(
            f"refusing splits {unknown}: only {ALLOWED_SPLITS} may be indexed. The test split is "
            "locked, and a detector that has seen it cannot be evaluated on it."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"splits": {}, "min_side_px": args.min_side_px}

    for split in args.splits:
        manifest = args.manifest_dir / split / "manifest.char.jsonl"
        if not manifest.is_file():
            raise SystemExit(f"manifest missing: {manifest}")
        rows: list[dict[str, Any]] = []
        totals = Counter()
        boxes_per_page: list[int] = []
        widths: list[float] = []
        heights: list[float] = []
        directions: Counter = Counter()
        missing_images: list[str] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            row, stats = page_rows(record, args.min_side_px)
            totals.update(stats)
            if row is None:
                continue
            image = Path(row["image_path"])
            if not image.is_file():
                # The index is only useful if the images are actually there; a missing one
                # would surface much later as a training crash.
                missing_images.append(row["page_id"])
                continue
            rows.append(row)
            boxes_per_page.append(len(row["boxes"]))
            widths.extend(box[2] - box[0] for box in row["boxes"])
            heights.extend(box[3] - box[1] for box in row["boxes"])
            directions.update(row["writing_direction"])

        out = args.output_dir / f"lines_{split}.jsonl"
        out.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        report["splits"][split] = {
            "pages": len(rows),
            "boxes": sum(boxes_per_page),
            "boxes_per_page": {
                "min": min(boxes_per_page, default=0),
                "median": st.median(boxes_per_page) if boxes_per_page else 0,
                "max": max(boxes_per_page, default=0),
            },
            "box_width_px": _quantiles(widths),
            "box_height_px": _quantiles(heights),
            "writing_direction": dict(directions),
            "missing_images": missing_images[:10],
            "missing_image_count": len(missing_images),
            "dropped": dict(totals),
        }
        summary = report["splits"][split]
        print(f"{split:11s} pages {summary['pages']:5d} boxes {summary['boxes']:6d} "
              f"per-page median {summary['boxes_per_page']['median']:.0f} "
              f"| width p50 {summary['box_width_px'].get('p50', 0):.0f} "
              f"height p50 {summary['box_height_px'].get('p50', 0):.0f} "
              f"| dirs {dict(directions)}")
        if summary["missing_image_count"]:
            print(f"  WARNING: {summary['missing_image_count']} pages have no image on disk")

    (args.output_dir / "line_detection_index_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
