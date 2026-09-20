"""Run the trained detector over pages and write boxes the routing bias can consume.

This is the join between the detector and the routing arms. Its output is what makes
``pred_static`` deployable: boxes that came from the image, in the same normalized coordinate
space the probe and ``layout_targets`` use, so "line 4" means one thing throughout.

## The coordinate space

Boxes are written normalized to the page, which is the space ``bridge.last_patch_positions``
lives in. Resizing preserves the aspect ratio and maps the whole page, so dividing a box by the
resized dimensions gives the page-normalized box without any need to know the original size --
and that is also why the detector can be trained at one resolution and used at another.

## Reading order

Boxes are sorted along the axis the direction implies, right to left for vertical columns. The
static arm biases their union and does not care about order, but an ordered list is what a tracked
arm would need, and getting it here rather than later means the ordering is derived once, from the
same boxes the bias uses.

## Threshold and the score it keeps

Detections below ``--score-threshold`` are dropped. The count kept per page is reported, because a
threshold too high silently removes columns and the routing would then be biasing a page with gaps
in it -- a failure that looks like "the bias does not help" rather than like a detection problem.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_line_detector import LinePages, collate, resize_scale  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-size", type=int, default=1000)
    parser.add_argument("--max-size", type=int, default=2000)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.2,
        help=(
            "detections below this are dropped. The audit's score sweep found recall and "
            "precision identical at 0.05, 0.10 and 0.20 -- nothing lands in between, the score "
            "distribution is confident-or-absent -- so anything at or below 0.2 keeps the same "
            "boxes, and 0.3 throws away 2.3 recall points that a union bias wants. 0.2 rather "
            "than 0.05 only keeps the file smaller"
        ),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def build_model(checkpoint_path: Path) -> torch.nn.Module:
    from torchvision.models.detection import fcos_resnet50_fpn

    model = fcos_resnet50_fpn(weights=None, weights_backbone=None)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = 1
    model.head.classification_head.cls_logits = torch.nn.Conv2d(
        model.head.classification_head.conv[0].out_channels,
        num_anchors,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    return model


def normalize_boxes(
    boxes: torch.Tensor, width: int, height: int
) -> list[list[float]]:
    """Boxes from resized pixel space to page-normalized coordinates.

    The resize maps the whole page and keeps the aspect ratio, so a box divided by the resized
    dimensions is already page-normalized whatever resolution the detector ran at.  That is what
    lets the detector be trained at one size and used at another, so it is worth stating rather
    than leaving as an inline division.
    """

    return [
        [
            float(box[0]) / width,
            float(box[1]) / height,
            float(box[2]) / width,
            float(box[3]) / height,
        ]
        for box in boxes
    ]


def order_boxes(
    boxes: list[list[float]], scores: list[float], direction: str
) -> tuple[list[list[float]], list[float]]:
    """Sort boxes into reading order along the axis the direction implies."""

    axis = 0 if direction == "vertical_rtl" else 1
    descending = direction == "vertical_rtl"
    order = sorted(
        range(len(boxes)),
        key=lambda index: (boxes[index][axis] + boxes[index][axis + 2]) / 2,
        reverse=descending,
    )
    return [boxes[index] for index in order], [scores[index] for index in order]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.checkpoint).to(device).eval()

    dataset = LinePages(args.index, args.min_size, args.max_size, args.limit)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate
    )

    rows: list[dict[str, Any]] = []
    kept_counts: list[int] = []
    score_values: list[float] = []
    with torch.no_grad():
        for images, batch_targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, batch_targets):
                # Normalized by the resized size: the resize maps the whole page and keeps the
                # aspect ratio, so this is the page-normalized box whatever resolution was used.
                height, width = images[0].shape[-2], images[0].shape[-1]
                keep = output["scores"] >= args.score_threshold
                boxes = output["boxes"][keep].cpu()
                scores = output["scores"][keep].cpu()
                normalized = normalize_boxes(boxes, width, height)
                ordered, ordered_scores = order_boxes(
                    normalized, [float(score) for score in scores], target["direction"]
                )
                rows.append(
                    {
                        "page_id": target["page_id"],
                        "direction": target["direction"],
                        "boxes": ordered,
                        "scores": ordered_scores,
                        "detected": len(ordered),
                        "truth_lines": int(target["boxes"].shape[0]),
                    }
                )
                kept_counts.append(len(ordered))
                score_values.extend(ordered_scores)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    truth = [row["truth_lines"] for row in rows]
    report = {
        "pages": len(rows),
        "score_threshold": args.score_threshold,
        "min_size": args.min_size,
        "detected_per_page": {
            "min": min(kept_counts, default=0),
            "median": st.median(kept_counts) if kept_counts else 0,
            "max": max(kept_counts, default=0),
        },
        "truth_lines_per_page": {
            "min": min(truth, default=0),
            "median": st.median(truth) if truth else 0,
            "max": max(truth, default=0),
        },
        "pages_with_no_detection": sum(1 for count in kept_counts if count == 0),
        "mean_score": st.mean(score_values) if score_values else None,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["pages_with_no_detection"]:
        # A page with no boxes is a page the static bias cannot touch at all, which would look
        # like the bias failing rather than the detector failing.
        print(
            f"WARNING: {report['pages_with_no_detection']} pages produced no boxes and cannot be "
            f"biased at all",
            file=sys.stderr,
        )
    (args.output.parent / (args.output.stem + "_report.json")).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
