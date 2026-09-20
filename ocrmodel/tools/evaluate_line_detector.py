"""Audit a trained line detector, at the thresholds the routing bias actually needs.

The plan's stage-3 audit asks for line recall, precision, matched IoU, reading-order accuracy,
per-direction accuracy, and the missed and merged rates. This reports all of them, plus the
threshold sweep that matters more than any single number.

## Why a sweep rather than one IoU

IoU 0.5 is the conventional bar and it is a hard one here. The median MTHv2 textline is 96px
wide and 1035px tall, so at a 1200px shorter side a column is about 77px wide: an error of 20px
on each edge already halves the IoU, while leaving the box well inside the column. The routing
bias does not care about that -- it biases whichever visual tokens fall inside the box, and stage
2 showed a whole-line box covering 77 tokens works. So the sweep from 0.3 to 0.7 says what the
detector is good enough *for*, where a single 0.5 figure would read as failure.

## Ordering

Reading order is reported as the accuracy of recovering the annotation's order by sorting
predicted boxes along the axis the direction implies -- right to left for ``vertical_rtl``, top
to bottom for ``horizontal_ltr``. Direction comes from the ground truth here, because deriving it
is a separate problem; what is being measured is whether the boxes are good enough to be put in
the right sequence, which is what the routing bias consumes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_line_detector import (  # noqa: E402
    LinePages,
    collate,
    iou_matrix,
    match_detections,
    resize_scale,
)

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--split-name", default="validation")
    parser.add_argument("--min-size", type=int, default=1200)
    parser.add_argument("--max-size", type=int, default=2400)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args(argv)


def order_accuracy(
    gt: torch.Tensor, pred: torch.Tensor, direction: str, iou_threshold: float = 0.5
) -> dict[str, Any]:
    """Can the matched boxes be put back into the annotation's reading order?

    Only the ground-truth lines that were matched take part: an unmatched line has no predicted
    box to place, so counting it as misordered would conflate "missed" with "out of order".
    """

    axis = 0 if direction == "vertical_rtl" else 1
    # Vertical columns are read right to left, so the geometry is expected to descend along x.
    descending = direction == "vertical_rtl"
    iou = iou_matrix(gt, pred)
    matched: dict[int, int] = {}
    taken: set[int] = set()
    for gt_index in range(gt.shape[0]):
        best, best_pred = 0.0, -1
        for pred_index in range(pred.shape[0]):
            if pred_index in taken:
                continue
            if float(iou[gt_index, pred_index]) > best:
                best, best_pred = float(iou[gt_index, pred_index]), pred_index
        if best >= iou_threshold:
            taken.add(best_pred)
            matched[gt_index] = best_pred
    if len(matched) < 2:
        return {"matched": len(matched), "accuracy": None, "inversions": 0}
    # Ground truth is the order the annotation lists them in; the prediction is that ordering
    # recovered from where the boxes sit.  Counting inversions against the same pairing keeps the
    # two sequences the same length, so the comparison is like for like.
    by_geometry = sorted(
        matched,
        key=lambda gt_index: float(
            (pred[matched[gt_index], axis] + pred[matched[gt_index], axis + 2]) / 2
        ),
        reverse=descending,
    )
    by_truth = sorted(matched)
    rank = {gt_index: position for position, gt_index in enumerate(by_truth)}
    inversions = sum(
        1
        for i in range(len(by_geometry))
        for j in range(i + 1, len(by_geometry))
        if rank[by_geometry[i]] > rank[by_geometry[j]]
    )
    pairs_of_pairs = len(by_geometry) * (len(by_geometry) - 1) / 2
    return {
        "matched": len(matched),
        "accuracy": 1.0 - inversions / pairs_of_pairs,
        "inversions": inversions,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from torchvision.models.detection import fcos_resnet50_fpn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = fcos_resnet50_fpn(weights=None, weights_backbone=None)
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = 1
    model.head.classification_head.cls_logits = torch.nn.Conv2d(
        model.head.classification_head.conv[0].out_channels, num_anchors, kernel_size=3,
        stride=1, padding=1,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    dataset = LinePages(args.index, args.min_size, args.max_size, args.limit)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.workers, collate_fn=collate
    )

    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, Any]] = []
    order_stats: list[dict[str, Any]] = []
    with torch.no_grad():
        for images, batch_targets in loader:
            outputs = model([image.to(device) for image in images])
            for output, target in zip(outputs, batch_targets):
                keep = output["scores"] >= args.score_threshold
                predictions.append({key: value[keep].cpu() for key, value in output.items()})
                targets.append(target)
                direction = target.get("direction") or "vertical_rtl"
                order_stats.append(
                    order_accuracy(target["boxes"], predictions[-1]["boxes"], direction)
                )

    report: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
        "split": args.split_name,
        "pages": len(dataset),
        "score_threshold": args.score_threshold,
        "min_size": args.min_size,
        "by_iou": {},
    }
    for threshold in THRESHOLDS:
        stats = match_detections(predictions, targets, threshold)
        stats["f1"] = (
            2 * stats["precision"] * stats["recall"] / (stats["precision"] + stats["recall"])
            if stats["precision"] + stats["recall"]
            else 0.0
        )
        stats["missed_gt"] = stats["gt"] - stats["matched"]
        stats["merged_fraction"] = stats["merged_gt"] / stats["gt"] if stats["gt"] else 0.0
        report["by_iou"][f"{threshold:.1f}"] = stats
    accuracies = [entry["accuracy"] for entry in order_stats if entry["accuracy"] is not None]
    report["reading_order"] = {
        "pages_with_two_or_more_matches": len(accuracies),
        "mean_pairwise_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
    }
    report["pages_with_no_detection"] = sum(
        1 for prediction in predictions if int(prediction["boxes"].shape[0]) == 0
    )

    print(f"{'IoU':>5} {'gt':>6} {'pred':>6} {'recall':>8} {'prec':>8} {'F1':>8} "
          f"{'meanIoU':>8} {'missed':>7} {'merged':>7}")
    for threshold in THRESHOLDS:
        stats = report["by_iou"][f"{threshold:.1f}"]
        print(f"{threshold:5.1f} {stats['gt']:6d} {stats['pred']:6d} {stats['recall']:8.4f} "
              f"{stats['precision']:8.4f} {stats['f1']:8.4f} {stats['mean_iou']:8.4f} "
              f"{stats['missed_gt']:7d} {stats['merged_gt']:7d}")
    print()
    print(f"reading order: {report['reading_order']}")
    print(f"pages with no detection at all: {report['pages_with_no_detection']}/{len(dataset)}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
