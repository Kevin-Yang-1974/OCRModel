"""Fine-tune a one-stage line detector so the routing bias can use predicted boxes.

Stage 2 showed the attention-routing gain survives coarsening from a character box to a whole
line: biasing every visual token on the line the reader is on reproduces the recorded
deletions-halved signature and beats the no-route control by 20% CER, paired CI excluding zero.
That arm is still an oracle -- the line boxes are annotation -- and the layout branch's own
predictions are not good enough to replace them. So the boxes come from here instead.

## Why FCOS

The columns on this data are long and thin: the median textline in the MTHv2 index is a few
hundred pixels tall and around ninety wide, so aspect ratios run past 10:1. Anchor-based heads
would need their anchor set tuned to that, and getting it wrong shows up as a detector that
misses exactly the thin columns the routing needs. FCOS is anchor-free and assigns by point
inside the box, which is far less sensitive to that.

## The protocol

Train on the train split, select the checkpoint on validation, and never touch test -- the index
builder refuses to emit it. Selection is by validation line F1 at IoU 0.5, not by loss: a
detector can lower its loss by getting confident about the boxes it already finds while missing
columns entirely, and missing columns is the failure that matters here.

## What is deliberately not done

No horizontal or vertical flips. They would be free augmentation for detection, but they change
which column comes first, and reading order is the thing the routing bias is supposed to deliver.
Augmenting in a way that makes the labels ambiguous with respect to the target is worse than a
smaller training set.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# One class, so torchvision's label-1-is-the-first-class convention means lines are label 1 and
# 0 stays background.
LINE_LABEL = 1
MATCH_IOU = 0.5


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-index", required=True, type=Path)
    parser.add_argument("--validation-index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--min-size",
        type=int,
        default=1200,
        help=(
            "shorter side of the resized page. The scans are 1500x3000 and the columns are ~90px "
            "wide; at torchvision's default 800 that becomes 48px, which is where thin columns "
            "start being missed"
        ),
    )
    parser.add_argument("--max-size", type=int, default=2400)
    parser.add_argument("--score-threshold", type=float, default=0.05,
                        help="detections below this score are dropped before matching")
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0, help="debug: use only N pages")
    parser.add_argument("--limit-validation", type=int, default=0)
    return parser.parse_args(argv)


class LinePages(Dataset):
    """Full-page images with their line boxes, resized so the aspect ratio is kept."""

    def __init__(self, index: Path, min_size: int, max_size: int, limit: int = 0) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self.rows = [
            json.loads(line)
            for line in index.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        row = self.rows[index]
        image = Image.open(row["image_path"]).convert("RGB")
        width, height = image.size
        scale = resize_scale(width, height, self.min_size, self.max_size)
        # PIL rather than torchvision's transform: it keeps the dataset usable (and testable)
        # wherever torch is, and bilinear on a document scan is what the detection preprocessing
        # does anyway.
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        resized = image.resize((new_width, new_height), Image.BILINEAR)
        tensor = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float() / 255.0
        boxes = torch.tensor(row["boxes"], dtype=torch.float32) * scale
        target = {
            "boxes": boxes,
            "labels": torch.full((boxes.shape[0],), LINE_LABEL, dtype=torch.int64),
            "page_id": row["page_id"],
            # The page's dominant direction, for the ordering metric only.  A page is
            # overwhelmingly one direction (99.8% of the index is vertical_rtl), and per-box
            # direction is a separate problem the evaluator does not take on.
            "direction": _dominant_direction(row.get("writing_direction") or []),
        }
        return tensor, target


def collate(batch):
    return tuple(zip(*batch))


def _dominant_direction(directions: list[str]) -> str:
    if not directions:
        return "unknown"
    return max(set(directions), key=directions.count)


def resize_scale(width: int, height: int, min_size: int, max_size: int) -> float:
    """Scale so the shorter side lands in [min_size, max_size], keeping the aspect ratio."""

    scale = min_size / min(float(width), float(height))
    if max(width, height) * scale > max_size:
        scale = max_size / max(float(width), float(height))
    return scale


def iou_matrix(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU, computed here rather than imported so the metric is testable on CPU.

    ``torchvision.ops.box_iou`` is the same arithmetic, but using it would make the detection
    metric depend on the detection stack, and the metric is the part worth unit-testing.
    """

    if gt.numel() == 0 or pred.numel() == 0:
        return torch.zeros((gt.shape[0], pred.shape[0]))
    lt = torch.max(gt[:, None, :2], pred[None, :, :2])
    rb = torch.min(gt[:, None, 2:], pred[None, :, 2:])
    wh = (rb - lt).clamp_min(0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_gt = (gt[:, 2] - gt[:, 0]).clamp_min(0) * (gt[:, 3] - gt[:, 1]).clamp_min(0)
    area_pred = (pred[:, 2] - pred[:, 0]).clamp_min(0) * (pred[:, 3] - pred[:, 1]).clamp_min(0)
    union = area_gt[:, None] + area_pred[None, :] - inter
    return inter / union.clamp_min(1e-9)


def match_detections(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, Any]],
    iou_threshold: float = MATCH_IOU,
) -> dict[str, float]:
    """Greedy IoU matching, reported as what the audit needs.

    Recall is the number that matters: a missed column is a stretch of page the routing bias
    cannot aim at, whereas a spurious extra box mostly overlaps a column that is already biased.
    Merged lines are counted separately because a box covering two columns biases both of them,
    which is a different failure from missing one.
    """

    matched_gt = 0
    matched_pred = 0
    total_gt = 0
    total_pred = 0
    ious: list[float] = []
    merged = 0
    for prediction, target in zip(predictions, targets):
        gt = target["boxes"]
        pred = prediction["boxes"]
        total_gt += int(gt.shape[0])
        total_pred += int(pred.shape[0])
        if gt.numel() == 0 or pred.numel() == 0:
            continue
        iou = iou_matrix(gt, pred)
        taken_pred: set[int] = set()
        for gt_index in range(gt.shape[0]):
            best, best_pred = 0.0, -1
            for pred_index in range(pred.shape[0]):
                if pred_index in taken_pred:
                    continue
                if float(iou[gt_index, pred_index]) > best:
                    best, best_pred = float(iou[gt_index, pred_index]), pred_index
            if best >= iou_threshold:
                taken_pred.add(best_pred)
                matched_gt += 1
                matched_pred += 1
                ious.append(best)
                # Two ground-truth lines sharing one prediction: the box spans both.
                others = sum(
                    1
                    for other in range(gt.shape[0])
                    if other != gt_index and float(iou[other, best_pred]) >= iou_threshold
                )
                merged += int(others > 0)
    return {
        "gt": total_gt,
        "pred": total_pred,
        "matched": matched_gt,
        "recall": matched_gt / total_gt if total_gt else 0.0,
        "precision": matched_pred / total_pred if total_pred else 0.0,
        "mean_iou": sum(ious) / len(ious) if ious else 0.0,
        "merged_gt": merged,
        "empty_predictions": sum(1 for p in predictions if int(p["boxes"].shape[0]) == 0),
    }


def f1(stats: dict[str, float]) -> float:
    precision, recall = stats["precision"], stats["recall"]
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


@torch.no_grad()
def evaluate(model, loader, device, score_threshold: float) -> dict[str, float]:
    model.eval()
    predictions: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, Any]] = []
    for images, batch_targets in loader:
        outputs = model([image.to(device) for image in images])
        for output in outputs:
            keep = output["scores"] >= score_threshold
            predictions.append({key: value[keep].cpu() for key, value in output.items()})
        targets.extend(batch_targets)
    return match_detections(predictions, targets)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn

    weights = FCOS_ResNet50_FPN_Weights.COCO_V1 if args.pretrained else None
    model = fcos_resnet50_fpn(weights=weights)
    # One class: replace the classification head's output layer, keeping the pretrained
    # regression towers, which is where most of the transferable structure lives.
    in_channels = model.head.classification_head.conv[0].out_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = 1
    model.head.classification_head.cls_logits = torch.nn.Conv2d(
        in_channels, num_anchors * 1, kernel_size=3, stride=1, padding=1
    )
    torch.nn.init.normal_(model.head.classification_head.cls_logits.weight, std=0.01)
    prior = -math.log((1 - 0.01) / 0.01)
    torch.nn.init.constant_(model.head.classification_head.cls_logits.bias, prior)
    model.to(device)

    train_set = LinePages(args.train_index, args.min_size, args.max_size, args.limit_train)
    val_set = LinePages(args.validation_index, args.min_size, args.max_size, args.limit_validation)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        collate_fn=collate, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=max(1, args.workers // 2),
        collate_fn=collate,
    )
    print(json.dumps({
        "event": "line_detector_started",
        "train_pages": len(train_set), "validation_pages": len(val_set),
        "device": str(device), "min_size": args.min_size, "epochs": args.epochs,
        "pretrained": bool(weights),
    }), flush=True)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history: list[dict[str, Any]] = []
    best = {"f1": -1.0, "epoch": -1}

    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        losses: list[float] = []
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items() if key != "page_id"}
                for target in targets
            ]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        stats = evaluate(model, val_loader, device, args.score_threshold)
        record = {
            "epoch": epoch,
            "train_loss": sum(losses) / max(1, len(losses)),
            "seconds": time.time() - started,
            "validation": stats,
            "validation_f1": f1(stats),
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        torch.save(
            {"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
            args.output_dir / "last.pt",
        )
        if record["validation_f1"] > best["f1"]:
            best = {"f1": record["validation_f1"], "epoch": epoch}
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "args": vars(args),
                 "validation": stats},
                args.output_dir / "best.pt",
            )
        (args.output_dir / "history.json").write_text(
            json.dumps({"history": history, "best": best}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps({"event": "line_detector_finished", "best": best}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
