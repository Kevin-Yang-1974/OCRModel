#!/usr/bin/env python3
"""Overlay the predicted mask and the ground-truth target on the page image.

The diagnosis in ``diagnose_decoder_mask_bias.py`` reduces the mask to numbers
(top-1 hit, soft IoU, in/out contrast).  Numbers hide the failure mode that
matters here: a mask can score a decent top-1 hit while being a broad blob that
covers half the page.  This renders the thing itself -- for a handful of target
tokens, the page with

* the **predicted** mask as a heat overlay, and
* the **ground-truth** target next to it,

so the two can be compared by eye, together with the character box the token is
supposed to write and the characters the model actually produced.

Cell geometry is taken from ``xywh`` (the router's own merged-grid centres), so
the overlay does not assume any particular row-major ordering of the grid: every
cell is painted at its own normalised rectangle.

Ground truth is used only to *draw* the target; it is never fed to the model.
"""

from __future__ import annotations

import argparse
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import torch
from PIL import Image, ImageDraw, ImageFilter

from layout_ocr.data import load_records, prepare_training_inputs
from layout_ocr.decoder_mask_checkpoint import load_config, load_lora_state, restore_router
from layout_ocr.decoder_mask_model import enable_eager_backend, install_decoder_mask_router
from layout_ocr.decoder_mask_router import _normalized_grid_xywh
from layout_ocr.lora import inject_decoder_lora, load_lora_state_dict
from layout_ocr.mask_targets import build_mask_targets, char_boxes

HEAT = (255, 92, 0)  # tint colour for a mask of 1.0
BOX = (0, 170, 90)


def _eos_ids(model: Any, processor: Any) -> set[int]:
    ids: set[int] = set()
    tokenizer = getattr(processor, "tokenizer", None)
    value = getattr(tokenizer, "eos_token_id", None)
    if value is not None:
        if isinstance(value, (list, tuple, set)):
            ids = {int(item) for item in value if item is not None}
        else:
            ids = {int(value)}
    gen = getattr(model, "generation_config", None)
    gvalue = getattr(gen, "eos_token_id", None)
    if gvalue is not None:
        if isinstance(gvalue, (list, tuple, set)):
            ids.update(int(item) for item in gvalue if item is not None)
        else:
            ids.add(int(gvalue))
    return ids


def _heat_layer(values: torch.Tensor, xywh: torch.Tensor, size: tuple[int, int]) -> Image.Image:
    """Paint each grid cell's value at its own normalised rectangle."""

    width, height = size
    layer = Image.new("L", size, 0)
    draw = ImageDraw.Draw(layer)
    cells = xywh[0].tolist()
    flat = values.tolist()
    for (cx, cy, cw, ch), value in zip(cells, flat):
        level = int(max(0.0, min(1.0, float(value))) * 255)
        if level == 0:
            continue
        x0 = (cx - cw / 2.0) * width
        x1 = (cx + cw / 2.0) * width
        y0 = (cy - ch / 2.0) * height
        y1 = (cy + ch / 2.0) * height
        draw.rectangle([x0, y0, x1, y1], fill=level)
    # The grid is coarse; smoothing keeps the overlay readable without inventing
    # structure (the underlying cells are still the only source of signal).
    return layer.resize(size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(3))


def _compose(
    page: Image.Image,
    values: torch.Tensor,
    xywh: torch.Tensor,
    box: list[float] | None,
) -> Image.Image:
    """Page with the mask as a heat overlay, its peak marked, and the GT box drawn.

    The alpha is gamma-compressed: a mask that sits at ~0.2 over most of the page
    (which is what this head produces) must read as a *faint* wash, otherwise the
    overlay hides the very property worth seeing -- that the peak barely stands
    out from the background.
    """

    heat = _heat_layer(values, xywh, page.size)
    base = page.convert("RGB").copy()
    tint = Image.new("RGB", base.size, HEAT)
    alpha = heat.point(lambda v: int(255 * (v / 255.0) ** 2.2 * 0.85))
    blended = Image.composite(tint, base, alpha)
    draw = ImageDraw.Draw(blended)
    width, height = blended.size
    if box is not None:
        draw.rectangle(
            [box[0] * width, box[1] * height, box[2] * width, box[3] * height],
            outline=BOX,
            width=max(2, width // 400),
        )
    peak = int(values.argmax())
    cx, cy = float(xywh[0][peak][0]) * width, float(xywh[0][peak][1]) * height
    radius = max(6, width // 90)
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=(20, 20, 200), width=max(3, width // 300))
    return blended


def _crop_around(page: Image.Image, box: list[float] | None, pad: float = 1.6) -> Image.Image:
    """A window around the ground-truth box, so the peak can be judged up close."""

    width, height = page.size
    if box is None:
        return page
    cx = (box[0] + box[2]) / 2.0 * width
    cy = (box[1] + box[3]) / 2.0 * height
    half_w = max((box[2] - box[0]) * width * pad, width * 0.06)
    half_h = max((box[3] - box[1]) * height * pad * 2.0, height * 0.03)
    return page.crop(
        (
            int(max(0, cx - half_w)),
            int(max(0, cy - half_h)),
            int(min(width, cx + half_w)),
            int(min(height, cy + half_h)),
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="render predicted vs target mask on the page")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, help="predictions.jsonl to show the produced text")
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=4, help="how many target tokens to render")
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False, local_files_only=True)
    size = dict(processor.image_processor.size)
    size["longest_edge"] = args.max_pixels
    processor.image_processor.size = size

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    model.eval()

    eos_ids = _eos_ids(model, processor)
    image_token_id = int(model.config.image_token_id)
    spatial_merge_size = int(model.model.visual.spatial_merge_size)

    inject_decoder_lora(model, rank=8, alpha=8.0)
    lora_state = load_lora_state(checkpoint_dir)
    if lora_state is not None:
        load_lora_state_dict(model, lora_state)
    config = load_config(checkpoint_dir)
    enable_eager_backend(model)
    runtime = install_decoder_mask_router(model, config, image_token_id, spatial_merge_size)
    restore_router(model, runtime, checkpoint_dir)
    runtime.router.eval()
    runtime.set_noise(0.0, 0.0)
    runtime.set_bias_strength(float(config.bias_max))

    records = load_records(args.manifest)
    record = records[args.page_index]
    inputs = prepare_training_inputs(processor, record, device, eos_ids)
    labels = inputs["labels"]
    prompt_length = int((labels == -100).to(torch.int32).cumprod(dim=1).sum().item())
    target_ids = inputs["input_ids"][0, prompt_length:]
    xywh, _ = _normalized_grid_xywh(inputs["image_grid_thw"], spatial_merge_size)
    targets = build_mask_targets(processor.tokenizer, record, target_ids, eos_ids, xywh)
    # Keep the runtime on the inference path.  G1/G2 ignore ``targets`` in the
    # deterministic router, but G3's VAE router would otherwise build its
    # ground-truth posterior and leak the target into the rendered mask.  A
    # non-None sentinel makes the runtime scan the whole teacher-forced target
    # span, while ``mask=None`` makes every router use its inference prior.  The
    # actual target below is only for drawing and diagnostics, never prediction.
    runtime.set_page(
        inputs["image_grid_thw"],
        inputs["input_ids"],
        prompt_length,
        SimpleNamespace(mask=None),
    )
    with torch.no_grad():
        model(**inputs)
    predicted = runtime.last_mask[0].detach().float().cpu()  # [T, N]
    ground_truth = targets.mask[0].detach().float().cpu()
    xywh_cpu = xywh.detach().float().cpu()

    page = Image.open(record["image_path"]).convert("RGB")
    boxes, _ = char_boxes(record)

    # Pick tokens that actually carry a box, spread across the page.
    boxed = [
        index
        for index in range(target_ids.numel())
        if bool(targets.spatial_valid[0, index]) and float(ground_truth[index].sum()) > 0
    ]
    if not boxed:
        raise SystemExit("no target token carries a ground-truth box on this page")
    step = max(1, len(boxed) // args.tokens)
    chosen = boxed[::step][: args.tokens]

    panels: list[tuple[Image.Image, Image.Image, Image.Image]] = []
    entries: list[dict[str, Any]] = []
    for index in chosen:
        span = targets.char_spans[index]
        box = None
        chars = ""
        if span is not None:
            covered = [boxes[i] for i in range(span[0], span[1]) if i < len(boxes)]
            present = [b for b in covered if b is not None]
            if present:
                box = [
                    min(b[0] for b in present),
                    min(b[1] for b in present),
                    max(b[2] for b in present),
                    max(b[3] for b in present),
                ]
            chars = record["page_text"][span[0] : span[1]]

        full = _compose(page, predicted[index], xywh_cpu, box)
        zoom_predicted = _crop_around(_compose(page, predicted[index], xywh_cpu, box), box)
        zoom_target = _crop_around(_compose(page, ground_truth[index], xywh_cpu, box), box)
        panels.append((full, zoom_predicted, zoom_target))

        # How much of the page the mask paints, and whether its peak is on the box.
        peak = int(predicted[index].argmax())
        inside = ground_truth[index] > 0.0
        entries.append(
            {
                "token_index": int(index),
                "char_span": list(span) if span else None,
                "characters": chars,
                "token_text": processor.tokenizer.decode([int(target_ids[index])], skip_special_tokens=True),
                "gt_box": box,
                "predicted_mask_max": float(predicted[index].max()),
                "predicted_mask_mean": float(predicted[index].mean()),
                "predicted_share_above_0.5": float((predicted[index] > 0.5).float().mean()),
                "predicted_share_above_0.2": float((predicted[index] > 0.2).float().mean()),
                "target_share_above_0.5": float(inside.float().mean()),
                "gt_box_cells": int(inside.sum()),
                "total_cells": int(predicted.shape[-1]),
                "mask_peak_cell": peak,
                "peak_inside_box": bool(inside[peak]) if box is not None else None,
            }
        )

    # One row per token: whole page | zoom on the box (predicted) | zoom (target).
    row_height = 420
    gap = 12
    columns: list[Image.Image] = []
    for full, zoom_predicted, zoom_target in panels:
        for panel in (full, zoom_predicted, zoom_target):
            scale = row_height / panel.size[1]
            columns.append(panel.resize((max(1, int(panel.size[0] * scale)), row_height)))
    row_width = sum(column.size[0] for column in columns[:3]) + gap * 4
    canvas = Image.new(
        "RGB", (row_width, (row_height + gap) * len(panels) + gap), (250, 250, 248)
    )
    for row in range(len(panels)):
        x = gap
        for column in columns[row * 3 : row * 3 + 3]:
            canvas.paste(column, (x, gap + row * (row_height + gap)))
            x += column.size[0] + gap
    png_path = output_dir / f"mask_overlay_{record['page_id']}.png"
    canvas.save(png_path)

    reference = record["page_text"]
    prediction = None
    if args.predictions and args.predictions.is_file():
        for line in args.predictions.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if str(entry.get("page_id")) == str(record["page_id"]):
                prediction = entry.get("prediction")
                break

    payload = {
        "status": "complete",
        "page_id": record["page_id"],
        "image": str(record["image_path"]),
        "png": str(png_path),
        "columns": "left = predicted mask, right = ground-truth target",
        "bias_max": float(config.bias_max),
        "reference_text": reference,
        "predicted_text": prediction,
        "tokens": entries,
    }
    (output_dir / "mask_overlay.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"event": "mask_overlay_complete", "png": str(png_path), "tokens": len(entries)}))


if __name__ == "__main__":
    main()
