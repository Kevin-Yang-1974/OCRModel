#!/usr/bin/env python3
"""Render paired validation patch-attention maps over original OCR pages."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--font",
        type=Path,
        default=Path(r"C:\Windows\Fonts\msyh.ttc"),
        help="Font with Chinese glyph coverage",
    )
    return parser.parse_args()


def overlay_page(
    image: Image.Image,
    patch_scores: np.ndarray,
    grid_shape: list[int],
    bbox: list[float],
    shared_vmax: float,
) -> tuple[Image.Image, float]:
    image = image.convert("RGB")
    width, height = image.size
    grid_h, grid_w = (int(value) for value in grid_shape)
    scores = np.asarray(patch_scores, dtype=np.float32).reshape(grid_h, grid_w)
    normalized = np.clip(scores / max(shared_vmax, 1e-12), 0.0, 1.0)

    rgba = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)
    rgba[..., :3] = (colormaps["inferno"](normalized)[..., :3] * 255).astype(np.uint8)
    rgba[..., 3] = (np.sqrt(normalized) * 150).astype(np.uint8)
    heat = Image.fromarray(rgba, "RGBA").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    composed = Image.alpha_composite(image.convert("RGBA"), heat).convert("RGB")

    x0, y0, x1, y1 = bbox
    box = (
        round(x0 * width),
        round(y0 * height),
        round(x1 * width),
        round(y1 * height),
    )
    draw = ImageDraw.Draw(composed)
    draw.rectangle(box, outline=(0, 0, 0), width=9)
    draw.rectangle(
        (box[0] + 2, box[1] + 2, box[2] - 2, box[3] - 2),
        outline=(95, 255, 88),
        width=5,
    )
    return composed, float(scores.max())


def attention_legend_image(
    width: int,
    height: int,
    shared_vmax: float,
    background: tuple[int, int, int] = (247, 248, 250),
) -> Image.Image:
    """Build the paired attention scale using the overlay's cmap and alpha rule."""
    values = np.linspace(0.0, shared_vmax, width, dtype=np.float32)
    normalized = np.clip(values / max(shared_vmax, 1e-12), 0.0, 1.0)
    colors = (colormaps["inferno"](normalized)[..., :3] * 255.0).astype(np.uint8)
    alpha = (np.sqrt(normalized) * 150.0).astype(np.uint8)
    base = np.asarray(background, dtype=np.float32)
    composited = colors.astype(np.float32) * (alpha[:, None] / 255.0) + base * (
        1.0 - alpha[:, None] / 255.0
    )
    pixels = np.clip(np.rint(composited), 0, 255).astype(np.uint8)
    return Image.fromarray(np.broadcast_to(pixels[None, :, :], (height, width, 3)).copy())


def main() -> None:
    args = parse_args()
    root = args.artifact_root.resolve()
    summary = json.loads(
        (root / "attention" / "capture_summary.json").read_text(encoding="utf-8")
    )
    selection = json.loads((root / "selected_cases.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("source_split") != "validation":
        raise ValueError("input must be a completed validation-only attention capture")
    records = summary.get("records") or []
    if len(records) != 10 or any(
        record.get("split") != "validation"
        or not record.get("generation_prefix_verified")
        for record in records
    ):
        raise ValueError("expected ten prefix-verified validation cases")
    cases = {str(case["case_id"]): case for case in selection.get("cases", [])}
    if len(cases) != 10:
        raise ValueError("selected case metadata must contain exactly ten entries")

    font_path = args.font
    if not font_path.exists() and Path(r"C:\Windows\Fonts\arial.ttf").exists():
        font_path = Path(r"C:\Windows\Fonts\arial.ttf")
    font_path_bold = font_path.with_name("msyhbd.ttc")
    if not font_path.exists():
        raise FileNotFoundError(f"font not found: {font_path}")
    font = ImageFont.truetype(str(font_path), 27)
    font_small = ImageFont.truetype(str(font_path), 21)
    font_tiny = ImageFont.truetype(str(font_path), 18)
    font_bold = ImageFont.truetype(
        str(font_path_bold if font_path_bold.exists() else font_path), 32
    )

    individual_dir = root / "individual"
    individual_dir.mkdir(exist_ok=True)
    index: list[dict[str, Any]] = []
    rendered: list[tuple[int, Image.Image, str]] = []

    for record in sorted(records, key=lambda value: int(value["case_id"])):
        case_id = str(record["case_id"])
        case = cases[case_id]
        page_id = str(record["page_id"])
        npz_path = root / "attention" / f"case-{case_id}-{page_id}.npz"
        with np.load(npz_path) as maps:
            baseline_scores = maps["baseline_raw"]
            routed_scores = maps["routed_raw"]
        shared_vmax = float(max(baseline_scores.max(), routed_scores.max()))
        if not np.isfinite(shared_vmax) or shared_vmax <= 0:
            raise FloatingPointError(f"{page_id}: invalid paired attention scale")

        image = Image.open(root / "images" / f"{page_id}.jpg")
        bbox = [float(value) for value in record["bbox"]]
        baseline_image, baseline_peak = overlay_page(
            image, baseline_scores, record["grid_shape"], bbox, shared_vmax
        )
        routed_image, routed_peak = overlay_page(
            image, routed_scores, record["grid_shape"], bbox, shared_vmax
        )

        kind_label = "删除纠正" if record["kind"] == "deletion" else "插入纠正"
        target_char = str(record["target_char"])
        baseline_char = str(record["baseline"]["generated_char"])
        routed_char = str(record["line_mask"]["generated_char"])
        target_line = (
            f"目标字「{target_char}」  |  Baseline query「{baseline_char}」  |  "
            f"Mask-routing query「{routed_char}」"
        )
        edit_line = (
            f"页级编辑 baseline {case['baseline_edits']} → method {case['mask_edits']}  |  "
            "热力：末层多头平均 raw patch attention（本组左右共用色标）  |  绿框：validation 目标字"
        )
        alignment_note = (
            "删除型 baseline 未输出目标字，此处显示 Levenshtein 对齐边界 query；绿框标出缺失的目标字"
            if record["kind"] == "deletion"
            else "插入型 baseline query 显示对齐出的首个额外字；绿框标出应识别的目标字"
        )

        panel_width = 760
        pad = 18
        max_image_height = 1320
        scale = min((panel_width - 2 * pad) / image.width, max_image_height / image.height)
        display_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        baseline_image = baseline_image.resize(display_size, Image.Resampling.LANCZOS)
        routed_image = routed_image.resize(display_size, Image.Resampling.LANCZOS)

        header_height, panel_header_height, footer_height, gap = 270, 82, 78, 24
        canvas_width = panel_width * 2 + gap
        image_top = header_height + panel_header_height
        canvas_height = image_top + display_size[1] + footer_height
        canvas = Image.new("RGB", (canvas_width, canvas_height), (247, 248, 250))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (24, 15),
            f"{int(case_id):02d}  {kind_label}  |  {page_id}",
            font=font_bold,
            fill=(20, 25, 35),
        )
        draw.text((24, 60), target_line, font=font_bold, fill=(28, 55, 40))
        draw.text((24, 111), edit_line, font=font_tiny, fill=(55, 60, 70))
        draw.text(
            (24, 145),
            f"热力数值：raw mean-head patch attention（概率）  |  本图动态范围：0–{shared_vmax:.6g}",
            font=font_tiny,
            fill=(55, 60, 70),
        )
        legend_x, legend_y, legend_width, legend_height = 24, 174, 700, 18
        canvas.paste(
            attention_legend_image(legend_width, legend_height, shared_vmax),
            (legend_x, legend_y),
        )
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            (legend_x, legend_y, legend_x + legend_width - 1, legend_y + legend_height - 1),
            outline=(70, 70, 75),
            width=1,
        )
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            tick_x = round(legend_x + fraction * (legend_width - 1))
            draw.line(
                (tick_x, legend_y + legend_height, tick_x, legend_y + legend_height + 5),
                fill=(45, 48, 55),
                width=1,
            )
            tick_value = f"{shared_vmax * fraction:.3g}"
            tick_box = draw.textbbox((0, 0), tick_value, font=font_tiny)
            tick_width = tick_box[2] - tick_box[0]
            tick_label_x = max(
                legend_x,
                min(tick_x - tick_width // 2, legend_x + legend_width - tick_width),
            )
            draw.text(
                (tick_label_x, legend_y + legend_height + 5),
                tick_value,
                font=font_tiny,
                fill=(55, 60, 70),
            )
        draw.text(
            (760, 173),
            "颜色按 Inferno 映射；透明度随注意力增大",
            font=font_tiny,
            fill=(55, 60, 70),
        )
        draw.text(
            (760, 199),
            "左右共用本图色标；不同样本按各自动态范围缩放",
            font=font_tiny,
            fill=(55, 60, 70),
        )
        draw.text(
            (24, 237),
            alignment_note,
            font=font_tiny,
            fill=(70, 70, 75),
        )

        panels = [
            (0, baseline_image, "Baseline", record["baseline"], baseline_peak),
            (panel_width + gap, routed_image, "Mask-routing", record["line_mask"], routed_peak),
        ]
        for x, panel_image, label, arm, peak in panels:
            draw.rectangle(
                (x, header_height, x + panel_width - 1, header_height + panel_header_height - 1),
                fill=(226, 232, 240),
            )
            draw.text((x + pad, header_height + 7), label, font=font_bold, fill=(22, 38, 60))
            draw.text(
                (x + pad, header_height + 47),
                f"显示 token「{arm['generated_char']}」  step={arm['generation_step']}  "
                f"visual mass={arm['visual_mass']:.4f}  peak={peak:.6f}",
                font=font_tiny,
                fill=(48, 55, 65),
            )
            image_x = x + (panel_width - display_size[0]) // 2
            canvas.paste(panel_image, (image_x, image_top))
            draw.rectangle(
                (x, image_top, x + panel_width - 1, image_top + display_size[1] - 1),
                outline=(170, 176, 186),
                width=2,
            )

        output_name = f"comparison-{int(case_id):02d}-{record['kind']}-{page_id}.png"
        output_path = individual_dir / output_name
        canvas.save(output_path, optimize=True)
        rendered.append((int(case_id), canvas.copy(), output_name))
        index.append(
            {
                "case_id": case_id,
                "split": "validation",
                "page_id": page_id,
                "kind": record["kind"],
                "target_char": target_char,
                "baseline_char": baseline_char,
                "mask_routing_char": routed_char,
                "bbox_normalized_xyxy": bbox,
                "baseline_edit_distance": case["baseline_edits"],
                "mask_routing_edit_distance": case["mask_edits"],
                "output": str(output_path),
                "attention_backend": record["attention_backend"],
                "generation_prefix_verified": record["generation_prefix_verified"],
            }
        )

    (root / "comparison_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (root / "comparison_index.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index[0].keys()))
        writer.writeheader()
        writer.writerows(index)

    cell_width, cell_height, columns = 780, 620, 2
    rows = math.ceil(len(rendered) / columns)
    contact = Image.new(
        "RGB", (cell_width * columns, cell_height * rows), (232, 235, 240)
    )
    draw = ImageDraw.Draw(contact)
    for position, (case_number, pair_image, filename) in enumerate(rendered):
        row, column = divmod(position, columns)
        x, y = column * cell_width, row * cell_height
        draw.rectangle(
            (x + 6, y + 6, x + cell_width - 7, y + cell_height - 7),
            fill=(255, 255, 255),
            outline=(195, 200, 208),
            width=2,
        )
        draw.text(
            (x + 18, y + 12),
            f"{case_number:02d}  {filename}",
            font=font_tiny,
            fill=(30, 38, 48),
        )
        pair_image.thumbnail((cell_width - 28, cell_height - 50), Image.Resampling.LANCZOS)
        contact.paste(pair_image, (x + (cell_width - pair_image.width) // 2, y + 42))
    contact.save(root / "contact_sheet.png", optimize=True)
    print(f"Rendered {len(rendered)} paired validation figures in {individual_dir}")


if __name__ == "__main__":
    main()
