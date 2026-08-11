from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a right-to-left vertical page without resizing the source columns."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-columns", type=int, default=26)
    parser.add_argument("--paper-threshold", type=int, default=50)
    parser.add_argument("--paper-row-fraction", type=float, default=0.50)
    parser.add_argument("--ink-threshold", type=int, default=60)
    parser.add_argument("--projection-smooth", type=int, default=11)
    parser.add_argument("--gap-threshold", type=float, default=3.0)
    parser.add_argument("--min-gap-width", type=int, default=3)
    parser.add_argument("--analysis-margin", type=int, default=36)
    parser.add_argument("--anandasky-patch-size", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask.tolist()):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(mask) - 1):
            end = index if not value else index + 1
            runs.append((start, end))
            start = None
    return runs


def longest_run(mask: np.ndarray) -> tuple[int, int]:
    runs = contiguous_runs(mask)
    if not runs:
        raise ValueError("No paper-like row interval was detected.")
    return max(runs, key=lambda run: run[1] - run[0])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_args(args: argparse.Namespace) -> None:
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image does not exist: {args.image}")
    if args.expected_columns < 1:
        raise ValueError("--expected-columns must be positive.")
    if not 0.0 < args.paper_row_fraction <= 1.0:
        raise ValueError("--paper-row-fraction must be in (0, 1].")
    if args.projection_smooth < 1 or args.projection_smooth % 2 == 0:
        raise ValueError("--projection-smooth must be a positive odd integer.")
    if args.analysis_margin < 0:
        raise ValueError("--analysis-margin cannot be negative.")
    if args.anandasky_patch_size < 1:
        raise ValueError("--anandasky-patch-size must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {manifest_path}. Pass --force only when replacement is intended."
        )

    source = Image.open(args.image).convert("RGB")
    source_gray = np.asarray(source.convert("L"))

    paper_fraction = (source_gray > args.paper_threshold).mean(axis=1)
    paper_rows = paper_fraction >= args.paper_row_fraction
    crop_y0, crop_y1 = longest_run(paper_rows)
    page = source.crop((0, crop_y0, source.width, crop_y1))
    page_gray = np.asarray(page.convert("L"))

    if 2 * args.analysis_margin >= page.height:
        raise ValueError("--analysis-margin removes the complete detected paper region.")
    analysis = page_gray[args.analysis_margin : page.height - args.analysis_margin]
    ink_projection = (analysis < args.ink_threshold).sum(axis=0).astype(np.float64)
    kernel = np.ones(args.projection_smooth, dtype=np.float64) / args.projection_smooth
    smoothed_projection = np.convolve(ink_projection, kernel, mode="same")
    gap_mask = smoothed_projection < args.gap_threshold
    gap_runs = [
        run
        for run in contiguous_runs(gap_mask)
        if run[1] - run[0] >= args.min_gap_width
    ]
    internal_gaps = [run for run in gap_runs if run[0] > 0 and run[1] < page.width]
    boundaries = [0] + [(start + end) // 2 for start, end in internal_gaps] + [page.width]

    detected_columns = len(boundaries) - 1
    if detected_columns != args.expected_columns:
        raise RuntimeError(
            "Column detection did not match the requested count: "
            f"detected={detected_columns}, expected={args.expected_columns}, "
            f"internal_gaps={internal_gaps}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ananda_dir = args.output_dir / "anandasky_columns"
    got_dir = args.output_dir / "got_columns"
    ananda_dir.mkdir(parents=True, exist_ok=True)
    got_dir.mkdir(parents=True, exist_ok=True)
    page.save(args.output_dir / "page_crop.png")

    page_pixels = np.asarray(page)
    padding_rgb = tuple(int(value) for value in np.median(page_pixels.reshape(-1, 3), axis=0))
    overlay = page.copy()
    draw = ImageDraw.Draw(overlay)
    for boundary in boundaries[1:-1]:
        draw.line((boundary, 0, boundary, page.height - 1), fill=(255, 0, 0), width=2)

    left_to_right = list(zip(boundaries[:-1], boundaries[1:]))
    columns: list[dict[str, object]] = []
    for reading_index, (x0, x1) in enumerate(reversed(left_to_right), start=1):
        filename = f"column_{reading_index:03d}.png"
        column = page.crop((x0, 0, x1, page.height))
        column.save(ananda_dir / filename)

        canvas_size = max(page.height, column.width)
        canvas = Image.new("RGB", (canvas_size, canvas_size), padding_rgb)
        paste_x = (canvas_size - column.width) // 2
        paste_y = (canvas_size - column.height) // 2
        canvas.paste(column, (paste_x, paste_y))
        canvas.save(got_dir / filename)

        patch_h = math.ceil(column.height / args.anandasky_patch_size)
        patch_w = math.ceil(column.width / args.anandasky_patch_size)
        draw.text((x0 + 2, 2), f"{reading_index:02d}", fill=(0, 255, 255))
        columns.append(
            {
                "reading_index": reading_index,
                "source_box": [x0, crop_y0, x1, crop_y1],
                "width": column.width,
                "height": column.height,
                "files": {
                    "anandasky": f"anandasky_columns/{filename}",
                    "got": f"got_columns/{filename}",
                },
                "got_padding": {
                    "canvas_size": canvas_size,
                    "paste_box": [
                        paste_x,
                        paste_y,
                        paste_x + column.width,
                        paste_y + column.height,
                    ],
                },
                "anandasky_patch_grid": [patch_h, patch_w],
                "anandasky_visual_tokens": patch_h * patch_w,
            }
        )

    overlay.save(args.output_dir / "columns_overlay.png")
    np.savetxt(
        args.output_dir / "vertical_projection.tsv",
        np.column_stack((np.arange(page.width), ink_projection, smoothed_projection)),
        delimiter="\t",
        header="x\tink_pixels\tsmoothed_ink_pixels",
        comments="",
        fmt=["%d", "%.0f", "%.6f"],
    )

    manifest = {
        "schema_version": 1,
        "input": {
            "path": str(args.image.resolve()),
            "sha256": sha256_file(args.image),
            "size": [source.width, source.height],
            "mode": source.mode,
        },
        "reading_order": "columns_right_to_left; characters_top_to_bottom",
        "resize_policy": {
            "anandasky": "No resize; use each original-aspect-ratio column.",
            "got": "No source resize; center each column on a square median-paper-color canvas before GOT's uniform 1024x1024 resize.",
        },
        "paper_crop": [0, crop_y0, source.width, crop_y1],
        "paper_padding_rgb": list(padding_rgb),
        "parameters": {
            "paper_threshold": args.paper_threshold,
            "paper_row_fraction": args.paper_row_fraction,
            "ink_threshold": args.ink_threshold,
            "projection_smooth": args.projection_smooth,
            "gap_threshold": args.gap_threshold,
            "min_gap_width": args.min_gap_width,
            "analysis_margin": args.analysis_margin,
            "expected_columns": args.expected_columns,
            "anandasky_patch_size": args.anandasky_patch_size,
        },
        "detected_gap_runs": [list(run) for run in gap_runs],
        "column_boundaries": boundaries,
        "columns": columns,
    }
    save_json(manifest_path, manifest)

    print("VERTICAL_PAGE_PREPROCESS_OK")
    print(f"input={args.image.resolve()}")
    print(f"input_size={source.width}x{source.height}")
    print(f"paper_crop=0,{crop_y0},{source.width},{crop_y1}")
    print(f"paper_size={page.width}x{page.height}")
    print(f"columns={len(columns)}")
    print(f"padding_rgb={padding_rgb}")
    print(f"manifest={manifest_path.resolve()}")


if __name__ == "__main__":
    main()
