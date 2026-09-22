#!/usr/bin/env python3
"""Measure the per-step bias dose of each mask-target shape, without a GPU run.

The line100-window attribution showed the whole-line target biases 77.62 visual
cells per decoding step and the 3-5 character window only 8.38, at the same
B=1.0 -- and that the difference, not the layout branch, accounts for the whole
CER gap. What that comparison cannot say is whether the operative quantity is
the per-key strength or the coverage, because the two moved together.

This tool puts the candidate target shapes on the coverage axis and reports the
dose distribution per step, so a stage can be checked for monotonicity before
any GPU budget is spent on it. It is a measurement, not an evaluation: it reads
ground-truth annotations, produces no predictions, and is never part of
acceptance or selection.

The rasters are built on the real merged visual grid, so the numbers are
comparable with the routing diagnostics recorded in validation_predictions.jsonl.
Note the one difference and why it is safe: routing counts cells whose centre
falls inside the rasterised polygon, while build_mask_targets assigns a cell to
a polygon if any of its corners does. On a grid this coarse the two agree to a
cell or two per step, which is far inside the spread this measurement is used
to separate. ``--reported`` compares against the recorded numbers so the gap is
visible rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import torch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="*",
        default=["token", "window", "anchored", "line"],
        help="target shapes to measure; 'anchored' is the window union its line remainder",
    )
    parser.add_argument("--window-min", type=int, default=3)
    parser.add_argument("--window-max", type=int, default=5)
    parser.add_argument("--max-pixels", type=int, default=4000000)
    parser.add_argument("--processor-mode", choices=("fast", "slow"), default="fast")
    parser.add_argument("--raster-mode", choices=("soft", "hard"), default="hard")
    parser.add_argument(
        "--merge-size",
        type=int,
        default=2,
        help="vision tower spatial merge; verified against the prompt's image-token count",
    )
    parser.add_argument("--pages", type=int, default=0, help="limit pages; 0 means all")
    parser.add_argument(
        "--reported",
        default="",
        metavar="MODE=TOKENS,...",
        help="recorded per-step hit tokens to compare against, e.g. 'line=77.62,window=8.38'",
    )
    parser.add_argument(
        "--reported-tolerance",
        type=float,
        default=1.0,
        help="absolute token tolerance for the reported check; exceeding it fails the run",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="report the mismatch instead of failing; only for diagnosing the tool itself",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def summarise(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    count = len(ordered)

    def quantile(fraction: float) -> float:
        return ordered[min(count - 1, int(fraction * count))]

    return {
        "steps": count,
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p05": quantile(0.05),
        "p95": quantile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
        "zero_share": sum(1 for v in ordered if v == 0.0) / count,
    }


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    args = parse_args(argv=None)
    from layout_ocr.data import prepare_training_inputs
    from layout_ocr.decoder_mask_router import _normalized_grid_xywh
    from layout_ocr.mask_targets import build_mask_targets
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(args.model_path), trust_remote_code=True, use_fast=args.processor_mode == "fast"
    )
    # The evaluator sets longest_edge to max_pixels; leaving it at the processor
    # default silently lands on a different visual grid and rescales every dose.
    # This is not cosmetic: the default grid for this model is roughly half the
    # merged cells of the 4M grid the routing arms ran on.
    processor.image_processor.size = {
        **processor.image_processor.size,
        "longest_edge": args.max_pixels,
    }
    device = torch.device(args.device)
    records = [
        json.loads(line)
        for line in args.validation_manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.pages:
        records = records[: args.pages]
    merge_size = int(args.merge_size)
    image_token_id = int(getattr(processor, "image_token_id", None) or 0)

    eos = {int(processor.tokenizer.eos_token_id)}
    per_mode: dict[str, list[float]] = {mode: [] for mode in args.modes}
    fallbacks: Counter[str] = Counter()
    lines_per_page: list[int] = []
    grid_checks: list[dict[str, int]] = []
    grid_sizes: list[int] = []
    for index, record in enumerate(records):
        inputs = prepare_training_inputs(processor, record, device, eos)
        # The assistant span of the chat template is exactly what the decoder
        # generates, so its token count is the decoding-step count.
        prompt_length = int((inputs["labels"] == -100).int().cumprod(1).sum())
        target_ids = inputs["input_ids"][0, prompt_length:]
        grid_thw = inputs["image_grid_thw"]
        xywh, merged_shape = _normalized_grid_xywh(grid_thw, merge_size)
        # Verify against the prompt: the placeholder count must equal the merged
        # grid size.  This catches a wrong merge size, but it does NOT catch a
        # wrong max_pixels -- a smaller grid is still self-consistent.  Only the
        # reported-value comparison below can catch that, which is why it exists.
        placeholder = int((inputs["input_ids"] == image_token_id).sum()) if image_token_id else 0
        if placeholder:
            grid_checks.append(
                {"merged_cells": merged_shape[0] * merged_shape[1], "image_tokens": placeholder}
            )
        grid_sizes.append(merged_shape[0] * merged_shape[1])
        for mode in args.modes:
            targets = build_mask_targets(
                processor.tokenizer,
                record,
                target_ids,
                eos,
                xywh,
                target_mode=mode,
                window_min=args.window_min,
                window_max=args.window_max,
                line_source="annotation",
                raster_mode=args.raster_mode,
            )
            # Only steps that actually carry a bias count; EOS and unmapped tokens
            # are skipped by the runtime and must not dilute the dose.
            valid = targets.spatial_valid[0]
            hit = (targets.mask[0] > 0).sum(dim=-1).float()[valid]
            per_mode[mode].extend(hit.tolist())
            fallbacks[f"{mode}:{targets.window_report['window_fallbacks']}"] += 0
            if mode == "window":
                fallbacks["window_fallbacks"] += targets.window_report["window_fallbacks"]
                lines_per_page.append(targets.window_report["lines"])
        if (index + 1) % 25 == 0:
            print(f"  {index + 1}/{len(records)} pages", flush=True)

    mismatched = [c for c in grid_checks if c["merged_cells"] != c["image_tokens"]]
    if mismatched:
        raise ValueError(
            f"merge size {merge_size} does not match the prompt's image-token count "
            f"(first mismatch: {mismatched[0]}); the grid would be wrong for every page"
        )
    report = {
        "pages": len(records),
        "window_min": args.window_min,
        "window_max": args.window_max,
        "raster_mode": args.raster_mode,
        "processor_mode": args.processor_mode,
        "merge_size": merge_size,
        "grid_verified_pages": len(grid_checks),
        "merged_cells_mean": statistics.fmean(grid_sizes) if grid_sizes else None,
        "merged_cells_min": min(grid_sizes) if grid_sizes else None,
        "merged_cells_max": max(grid_sizes) if grid_sizes else None,
        "modes": {mode: summarise(values) for mode, values in per_mode.items() if values},
        "window_fallbacks": fallbacks["window_fallbacks"],
        "mean_lines_per_page": statistics.fmean(lines_per_page) if lines_per_page else None,
        "reads_ground_truth": True,
        "usable_for_selection": False,
        "test_manifest_read": False,
    }
    recorded: dict[str, float] = {}
    for item in args.reported.split(","):
        if not item.strip():
            continue
        name, _, value = item.partition("=")
        recorded[name.strip()] = float(value)
    if recorded:
        report["reported"] = {
            mode: {
                "recorded": value,
                "measured": report["modes"][mode]["mean"],
                "difference": report["modes"][mode]["mean"] - value,
                "relative": report["modes"][mode]["mean"] / value - 1.0,
            }
            for mode, value in recorded.items()
            if mode in report["modes"]
        }
        off = {
            mode: stats["relative"]
            for mode, stats in report["reported"].items()
            if abs(stats["difference"]) > args.reported_tolerance
        }
        report["reported_check"] = {"passed": not off, "tolerance": args.reported_tolerance, "off": off}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for mode, stats in report["modes"].items():
        print(
            f"{mode:10s} mean {stats['mean']:8.2f}  median {stats['median']:8.2f}  "
            f"p05 {stats['p05']:7.2f}  p95 {stats['p95']:8.2f}  zero {stats['zero_share']:.3f}"
        )
    print(
        "merged cells per page: mean {:.0f} min {} max {}".format(
            report["merged_cells_mean"] or 0,
            report["merged_cells_min"],
            report["merged_cells_max"],
        )
    )
    print(json.dumps({k: v for k, v in report.items() if k not in ("modes",)}, ensure_ascii=False))
    check = report.get("reported_check")
    if check and not check["passed"] and not args.allow_mismatch:
        # The whole value of this tool is that its third number can be trusted.
        # If it cannot reproduce the two recorded arms, it is measuring something
        # else -- a wrong grid, a wrong denominator -- and its anchored figure
        # would be just as wrong.  Fail rather than report it.
        raise SystemExit(
            "reported-value self-check failed: "
            + json.dumps(report["reported"], ensure_ascii=False)
            + "; the measured doses are not comparable with the recorded arms"
        )


if __name__ == "__main__":
    main()
