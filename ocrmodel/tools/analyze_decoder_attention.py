#!/usr/bin/env python3
"""Summarise decoder visual attention and render a few page overlays.

The probe JSONL contains the scalar/per-head measurements for every validation
page.  Selected pages additionally carry compressed ``npz`` samples containing
the visual-conditional attention distribution over patch tokens.  This tool keeps
those two evidence levels separate:

* all-page statistics use the JSONL reductions, and
* heatmaps use only the pre-registered patch samples.

The heatmap is therefore a visualization of ``a_vis`` (softmax conditional on
visual keys), not a claim that the model assigned that mass against the text keys.
The latter is reported separately as ``m_t`` and ``lse_vis - lse_text``.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def finite_values(values: Iterable[Any]) -> np.ndarray:
    converted: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            converted.append(number)
    return np.asarray(converted, dtype=np.float64)


def describe(values: Iterable[Any]) -> dict[str, Any]:
    array = finite_values(values)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def effective_fraction(entropy_norm: float, visual_tokens: int) -> float | None:
    if visual_tokens <= 1 or not math.isfinite(entropy_norm):
        return None
    # entropy_norm = H / log(V), so exp(H) / V is the effective fraction of
    # visual tokens used by the distribution.  Uniform attention is exactly 1.
    return float(math.exp((entropy_norm - 1.0) * math.log(visual_tokens)))


def row_measurements(reports: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_head: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pages: set[str] = set()
    all_rows: list[dict[str, Any]] = []
    for report in reports:
        page_id = str(report.get("page_id"))
        pages.add(page_id)
        visual_tokens = int(report.get("visual_tokens") or 0)
        for step in report.get("steps") or []:
            for row in step.get("heads") or []:
                item = {
                    "page_id": page_id,
                    "step": int(step.get("step", 0)),
                    "layer": int(row.get("layer", -1)),
                    "head": int(row.get("head", -1)),
                    "visual_tokens": visual_tokens,
                    "m_t": row.get("m_t"),
                    "entropy_norm": row.get("entropy_norm"),
                    "effective_token_fraction": effective_fraction(
                        float(row.get("entropy_norm", float("nan"))), visual_tokens
                    ),
                    "visual_text_logit_gap": (
                        float(row["lse_vis"]) - float(row["lse_text"])
                        if row.get("lse_vis") is not None and row.get("lse_text") is not None
                        else None
                    ),
                }
                all_rows.append(item)
                by_layer[str(item["layer"])].append(item)
                by_head[f"{item['layer']}:{item['head']}"].append(item)

    def summarise_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "rows": len(rows),
            "pages": len({row["page_id"] for row in rows}),
            "steps": len({(row["page_id"], row["step"]) for row in rows}),
            "visual_mass_m_t": describe(row["m_t"] for row in rows),
            "visual_conditional_entropy_norm": describe(row["entropy_norm"] for row in rows),
            "effective_visual_token_fraction": describe(
                row["effective_token_fraction"] for row in rows
            ),
            "visual_text_logit_gap": describe(row["visual_text_logit_gap"] for row in rows),
        }

    global_summary = summarise_group(all_rows)
    global_summary["unique_visual_token_counts"] = sorted(
        {int(row["visual_tokens"]) for row in all_rows if row["visual_tokens"] > 0}
    )
    return by_layer, {
        "all": global_summary,
        "per_layer": {
            key: summarise_group(rows)
            for key, rows in sorted(by_layer.items(), key=lambda item: int(item[0]))
        },
        "per_head": {
            key: summarise_group(rows)
            for key, rows in sorted(by_head.items(), key=lambda item: tuple(map(int, item[0].split(":"))))
        },
        "pages": len(pages),
        "rows": len(all_rows),
    }


def map_summary(probabilities: np.ndarray) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    values = np.clip(values, 0.0, None)
    total = float(values.sum())
    if total <= 0.0:
        return {"visual_tokens": int(values.size), "status": "empty"}
    values = values / total
    entropy = float(-(values * np.log(np.clip(values, 1e-12, None))).sum())
    visual_tokens = int(values.size)
    normalized_entropy = entropy / math.log(visual_tokens) if visual_tokens > 1 else 0.0
    ordered = np.sort(values)[::-1]
    return {
        "visual_tokens": visual_tokens,
        "normalized_entropy": normalized_entropy,
        "effective_token_fraction": float(math.exp(entropy) / max(1, visual_tokens)),
        "top1_mass": float(ordered[:1].sum()),
        "top5_mass": float(ordered[:5].sum()),
        "top10_mass": float(ordered[:10].sum()),
        "top1_lift_over_uniform": float(ordered[0] * visual_tokens),
        "top5_lift_over_uniform": float(ordered[:5].sum() * visual_tokens / min(5, visual_tokens)),
    }


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")


def _grid_values(positions: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Put normalized patch-center values on a regular y/x grid."""

    rounded = np.round(np.asarray(positions, dtype=np.float64), 6)
    xs = np.unique(rounded[:, 0])
    ys = np.unique(rounded[:, 1])
    x_index = {float(value): index for index, value in enumerate(xs)}
    y_index = {float(value): index for index, value in enumerate(ys)}
    grid = np.zeros((len(ys), len(xs)), dtype=np.float64)
    counts = np.zeros_like(grid)
    for (x, y), value in zip(rounded.tolist(), np.asarray(values).tolist()):
        row = y_index[float(y)]
        column = x_index[float(x)]
        grid[row, column] += float(value)
        counts[row, column] += 1.0
    return grid / np.maximum(counts, 1.0)


def _colour_map(levels: np.ndarray) -> np.ndarray:
    """A compact blue-cyan-yellow-red heatmap without a matplotlib dependency."""

    values = np.clip(levels, 0.0, 1.0)
    stops = np.asarray(
        [[35, 65, 210], [20, 205, 220], [255, 225, 55], [210, 30, 35]],
        dtype=np.float64,
    )
    scaled = values * (len(stops) - 1)
    low = np.floor(scaled).astype(np.int64)
    high = np.minimum(low + 1, len(stops) - 1)
    fraction = (scaled - low)[..., None]
    return (stops[low] * (1.0 - fraction) + stops[high] * fraction).astype(np.uint8)


def render_overlay(
    page_path: Path,
    positions: np.ndarray,
    values: np.ndarray,
    output_path: Path,
    *,
    title: str,
    max_edge: int,
) -> dict[str, Any]:
    with Image.open(page_path) as source:
        page = source.convert("RGB")
    scale = min(1.0, max_edge / max(page.size)) if max_edge > 0 else 1.0
    if scale < 1.0:
        page = page.resize(
            (max(1, int(round(page.width * scale))), max(1, int(round(page.height * scale)))),
            Image.Resampling.LANCZOS,
        )

    distribution = np.asarray(values, dtype=np.float64).mean(axis=0)
    distribution = np.clip(distribution, 0.0, None)
    distribution /= max(float(distribution.sum()), 1e-12)
    grid = _grid_values(positions, distribution)
    peak = float(grid.max())
    level = grid / max(peak, 1e-12)
    heat_small = Image.fromarray(np.asarray(level * 255.0, dtype=np.uint8), mode="L")
    heat = np.asarray(
        heat_small.resize(page.size, Image.Resampling.BILINEAR), dtype=np.float64
    ) / 255.0
    colours = _colour_map(heat)
    alpha = np.clip(np.sqrt(heat) * 0.82, 0.0, 0.82)[..., None]
    base = np.asarray(page, dtype=np.float64)
    blended = (base * (1.0 - alpha) + colours.astype(np.float64) * alpha).astype(np.uint8)
    canvas = Image.fromarray(blended, mode="RGB")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, min(canvas.width, 460), 28), fill=(0, 0, 0))
    draw.text((8, 7), title, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return {
        "output": str(output_path),
        "source_image": str(page_path),
        "peak_probability": peak,
        "map": map_summary(distribution),
    }


def load_npz_maps(root: Path) -> list[dict[str, Any]]:
    maps: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.npz")):
        with np.load(path) as data:
            page_id = str(data["page_id"].reshape(-1)[0])
            attention = np.asarray(data["attention"], dtype=np.float32)
            steps = np.asarray(data["steps"], dtype=np.int64)
            layers = np.asarray(data["layers"], dtype=np.int64)
            heads = np.asarray(data["heads"], dtype=np.int64)
            positions = np.asarray(data["positions"], dtype=np.float32)
        if attention.ndim != 3 or len(steps) != attention.shape[0] or len(layers) != attention.shape[0]:
            raise ValueError(f"invalid attention map shape in {path}")
        maps.append(
            {
                "path": str(path),
                "page_id": page_id,
                "attention": attention,
                "steps": steps,
                "layers": layers,
                "heads": heads,
                "positions": positions,
            }
        )
    return maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="summarise GLM-OCR decoder visual attention")
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heatmap-dir", type=Path, required=True)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--render-layer", type=int, default=8)
    parser.add_argument("--render-pages", type=int, default=4)
    parser.add_argument("--render-steps-per-page", type=int, default=3)
    parser.add_argument("--max-edge", type=int, default=2400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = load_jsonl(args.probe)
    manifest_rows = {str(row["page_id"]): row for row in load_jsonl(args.manifest)}
    if len(reports) != len(manifest_rows):
        raise ValueError(
            f"probe pages {len(reports)} do not match validation subset pages {len(manifest_rows)}"
        )
    report_ids = {str(report.get("page_id")) for report in reports}
    if report_ids != set(manifest_rows):
        missing = sorted(set(manifest_rows) - report_ids)
        extra = sorted(report_ids - set(manifest_rows))
        raise ValueError(f"probe page mismatch: missing={missing[:3]} extra={extra[:3]}")
    if args.predictions and args.predictions.is_file():
        prediction_ids = {str(row["page_id"]) for row in load_jsonl(args.predictions)}
        if prediction_ids != set(manifest_rows):
            raise ValueError("prediction pages do not match the validation subset")

    _, distribution = row_measurements(reports)
    maps = load_npz_maps(args.heatmap_dir)
    sample_rows: list[dict[str, Any]] = []
    for item in maps:
        for index in range(item["attention"].shape[0]):
            sample_rows.append(
                {
                    "page_id": item["page_id"],
                    "layer": int(item["layers"][index]),
                    "step": int(item["steps"][index]),
                    **map_summary(item["attention"][index].mean(axis=0)),
                }
            )
    distribution["patch_sample_maps"] = {
        "maps": len(sample_rows),
        "pages": len({row["page_id"] for row in sample_rows}),
        "normalized_entropy": describe(row.get("normalized_entropy") for row in sample_rows),
        "effective_token_fraction": describe(
            row.get("effective_token_fraction") for row in sample_rows
        ),
        "top1_mass": describe(row.get("top1_mass") for row in sample_rows),
        "top5_mass": describe(row.get("top5_mass") for row in sample_rows),
        "top10_mass": describe(row.get("top10_mass") for row in sample_rows),
        "top1_lift_over_uniform": describe(
            row.get("top1_lift_over_uniform") for row in sample_rows
        ),
        "top5_lift_over_uniform": describe(
            row.get("top5_lift_over_uniform") for row in sample_rows
        ),
        "uniform_reference": {
            "normalized_entropy": 1.0,
            "effective_token_fraction": 1.0,
            "top1_lift_over_uniform": 1.0,
            "top5_lift_over_uniform": 1.0,
        },
    }

    selected_page_ids = sorted({item["page_id"] for item in maps})[: max(0, args.render_pages)]
    overlays: list[dict[str, Any]] = []
    for item in maps:
        page_id = item["page_id"]
        if page_id not in selected_page_ids:
            continue
        indices = [
            index
            for index, layer in enumerate(item["layers"].tolist())
            if int(layer) == args.render_layer
        ]
        if not indices:
            continue
        ordered_steps = sorted(indices, key=lambda index: int(item["steps"][index]))
        count = min(args.render_steps_per_page, len(ordered_steps))
        chosen = [
            ordered_steps[index]
            for index in np.linspace(0, len(ordered_steps) - 1, count, dtype=np.int64).tolist()
        ]
        for index in chosen:
            step = int(item["steps"][index])
            output = args.output_dir / "visualizations" / (
                f"{safe_name(page_id)}_layer{args.render_layer}_step{step:04d}_mean.png"
            )
            record = manifest_rows[page_id]
            overlays.append(
                {
                    "page_id": page_id,
                    "layer": args.render_layer,
                    "step": step,
                    "heads": item["heads"].tolist(),
                    **render_overlay(
                        Path(record["image_path"]),
                        item["positions"],
                        item["attention"][index],
                        output,
                        title=f"layer={args.render_layer} step={step} heads=mean",
                        max_edge=args.max_edge,
                    ),
                }
            )

    summary = {
        "status": "complete",
        "pages": len(manifest_rows),
        "probe_pages": len(reports),
        "probe_layers": sorted({int(layer) for report in reports for layer in report.get("layers", [])}),
        "probe_heads": sorted({int(head) for report in reports for head in (report.get("heads") or [])}),
        "visual_attention_definition": "softmax conditional on visual keys; m_t reports total visual mass against text keys",
        "distribution": distribution,
        "heatmap_samples": {
            "directory": str(args.heatmap_dir),
            "files": len(maps),
            "render_layer": args.render_layer,
            "render_pages": selected_page_ids,
            "overlays": overlays,
        },
        "validation_manifest": str(args.manifest),
        "probe": str(args.probe),
        "predictions": str(args.predictions) if args.predictions else None,
    }
    output = args.output_dir / "attention_summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "event": "decoder_attention_analysis_complete",
        "pages": len(manifest_rows),
        "probe_rows": distribution["rows"],
        "sample_maps": len(sample_rows),
        "overlays": len(overlays),
        "summary": str(output),
    }, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
