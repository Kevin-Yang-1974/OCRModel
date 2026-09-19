"""Quantify the spatial kernel proposed in plans/LAYOUT_FUSION_REDESIGN.md section 2.2.

Section 2.2 replaces the learned transport with a fixed geometric kernel over
predicted box centres::

    distances       = cdist(patch_positions, box_centres)     # normalized coords
    spatial_weights = exp(-distances / 0.1)
    spatial_weights = spatial_weights / spatial_weights.sum(dim=-1, keepdim=True)

Both operands live in normalized page coordinates, so a distance of 0.1 is a tenth
of the page.  The script measures what that does to the weights on real Dunhuang
region geometry: whether the row-normalized kernel is a smooth interpolation over
nearby boxes (the design intent) or a saturated nearest-centre lookup.

It also reports the smallest row sum.  The proposal divides by that sum with no
``clamp_min``, and in low precision a row whose entries all underflow to zero
would produce NaN.  This is a static analysis of the kernel only -- it does not
run the model and says nothing about whether the arm would help OCR.

Usage::

    python tools/analyze_spatial_kernel.py --annotations <dir> [--sigma 0.1]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from layout_ocr.glm_bridge import patch_grid_positions  # noqa: E402


def load_box_centres(
    annotations: str, num_queries: int | None = None
) -> tuple[list[torch.Tensor], list[float]]:
    """Return per-page normalized box centres and per-page median box diagonal.

    ``num_queries`` subsamples each page down to the adapter's actual box count.
    This matters: the kernel in section 2.2 runs over the ``num_queries`` *predicted*
    boxes, not over the page's ground-truth regions, and a page of Dunhuang text
    lines has far more regions than the 32 queries the Q32 lineage uses.  Using the
    ground-truth count would make the box centres look denser than they are and
    flatter the kernel.  Subsampling is a deterministic stride, so it spreads the
    retained centres over the page without needing the (unavailable) predicted
    boxes.
    """

    pages: list[torch.Tensor] = []
    diagonals: list[float] = []
    for path in sorted(glob.glob(os.path.join(annotations, "*.json"))):
        record = json.loads(open(path, encoding="utf-8").read())
        corners = []
        for region in record.get("regions", []):
            polygon = region.get("polygon_px")
            if not polygon:
                continue
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            corners.append((min(xs), min(ys), max(xs), max(ys)))
        if len(corners) < 2:
            continue
        page_height, page_width = float(record["page_size"][1]), float(record["page_size"][0])
        boxes = torch.tensor(corners, dtype=torch.float32)
        boxes = boxes / torch.tensor(
            [page_width, page_height, page_width, page_height], dtype=torch.float32
        )
        boxes = boxes.clamp(0.0, 1.0)
        if num_queries is not None and boxes.shape[0] > num_queries:
            stride = boxes.shape[0] / num_queries
            index = torch.floor(torch.arange(num_queries, dtype=torch.float32) * stride)
            boxes = boxes[index.long().clamp(max=boxes.shape[0] - 1)]
        centres = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        width = (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
        height = (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
        diagonal = torch.sqrt(width**2 + height**2)
        pages.append(centres)
        diagonals.append(float(diagonal.median()))
    return pages, diagonals


def kernel_stats(
    patch_positions: torch.Tensor, centres: torch.Tensor, sigma: float, gaussian: bool
) -> dict:
    """Summarize one kernel over a page's boxes.

    The decisive statistic is ``weight_flatness``.  Section 2.2 replaces the learned
    transport with a fixed kernel, and the whole point of a fixed kernel is that it
    is *patch-specific*: patch ``p`` is supposed to receive its own mixture of box
    features.  ``weight_flatness`` is the same cross-patch variation measure the
    write-back probe already reports for ``layout_context`` (``lc_flat``), applied to
    the kernel's rows over the box axis.  A low value means every patch is handed
    nearly the same mixture, so the arm would inject a nearly patch-invariant offset
    -- the degeneracy that motivated this redesign in the first place, reintroduced
    by the geometry.
    """

    distances = torch.cdist(patch_positions.float(), centres.float())
    if gaussian:
        weights = torch.exp(-(distances**2) / (2 * sigma**2))
    else:
        weights = torch.exp(-distances / sigma)
    row_sums = weights.sum(dim=-1)
    normalized = weights / row_sums.unsqueeze(-1).clamp_min(1e-30)
    entropy = -(normalized * normalized.clamp_min(1e-30).log()).sum(dim=-1)
    centred = normalized - normalized.mean(dim=0, keepdim=True)
    weight_flatness = centred.norm(dim=-1).mean() / normalized.norm(dim=-1).mean().clamp_min(
        1e-12
    )
    top = normalized.max(dim=-1).values
    return {
        "sigma": sigma,
        "kernel": "gaussian" if gaussian else "laplacian",
        "weight_flatness": float(weight_flatness),
        "effective_boxes": float(entropy.exp().mean()),
        "fraction_top_above_0.99": float((top > 0.99).float().mean()),
        "distances_median": float(distances.median()),
        "min_row_sum": float(row_sums.min()),
        "min_row_sum_fp16": float(row_sums.half().min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, help="directory of page annotation json")
    parser.add_argument("--sigma", type=float, default=0.1, help="the plan's fixed sigma")
    parser.add_argument("--grid-height", type=int, default=64)
    parser.add_argument("--grid-width", type=int, default=48)
    parser.add_argument("--spatial-merge-size", type=int, default=2)
    parser.add_argument(
        "--num-queries",
        type=int,
        default=32,
        help="adapter box count to subsample to; 0 keeps every ground-truth region",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    grid = torch.tensor([[1, args.grid_height, args.grid_width]])
    positions = patch_grid_positions(grid, args.spatial_merge_size)[0]
    queries = None if args.num_queries <= 0 else args.num_queries
    pages, diagonals = load_box_centres(args.annotations, queries)
    if not pages:
        raise SystemExit(f"no usable annotations under {args.annotations}")

    median_diagonal = statistics.median(diagonals)
    results = {
        "pages": len(pages),
        "patches": int(positions.shape[0]),
        "num_queries": queries,
        "median_boxes_per_page": statistics.median([int(p.shape[0]) for p in pages]),
        "median_box_diagonal_normalized": median_diagonal,
    }

    # The plan's kernel, evaluated at its own sigma.
    for page in pages[:32]:
        stats = kernel_stats(positions, page, args.sigma, gaussian=False)
        results.setdefault("laplacian_fixed_sigma", []).append(stats)
    # A Gaussian at the plan's sigma, to separate "wrong kernel family" from
    # "wrong sigma units".
    for page in pages[:32]:
        stats = kernel_stats(positions, page, args.sigma, gaussian=True)
        results.setdefault("gaussian_fixed_sigma", []).append(stats)
    # A Gaussian whose sigma tracks the actual box size on the page.
    for page, diagonal in zip(pages[:32], diagonals[:32]):
        stats = kernel_stats(positions, page, max(diagonal * 0.5, 1e-3), gaussian=True)
        results.setdefault("gaussian_box_scaled_sigma", []).append(stats)

    summary = {}
    for arm, rows in results.items():
        if not isinstance(rows, list) or not rows:
            continue
        summary[arm] = {
            key: statistics.median([row[key] for row in rows])
            for key in rows[0]
            if key not in {"sigma", "kernel"}
        }
        summary[arm]["sigma"] = rows[0]["sigma"]
        summary[arm]["kernel"] = rows[0]["kernel"]
    results["summary"] = summary
    print(json.dumps({k: v for k, v in results.items() if k != "summary"}, indent=2))
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
