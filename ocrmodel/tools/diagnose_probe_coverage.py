"""Ask whether the stage-1 coverage miss is a property of the readout or of the metric.

Stage 1 failed its gate on exactly one criterion: 46.3% of alignable characters sat at
``top_line_mass >= 0.5`` against a 50% target, while the accuracy on those characters was
99.4% against a 90% target.  A criterion that is missed by 3.7 points while its companion
is exceeded by a wide margin is worth understanding before anything is decided with it.

``top_line_mass`` is an absolute share, but the share a page *can* give one line depends on
how many lines it has.  Under a uniform distribution over ``N`` lines no line exceeds
``1/N``, so reaching 0.5 demands concentrating ``N/2`` times above uniform: easy on a
ten-line page and much harder on a forty-line one.  The threshold therefore mixes "the
readout is not confident here" with "this page has more lines", and the check is whether
coverage tracks the line count.

This is a diagnostic.  Whatever it finds about a better-defined threshold has to be
pre-registered for a future run and not used to re-score stage 1, which was judged under
the threshold chosen before it ran.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_attention_localization import (  # noqa: E402
    load_jsonl,
    load_pages,
    score_page,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--layers", type=int, nargs="+", default=[8])
    parser.add_argument("--aggregate", default="mean", choices=["mean", "best"])
    parser.add_argument("--check-pages", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def correlation(pairs: list[tuple[float, float]]) -> float | None:
    """Pearson correlation, or ``None`` when either side has no spread."""

    if len(pairs) < 3:
        return None
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    if st.pstdev(xs) == 0 or st.pstdev(ys) == 0:
        return None
    mx, my = st.mean(xs), st.mean(ys)
    covariance = sum((x - mx) * (y - my) for x, y in pairs)
    return covariance / (
        len(pairs) * st.pstdev(xs) * st.pstdev(ys)
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    reports = {row["page_id"]: row for row in load_jsonl(args.probe)}
    predictions = {row["page_id"]: row for row in load_jsonl(args.predictions)}
    manifest = {row["page_id"]: row for row in load_jsonl(args.manifest)}
    keep = load_pages(args.check_pages)

    pages: list[dict[str, Any]] = []
    for page_id, report in reports.items():
        if keep is not None and page_id not in keep:
            continue
        record = manifest.get(page_id)
        prediction = predictions.get(page_id)
        if record is None or prediction is None or not record.get("characters"):
            continue
        page = score_page(
            report,
            prediction["reference"],
            prediction.get("prediction", ""),
            record.get("regions") or [],
            record["characters"],
            layers=args.layers,
            heads=None,  # the run recorded only the frozen heads
            aggregate=args.aggregate,
        )
        rows = page["rows"]
        if not rows:
            continue
        regions = page["num_regions"] or 1
        confidences = [row["confidence"] for row in rows]
        pages.append(
            {
                "page_id": page_id,
                "regions": regions,
                "scored": len(rows),
                "coverage_0_5": sum(1 for c in confidences if c >= 0.5) / len(rows),
                "median_confidence": st.median(confidences),
                "mean_confidence": st.mean(confidences),
                # How many times above a uniform distribution over the page's lines the
                # readout concentrates.  This is the scale-free version of the same
                # quantity, and it is the one that can be compared across pages.
                "multiple_of_uniform": st.mean(confidences) * regions,
                "rows": rows,
            }
        )

    if not pages:
        raise SystemExit("no pages with a character channel were scored")

    all_rows = [row for page in pages for row in page["rows"]]
    total = len(all_rows)
    result: dict[str, Any] = {
        "pages": len(pages),
        "scored": total,
        "coverage_0_5": sum(1 for r in all_rows if r["confidence"] >= 0.5) / total,
        "accuracy_overall": sum(1 for r in all_rows if r["pred"] == r["truth"]) / total,
        "regions_median": st.median([page["regions"] for page in pages]),
        "per_page": [
            {key: value for key, value in page.items() if key != "rows"} for page in pages
        ],
    }
    # Does coverage track the line count, as the metric's construction predicts?
    result["correlation_coverage_vs_regions"] = correlation(
        [(page["regions"], page["coverage_0_5"]) for page in pages]
    )
    result["correlation_confidence_vs_regions"] = correlation(
        [(page["regions"], page["median_confidence"]) for page in pages]
    )
    # The scale-free readout should be far more stable across pages if the confound is real.
    result["multiple_of_uniform_median"] = st.median(
        [page["multiple_of_uniform"] for page in pages]
    )
    result["correlation_multiple_vs_regions"] = correlation(
        [(page["regions"], page["multiple_of_uniform"]) for page in pages]
    )

    coverages = sorted(page["coverage_0_5"] for page in pages)
    result["per_page_coverage"] = {
        "min": coverages[0],
        "p25": coverages[len(coverages) // 4],
        "median": st.median(coverages),
        "p75": coverages[3 * len(coverages) // 4],
        "max": coverages[-1],
        "pages_at_or_above_0_5": sum(1 for c in coverages if c >= 0.5),
    }

    # By line count, so a page-level confound shows up as a trend rather than as noise.
    buckets: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        regions = page["regions"]
        label = "<=16" if regions <= 16 else "17-24" if regions <= 24 else "25-32" if regions <= 32 else ">32"
        buckets.setdefault(label, []).append(page)
    result["by_region_bucket"] = {}
    for label in ("<=16", "17-24", "25-32", ">32"):
        group = buckets.get(label)
        if not group:
            continue
        rows = [row for page in group for row in page["rows"]]
        result["by_region_bucket"][label] = {
            "pages": len(group),
            "scored": len(rows),
            "coverage_0_5": sum(1 for r in rows if r["confidence"] >= 0.5) / len(rows),
            "accuracy": sum(1 for r in rows if r["pred"] == r["truth"]) / len(rows),
            "median_regions": st.median([page["regions"] for page in group]),
        }

    # The trade-off the threshold sits on: for each bar, how much is covered and how
    # accurate is what is covered.  A definition worth pre-registering is one with a
    # point that clears both targets at once.
    sweep = []
    for step in range(1, 20):
        threshold = step / 20
        subset = [row for row in all_rows if row["confidence"] >= threshold]
        sweep.append(
            {
                "threshold": threshold,
                "coverage": len(subset) / total,
                "accuracy": (
                    sum(1 for row in subset if row["pred"] == row["truth"]) / len(subset)
                    if subset
                    else None
                ),
            }
        )
    result["threshold_sweep"] = sweep

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")

    cover = result["per_page_coverage"]
    print(f"pages {len(pages)}  scored {total}  overall coverage@0.5 "
          f"{result['coverage_0_5']:.4f}  accuracy {result['accuracy_overall']:.4f}")
    print(f"regions per page: median {result['regions_median']:.1f}")
    print()
    print("per-page coverage@0.5:")
    print(f"  min {cover['min']:.3f}  p25 {cover['p25']:.3f}  median {cover['median']:.3f}  "
          f"p75 {cover['p75']:.3f}  max {cover['max']:.3f}")
    print(f"  pages at or above 0.5: {cover['pages_at_or_above_0_5']}/{len(pages)}")
    print()
    print("is coverage tracking the line count?")
    print(f"  corr(coverage@0.5, regions)      {result['correlation_coverage_vs_regions']}")
    print(f"  corr(median confidence, regions) {result['correlation_confidence_vs_regions']}")
    print(f"  corr(multiple-of-uniform, regions) {result['correlation_multiple_vs_regions']}")
    print(f"  median multiple-of-uniform       {result['multiple_of_uniform_median']:.2f}x")
    print()
    print("by line count:")
    header = f"  {'bucket':>7} {'pages':>6} {'scored':>7} {'coverage':>9} {'accuracy':>9} {'med regions':>12}"
    print(header)
    for label, stats in result["by_region_bucket"].items():
        print(f"  {label:>7} {stats['pages']:6d} {stats['scored']:7d} "
              f"{stats['coverage_0_5']:9.4f} {stats['accuracy']:9.4f} "
              f"{stats['median_regions']:12.1f}")
    print()
    print("threshold sweep (coverage, accuracy of the covered):")
    for point in sweep:
        acc = point["accuracy"]
        mark = ""
        if point["coverage"] >= 0.5 and acc is not None and acc >= 0.9:
            mark = "  <- clears both targets"
        print(f"  t={point['threshold']:.2f}  coverage {point['coverage']:.4f}  "
              f"accuracy {'n/a' if acc is None else f'{acc:.4f}'}{mark}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
