"""Aggregate the write-back intervention matrix and test each arm against ``full``.

Reads the per-arm ``summary.json`` and probe streams written by
``tools/training/run_glmocr_layout_writeback_intervention_a100.sh``, prints the
matrix, and runs a page-paired bootstrap of each arm's CER against the baseline.

The pairing matters: all arms score the same pages, so the variance that matters is
the per-page difference, not the per-arm CER.  The bootstrap follows
``tools/analyze_cer_significance.py`` (resample pages jointly, 10k iterations).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_cer_significance import levenshtein  # noqa: E402

LABELS = {
    "full": "H (baseline)",
    "zero": "0 (floor, gate=0)",
    "global": "mean_p H broadcast (patch-common only)",
    "spatial": "H - mean_p H (spatial structure only)",
    "spatial_scaled": "H - mean_p H rescaled to full amplitude (confound removed)",
    "shuffle": "patch-axis permutation (content destroyed)",
    "noise": "flatness+norm matched Gaussian",
    "global_perm": "mean_p H with permuted hidden axis (direction destroyed)",
}


def page_edits(path: Path) -> dict[str, tuple[int, int]]:
    """Per-page (edit distance, reference length) for a prediction dump."""

    rows: dict[str, tuple[int, int]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            reference = "".join(record["reference"].split())
            prediction = "".join(record["prediction"].split())
            rows[record.get("page_id", "")] = (levenshtein(reference, prediction), len(reference))
    return rows


def cer(rows: dict[str, tuple[int, int]]) -> float:
    edits = sum(value[0] for value in rows.values())
    length = sum(value[1] for value in rows.values())
    return edits / length if length else float("nan")


def paired_bootstrap(
    baseline: dict[str, tuple[int, int]],
    arm: dict[str, tuple[int, int]],
    iterations: int,
    seed: int,
) -> dict:
    pages = sorted(set(baseline) & set(arm))
    if not pages:
        return {"pages": 0}
    differences = [
        (arm[page][0] - baseline[page][0], baseline[page][1]) for page in pages
    ]
    generator = random.Random(seed)
    observed_num = sum(d[0] for d in differences)
    observed_den = sum(d[1] for d in differences) or 1
    observed = observed_num / observed_den
    samples = []
    total = len(differences)
    for _ in range(iterations):
        numerator = 0
        denominator = 0
        for _ in range(total):
            edits, length = differences[generator.randrange(total)]
            numerator += edits
            denominator += length
        samples.append(numerator / (denominator or 1))
    samples.sort()
    lower = samples[int(0.025 * iterations)]
    upper = samples[int(0.975 * iterations) - 1]
    probability = sum(1 for value in samples if value <= 0) / iterations
    return {
        "pages": total,
        "delta_cer": observed,
        "ci": [lower, upper],
        "p_le_zero": probability,
        "significant": lower < 0 and upper < 0 or (lower > 0 and upper > 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline", default="full")
    args = parser.parse_args()

    arms_dir = args.root / "arms"
    payload: dict = {"status": "complete", "arms": {}}
    per_page: dict[str, dict[str, tuple[int, int]]] = {}

    for arm, label in LABELS.items():
        arm_dir = arms_dir / arm
        summary_path = arm_dir / "summary.json"
        if not summary_path.exists():
            payload["arms"][arm] = {"label": label, "status": "missing"}
            continue
        metrics = json.loads(summary_path.read_text(encoding="utf-8"))["validation"]
        record = {
            "label": label,
            "pages": metrics["pages"],
            "cer": metrics["cer"],
            "substitutions": metrics["substitutions"],
            "insertions": metrics["insertions"],
            "deletions": metrics["deletions"],
            "generation_limit_hits": metrics["generation_limit_hits"],
            "effective_residual_scale": metrics.get("effective_residual_scale"),
        }
        probe_path = arm_dir.with_name(arm_dir.name + ".probe.jsonl")
        if probe_path.exists():
            probe = [
                json.loads(line)
                for line in probe_path.read_text(encoding="utf-8").splitlines()
            ]
            if probe:
                record["probe"] = {
                    "records": len(probe),
                    "lc_flat": statistics.median([r["lc_flat"] for r in probe]),
                    "inj_over_vt": statistics.median(
                        [r["inj_over_vt"] for r in probe if r.get("inj_over_vt") is not None]
                    ),
                    "lc_global_share": statistics.median(
                        [r["lc_global_share"] for r in probe]
                    ),
                    "lc_spatial_share": statistics.median(
                        [r["lc_spatial_share"] for r in probe]
                    ),
                    "arms_seen": sorted({r["intervention"] for r in probe}),
                }
        attenuation_path = arm_dir.with_name(arm_dir.name + ".attenuation.jsonl")
        if attenuation_path.exists():
            rows = [
                json.loads(line)
                for line in attenuation_path.read_text(encoding="utf-8").splitlines()
            ]
            if rows:
                record["merger_attenuation"] = {
                    "records": len(rows),
                    "in_rel": statistics.median([r["in_rel"] for r in rows]),
                    "out_rel": statistics.median([r["out_rel"] for r in rows]),
                    "attenuation": statistics.median([r["attenuation"] for r in rows]),
                }
        payload["arms"][arm] = record
        predictions = arm_dir / "validation_predictions.jsonl"
        if predictions.exists():
            per_page[arm] = page_edits(predictions)

    print(
        f"{'arm':9s} {'CER':>10s} {'edits':>7s} {'scale':>7s} "
        f"{'lc_flat':>8s} {'inj/vt':>8s} {'gshare':>7s} {'sshare':>7s}"
    )
    for arm, row in payload["arms"].items():
        if row.get("status") == "missing":
            print(f"{arm:9s} {'MISSING':>10s}")
            continue
        probe = row.get("probe", {})
        edits = row["substitutions"] + row["insertions"] + row["deletions"]
        print(
            f"{arm:9s} {row['cer']:10.6f} {edits:7d} "
            f"{(row['effective_residual_scale'] or 0):7.4f} "
            f"{probe.get('lc_flat', float('nan')):8.4f} "
            f"{probe.get('inj_over_vt', float('nan')):8.4f} "
            f"{probe.get('lc_global_share', float('nan')):7.3f} "
            f"{probe.get('lc_spatial_share', float('nan')):7.3f}"
        )

    for arm, row in payload["arms"].items():
        if "merger_attenuation" in row:
            m = row["merger_attenuation"]
            print(
                f"seam ({arm}): in_rel={m['in_rel']:.5f} out_rel={m['out_rel']:.5f} "
                f"attenuation={m['attenuation']:.6f}"
            )

    if args.baseline in per_page:
        print()
        print(f"paired bootstrap vs {args.baseline} ({args.iterations} iterations)")
        for arm in LABELS:
            if arm == args.baseline or arm not in per_page:
                continue
            result = paired_bootstrap(
                per_page[args.baseline], per_page[arm], args.iterations, args.seed
            )
            payload["arms"][arm]["vs_baseline"] = result
            print(
                f"  {arm:9s} delta={result['delta_cer']:+.6f} "
                f"CI=[{result['ci'][0]:+.6f},{result['ci'][1]:+.6f}] "
                f"P(<=0)={result['p_le_zero']:.3f} "
                f"{'SIGNIFICANT' if result['significant'] else 'not significant'}"
            )

    (args.root / "intervention_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {args.root / 'intervention_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
