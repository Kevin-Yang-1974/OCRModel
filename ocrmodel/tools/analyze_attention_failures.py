#!/usr/bin/env python3
"""Locate bad-generation events and align them with decoder attention steps.

This is a text-level diagnostic.  MTHv2's page manifest used by the baseline run
has regions but no character boxes, so a dropped reference character cannot be
assigned to an image column without inventing a spatial label.  The tool still
aligns reference/prediction strings to the probe's emitted fragments and reports
the visual-attention reductions at nearby missing-character and consecutive-cycle
steps.  When a selected page has a retained NPZ sample, it renders the nearest
available patch heatmap for each event.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from analyze_attention_localization import align, neighbourhood_flags, page_steps


DEFAULT_LAYERS = (0, 4, 8, 12)
DEFAULT_HEADS = (2, 3, 8, 10, 11, 12, 14, 15)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite_mean(values: Iterable[Any]) -> float | None:
    numbers: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return sum(numbers) / len(numbers) if numbers else None


def edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[right_index - 1] + 1,
                    previous[right_index - 1] + int(left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def cycle_segments(text: str, *, max_period: int = 64) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    index = 0
    while index < len(text):
        best: tuple[int, int] | None = None
        for period in range(1, max_period + 1):
            if index + period * 3 > len(text):
                continue
            unit = text[index : index + period]
            repeats = 1
            while text[index + repeats * period : index + (repeats + 1) * period] == unit:
                repeats += 1
            if repeats >= 3 and (best is None or period * repeats > best[0] * best[1]):
                best = (period, repeats)
        if best is None:
            index += 1
            continue
        period, repeats = best
        segments.append(
            {
                "start": index,
                "end": index + period * repeats,
                "period": period,
                "repeats": repeats,
                "unit": text[index : index + period],
            }
        )
        index += period * repeats
    return segments


def page_quality(row: dict[str, Any]) -> dict[str, Any]:
    reference = str(row.get("reference") or "")
    prediction = str(row.get("prediction") or "")
    distance = edit_distance(prediction, reference)
    cycle = cycle_segments(prediction)
    repeated = bool(row.get("repeated_cycle_detected")) or bool(cycle)
    limit_hit = bool(row.get("generation_limit_hit"))
    return {
        "page_id": str(row["page_id"]),
        "reference_chars": len(reference),
        "prediction_chars": len(prediction),
        "edit_distance": distance,
        "cer": distance / max(1, len(reference)),
        "generation_length": int(row.get("generation_length") or len(prediction)),
        "generation_limit_hit": limit_hit,
        "generation_eos_hit": bool(row.get("generation_eos_hit")),
        "repeated_cycle_detected": repeated,
        "repeated_cycle_rate": float(row.get("repeated_cycle_rate") or 0.0),
        "loop_continuation": row.get("loop_continuation"),
        "cycle_segments": cycle,
        "density_bucket": row.get("density_bucket"),
        "reference": reference,
        "prediction": prediction,
    }


def step_rows(report: dict[str, Any], layers: set[int], heads: set[int]) -> dict[int, dict[str, Any]]:
    grouped = page_steps(report)
    result: dict[int, dict[str, Any]] = {}
    for step, record in grouped.items():
        selected = [
            row
            for row in record.get("heads", [])
            if int(row.get("layer", -1)) in layers and int(row.get("head", -1)) in heads
        ]
        result[step] = {"emitted": record.get("emitted"), "rows": selected}
    return result


def scalar_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "heads": len(rows),
        "visual_mass_m_t": finite_mean(row.get("m_t") for row in rows),
        "entropy_norm": finite_mean(row.get("entropy_norm") for row in rows),
        "visual_text_logit_gap": finite_mean(
            (float(row["lse_vis"]) - float(row["lse_text"]))
            if row.get("lse_vis") is not None and row.get("lse_text") is not None
            else None
            for row in rows
        ),
    }


def metric_delta(values: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in ("visual_mass_m_t", "entropy_norm", "visual_text_logit_gap"):
        value = values.get(key)
        reference = baseline.get(key)
        if value is None or reference is None:
            delta[key] = None
        else:
            delta[key] = float(value) - float(reference)
    return delta


def step_context(
    steps: dict[int, dict[str, Any]], center: int, radius: int
) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for step in sorted(steps):
        if abs(step - center) > radius:
            continue
        context.append(
            {
                "step": step,
                "emitted": steps[step].get("emitted"),
                "attention": scalar_metrics(steps[step].get("rows", [])),
            }
        )
    return context


def nearest_step_for_reference(
    mapping: list[int | None], char_steps: list[int], reference_index: int, steps: list[int]
) -> int:
    candidates = [
        (abs(int(mapped) - reference_index), char_steps[index])
        for index, mapped in enumerate(mapping)
        if mapped is not None and index < len(char_steps)
    ]
    if candidates:
        return min(candidates)[1]
    return min(steps, key=lambda step: abs(step - reference_index)) if steps else 0


def snippet(text: str, index: int, radius: int = 12) -> str:
    start = max(0, index - radius)
    end = min(len(text), index + radius + 1)
    return text[start:end]


def page_events(
    quality: dict[str, Any],
    report: dict[str, Any],
    layers: set[int],
    heads: set[int],
    *,
    radius: int,
    max_events: int,
) -> dict[str, Any]:
    steps = step_rows(report, layers, heads)
    ordered_steps = sorted(steps)
    fragments = [steps[step].get("emitted") or "" for step in ordered_steps]
    generated = "".join(fragments)
    reference = quality["reference"]
    mapping = align(generated, reference)
    char_steps: list[int] = []
    for step, fragment in zip(ordered_steps, fragments):
        char_steps.extend([step] * len(fragment))
    step_char_positions: defaultdict[int, list[int]] = defaultdict(list)
    for char_index, step in enumerate(char_steps):
        step_char_positions[step].append(char_index)
    page_attention = scalar_metrics(
        [row for step in ordered_steps for row in steps[step].get("rows", [])]
    )
    near_dropped, near_repeated = neighbourhood_flags(generated, reference, mapping, radius)

    mapped = {index for index in mapping if index is not None}
    dropped = [index for index in range(len(reference)) if index not in mapped]
    dropped_step_counts: Counter[int] = Counter()
    dropped_step_refs: defaultdict[int, list[int]] = defaultdict(list)
    for reference_index in dropped:
        step = nearest_step_for_reference(mapping, char_steps, reference_index, ordered_steps)
        dropped_step_counts[step] += 1
        dropped_step_refs[step].append(reference_index)

    missing_events: list[dict[str, Any]] = []
    for step, _count in dropped_step_counts.most_common(max_events):
        references = dropped_step_refs[step]
        reference_index = references[0]
        missing_events.append(
            {
                "type": "missing_reference_character",
                "step": step,
                "reference_index": reference_index,
                "missing_char": reference[reference_index],
                "missing_count_near_step": len(references),
                "reference_context": snippet(reference, reference_index),
                "generated_context": snippet(
                    generated,
                    step_char_positions.get(step, [0])[0] if generated else 0,
                ),
                "attention": scalar_metrics(steps.get(step, {}).get("rows", [])),
                "attention_delta_to_page_mean": metric_delta(
                    scalar_metrics(steps.get(step, {}).get("rows", [])), page_attention
                ),
                "context": step_context(steps, step, radius),
            }
        )

    cycles = quality["cycle_segments"]
    loop_events: list[dict[str, Any]] = []
    for cycle in cycles[:max_events]:
        start = int(cycle["start"])
        end = min(len(char_steps), int(cycle["end"]))
        center_step = char_steps[start] if start < len(char_steps) else (ordered_steps[-1] if ordered_steps else 0)
        loop_events.append(
            {
                "type": "consecutive_cycle",
                "step": center_step,
                "generated_start": start,
                "generated_end": end,
                "period": cycle["period"],
                "repeats": cycle["repeats"],
                "unit": cycle["unit"],
                "generated_context": generated[max(0, start - 12) : min(len(generated), end + 12)],
                "attention": scalar_metrics(steps.get(center_step, {}).get("rows", [])),
                "attention_delta_to_page_mean": metric_delta(
                    scalar_metrics(steps.get(center_step, {}).get("rows", [])), page_attention
                ),
                "context": step_context(steps, center_step, radius),
            }
        )

    return {
        "generated_chars_from_probe": len(generated),
        "steps": len(ordered_steps),
        "dropped_reference_chars": len(dropped),
        "repeated_generated_chars": sum(
            1 for flag in near_repeated if flag
        ),
        "near_dropped_generated_chars": sum(1 for flag in near_dropped if flag),
        "near_repeated_generated_chars": sum(1 for flag in near_repeated if flag),
        "page_attention_mean": page_attention,
        "missing_events": missing_events,
        "loop_events": loop_events,
    }


def choose_pages(qualities: list[dict[str, Any]], top_pages: int) -> tuple[list[str], dict[str, list[str]]]:
    by_id = {row["page_id"]: row for row in qualities}
    worst = sorted(qualities, key=lambda row: (row["cer"], row["edit_distance"]), reverse=True)[:top_pages]
    loops = sorted(
        qualities,
        key=lambda row: (
            int(row["generation_limit_hit"]),
            int(row["repeated_cycle_detected"]),
            row["repeated_cycle_rate"],
            row["generation_length"],
        ),
        reverse=True,
    )[:top_pages]
    reasons: defaultdict[str, list[str]] = defaultdict(list)
    for row in worst:
        reasons[row["page_id"]].append("worst_cer")
    for row in loops:
        if row["generation_limit_hit"] or row["repeated_cycle_detected"]:
            reasons[row["page_id"]].append("loop_or_limit")
    selected = list(reasons)
    return selected, dict(reasons)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heatmap-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--top-pages", type=int, default=4)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--heads", type=int, nargs="+", default=list(DEFAULT_HEADS))
    parser.add_argument("--neighbourhood", type=int, default=3)
    parser.add_argument("--max-events-per-page", type=int, default=3)
    parser.add_argument("--max-edge", type=int, default=2400)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_jsonl(args.predictions)
    qualities = [page_quality(row) for row in predictions]
    selected_ids, reasons = choose_pages(qualities, max(1, args.top_pages))

    npz_maps: list[dict[str, Any]] = []
    if args.heatmap_dir and args.heatmap_dir.is_dir():
        from analyze_decoder_attention import load_npz_maps

        npz_maps = load_npz_maps(args.heatmap_dir)
        for item in npz_maps:
            page_id = item["page_id"]
            if page_id not in reasons:
                reasons[page_id] = ["pre_registered_heatmap"]
                selected_ids.append(page_id)

    selected_set = set(selected_ids)
    reports: dict[str, dict[str, Any]] = {}
    with args.probe.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            report = json.loads(line)
            page_id = str(report.get("page_id"))
            if page_id in selected_set:
                reports[page_id] = report

    quality_by_id = {row["page_id"]: row for row in qualities}
    pages: list[dict[str, Any]] = []
    for page_id in selected_ids:
        quality = quality_by_id[page_id]
        page = dict(quality)
        page["selection_reasons"] = reasons[page_id]
        if page_id in reports:
            page["event_alignment"] = page_events(
                quality,
                reports[page_id],
                set(args.layers),
                set(args.heads),
                radius=args.neighbourhood,
                max_events=args.max_events_per_page,
            )
        else:
            page["event_alignment"] = {"status": "probe_page_missing"}
        page.pop("reference", None)
        page.pop("prediction", None)
        pages.append(page)

    overlays: list[dict[str, Any]] = []
    if npz_maps and args.manifest:
        from analyze_decoder_attention import render_overlay, safe_name

        manifest = {str(row["page_id"]): row for row in load_jsonl(args.manifest)}
        for item in npz_maps:
            page_id = item["page_id"]
            page = next((row for row in pages if row["page_id"] == page_id), None)
            if page is None or page.get("event_alignment", {}).get("status"):
                continue
            events = page["event_alignment"].get("missing_events", []) + page["event_alignment"].get("loop_events", [])
            layer_indices = [
                index for index, layer in enumerate(item["layers"].tolist()) if int(layer) == 8
            ]
            if not layer_indices or page_id not in manifest:
                continue
            for event in events:
                target = int(event["step"])
                index = min(layer_indices, key=lambda candidate: abs(int(item["steps"][candidate]) - target))
                sampled = int(item["steps"][index])
                output = args.output_dir / "failure_visualizations" / (
                    f"{safe_name(page_id)}_{event['type']}_target{target:04d}_sample{sampled:04d}.png"
                )
                rendered = render_overlay(
                    Path(manifest[page_id]["image_path"]),
                    item["positions"],
                    item["attention"][index],
                    output,
                    title=f"layer=8 target={target} sample={sampled} {event['type']}",
                    max_edge=args.max_edge,
                )
                overlays.append(
                    {
                        "page_id": page_id,
                        "event_type": event["type"],
                        "target_step": target,
                        "sampled_step": sampled,
                        "step_delta": sampled - target,
                        **rendered,
                    }
                )

    result = {
        "status": "complete",
        "pages_total": len(qualities),
        "pages_selected": len(pages),
        "layers": args.layers,
        "heads": args.heads,
        "text_alignment_note": "MTHv2 page manifest has no character boxes; missing-character events are text-aligned and are not assigned to image columns.",
        "pages": pages,
        "failure_visualizations": overlays,
    }
    output = args.output_dir / "attention_failures.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "attention_failure_analysis_complete",
                "pages_total": len(qualities),
                "pages_selected": len(pages),
                "overlays": len(overlays),
                "output": str(output),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
