"""Check the probe's recomputed attention against the model's own eager weights.

Stage 0 of ``plans/LAYOUT_ATTENTION_TRACKING.md`` requires this before any of the
probe's numbers is believed.  Everything the probe reports rests on one unverified
claim: that re-running ``q_proj``/``k_proj``, applying the module's rotary embedding and
scaling by ``head_dim**-0.5`` reproduces the logits the model actually attended with.
If that is wrong the probe still produces a full, smooth, plausible-looking set of
numbers -- it is a measurement, so being wrong and being right look the same.

## Why the prefill, and why eager

The probe's visual keys are captured during the **prefill** and reused for every later
decode step, so the prefill is precisely where the transform has to be right.  A decode
step would only test the case the probe derives from the prefill; testing the prefill
tests the source.

``attn_implementation="eager"`` is what makes the model's own weights visible at all:
the sdpa path never materializes them.  That means this comparison is eager against a
recomputation of the same eager forward, so any difference is the transform's, not a
kernel's -- which is the point.  The kernel path used in a real run (sdpa) is not
compared here, and a difference between sdpa and eager on this model is a separate
question the routing work already measured as nil for the mask path.

The visual-conditional distribution and the total mass are compared per head and per
query position.  ``m_t`` is the sensitive one: it is a ratio over the *whole* key span,
so a missing term in the text half or a wrong scale on the mask moves it even when the
visual half is perfect.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from layout_ocr.attention_probe import (  # noqa: E402
    AttentionProbe,
    _find_attention_modules,
    probe_layers,
)
from layout_ocr.data import load_records, prepare_inference_inputs  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--page-index", type=int, default=0)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--query-positions", type=int, default=0,
                        help="compare only the first N query positions; 0 compares all")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _configure_processor(processor: Any, max_pixels: int) -> None:
    """Same resolution setting the eval applies, so the sequence lengths match a run."""

    from layout_ocr.train_screen import configure_processor

    configure_processor(processor, max_pixels)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True, local_files_only=True)
    _configure_processor(processor, args.max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="eager",  # the only path that returns its weights
        local_files_only=True,
    )
    model.to(device).eval()

    records = load_records(args.manifest)
    record = records[args.page_index]
    inputs = prepare_inference_inputs(processor, record, device)

    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(model.config, "text_config").image_token_id
    positions = (inputs["input_ids"][0] == int(image_token_id)).nonzero().flatten()
    visual_start = int(positions[0].item())
    visual_count = int(positions.numel())
    prompt_length = int(inputs["input_ids"].shape[1])

    layers = tuple(args.layers) if args.layers else probe_layers()
    modules = dict(_find_attention_modules(model))
    missing = [layer for layer in layers if layer not in modules]
    if missing:
        raise SystemExit(f"model has no layers {missing}")

    captured: dict[int, dict[str, Any]] = {}

    def capture(layer: int):
        def hook(module, hook_args, hook_kwargs, output):
            weights = output[1] if isinstance(output, tuple) and len(output) > 1 else None
            hidden = hook_kwargs.get("hidden_states")
            if hidden is None and hook_args:
                hidden = hook_args[0]
            captured[layer] = {
                "weights": weights,
                "hidden": hidden,
                "position_embeddings": hook_kwargs.get("position_embeddings"),
                "mask": hook_kwargs.get("attention_mask"),
            }
        return hook

    handles = [
        modules[layer].register_forward_hook(capture(layer), with_kwargs=True) for layer in layers
    ]
    with torch.no_grad():
        model(**inputs)
    for handle in handles:
        handle.remove()

    # Replicate the probe's path over exactly the tensors that forward used.
    probe = AttentionProbe(None, int(image_token_id), layers=layers)
    results: dict[str, Any] = {
        "page_id": record["page_id"],
        "prompt_length": prompt_length,
        "visual_start": visual_start,
        "visual_count": visual_count,
        "layers": list(layers),
        "per_layer": {},
    }
    for layer in layers:
        entry = captured.get(layer) or {}
        weights = entry.get("weights")
        if weights is None:
            results["per_layer"][str(layer)] = {
                "status": "no_weights",
                "note": "the eager forward did not return attention weights for this layer",
            }
            continue
        hidden = entry["hidden"]
        # The probe's own shape discovery, projection and rotary application, on the
        # prefill's inputs.
        probe._read_shape(modules[layer], layer)
        projected = probe._project(modules[layer], layer, hidden, entry["position_embeddings"])
        if projected is None:
            results["per_layer"][str(layer)] = {
                "status": "transform_failed",
                "reason": probe.report()["transform_failed"].get(str(layer)),
            }
            continue
        query, key = projected
        # Store the keys the way the prefill does, then reduce -- so the comparison
        # covers _store_prompt and _split_mask as well as _reduce.
        probe.visual_start = visual_start
        probe.visual_count = visual_count
        probe._store_prompt(layer, key)
        mask = entry["mask"]
        mask_vis = mask_text = None
        if mask is not None:
            mask_vis, mask_text = probe._split_mask(mask, layer)
        visual = probe._visual_keys[layer]
        mass, dist, _lse_vis, _lse_text = probe._reduce(
            layer, query, visual, probe._text_keys(layer), mask_vis, mask_text
        )

        # The model's own weights: [1, num_heads, q_len, kv_len], already softmaxed and
        # already masked, over the same key span.
        real = weights.float()
        if real.shape[-1] != prompt_length:
            results["per_layer"][str(layer)] = {
                "status": "length_mismatch",
                "weights_kv": int(real.shape[-1]),
                "prompt_length": prompt_length,
            }
            continue
        span = slice(visual_start, visual_start + visual_count)
        real_vis = real[:, :, :, span]  # [1, heads, q_len, V]
        real_mass = real_vis.sum(dim=-1)  # [1, heads, q_len]
        real_dist = real_vis / real_mass.unsqueeze(-1).clamp_min(1e-12)

        heads = mass.shape[0]
        real_mass = real_mass[0, :heads]
        real_dist = real_dist[0, :heads]
        positions = slice(0, args.query_positions) if args.query_positions else slice(None)
        mine_mass = mass[:, positions]
        mine_dist = dist[:, positions]
        ref_mass = real_mass[:, positions]
        ref_dist = real_dist[:, positions]

        mass_error = (mine_mass - ref_mass).abs()
        dist_error = (mine_dist - ref_dist).abs()
        results["per_layer"][str(layer)] = {
            "status": "compared",
            "heads": int(heads),
            "query_positions": int(mine_mass.shape[1]),
            "dtype": str(dist.dtype),
            "weights_dtype": str(weights.dtype),
            "mass_max_abs_error": float(mass_error.max()),
            "mass_mean_abs_error": float(mass_error.mean()),
            "dist_max_abs_error": float(dist_error.max()),
            "dist_mean_abs_error": float(dist_error.mean()),
            "mass_max_ref": float(ref_mass.max()),
            "dist_row_sum_max_error": float((mine_dist.sum(-1) - 1).abs().max()),
        }

    text = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)

    compared = [v for v in results["per_layer"].values() if v.get("status") == "compared"]
    if not compared:
        print("\n没有一层比对成功 —— 变换未被验证。", file=sys.stderr)
        return 1
    worst = max(v["mass_max_abs_error"] for v in compared)
    worst_dist = max(v["dist_max_abs_error"] for v in compared)
    print(f"\n最大误差：m_t {worst:.3e}，a_vis {worst_dist:.3e}（bfloat16 logits）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
