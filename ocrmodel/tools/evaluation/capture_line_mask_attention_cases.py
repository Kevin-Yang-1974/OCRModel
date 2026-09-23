#!/usr/bin/env python3
"""Capture final-layer visual-patch attention for archived validation corrections.

The selected examples and target character boxes are post-hoc analysis inputs only.
Generation still receives the page image and fixed OCR prompt; no references or boxes
are injected into either model. The saved map is the mean over heads of the final
decoder layer's attention to visual patch keys, plus its unnormalised visual mass.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def compact(text: str) -> str:
    return "".join(text.split())


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["page_id"])] = row
    return rows


def char_to_generation_step(token_ids: list[int], tokenizer: Any) -> tuple[list[int], str]:
    """Map whitespace-free decoded character positions to generation forward steps."""

    decoded = ""
    steps: list[int] = []
    for step, _ in enumerate(token_ids):
        current = compact(tokenizer.decode(token_ids[: step + 1], skip_special_tokens=True))
        common = 0
        limit = min(len(decoded), len(current))
        while common < limit and decoded[common] == current[common]:
            common += 1
        steps = steps[:common] + [step] * max(0, len(current) - common)
        decoded = current
    return steps, decoded


def install_sdpa_attention_recorder(attention_module: Any, torch: Any) -> tuple[Any, Any]:
    """Keep SDPA outputs unchanged and expose last-query probabilities for maps."""

    transformer_module = importlib.import_module(attention_module.__class__.__module__)
    registry = transformer_module.ALL_ATTENTION_FUNCTIONS
    original_sdpa = registry["sdpa"]

    def sdpa_with_last_query(module: Any, query: Any, key: Any, value: Any,
                             attention_mask: Any, **kwargs: Any) -> tuple[Any, Any]:
        output, _ = original_sdpa(
            module, query, key, value, attention_mask, **kwargs
        )
        if module is not attention_module:
            return output, None

        key_states = transformer_module.repeat_kv(key, module.num_key_value_groups)
        query_last = query[:, :, -1:, :]
        scores = torch.matmul(query_last, key_states.transpose(2, 3))
        scores = scores * kwargs.get("scaling", module.scaling)
        if attention_mask is not None:
            mask = attention_mask
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            if mask.shape[-2] != 1:
                mask = mask[..., -1:, :]
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + mask
        weights = torch.nn.functional.softmax(
            scores, dim=-1, dtype=torch.float32
        ).to(query.dtype)
        return output, weights

    registry["sdpa"] = sdpa_with_last_query
    return registry, original_sdpa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--mask-checkpoint", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--routed-predictions", type=Path, required=True)
    parser.add_argument("--selected-cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hard-token-cap", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    code_root = args.code_root.resolve()
    sys.path.insert(0, str(code_root / "src"))
    sys.path.insert(0, str(code_root / "tools" / "evaluation"))

    import torch

    from evaluate_window_mask_routing import eos_ids, load_backbone
    from layout_ocr.data import prepare_inference_inputs
    from layout_ocr.line_mask_head import LineMaskConfig, LineMaskHead
    from layout_ocr.line_mask_runtime import LineMaskRuntime

    selected = json.loads(args.selected_cases.read_text(encoding="utf-8"))
    if selected.get("source") != "validation archive: baseline-pred + formal-20260923/validation-epoch8":
        raise ValueError("selected cases must come from the registered validation archive")
    cases = selected.get("cases") or []
    if not cases or any(case.get("split") != "validation" for case in cases):
        raise ValueError("selected cases must be non-empty and validation-only")
    if selected.get("test_used_for_selection") is not False:
        raise ValueError("test split must not be used to select cases")

    manifest = load_jsonl(args.validation_manifest)
    baseline = load_jsonl(args.baseline_predictions)
    routed = load_jsonl(args.routed_predictions)
    by_page: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        page_id = str(case["page_id"])
        if page_id not in manifest or page_id not in baseline or page_id not in routed:
            raise KeyError(f"validation archive is missing page {page_id}")
        if manifest[page_id].get("split") != "validation":
            raise ValueError(f"non-validation page selected: {page_id}")
        by_page.setdefault(page_id, []).append(case)

    device = torch.device(args.device)
    model, processor = load_backbone(args.model_path, args.backbone_checkpoint, device)
    checkpoint = torch.load(args.mask_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 8 or checkpoint.get("step") != 3456:
        raise ValueError("expected the validation-selected epoch-8/step-3456 head")
    head = LineMaskHead(LineMaskConfig(**checkpoint["config"])).to(device)
    head.load_state_dict(checkpoint["head"], strict=True)
    head.eval()
    runtime = LineMaskRuntime(model, processor.tokenizer, head)

    last_attention = runtime.text.layers[-1].self_attn
    attention_registry, original_sdpa = install_sdpa_attention_recorder(last_attention, torch)
    image_token_id = runtime.bridge.image_token_id
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=False)
    results: list[dict[str, Any]] = []

    try:
        for page_number, (page_id, page_cases) in enumerate(sorted(by_page.items()), start=1):
            record = manifest[page_id]
            inputs = prepare_inference_inputs(
                processor, {"image_path": record["image_path"]}, device
            )
            visual_positions = (inputs["input_ids"][0] == image_token_id).nonzero().flatten()
            grid_h, grid_w = (int(value) for value in inputs["image_grid_thw"][0, 1:].tolist())
            merge = runtime.merge
            grid_shape = (grid_h // merge, grid_w // merge)
            if visual_positions.numel() != grid_shape[0] * grid_shape[1]:
                raise ValueError(
                    f"{page_id}: visual-token count {visual_positions.numel()} does not match grid {grid_shape}"
                )

            arm_results: dict[str, dict[str, Any]] = {}
            for arm, enabled, archived in (
                ("baseline", False, baseline[page_id]["prediction"]),
                ("line_mask", True, routed[page_id]["prediction"]),
            ):
                char_indices = [
                    int(case["baseline_i"] if arm == "baseline" else case["mask_i"])
                    for case in page_cases
                ]
                needed_char = max(char_indices)
                max_new_tokens = min(
                    args.hard_token_cap,
                    max(64, 2 * (needed_char + 1) + 32),
                )
                maps: list[np.ndarray] = []

                def capture_patch_attention(module: Any, inputs_: Any, output: Any) -> None:
                    weights = output[1]
                    if weights is None:
                        raise RuntimeError("SDPA diagnostic did not return final-layer probabilities")
                    # Output step t predicts generated token t; for prefill use the
                    # final prompt query, and for cached decode use its sole query.
                    patch_mass = weights[0, :, -1, visual_positions].float().mean(dim=0)
                    maps.append(patch_mass.detach().cpu().numpy().astype(np.float32, copy=False))

                handle = last_attention.register_forward_hook(capture_patch_attention)
                try:
                    runtime.enabled = enabled
                    runtime.set_page(inputs, page_id)
                    with torch.inference_mode():
                        generated = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            use_cache=True,
                            eos_token_id=eos_ids(model, processor),
                        )
                    prompt_length = int(inputs["input_ids"].shape[1])
                    token_ids = [int(value) for value in generated[0, prompt_length:].tolist()]
                    char_steps, generated_text = char_to_generation_step(
                        token_ids, processor.tokenizer
                    )
                finally:
                    handle.remove()

                archived_text = compact(archived)
                if len(generated_text) <= needed_char or len(archived_text) <= needed_char:
                    raise RuntimeError(
                        f"{page_id}/{arm}: generation ended before target character {needed_char}"
                    )
                prefix_equal = generated_text[: needed_char + 1] == archived_text[: needed_char + 1]
                if not prefix_equal:
                    raise RuntimeError(
                        f"{page_id}/{arm}: replay prefix differs from archived validation output "
                        f"through character {needed_char}"
                    )
                if len(maps) != len(token_ids):
                    raise RuntimeError(
                        f"{page_id}/{arm}: captured {len(maps)} attention rows for {len(token_ids)} generated tokens"
                    )

                raw_by_case: dict[str, dict[str, Any]] = {}
                for case in page_cases:
                    char_index = int(
                        case["baseline_i"] if arm == "baseline" else case["mask_i"]
                    )
                    if char_index >= len(char_steps):
                        raise RuntimeError(f"{page_id}/{arm}: no token mapping for char {char_index}")
                    step = char_steps[char_index]
                    raw = np.asarray(maps[step], dtype=np.float32)
                    mass = float(raw.sum())
                    if not np.isfinite(raw).all() or not np.isfinite(mass) or mass <= 0:
                        raise FloatingPointError(f"{page_id}/{arm}: non-finite/empty patch attention")
                    conditional = raw / mass
                    expected_char = str(case["baseline_char"] if arm == "baseline" else case["gt"])
                    if generated_text[char_index] != expected_char:
                        raise RuntimeError(
                            f"{page_id}/{arm}: selected character differs from archive metadata"
                        )
                    raw_by_case[str(case["case_id"])] = {
                        "raw": raw,
                        "conditional": conditional,
                        "visual_mass": mass,
                        "generation_step": int(step),
                        "token_id": int(token_ids[step]),
                        "generated_char": generated_text[char_index],
                        "char_index": char_index,
                        "prefix_matches_archive": True,
                    }
                arm_results[arm] = raw_by_case
                print(
                    json.dumps(
                        {
                            "page": page_number,
                            "pages": len(by_page),
                            "page_id": page_id,
                            "arm": arm,
                            "captured_steps": len(maps),
                            "grid": grid_shape,
                            "prefix_matches_archive": prefix_equal,
                        }
                    ),
                    flush=True,
                )

            for case in page_cases:
                case_id = str(case["case_id"])
                baseline_map = arm_results["baseline"][case_id]
                routed_map = arm_results["line_mask"][case_id]
                npz_path = output_root / f"case-{case_id}-{page_id}.npz"
                np.savez_compressed(
                    npz_path,
                    baseline_raw=baseline_map["raw"],
                    baseline_conditional=baseline_map["conditional"],
                    routed_raw=routed_map["raw"],
                    routed_conditional=routed_map["conditional"],
                    grid_shape=np.asarray(grid_shape, dtype=np.int32),
                )
                results.append(
                    {
                        "case_id": case_id,
                        "page_id": page_id,
                        "kind": case["kind"],
                        "target_char": case["gt"],
                        "reference_char_index": case["ref_i"],
                        "bbox": case["bbox"],
                        "baseline": {
                            key: value
                            for key, value in baseline_map.items()
                            if key not in {"raw", "conditional"}
                        },
                        "line_mask": {
                            key: value
                            for key, value in routed_map.items()
                            if key not in {"raw", "conditional"}
                        },
                        "grid_shape": list(grid_shape),
                        "attention_definition": "mean last-layer head probability over image patch keys; raw sum is total visual mass; conditional map sums to one",
                        "attention_backend": "SDPA output unchanged; last-query probabilities recomputed for visualization",
                        "generation_prefix_verified": True,
                        "split": "validation",
                        "test_used_for_selection": False,
                    }
                )
    finally:
        attention_registry["sdpa"] = original_sdpa
        runtime.remove()

    (output_root / "capture_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "source_split": "validation",
                "source_archive": str(args.selected_cases),
                "pages": len(by_page),
                "cases": len(results),
                "test_used_for_selection": False,
                "records": results,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
