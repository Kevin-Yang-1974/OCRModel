#!/usr/bin/env python3
"""Evaluate GOT-OCR2.0 or GLM-OCR on MTHv2 column crops.

The input is the line/column JSONL produced by the MTHv2 AnandaSky converter.
Each model sees the same image and reference text; only the model-specific
processor is different.  An optional GLM adapter checkpoint lets this same
evaluator score the improved GLM arm against the untouched checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["got", "glm"], required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--got-source-root", type=Path)
    parser.add_argument(
        "--glm-checkpoint-dir",
        type=Path,
        help="Optional GLM layout/semantic adapter checkpoint directory.",
    )
    parser.add_argument(
        "--glm-metadata",
        type=Path,
        help="Training metadata corresponding to --glm-checkpoint-dir.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    return rows


def normalize(text: str) -> str:
    return "".join(str(text).split())


def reference_text(row: dict[str, Any]) -> str:
    """Return the reference field shared by page and line manifests."""
    value = row.get("text", row.get("page_text"))
    if value is None:
        raise KeyError("manifest row missing text/page_text")
    return str(value)


def alignment(reference: str, prediction: str) -> tuple[int, Counter[str], Counter[str]]:
    rows, cols = len(reference) + 1, len(prediction) + 1
    costs = [[0] * cols for _ in range(rows)]
    moves = [[""] * cols for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], moves[i][0] = i, "D"
    for j in range(1, cols):
        costs[0][j], moves[0][j] = j, "I"
    for i in range(1, rows):
        for j in range(1, cols):
            substitution = costs[i - 1][j - 1] + (reference[i - 1] != prediction[j - 1])
            costs[i][j], moves[i][j] = min(
                (substitution, "M"), (costs[i - 1][j] + 1, "D"), (costs[i][j - 1] + 1, "I")
            )
    errors: Counter[str] = Counter()
    matches: Counter[str] = Counter()
    i, j = len(reference), len(prediction)
    while i or j:
        move = moves[i][j]
        if move == "M":
            if reference[i - 1] == prediction[j - 1]:
                matches[reference[i - 1]] += 1
            else:
                errors["substitutions"] += 1
            i, j = i - 1, j - 1
        elif move == "D":
            errors["deletions"] += 1
            i -= 1
        else:
            errors["insertions"] += 1
            j -= 1
    return costs[-1][-1], errors, matches


def metrics(pairs: list[tuple[str, str]], train_counts: Counter[str]) -> dict[str, Any]:
    total_errors = 0
    reference_characters = 0
    counts: Counter[str] = Counter()
    exact = 0
    rare_reference = {k: 0 for k in (1, 3, 5)}
    rare_matches = {k: 0 for k in (1, 3, 5)}
    for reference, prediction in pairs:
        reference, prediction = normalize(reference), normalize(prediction)
        edits, error_counts, matches = alignment(reference, prediction)
        total_errors += edits
        counts.update(error_counts)
        reference_characters += len(reference)
        exact += reference == prediction
        reference_counts = Counter(reference)
        for k in (1, 3, 5):
            rare = {char for char, count in train_counts.items() if 0 < count <= k}
            rare_reference[k] += sum(reference_counts[char] for char in rare)
            rare_matches[k] += sum(matches[char] for char in rare)
    result: dict[str, Any] = {
        "pages": len(pairs),
        "reference_characters": reference_characters,
        "character_errors": total_errors,
        "insertions": counts["insertions"],
        "deletions": counts["deletions"],
        "substitutions": counts["substitutions"],
        "cer": total_errors / max(1, reference_characters),
        "exact_page_rate": exact / max(1, len(pairs)),
    }
    for k in (1, 3, 5):
        result[f"low_frequency_k{k}_reference_characters"] = rare_reference[k]
        result[f"low_frequency_k{k}_recall"] = (
            rare_matches[k] / rare_reference[k] if rare_reference[k] else None
        )
    return result


def square_canvas(image: Image.Image) -> Image.Image:
    size = max(image.width, image.height)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def load_got(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    if args.got_source_root is None:
        raise ValueError("--got-source-root is required for --model got")
    sys.path.insert(0, str(args.got_source_root))
    from transformers import AutoTokenizer
    from GOT.model import GOTQwenForCausalLM
    from GOT.model.plug.blip_process import BlipImageEvalProcessor
    from GOT.utils.conversation import SeparatorStyle, conv_templates
    from GOT.utils.utils import KeywordsStoppingCriteria, disable_torch_init

    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True, local_files_only=True)
    model = GOTQwenForCausalLM.from_pretrained(
        args.model_dir, low_cpu_mem_usage=True, use_safetensors=True,
        local_files_only=True, pad_token_id=151643, torch_dtype=torch.float16,
    ).to(device=device, dtype=torch.float16).eval()
    processor = BlipImageEvalProcessor(image_size=1024)
    question = "<img>" + "<imgpad>" * 256 + "</img>\nOCR: "
    conversation = conv_templates["mpt"].copy()
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    input_ids = torch.as_tensor(tokenizer([prompt]).input_ids, device=device)
    stop_str = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2

    def predict(image: Image.Image) -> str:
        canvas = square_canvas(image)
        low = processor(canvas).unsqueeze(0).to(device=device, dtype=torch.float16)
        high = processor(canvas.copy()).unsqueeze(0).to(device=device, dtype=torch.float16)
        stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            output = model.generate(
                input_ids, images=[(low, high)], do_sample=False, num_beams=1,
                no_repeat_ngram_size=20, max_new_tokens=args.max_new_tokens,
                use_cache=True, stopping_criteria=[stopping],
            )
        text = tokenizer.decode(output[0, input_ids.shape[1]:]).strip()
        return text[:-len(stop_str)].strip() if text.endswith(stop_str) else text

    return {"predict": predict, "name": "GOT-OCR2.0", "model": model}


def load_glm(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_dir, use_fast=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()
    bridge = None
    if args.glm_checkpoint_dir is not None:
        if args.glm_metadata is None:
            raise ValueError("--glm-metadata is required with --glm-checkpoint-dir")
        from layout_ocr.glm_bridge import install_layout_adapter
        from layout_ocr.train_screen import (
            inject_decoder_lora,
            load_adapter_checkpoint,
            load_decoder_lora_checkpoint,
        )

        metadata = json.loads(args.glm_metadata.read_text(encoding="utf-8"))
        adapter_config = metadata.get("adapter_config") or {}
        bridge = install_layout_adapter(
            model,
            str(metadata.get("mode", "layout_ot")),
            int(adapter_config.get("num_queries", metadata.get("num_queries", 64))),
            max_residual_scale=adapter_config.get("max_residual_scale", 0.03),
            initial_residual_scale=float(adapter_config.get("initial_residual_scale", 0.0)),
            use_validity_head=bool(adapter_config.get("use_validity_head", False)),
            initial_valid_probability=float(adapter_config.get("initial_valid_probability", 0.05)),
            validity_gating_mode=adapter_config.get("validity_gating_mode", "legacy_normalized"),
            validity_use_transport_evidence=bool(adapter_config.get("validity_use_transport_evidence", False)),
            adapter_precision="fp32",
            region_autoregressive=bool(adapter_config.get("region_autoregressive", False)),
            region_decoder_hidden_size=int(adapter_config.get("region_decoder_hidden_size", 256)),
            region_decoder_layers=int(adapter_config.get("region_decoder_layers", 2)),
            region_decoder_num_heads=int(adapter_config.get("region_decoder_num_heads", 8)),
            region_pointer_mask=bool(adapter_config.get("region_pointer_mask", True)),
            region_spatial_penalty=float(adapter_config.get("region_spatial_penalty", 4.0)),
            region_spatial_iou_threshold=float(adapter_config.get("region_spatial_iou_threshold", 0.8)),
            box_head_mlp=bool(adapter_config.get("box_head_mlp", False)),
            box_head_hidden=int(adapter_config.get("box_head_hidden", 0)),
            sem_adapter_mlp=bool(adapter_config.get("sem_adapter_mlp", False)),
            sem_adapter_hidden=int(adapter_config.get("sem_adapter_hidden", 0)),
            query_refine_layers=int(adapter_config.get("query_refine_layers", 0)),
        )
        checkpoint_dir = args.glm_checkpoint_dir.resolve()
        load_adapter_checkpoint(checkpoint_dir, bridge)
        if metadata.get("decoder_adaptation", "frozen") == "lora":
            lora_config = metadata.get("decoder_lora_config") or {}
            inject_decoder_lora(
                model,
                rank=int(lora_config.get("rank", 8)),
                alpha=float(lora_config.get("alpha", 8)),
                dropout=float(lora_config.get("dropout", 0)),
            )
            load_decoder_lora_checkpoint(checkpoint_dir, model)
        model.eval()
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "Text Recognition:"}
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def predict(image: Image.Image) -> str:
        inputs = processor(text=[prompt], images=[image], return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        if bridge is not None:
            # The adapter is installed at the visual pre-merge seam and needs
            # the processor's patch grid for every visual forward.  Training
            # and the page evaluator set this explicitly; omitting it here
            # makes every method-arm sample fail before generation.
            bridge.set_grid_thw(inputs["image_grid_thw"])
            bridge.set_region_targets(None)
            bridge.set_region_decode_controls(pointer_mask=None, spatial_penalty=None)
        with torch.inference_mode():
            output = model.generate(
                **inputs, do_sample=False, use_cache=True, max_new_tokens=args.max_new_tokens
            )
        input_length = inputs["input_ids"].shape[-1]
        return processor.batch_decode(output[:, input_length:], skip_special_tokens=True)[0].strip()

    return {
        "predict": predict,
        "name": "GLM-OCR+layout-semantic-adapter" if bridge is not None else "GLM-OCR",
        "model": model,
    }


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("invalid generation limit")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", 0)
    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    train_counts: Counter[str] = Counter()
    for row in read_jsonl(args.train_manifest):
        train_counts.update(normalize(reference_text(row)))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runner = load_got(args, device) if args.model == "got" else load_glm(args, device)
    predictions_path = args.output_dir / "predictions.jsonl"
    pairs: list[tuple[str, str]] = []
    errors = 0
    started_at = time.time()
    with predictions_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            image_path = args.manifest.parent / row["image"]
            reference = reference_text(row)
            result: dict[str, Any] = {
                "index": index, "id": row.get("id", str(index)),
                "image": str(image_path), "reference": reference,
            }
            try:
                with Image.open(image_path) as source:
                    prediction = runner["predict"](source.convert("RGB"))
                result["prediction"] = prediction
                pairs.append((reference, prediction))
            except Exception as exc:
                errors += 1
                result["error"] = f"{type(exc).__name__}: {exc}"
                pairs.append((reference, ""))
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if index == 0 or (index + 1) % 100 == 0:
                print(f"{args.model.upper()}_COLUMNS_PROGRESS {index + 1}/{len(rows)}", flush=True)
    summary = {
        "status": "complete" if errors == 0 else "completed_with_errors",
        "model": runner["name"], "model_dir": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()), "train_manifest": str(args.train_manifest.resolve()),
        "records": len(rows), "max_new_tokens": args.max_new_tokens,
        "errors": errors, "elapsed_seconds": time.time() - started_at,
        "test_used_for_selection": False, "metrics": metrics(pairs, train_counts),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "COLUMN_EVAL_COMPLETED").touch()
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
