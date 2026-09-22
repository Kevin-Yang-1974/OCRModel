#!/usr/bin/env python3
"""Free-generation oracle CER: inject the ground-truth mask and decode normally.

This is the ceiling counterpart to ``oracle_decoder_mask.py`` (which re-scores a
*fixed* trajectory's cross-entropy).  That tool answers "does the correct bias
make a fixed trajectory more likely?"; this one answers the question that matters
for the screen -- "what CER does *perfect* routing achieve?" -- by swapping the
head's predicted mask for the ground-truth mask during greedy generation, so the
result is directly comparable to B0's zero-shot CER (0.287 on the 64-page
validation set).

Two oracle modes, matching the training arms:

* ``token``  -- every token attends to the union of the character boxes it
                covers (the single-character/small-span target used to isolate
                mask shape from the backbone checkpoint).
* ``line``   -- every token attends to the full bounding hull of its text line
                (the strongest spatial cue available, the line-level ceiling).
* ``window`` -- every token attends to the 3-5 character hull around its span
                (the same target the G1 training arm optimises).

The mask can be injected positionally or with the attention-routing ``synced``
pointer.  The latter reads the generated token ids at the top-level generation
hook, advances monotonically through ``page_text`` with the same bounded
lookahead, and maps the reached character back to the target-token mask.  This
keeps the oracle attached to the model's actual reading position after an
insertion or skip instead of silently drifting by decode step.

This reads evaluation ground truth.  It is a diagnostic only: it must never
enter checkpoint selection or be reported as a deployable result.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import torch

from layout_ocr.data import load_records, prepare_inference_inputs, prepare_training_inputs
from layout_ocr.decoder_mask_model import install_decoder_mask_router
from layout_ocr.decoder_mask_router import DecoderMaskConfig, _normalized_grid_xywh
from layout_ocr.glm_bridge import install_layout_adapter
from layout_ocr.lora import inject_decoder_lora
from layout_ocr.mask_targets import build_mask_targets
from layout_ocr.metrics import aggregate_ocr_metrics
from layout_ocr.train_screen import (
    configure_deterministic_execution,
    load_adapter_checkpoint,
    load_decoder_lora_checkpoint,
)


def _eos_ids(model: object, processor: object) -> set[int]:
    tokenizer = getattr(processor, "tokenizer", None)
    ids: set[int] = set()
    value = getattr(tokenizer, "eos_token_id", None)
    if value is not None:
        if isinstance(value, (list, tuple, set)):
            ids = {int(item) for item in value if item is not None}
        else:
            ids = {int(value)}
    gen = getattr(model, "generation_config", None)
    gvalue = getattr(gen, "eos_token_id", None)
    if gvalue is not None:
        if isinstance(gvalue, (list, tuple, set)):
            ids.update(int(item) for item in gvalue if item is not None)
        else:
            ids.add(int(gvalue))
    return ids


def _image_token_id(model: object) -> int:
    value = getattr(model.config, "image_token_id", None)
    if value is None:
        value = getattr(getattr(model.config, "text_config", None), "image_token_id", None)
    if value is None:
        raise RuntimeError("could not resolve the model image token id")
    return int(value)


class _OracleState:
    """Per-page reference masks plus positional or synced generation state."""

    LOOKAHEAD = 16

    def __init__(self, runtime: object, tokenizer: object, pointer_mode: str) -> None:
        self.runtime = runtime
        self.tokenizer = tokenizer
        self.pointer_mode = pointer_mode
        self.ref: torch.Tensor | None = None  # [1, T, N] reference-token masks
        self.zero: torch.Tensor | None = None  # [1, 1, N] no-bias fallback past T
        self.char_to_token: list[int | None] = []
        self.reference: str | None = None
        self.prompt_length: int | None = None
        self.position: int = 0
        self._last_observed = -1
        self.step: int = 1  # positional index of the next target token

    def set_page(
        self,
        ref: torch.Tensor,
        *,
        reference: str,
        prompt_length: int,
        char_spans: list[tuple[int, int] | None],
    ) -> None:
        self.ref = ref
        self.zero = torch.zeros(1, 1, ref.shape[-1], device=ref.device, dtype=ref.dtype)
        self.reference = reference
        self.prompt_length = int(prompt_length)
        self.position = 0
        self._last_observed = -1
        self.step = 1
        self.char_to_token = [None] * len(reference)
        for token_index, span in enumerate(char_spans):
            if span is None:
                continue
            start, end = span
            for char_index in range(max(0, start), min(len(reference), end)):
                if self.char_to_token[char_index] is None:
                    self.char_to_token[char_index] = token_index

    def _advance(self, text: str) -> None:
        if self.reference is None:
            return
        for char in text:
            if self.position < len(self.reference) and self.reference[self.position] == char:
                self.position += 1
                continue
            limit = min(len(self.reference), self.position + self.LOOKAHEAD)
            found = self.reference.find(char, self.position, limit)
            if found != -1:
                self.position = found + 1

    def _synced_index(self) -> int | None:
        if self.position >= len(self.char_to_token):
            return None
        return self.char_to_token[self.position]

    def hook(self, module: object, args: tuple, kwargs: dict) -> None:
        if self.ref is None:
            return
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        q_len = 0 if ids is None else int(ids.shape[1])
        if q_len > 1:
            # Prefill: the last prompt position predicts the first target token,
            # which is supervised by reference token 0.  Reset the cursor.
            self.step = 1
            self.runtime.oracle_mask = self.ref[:, 0:1]
            return

        if self.pointer_mode == "synced":
            cache_position = kwargs.get("cache_position")
            if not isinstance(cache_position, torch.Tensor) or self.prompt_length is None:
                index = self._synced_index()
            else:
                positions = cache_position.flatten()
                generated = positions >= self.prompt_length
                if (
                    isinstance(ids, torch.Tensor)
                    and bool(generated.any())
                    and int(positions[generated][0].item()) > self._last_observed
                ):
                    self._last_observed = int(positions[generated][-1].item())
                    self._advance(self.tokenizer.decode(ids[0][generated], skip_special_tokens=True))
                index = self._synced_index()
        else:
            index = self.step
            self.step += 1

        if index is not None and index < self.ref.shape[1]:
            self.runtime.oracle_mask = self.ref[:, index : index + 1]
        else:
            self.runtime.oracle_mask = self.zero


def _decode_tokens(tokenizer: object, tokens: torch.Tensor, eos_ids: set[int]) -> str:
    ids = tokens.tolist()
    first_eos = next((i for i, t in enumerate(ids) if int(t) in eos_ids), None)
    if first_eos is not None:
        ids = ids[: first_eos + 1]
    return tokenizer.decode(ids, skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="oracle-mask free-generation CER")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="optional trained layout+decoder checkpoint to load over the base model",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, help="train manifest for character counts")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--oracle-mode", choices=("token", "line", "window"), required=True)
    parser.add_argument("--layout-mode", choices=("geometry", "attention"), default="geometry")
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--decoder-lora-rank", type=int, default=8)
    parser.add_argument("--decoder-lora-alpha", type=float, default=8.0)
    parser.add_argument("--window-size", nargs=2, type=int, default=[3, 5], metavar=("MIN", "MAX"))
    parser.add_argument("--bias-max", type=float, default=2.0)
    parser.add_argument("--max-pixels", type=int, default=4000000)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--processor-mode", choices=("fast", "slow"), default="slow")
    parser.add_argument("--raster-mode", choices=("soft", "hard"), default="soft")
    parser.add_argument("--pointer-mode", choices=("step", "synced"), default="step")
    parser.add_argument(
        "--split-layer",
        type=int,
        default=8,
        help="first layer receiving the bias; 0 injects into every decoder layer",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pages", type=int, default=0, help="limit pages; 0 means all")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _train_character_counts(records: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record["page_text"])
    return counts


def main() -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    reproducibility = configure_deterministic_execution()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = args.output_dir.resolve()
    # Refuse only a *completed* prior run, not a directory the launcher created
    # ahead of time to hold its own run.log.  A fresh run may legitimately find
    # the directory already present and empty.
    if (output_dir / "summary.json").exists() or (output_dir / "predictions.jsonl").exists():
        raise FileExistsError(f"{output_dir} already contains results; refusing to overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(
        args.model_path, use_fast=args.processor_mode == "fast", local_files_only=True
    )
    size = dict(processor.image_processor.size)
    size["longest_edge"] = args.max_pixels
    processor.image_processor.size = size

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device)
    model.eval()

    checkpoint_dir = args.checkpoint_dir.resolve() if args.checkpoint_dir else None
    layout_bridge = None
    if checkpoint_dir is not None:
        if not (checkpoint_dir / "adapter.safetensors").is_file():
            raise FileNotFoundError(f"layout adapter checkpoint is missing: {checkpoint_dir / 'adapter.safetensors'}")
        if not (checkpoint_dir / "decoder_lora.safetensors").is_file():
            raise FileNotFoundError(f"decoder LoRA checkpoint is missing: {checkpoint_dir / 'decoder_lora.safetensors'}")
        layout_bridge = install_layout_adapter(
            model,
            args.layout_mode,
            num_queries=args.num_queries,
            max_residual_scale=0.03,
            initial_residual_scale=0.0,
            adapter_precision="fp32",
        )
        load_adapter_checkpoint(checkpoint_dir, layout_bridge)
        inject_decoder_lora(
            model,
            rank=args.decoder_lora_rank,
            alpha=args.decoder_lora_alpha,
            dropout=0.0,
        )
        load_decoder_lora_checkpoint(checkpoint_dir, model)
        model.eval()

    eos_ids = _eos_ids(model, processor)
    if not eos_ids:
        raise RuntimeError("could not resolve EOS token ids")
    image_token_id = _image_token_id(model)
    merge = int(model.model.visual.spatial_merge_size)

    # A fresh, untrained head: the oracle overrides its prediction entirely, so
    # only the bias mechanism and the geometry matter.  target_mode stays "token"
    # in the config (valid) -- the oracle target is built directly below.
    config = DecoderMaskConfig(
        visual_source="merged",
        head="mlp",
        bias_max=args.bias_max,
        window_min=args.window_size[0],
        window_max=args.window_size[1],
        split_layer=args.split_layer,
    )
    runtime = install_decoder_mask_router(model, config, image_token_id, merge)
    runtime.router.eval()
    runtime.set_bias_strength(args.bias_max)
    runtime.set_noise(0.0, 0.0)

    state = _OracleState(runtime, processor.tokenizer, args.pointer_mode)
    handle = model.register_forward_pre_hook(state.hook, with_kwargs=True)

    records = load_records(args.manifest)
    if args.pages > 0:
        records = records[: args.pages]
    train_counts = (
        _train_character_counts(load_records(args.train_manifest))
        if args.train_manifest
        else Counter()
    )

    pairs: list[tuple[str, str]] = []
    predictions: list[dict] = []
    total_tokens = 0
    limit_hits = 0
    start = time.time()
    for record in records:
        # Ground-truth masks on the head's merged grid, aligned to the reference
        # token sequence.  The generation hook can select them by step or by the
        # character reached by the synced pointer.
        training_inputs = prepare_training_inputs(processor, record, device, eos_ids)
        labels = training_inputs["labels"]
        t_prompt_len = int((labels == -100).to(torch.int32).cumprod(dim=1).sum().item())
        target_ids = training_inputs["input_ids"][0, t_prompt_len:]
        xywh, _ = _normalized_grid_xywh(training_inputs["image_grid_thw"], merge)
        targets = build_mask_targets(
            processor.tokenizer,
            record,
            target_ids,
            eos_ids,
            xywh,
            target_mode=args.oracle_mode,
            window_min=args.window_size[0],
            window_max=args.window_size[1],
            line_source="auto",
            raster_mode=args.raster_mode,
        )
        if args.oracle_mode == "line" and targets.window_report.get("line_source") != "annotation":
            raise RuntimeError(
                f"line-level oracle needs a 'line_index' in the manifest, but page "
                f"{record['page_id']} reported {targets.window_report.get('line_source')}"
            )
        inference_inputs = prepare_inference_inputs(processor, record, device)
        prompt_len = int(inference_inputs["input_ids"].shape[1])
        state.set_page(
            targets.mask,
            reference=record["page_text"],
            prompt_length=prompt_len,
            char_spans=targets.char_spans,
        )
        if layout_bridge is not None:
            layout_bridge.set_grid_thw(inference_inputs["image_grid_thw"])
        runtime.set_page(
            inference_inputs["image_grid_thw"], inference_inputs["input_ids"], prompt_len, None
        )
        generated = model.generate(
            **inference_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            eos_token_id=sorted(eos_ids),
        )
        tokens = generated[0, prompt_len:]
        total_tokens += int(tokens.numel())
        hit_limit = not any(int(t) in eos_ids for t in tokens.tolist())
        limit_hits += int(hit_limit)
        prediction = _decode_tokens(processor.tokenizer, tokens, eos_ids)
        pairs.append((record["page_text"], prediction))
        predictions.append(
            {
                "page_id": record["page_id"],
                "reference": record["page_text"],
                "prediction": prediction,
                "generated_tokens": int(tokens.numel()),
                "generation_limit_hit": hit_limit,
            }
        )
        runtime.clear_page()

    handle.remove()
    elapsed = time.time() - start
    metrics = aggregate_ocr_metrics(pairs, train_counts)
    summary = {
        "status": "complete",
        "oracle_mode": args.oracle_mode,
        "bias_max": args.bias_max,
        "window_size": list(args.window_size) if args.oracle_mode == "window" else None,
        "model_path": str(args.model_path),
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "manifest": str(args.manifest),
        "pages": len(records),
        "max_new_tokens": args.max_new_tokens,
        "max_pixels": args.max_pixels,
        "processor_mode": args.processor_mode,
        "attention_implementation": "sdpa",
        "reproducibility": reproducibility,
        "seed": args.seed,
        "raster_mode": args.raster_mode,
        "pointer_mode": args.pointer_mode,
        "split_layer": args.split_layer,
        "metrics": metrics,
        "resource": {
            "elapsed_seconds": elapsed,
            "generated_tokens": total_tokens,
            "tokens_per_second": total_tokens / elapsed if elapsed > 0 else None,
            "generation_limit_hits": limit_hits,
            "generation_limit_hit_rate": limit_hits / max(1, len(records)),
        },
        "reads_ground_truth": True,
        "usable_for_selection": False,
        "test_used_for_selection": False,
    }
    (output_dir / "predictions.jsonl").write_text(
        "\n".join(json.dumps(entry, ensure_ascii=False) for entry in predictions) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
