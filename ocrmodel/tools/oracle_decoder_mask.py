#!/usr/bin/env python3
"""Oracle-mask intervention: is the *idea* of routing useful, head aside?

The screen can show that a head beats the no-head baseline, but not whether the
gain comes from the routing bias or merely from the auxiliary supervision the
head's loss adds to the shared LoRA.  This tool separates the two by swapping the
predicted mask for the **ground-truth** mask at inference and asking a single
question:

    holding the model's own token trajectory fixed, does the ground-truth bias
    make that trajectory more likely?

If yes, the routing signal is real and the head is what is failing.  If no, no
amount of head engineering will help, because the information the bias would
carry is worth nothing to the decoder.

Why the trajectory is generated first and re-scored, rather than the mask being
injected during generation: a decode loop that injects a per-step mask has to
reproduce ``generate``'s cache/position bookkeeping exactly, and any mismatch
shows up as a spuriously bad score rather than as an error.  Scoring a fixed
trajectory needs no such bookkeeping and answers the same question, because the
only thing that changes between the two arms is the mask the attention sees.

Both the reference trajectory (teacher forcing) and the model's own generated
trajectory can be scored; the generated one is the informative case, since
teacher-forced likelihood is a known-poor proxy for free generation.

This reads evaluation ground truth.  It is a diagnostic only: it must never
enter checkpoint selection or be reported as a deployable result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import torch

from layout_ocr.data import load_records, prepare_inference_inputs, prepare_training_inputs
from layout_ocr.decoder_mask_checkpoint import load_config, load_lora_state, restore_router
from layout_ocr.decoder_mask_model import enable_eager_backend, install_decoder_mask_router
from layout_ocr.decoder_mask_router import fine_grid_xywh, _normalized_grid_xywh
from layout_ocr.lora import inject_decoder_lora, load_lora_state_dict
from layout_ocr.mask_targets import build_mask_targets, token_char_spans, char_boxes


def _eos_ids(model: Any, processor: Any) -> set[int]:
    ids: set[int] = set()
    value = getattr(getattr(processor, "tokenizer", None), "eos_token_id", None)
    if value is not None:
        ids = {int(v) for v in value} if isinstance(value, (list, tuple, set)) else {int(value)}
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is not None:
        if isinstance(value, (list, tuple, set)):
            ids.update(int(v) for v in value)
        else:
            ids.add(int(value))
    return ids


def _cursor_of_prefix(generated: str, page_text: str) -> int:
    """How many ``page_text`` characters the generated prefix has consumed.

    A monotone alignment, done incrementally from the front: walk the generated
    string and advance through the page while characters agree, so a divergence
    stalls the cursor instead of desynchronising the rest of the page.
    """

    cursor = 0
    page = "".join(page_text.split())
    produced = "".join(generated.split())
    for char in produced:
        if cursor < len(page) and char == page[cursor]:
            cursor += 1
    return cursor


def _oracle_row(cursor: int, spans: list[tuple[int, int] | None], n_rows: int) -> int | None:
    """Reference-token index whose character span covers (or follows) ``cursor``."""

    for index, span in enumerate(spans):
        if span is None:
            continue
        if span[1] > cursor:
            return index
    for index in range(len(spans) - 1, -1, -1):
        if spans[index] is not None:
            return index
    return None


def _score(
    model: Any,
    processor: Any,
    inputs: dict[str, Any],
    reference_ids: torch.Tensor,
    oracle_rows: list[int | None],
    oracle_mask: torch.Tensor,
    runtime: Any,
    use_oracle: bool,
) -> dict[str, float]:
    """Cross-entropy of ``reference_ids`` under the predicted or the oracle mask."""

    prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]
    input_ids = torch.cat((prompt_ids, reference_ids.unsqueeze(0)), dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[0, prompt_len:] = reference_ids

    runtime.set_page(
        inputs["image_grid_thw"], input_ids, prompt_len, None
    )
    # Generation-style page: the head needs a query position per scored token.
    runtime.query_positions = torch.arange(prompt_len - 1, input_ids.shape[1] - 1, device=input_ids.device)
    runtime.oracle_mask = None
    if use_oracle:
        rows = [row if row is not None else 0 for row in oracle_rows]
        runtime.oracle_mask = oracle_mask[0][rows].unsqueeze(0)  # [1, T, N]
    # No attention_mask: the sequence here is a freshly built prompt+trajectory
    # concatenation, so any mask carried over from the loader would be the wrong
    # length.  A batch of one with no padding does not need one.
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            pixel_values=inputs["pixel_values"],
            image_grid_thw=inputs["image_grid_thw"],
            labels=labels,
        )
    runtime.oracle_mask = None
    loss = float(outputs.loss.detach().item())
    logits = outputs.logits[0, prompt_len - 1 : -1]
    predicted = logits.argmax(dim=-1)
    correct = int((predicted == reference_ids).sum().item())
    return {"loss": loss, "token_accuracy": correct / max(1, int(reference_ids.numel()))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="oracle-mask intervention on a trained checkpoint")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    parser.add_argument("--trajectory", choices=("generated", "reference"), default="generated")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device else torch.device("cuda")

    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False, local_files_only=True)
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

    eos_ids = _eos_ids(model, processor)
    image_token_id = int(model.config.image_token_id)
    merge = int(model.model.visual.spatial_merge_size)
    inject_decoder_lora(model, rank=8, alpha=8.0)
    state = load_lora_state(checkpoint_dir)
    if state is not None:
        load_lora_state_dict(model, state)
    config = load_config(checkpoint_dir)
    enable_eager_backend(model)
    runtime = install_decoder_mask_router(model, config, image_token_id, merge)
    restore_router(model, runtime, checkpoint_dir)
    runtime.router.eval()
    runtime.set_noise(0.0, 0.0)
    runtime.set_bias_strength(float(config.bias_max))

    records = load_records(args.manifest)[: args.pages]
    rows_out: list[dict[str, Any]] = []
    totals = {"predicted": 0.0, "oracle": 0.0, "n": 0, "pad": 0.0, "oap": 0.0, "rows": 0}
    for record in records:
        inference_inputs = prepare_inference_inputs(processor, record, device)
        prompt_len = int(inference_inputs["input_ids"].shape[1])
        if args.trajectory == "generated":
            runtime.set_page(
                inference_inputs["image_grid_thw"], inference_inputs["input_ids"], prompt_len, None
            )
            with torch.no_grad():
                generated = model.generate(
                    **inference_inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    eos_token_id=sorted(eos_ids),
                )
            trajectory = generated[0, prompt_len:]
            decoded = processor.tokenizer.decode(trajectory.tolist(), skip_special_tokens=True)
        else:
            training_inputs = prepare_training_inputs(processor, record, device, eos_ids)
            labels = training_inputs["labels"]
            prompt_len = int((labels == -100).to(torch.int32).cumprod(dim=1).sum().item())
            trajectory = training_inputs["input_ids"][0, prompt_len:]
            decoded = record["page_text"]
            inference_inputs = training_inputs

        # Ground-truth masks on the head's own grid, plus each reference token's
        # character span, so the oracle row for a reading position is a lookup.
        if config.visual_source == "fine":
            xywh, _ = fine_grid_xywh(inference_inputs["image_grid_thw"], 1)
        else:
            xywh, _ = _normalized_grid_xywh(inference_inputs["image_grid_thw"], merge)
        targets = build_mask_targets(
            processor.tokenizer, record, trajectory, eos_ids, xywh
        )
        boxes, char_statuses = char_boxes(record)
        spans, _, _ = token_char_spans(
            processor.tokenizer, record["page_text"], trajectory, char_statuses
        )

        cursor = _cursor_of_prefix(decoded, record["page_text"])
        oracle_rows = [_oracle_row(cursor, spans, trajectory.numel())] * int(trajectory.numel())
        predicted = _score(model, processor, inference_inputs, trajectory, oracle_rows, targets.mask, runtime, False)
        oracle = _score(model, processor, inference_inputs, trajectory, oracle_rows, targets.mask, runtime, True)
        row = {
            "page_id": record["page_id"],
            "trajectory": args.trajectory,
            "scored_tokens": int(trajectory.numel()),
            "cursor": cursor,
            "predicted_loss": predicted["loss"],
            "oracle_loss": oracle["loss"],
            "delta": predicted["loss"] - oracle["loss"],  # >0 means the oracle is better
            "predicted_accuracy": predicted["token_accuracy"],
            "oracle_accuracy": oracle["token_accuracy"],
        }
        rows_out.append(row)
        totals["predicted"] += predicted["loss"]
        totals["oracle"] += oracle["loss"]
        totals["pad"] += predicted["token_accuracy"]
        totals["oap"] += oracle["token_accuracy"]
        totals["n"] += 1
        print(json.dumps(row, ensure_ascii=False))

    n = max(1, totals["n"])
    payload = {
        "status": "complete",
        "checkpoint_dir": str(checkpoint_dir),
        "manifest": str(args.manifest),
        "trajectory": args.trajectory,
        "pages": totals["n"],
        "mean_predicted_loss": totals["predicted"] / n,
        "mean_oracle_loss": totals["oracle"] / n,
        "mean_delta_loss": (totals["predicted"] - totals["oracle"]) / n,
        "mean_predicted_accuracy": totals["pad"] / n,
        "mean_oracle_accuracy": totals["oap"] / n,
        "reads_ground_truth": True,
        "usable_for_selection": False,
        "pages_detail": rows_out,
    }
    (output_dir / "oracle.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: payload[k] for k in (
        "mean_predicted_loss", "mean_oracle_loss", "mean_delta_loss",
        "mean_predicted_accuracy", "mean_oracle_accuracy")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
