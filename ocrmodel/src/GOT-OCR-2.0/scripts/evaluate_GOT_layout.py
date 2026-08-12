from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from GOT.model.GOT_ocr_2_0 import GOTQwenForCausalLM
from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.utils.utils import KeywordsStoppingCriteria, disable_torch_init
from layout_page_dataset import make_layout_page_validation_data_module
from layout_validation_metrics import (
    DIRECTION_LABELS,
    LayoutValidationAccumulator,
    OCRValidationAccumulator,
)
from local_tokenizer import load_local_tokenizer, tokenizer_candidates


SUMMARY_NAME = "layout_validation_metrics.json"
PREDICTIONS_NAME = "layout_validation_predictions.jsonl"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate whole-page GOT2 OCR generation and optional VLQA explanations "
            "without supplying layout metadata to the model."
        )
    )
    parser.add_argument("--model-name-or-path", type=Path, required=True)
    parser.add_argument(
        "--model-kind",
        choices=("baseline", "vlqa"),
        default="vlqa",
        help=(
            "Strict checkpoint protocol. baseline rejects VLQA, while vlqa requires "
            "a complete layout adapter."
        ),
    )
    parser.add_argument(
        "--require-vlqa-stage",
        choices=("p1", "p2"),
        help="Optionally require layout_training_metrics.json to report this stage.",
    )
    parser.add_argument("--tokenizer-name-or-path", type=Path)
    parser.add_argument("--layout-manifest", type=Path, required=True)
    parser.add_argument("--layout-image-root", type=Path)
    parser.add_argument("--layout-split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-regions", type=positive_int, default=16)
    parser.add_argument("--max-records", type=nonnegative_int, default=0)
    parser.add_argument("--model-max-length", type=positive_int, default=2048)
    parser.add_argument("--max-new-tokens", type=positive_int, default=2048)
    parser.add_argument("--no-repeat-ngram-size", type=nonnegative_int, default=20)
    parser.add_argument("--object-threshold", type=probability, default=0.5)
    parser.add_argument("--iou-threshold", type=probability, default=0.5)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def move_images(
    images: Sequence[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            image.to(device=device, non_blocking=True),
            image_high.to(device=device, non_blocking=True),
        )
        for image, image_high in images
    ]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def trim_generation(decoded: str, stop_string: str) -> str:
    output = decoded.strip()
    if stop_string and output.endswith(stop_string):
        output = output[: -len(stop_string)]
    return output.strip()


def require_layout_predictions(outputs: Any, max_regions: int) -> None:
    expected_shapes = {
        "layout_object_logits": (1, max_regions),
        "layout_bbox_xyxy": (1, max_regions, 4),
        "layout_direction_logits": (1, max_regions, len(DIRECTION_LABELS)),
    }
    for name, expected_shape in expected_shapes.items():
        value = getattr(outputs, name, None)
        if value is None:
            raise RuntimeError(f"Model forward did not return {name}.")
        if tuple(value.shape) != expected_shape:
            raise RuntimeError(
                f"Model returned {name} shape {tuple(value.shape)}, expected {expected_shape}."
            )
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Model returned non-finite values in {name}.")


def validate_model_protocol(
    *,
    model: GOTQwenForCausalLM,
    model_path: Path,
    model_kind: str,
    max_regions: int,
    required_vlqa_stage: str | None,
) -> str | None:
    adapter = model.get_model().layout_adapter
    use_vlqa = getattr(model.config, "use_vlqa", False) is True
    if model_kind == "baseline":
        if required_vlqa_stage is not None:
            raise ValueError("--require-vlqa-stage is only valid for --model-kind vlqa.")
        if use_vlqa or adapter is not None:
            raise RuntimeError(
                "Baseline evaluation requires an original GOT2 checkpoint without VLQA."
            )
        return None

    if not use_vlqa or adapter is None:
        raise RuntimeError("VLQA evaluation requires config.use_vlqa=true and layout_adapter.")
    model_queries = int(getattr(model.config, "vlqa_num_queries", -1))
    if model_queries != max_regions:
        raise RuntimeError(
            f"Model VLQA query count {model_queries} does not match --max-regions "
            f"{max_regions}."
        )

    metrics_path = model_path / "layout_training_metrics.json"
    checkpoint_stage = None
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict):
            raise TypeError(f"Expected a JSON object: {metrics_path}")
        stage = metrics.get("layout_stage")
        if stage in {"p1", "p2"}:
            checkpoint_stage = str(stage)
    if required_vlqa_stage is not None and checkpoint_stage != required_vlqa_stage:
        raise RuntimeError(
            "VLQA checkpoint stage mismatch: "
            f"{checkpoint_stage!r} != {required_vlqa_stage!r}."
        )
    return checkpoint_stage


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    model_path = args.model_name_or_path.expanduser().resolve()
    manifest = args.layout_manifest.expanduser().resolve()
    image_root = (
        args.layout_image_root.expanduser().resolve()
        if args.layout_image_root is not None
        else manifest.parent
    )
    output_dir = args.output_dir.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    tokenizer_path = (
        args.tokenizer_name_or_path.expanduser().resolve()
        if args.tokenizer_name_or_path is not None
        else model_path
    )
    tokenizer_paths = tokenizer_candidates(
        tokenizer_path,
        os.environ.get("GOT_TOKENIZER_MODEL"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_NAME
    predictions_path = output_dir / PREDICTIONS_NAME
    if summary_path.exists() or predictions_path.exists():
        raise FileExistsError(
            f"Validation outputs already exist under {output_dir}; refusing to overwrite them."
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation was requested but torch.cuda.is_available() is false.")
    dtype = resolve_dtype(args.dtype)
    disable_torch_init()
    tokenizer, tokenizer_path = load_local_tokenizer(
        AutoTokenizer,
        tokenizer_paths,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
    )
    tokenizer.model_max_length = args.model_max_length
    model = GOTQwenForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        pad_token_id=151643,
        torch_dtype=dtype,
        local_files_only=True,
    ).eval()
    model.to(device=device, dtype=dtype)
    checkpoint_stage = validate_model_protocol(
        model=model,
        model_path=model_path,
        model_kind=args.model_kind,
        max_regions=args.max_regions,
        required_vlqa_stage=args.require_vlqa_stage,
    )

    image_processor = BlipImageEvalProcessor(image_size=1024)
    data_module = make_layout_page_validation_data_module(
        tokenizer=tokenizer,
        manifest=manifest,
        image_root=image_root,
        split=args.layout_split,
        max_regions=args.max_regions,
        image_processor=image_processor,
        image_token_len=256,
        max_records=args.max_records,
    )
    dataset = data_module["eval_dataset"]
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=data_module["data_collator"],
    )
    layout_accumulator = (
        LayoutValidationAccumulator(
            object_threshold=args.object_threshold,
            iou_threshold=args.iou_threshold,
        )
        if args.model_kind == "vlqa"
        else None
    )
    ocr_accumulator = OCRValidationAccumulator()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    total_layout_seconds = 0.0
    total_generation_seconds = 0.0
    completed_pages = 0
    temporary_predictions = predictions_path.with_suffix(predictions_path.suffix + ".tmp")
    try:
        with temporary_predictions.open("x", encoding="utf-8", newline="\n") as handle:
            for batch in loader:
                if len(batch["page_id"]) != 1:
                    raise RuntimeError("Layout validation currently requires batch_size=1.")
                input_ids = batch["input_ids"].to(device=device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(
                    device=device,
                    non_blocking=True,
                )
                images = move_images(batch["images"], device)

                layout_outputs = None
                layout_seconds = 0.0
                if args.model_kind == "vlqa":
                    synchronize(device)
                    layout_started = time.perf_counter()
                    with torch.inference_mode(), torch.autocast(
                        device_type=device.type,
                        dtype=dtype,
                        enabled=device.type == "cuda" and dtype != torch.float32,
                    ):
                        layout_outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            images=images,
                            use_cache=False,
                            return_dict=True,
                        )
                    synchronize(device)
                    layout_seconds = time.perf_counter() - layout_started
                    require_layout_predictions(layout_outputs, args.max_regions)

                stop_string = batch["stop_string"][0]
                stopping_criteria = KeywordsStoppingCriteria(
                    [stop_string],
                    tokenizer,
                    input_ids,
                )
                generation_kwargs: dict[str, Any] = {
                    "attention_mask": attention_mask,
                    "images": images,
                    "do_sample": False,
                    "num_beams": 1,
                    "max_new_tokens": args.max_new_tokens,
                    "stopping_criteria": [stopping_criteria],
                }
                if args.no_repeat_ngram_size:
                    generation_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

                synchronize(device)
                generation_started = time.perf_counter()
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=dtype,
                    enabled=device.type == "cuda" and dtype != torch.float32,
                ):
                    output_ids = model.generate(input_ids, **generation_kwargs)
                synchronize(device)
                generation_seconds = time.perf_counter() - generation_started

                decoded = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
                predicted_text = trim_generation(decoded, stop_string)
                if layout_outputs is not None:
                    object_scores = (
                        layout_outputs.layout_object_logits[0]
                        .float()
                        .sigmoid()
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    predicted_boxes = (
                        layout_outputs.layout_bbox_xyxy[0]
                        .float()
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    predicted_directions = (
                        layout_outputs.layout_direction_logits[0]
                        .float()
                        .argmax(dim=-1)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    assert layout_accumulator is not None
                    page_metrics = layout_accumulator.add_page(
                        reference_text=batch["page_text"][0],
                        predicted_text=predicted_text,
                        regions=batch["regions"][0],
                        annotation_status=batch["layout_annotation_status"][0],
                        object_scores=object_scores,
                        predicted_boxes=predicted_boxes,
                        predicted_directions=predicted_directions,
                    )
                    query_predictions = [
                        {
                            "query_index": query_index,
                            "object_probability": object_scores[query_index],
                            "bbox_xyxy": predicted_boxes[query_index],
                            "writing_direction": DIRECTION_LABELS[
                                predicted_directions[query_index]
                            ],
                        }
                        for query_index in range(args.max_regions)
                    ]
                else:
                    page_metrics = {
                        "ocr": ocr_accumulator.add_page(
                            batch["page_text"][0],
                            predicted_text,
                        )
                    }
                    query_predictions = None
                prediction_record = {
                    "page_id": batch["page_id"][0],
                    "image": batch["image_path"][0],
                    "reference_text": batch["page_text"][0],
                    "predicted_text": predicted_text,
                    "layout_annotation_status": batch["layout_annotation_status"][0],
                    "regions": batch["regions"][0],
                    "layout_predictions": query_predictions,
                    "metrics": page_metrics,
                    "runtime_seconds": {
                        "layout_forward": layout_seconds,
                        "ocr_generation": generation_seconds,
                    },
                }
                handle.write(compact_json(prediction_record) + "\n")
                handle.flush()
                total_layout_seconds += layout_seconds
                total_generation_seconds += generation_seconds
                completed_pages += 1
        temporary_predictions.replace(predictions_path)
    except BaseException:
        temporary_predictions.unlink(missing_ok=True)
        raise

    if completed_pages != len(dataset):
        raise RuntimeError(
            f"Validation processed {completed_pages} pages but dataset contains {len(dataset)}."
        )
    total_inference_seconds = total_layout_seconds + total_generation_seconds
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    summary = {
        "schema_version": 1,
        "status": "ok",
        "purpose": "whole-page checkpoint evaluation; not a training metric",
        "model_kind": args.model_kind,
        "checkpoint_stage": checkpoint_stage,
        "model": str(model_path),
        "tokenizer": str(tokenizer_path),
        "manifest": str(manifest),
        "image_root": str(image_root),
        "split": args.layout_split,
        "pages": completed_pages,
        "input_protocol": {
            "model_inputs": ["whole_page_image", "ocr_prompt"],
            "ocr_prompt": "OCR: ",
            "layout_metadata_as_model_input": False,
            "layout_metadata_usage": "offline_metrics_only",
        },
        "decoding": {
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": args.max_new_tokens,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
        },
        "layout": {
            "max_regions": args.max_regions,
            "direction_labels": list(DIRECTION_LABELS),
            "prediction_output": (
                "optional_explanation_and_evaluation"
                if args.model_kind == "vlqa"
                else "not_available_for_original_got2"
            ),
        },
        "metrics": (
            layout_accumulator.summary()
            if layout_accumulator is not None
            else {"ocr": ocr_accumulator.summary(), "layout": None}
        ),
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "layout_adapter": (
                sum(
                    parameter.numel()
                    for parameter in model.get_model().layout_adapter.parameters()
                )
                if model.get_model().layout_adapter is not None
                else 0
            ),
        },
        "runtime": {
            "device": str(device),
            "dtype": args.dtype,
            "layout_forward_seconds": total_layout_seconds,
            "ocr_generation_seconds": total_generation_seconds,
            "total_inference_seconds": total_inference_seconds,
            "mean_seconds_per_page": total_inference_seconds / completed_pages,
            "pages_per_second": (
                completed_pages / total_inference_seconds
                if total_inference_seconds > 0.0
                else None
            ),
            "peak_cuda_memory_bytes": peak_memory_bytes,
        },
        "predictions": str(predictions_path),
    }
    write_json(summary_path, summary)
    print(
        compact_json(
            {
                "event": "got_layout_validation_completed",
                "summary": str(summary_path),
                "model_kind": args.model_kind,
                "pages": completed_pages,
                "metrics": summary["metrics"],
            }
        ),
        flush=True,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    evaluate(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
