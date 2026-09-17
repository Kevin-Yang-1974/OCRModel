#!/usr/bin/env python3
"""Evaluate a trained GLM-OCR adapter and decoder-LoRA on AncientDoc test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch


CODE_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = CODE_ROOT / "src"
for _path in (CODE_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from layout_ocr.data import prepare_inference_inputs  # noqa: E402
from layout_ocr.lora import trainable_parameter_report  # noqa: E402
from layout_ocr.stabilization import (  # noqa: E402
    RepeatSuppressionConfig,
    generate_with_loop_recovery,
)
from layout_ocr.train_screen import (  # noqa: E402
    adapter_finite_report,
    configure_deterministic_execution,
    decoder_lora_finite_report,
    eos_token_ids,
    inject_decoder_lora,
    load_adapter_checkpoint,
    load_decoder_lora_checkpoint,
    load_model,
)
from tools.sota.ancientdoc import (  # noqa: E402
    DATASET_ID,
    EVALUATION_SPLIT,
    SOURCE_SPLIT,
    iter_manifest,
    manifest_contract,
)
from tools.sota.schema import PredictionRecord, jsonl_record, normalize_text  # noqa: E402


PROMPT = "Text Recognition:"
DEFAULT_MAX_PIXELS = 1003520
DEFAULT_MAX_OUTPUT_TOKENS = 1536
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 16.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _model_args(args: argparse.Namespace, adapter_config: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(
        model_path=args.model_path,
        processor_mode=args.processor_mode,
        max_pixels=args.max_pixels,
        mode=args.mode,
        num_queries=args.num_queries,
        residual_scale_cap=adapter_config["max_residual_scale"],
        initial_residual_scale=adapter_config["initial_residual_scale"],
        use_validity_head=adapter_config["use_validity_head"],
        initial_valid_probability=adapter_config["initial_valid_probability"],
        validity_gating_mode=adapter_config["validity_gating_mode"],
        validity_use_transport_evidence=adapter_config["validity_use_transport_evidence"],
        adapter_precision="fp32",
        region_autoregressive=adapter_config["region_autoregressive"],
        region_decoder_hidden_size=adapter_config["region_decoder_hidden_size"],
        region_decoder_layers=adapter_config["region_decoder_layers"],
        region_decoder_num_heads=adapter_config["region_decoder_num_heads"],
        region_pointer_mask=adapter_config["region_pointer_mask"],
        region_spatial_penalty=adapter_config["region_spatial_penalty"],
        region_spatial_iou_threshold=adapter_config["region_spatial_iou_threshold"],
    )


def _load_variant(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    adapter_config = _read_json(args.checkpoint_dir / "adapter_config.json")
    if adapter_config.get("mode") != args.mode:
        raise ValueError(
            f"adapter mode mismatch: config={adapter_config.get('mode')!r}, requested={args.mode!r}"
        )
    if int(adapter_config.get("num_queries", -1)) != args.num_queries:
        raise ValueError(
            f"adapter query mismatch: config={adapter_config.get('num_queries')!r}, requested={args.num_queries}"
        )
    if int(adapter_config.get("hidden_size", -1)) != 1536:
        raise ValueError(f"unexpected adapter hidden size: {adapter_config.get('hidden_size')!r}")

    model_args = _model_args(args, adapter_config)
    model, processor, bridge = load_model(model_args, torch.device(args.device))
    lora_config = inject_decoder_lora(
        model,
        rank=args.decoder_lora_rank,
        alpha=args.decoder_lora_alpha,
        dropout=args.decoder_lora_dropout,
    )
    load_decoder_lora_checkpoint(args.checkpoint_dir, model)
    load_adapter_checkpoint(args.checkpoint_dir, bridge)
    model.eval()
    bridge.adapter.eval()
    model.config.use_cache = True

    load_report = {
        "status": "loaded",
        "provider": "project_glm_ocr_layout_adapter",
        "device": args.device,
        "dtype": "bf16_backbone_fp32_adapter_lora",
        "mode": args.mode,
        "num_queries": args.num_queries,
        "adapter_config": adapter_config,
        "decoder_lora_config": lora_config,
        "adapter": adapter_finite_report(bridge.adapter),
        "decoder_lora": decoder_lora_finite_report(model),
        "parameters": trainable_parameter_report(model),
    }
    if not load_report["adapter"]["parameters_finite"]:
        raise FloatingPointError("adapter checkpoint contains non-finite parameters")
    if not load_report["decoder_lora"]["parameters_finite"]:
        raise FloatingPointError("decoder LoRA checkpoint contains non-finite parameters")
    return model, processor, bridge, adapter_config, load_report


def _protocol(
    args: argparse.Namespace,
    contract: dict[str, Any],
    adapter_config: dict[str, Any],
    lora_config: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_files = {
        name: {
            "path": str(args.checkpoint_dir / name),
            "sha256": _sha256(args.checkpoint_dir / name),
        }
        for name in ("adapter.safetensors", "decoder_lora.safetensors", "adapter_config.json")
    }
    return {
        "dataset": {
            "dataset_id": DATASET_ID,
            "source_split": SOURCE_SPLIT,
            "evaluation_split": EVALUATION_SPLIT,
            "source_label": str(args.source_label.resolve()),
            "source_label_sha256": _sha256(args.source_label),
            "manifest": contract,
        },
        "model": {
            "name": f"GLM-OCR synthetic {args.mode}",
            "family": "glm_ocr_trained_adapter_zero_shot",
            "checkpoint_id": "zai-org/GLM-OCR",
            "base_model_path": str(args.model_path),
        },
        "checkpoint": checkpoint_files,
        "adapter": adapter_config,
        "decoder_lora": lora_config,
        "prompt": PROMPT,
        "generation": {
            "do_sample": False,
            "mode": "plain",
            "max_output_tokens": args.max_eval_new_tokens,
        },
        "processor": {
            "mode": args.processor_mode,
            "max_pixels": args.max_pixels,
        },
        "model_inputs": ["whole_page_image", "ocr_prompt"],
        "bbox_as_input": False,
        "direction_as_input": False,
        "reading_order_as_input": False,
        "test_used_for_selection": False,
        "selection": None,
        "device": args.device,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "evaluation_pages": args.limit or contract["expected_count"],
    }


def _plain_generation_config() -> RepeatSuppressionConfig:
    return RepeatSuppressionConfig(enabled=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("content_only", "geometry"), required=True)
    parser.add_argument("--num-queries", type=int, default=32)
    parser.add_argument("--decoder-lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--decoder-lora-alpha", type=float, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--decoder-lora-dropout", type=float, default=0.0)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--source-label", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--processor-mode", choices=("fast", "slow"), default="fast")
    parser.add_argument("--max-eval-new-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--allow-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_test:
        raise SystemExit("AncientDoc test is locked; pass --allow-test for the explicit requested run.")
    if args.num_queries != 32:
        raise SystemExit("this synthetic-weight protocol requires --num-queries 32")
    if args.decoder_lora_rank <= 0 or args.decoder_lora_alpha <= 0:
        raise SystemExit("decoder LoRA rank and alpha must be positive")
    if args.max_eval_new_tokens <= 0 or args.max_pixels <= 0:
        raise SystemExit("generation and image budgets must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive when provided")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("invalid shard index/count")
    if not args.source_label.is_file():
        raise SystemExit(f"source label does not exist: {args.source_label}")
    if not args.checkpoint_dir.is_dir():
        raise SystemExit(f"checkpoint directory does not exist: {args.checkpoint_dir}")
    for required in ("adapter.safetensors", "decoder_lora.safetensors", "adapter_config.json"):
        if not (args.checkpoint_dir / required).is_file():
            raise SystemExit(f"checkpoint file is missing: {args.checkpoint_dir / required}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    contract = manifest_contract(args.manifest)
    load_started = time.perf_counter()
    protocol: dict[str, Any] | None = None
    load_status: dict[str, Any]
    model = processor = bridge = None
    try:
        configure_deterministic_execution()
        if not torch.cuda.is_available():
            raise RuntimeError("GLM-OCR AncientDoc test requires CUDA")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        model, processor, bridge, adapter_config, load_status = _load_variant(args)
        protocol = _protocol(
            args,
            contract,
            adapter_config,
            load_status["decoder_lora_config"],
        )
        load_status["load_seconds"] = time.perf_counter() - load_started
    except Exception as exc:
        load_status = {
            "status": "blocked",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc()[-3000:],
            "load_seconds": time.perf_counter() - load_started,
        }
        _write_json(args.output_dir / "load_status.json", load_status)
        _write_json(
            args.output_dir / "run_summary.json",
            {
                "event": "ancientdoc_glmocr_adapter_eval_complete",
                "status": "blocked",
                "model": f"GLM-OCR synthetic {args.mode}",
                "mode": args.mode,
                "num_queries": args.num_queries,
                "pages": 0,
                "ok": 0,
                "failed": 0,
                "test_used_for_selection": False,
            },
        )
        (args.output_dir / "predictions.jsonl").write_text("", encoding="utf-8")
        print(json.dumps(load_status, ensure_ascii=False), flush=True)
        return 3

    _write_json(args.output_dir / "protocol.json", protocol)
    _write_json(args.output_dir / "load_status.json", load_status)

    eos_ids = eos_token_ids(model, processor)
    generation_config = _plain_generation_config()
    prediction_path = args.output_dir / "predictions.jsonl"
    seen = ok = failed = 0
    started = time.perf_counter()
    with prediction_path.open("w", encoding="utf-8", newline="\n") as output:
        for index, record in enumerate(
            iter_manifest(args.manifest, args.image_root, limit=args.limit)
        ):
            if index % args.shard_count != args.shard_index:
                continue
            page_started = time.perf_counter()
            try:
                inputs = prepare_inference_inputs(processor, record, torch.device(args.device))
                bridge.set_grid_thw(inputs["image_grid_thw"])
                bridge.set_region_targets(None)
                bridge.set_region_decode_controls(pointer_mask=None, spatial_penalty=None)
                prompt_length = int(inputs["input_ids"].shape[1])
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
                with torch.inference_mode():
                    generated = generate_with_loop_recovery(
                        model,
                        inputs,
                        prompt_length=prompt_length,
                        eos_token_ids=eos_ids,
                        config=generation_config,
                        max_new_tokens=args.max_eval_new_tokens,
                        mode="plain",
                    )
                if not isinstance(generated, torch.Tensor):
                    generated = getattr(generated, "sequences", None)
                if not isinstance(generated, torch.Tensor) or generated.ndim != 2:
                    raise RuntimeError("generation did not return a rank-2 token tensor")
                generated_tokens = generated[0, prompt_length:]
                decoded = processor.decode(generated_tokens, skip_special_tokens=True)
                generated_list = generated_tokens.detach().cpu().tolist()
                eos_hit = bool(eos_ids and any(int(token) in eos_ids for token in generated_list))
                elapsed = time.perf_counter() - page_started
                runtime = {
                    "latency_seconds": elapsed,
                    "pages_per_second": 1.0 / elapsed if elapsed else None,
                    "max_output_tokens": args.max_eval_new_tokens,
                    "generation_length": len(generated_list),
                    "eos_hit": eos_hit,
                    "generation_limit_hit": len(generated_list) >= args.max_eval_new_tokens,
                    "device": args.device,
                    "mode": args.mode,
                }
                if torch.cuda.is_available():
                    runtime["peak_memory_mib"] = torch.cuda.max_memory_allocated(
                        torch.device(args.device)
                    ) / (1024 * 1024)
                prediction = PredictionRecord(
                    record["page_id"],
                    record["image"],
                    f"GLM-OCR synthetic {args.mode}",
                    {"text": decoded},
                    normalize_text(decoded),
                    "ok",
                    runtime,
                    None,
                )
                ok += 1
            except Exception as exc:
                prediction = PredictionRecord(
                    record["page_id"],
                    record["image"],
                    f"GLM-OCR synthetic {args.mode}",
                    None,
                    "",
                    "failed",
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback_tail": traceback.format_exc()[-2000:],
                        "wall_seconds": time.perf_counter() - page_started,
                        "device": args.device,
                        "mode": args.mode,
                    },
                    None,
                )
                failed += 1
            output.write(jsonl_record(prediction) + "\n")
            output.flush()
            seen += 1
            if args.progress_every > 0 and seen % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "ancientdoc_glmocr_adapter_progress",
                            "mode": args.mode,
                            "seen": seen,
                            "ok": ok,
                            "failed": failed,
                            "shard_index": args.shard_index,
                            "shard_count": args.shard_count,
                            "elapsed_seconds": round(time.perf_counter() - started, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    summary = {
        "event": "ancientdoc_glmocr_adapter_eval_complete",
        "dataset": DATASET_ID,
        "source_split": SOURCE_SPLIT,
        "split": EVALUATION_SPLIT,
        "model": f"GLM-OCR synthetic {args.mode}",
        "mode": args.mode,
        "pages": seen,
        "ok": ok,
        "failed": failed,
        "status": "loaded",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "num_queries": args.num_queries,
        "decoder_lora_rank": args.decoder_lora_rank,
        "decoder_lora_alpha": args.decoder_lora_alpha,
        "max_output_tokens": args.max_eval_new_tokens,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "predictions": str(prediction_path),
        "test_used_for_selection": False,
    }
    _write_json(args.output_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
