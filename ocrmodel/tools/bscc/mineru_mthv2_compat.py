#!/usr/bin/env python3
"""Prepare MTHv2 pages for the official-shaped MinerU SFT entry point.

The public MinerU fine-tuning tutorial provides ``run_mineru.py sft`` but the
downloadable ``mineru_ext`` package is not part of the public repository.  This
module supplies the BSCC compatibility boundary: it converts the existing
MTHv2 whole-page manifests to the standard MS-SWIFT multimodal JSONL schema and
writes the absolute-path YAML consumed by the compatibility entry point.

Only train and validation are opened here.  The test manifest is deliberately
not part of this preparation path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


COMPAT_VERSION = "mineru_mthv2_compat_v1"
MODEL_NAME = "MinerU2.5-Pro-2605-1.2B"
PROMPT = "<image>\nText Recognition:"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _response_from_row(row: Dict[str, Any], manifest: Path, line_number: int) -> str:
    page_text = row.get("page_text")
    if isinstance(page_text, str) and page_text:
        return page_text

    for message in row.get("conversations") or []:
        if message.get("from") in {"gpt", "assistant"}:
            value = message.get("value")
            if isinstance(value, str) and value:
                return value
    raise ValueError(f"empty MTHv2 target at {manifest}:{line_number}")


def _image_path(manifest: Path, image_value: Any, line_number: int) -> Path:
    if not isinstance(image_value, str) or not image_value:
        raise ValueError(f"missing image at {manifest}:{line_number}")
    candidate = Path(image_value)
    if not candidate.is_absolute():
        candidate = manifest.parent / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"MTHv2 image does not exist at {manifest}:{line_number}: {candidate}")
    return candidate


def _read_split(dataset_root: Path, split: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    manifest = (dataset_root / split / "manifest.jsonl").resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"missing MTHv2 {split} manifest: {manifest}")

    records: List[Dict[str, Any]] = []
    target_lengths: List[int] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            target = _response_from_row(row, manifest, line_number)
            image = _image_path(manifest, row.get("image"), line_number)
            records.append({
                "messages": [
                    {"role": "user", "content": PROMPT},
                    {"role": "assistant", "content": target},
                ],
                "images": [str(image)],
            })
            target_lengths.append(len(target))

    if not records:
        raise ValueError(f"empty MTHv2 {split} manifest: {manifest}")
    return records, {
        "manifest": str(manifest),
        "manifest_sha256": _sha256(manifest),
        "count": len(records),
        "target_chars_min": min(target_lengths),
        "target_chars_max": max(target_lengths),
        "target_chars_mean": round(sum(target_lengths) / len(target_lengths), 3),
    }


def _write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    return _sha256(path)


def _yaml_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _write_config(
    path: Path,
    *,
    model_dir: Path,
    train_jsonl: Path,
    validation_jsonl: Path,
    output_dir: Path,
    max_steps: int,
    max_length: int,
    max_pixels: int,
) -> None:
    config = "\n".join([
        f"model: {_yaml_quote(str(model_dir))}",
        "model_type: qwen2_vl",
        "torch_dtype: bfloat16",
        "attn_impl: sdpa",
        "tuner_type: lora",
        "lora_rank: 8",
        "lora_alpha: 32",
        "lora_dropout: 0.05",
        "target_modules: all-linear",
        "freeze_vit: true",
        "freeze_aligner: true",
        "dataset:",
        f"  - {_yaml_quote(str(train_jsonl))}",
        "val_dataset:",
        f"  - {_yaml_quote(str(validation_jsonl))}",
        "split_dataset_ratio: 0",
        "data_seed: 42",
        "dataset_shuffle: true",
        "val_dataset_shuffle: false",
        "dataset_num_proc: 4",
        "load_from_cache_file: false",
        f"max_length: {max_length}",
        f"max_pixels: {max_pixels}",
        "truncation_strategy: delete",
        "use_chat_template: true",
        "loss_scale: default",
        "per_device_train_batch_size: 1",
        "per_device_eval_batch_size: 1",
        "gradient_accumulation_steps: 1",
        "gradient_checkpointing: true",
        "vit_gradient_checkpointing: false",
        "learning_rate: 0.0001",
        "warmup_ratio: 0.05",
        "lr_scheduler_type: cosine",
        f"max_steps: {max_steps}",
        "save_strategy: steps",
        "save_steps: 1000",
        "save_total_limit: 5",
        "eval_strategy: steps",
        "eval_steps: 1000",
        "logging_steps: 10",
        "dataloader_num_workers: 4",
        "seed: 42",
        "output_dir: " + _yaml_quote(str(output_dir)),
        "add_version: false",
        "report_to:",
        "  - tensorboard",
        "",
    ])
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(config)


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    protocol_file = Path(args.protocol_file).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = output_dir / "train.jsonl"
    validation_jsonl = output_dir / "validation.jsonl"
    metadata_file = output_dir / "metadata.json"
    expected = (train_jsonl, validation_jsonl, metadata_file)
    existing = any(output_dir.iterdir())

    if existing and not all(path.is_file() for path in expected):
        raise FileExistsError(f"refusing to reuse partial compatibility data directory: {output_dir}")

    if all(path.is_file() for path in expected):
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata.get("compat_version") != COMPAT_VERSION:
            raise FileExistsError(f"compatibility data directory belongs to another generator: {output_dir}")
        train_info = metadata["splits"]["train"]
        validation_info = metadata["splits"]["validation"]
        print(json.dumps({
            "event": "mineru_mthv2_data_reused",
            "output_dir": str(output_dir),
            "train_count": train_info["count"],
            "validation_count": validation_info["count"],
            "test_manifest_read": False,
        }, ensure_ascii=False, separators=(",", ":")))
    else:
        train_records, train_info = _read_split(dataset_root, "train")
        validation_records, validation_info = _read_split(dataset_root, "validation")
        train_info["jsonl"] = str(train_jsonl)
        train_info["jsonl_sha256"] = _write_jsonl(train_jsonl, train_records)
        validation_info["jsonl"] = str(validation_jsonl)
        validation_info["jsonl_sha256"] = _write_jsonl(validation_jsonl, validation_records)
        metadata = {
            "compat_version": COMPAT_VERSION,
            "model_name": MODEL_NAME,
            "task": "mthv2_whole_page_text_recognition",
            "prompt": PROMPT,
            "dataset_root": str(dataset_root),
            "splits": {"train": train_info, "validation": validation_info},
            "test_manifest_read": False,
            "selection_pending": True,
        }
        metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if protocol_file.exists():
        raise FileExistsError(f"protocol file already exists; use a new run id: {protocol_file}")
    protocol_file.parent.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol_version": "mthv2_train_validation_no_test_v1",
        "run_family": "MinerU2.5-Pro",
        "model": MODEL_NAME,
        "task": "whole_page_text_recognition_sft",
        "entrypoint": "python run_mineru.py sft --config <absolute_config>",
        "compatibility_mode": "bscc_ms_swift_forwarder",
        "source": "MTHv2 converted mthv2_layout_page_v1_real",
        "train_manifest": str(dataset_root / "train" / "manifest.jsonl"),
        "validation_manifest": str(dataset_root / "validation" / "manifest.jsonl"),
        "train_count": metadata["splits"]["train"]["count"],
        "validation_count": metadata["splits"]["validation"]["count"],
        "test_manifest_read": False,
        "selection_pending": True,
        "test_policy": "MTHv2 test is not opened by preparation or training.",
    }
    protocol_file.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    config_path = None
    if args.config_path:
        if not args.model_dir or not args.run_output_dir:
            raise ValueError("--model-dir and --run-output-dir are required with --config-path")
        config_path = Path(args.config_path).resolve()
        if config_path.exists():
            raise FileExistsError(f"config file already exists; use a new run id: {config_path}")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        _write_config(
            config_path,
            model_dir=Path(args.model_dir).resolve(),
            train_jsonl=train_jsonl,
            validation_jsonl=validation_jsonl,
            output_dir=Path(args.run_output_dir).resolve(),
            max_steps=args.max_steps,
            max_length=args.max_length,
            max_pixels=args.max_pixels,
        )

    result = {
        "event": "mineru_mthv2_compat_ready",
        "compat_version": COMPAT_VERSION,
        "output_dir": str(output_dir),
        "protocol_file": str(protocol_file),
        "config_file": str(config_path) if config_path else None,
        "train_count": metadata["splits"]["train"]["count"],
        "validation_count": metadata["splits"]["validation"]["count"],
        "test_manifest_read": False,
        "selection_pending": True,
    }
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-file", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--config-path")
    parser.add_argument("--run-output-dir")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--max-pixels", type=int, default=1003520)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
