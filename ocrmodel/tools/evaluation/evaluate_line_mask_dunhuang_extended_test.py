#!/usr/bin/env python3
"""Evaluate baseline and the validation-locked line mask on a Q32 test extension.

The original 59 Q32 test pages have OCR references. The 77 added Dunhuang
images are included in both inference arms but are unscored because their RGN
files contain regions only, not page transcription.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def read_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"empty manifest: {path}")
    page_ids: set[str] = set()
    for row in rows:
        page_id = str(row.get("page_id", ""))
        if not page_id or page_id in page_ids:
            raise ValueError(f"duplicate or missing page_id in {path}")
        page_ids.add(page_id)
        image = Path(str(row.get("image_path") or row.get("image") or ""))
        if not image.is_absolute():
            image = path.parent / image
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(f"missing image for {page_id}: {image}")
        row["image_path"] = str(image)
        if row.get("split", row.get("official_split")) != "test":
            raise ValueError(f"non-test record in extended test manifest: {page_id}")
        if row.get("reference_available"):
            if not isinstance(row.get("page_text"), str) or not row["page_text"]:
                raise ValueError(f"labeled test row has no transcript: {page_id}")
        elif row.get("page_text") is not None:
            raise ValueError(f"unlabeled test row unexpectedly contains text: {page_id}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--mask-checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.shard_count != 5 or not 0 <= args.shard_index < args.shard_count:
        parser.error("this evaluation uses five disjoint physical-GPU shards")
    return args


def main() -> None:
    args = parse_args()
    code_root = args.code_root.resolve()
    sys.path.insert(0, str(code_root / "src"))
    sys.path.insert(0, str(code_root / "tools" / "evaluation"))

    import torch

    from evaluate_window_mask_routing import eos_ids, load_backbone
    from layout_ocr.line_mask_head import LineMaskConfig, LineMaskHead
    from layout_ocr.line_mask_runtime import LineMaskRuntime
    from layout_ocr.metrics import aggregate_ocr_metrics
    from layout_ocr.stabilization import repetition_diagnostics
    from layout_ocr.data import prepare_inference_inputs

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("test_used_for_selection") is not False:
        raise ValueError("extended test protocol is not selection-locked")
    records = read_records(args.test_manifest)
    if len(records) != int(protocol["split_pages"]["expanded_test_total"]):
        raise ValueError("expanded manifest size differs from the locked protocol")
    if len({str(row["page_id"]) for row in records}) != len(records):
        raise ValueError("expanded test manifest has duplicate page IDs")

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if (
        selection.get("epoch") != 8
        or selection.get("step") != 3456
        or selection.get("test_manifest_read") is not False
        or selection.get("test_used_for_selection") is not False
        or Path(selection.get("checkpoint", "")).resolve() != args.mask_checkpoint.resolve()
    ):
        raise ValueError("mask selection is not the validation-locked epoch8/step3456 checkpoint")
    checkpoint = torch.load(args.mask_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 8 or checkpoint.get("step") != 3456:
        raise ValueError("mask checkpoint payload is not epoch8/step3456")
    config = LineMaskConfig(**checkpoint["config"])

    model, processor = load_backbone(args.model_path, args.backbone_checkpoint, args.device)
    head = LineMaskHead(config).to(args.device)
    head.load_state_dict(checkpoint["head"], strict=True)
    if not all(bool(torch.isfinite(tensor).all()) for tensor in head.state_dict().values()):
        raise FloatingPointError("selected mask checkpoint contains non-finite tensors")
    head.eval()
    runtime = LineMaskRuntime(model, processor.tokenizer, head)
    eos = eos_ids(model, processor)

    shard = records[args.shard_index :: args.shard_count]
    if not shard:
        raise ValueError(f"empty shard {args.shard_index} for {len(records)} pages")
    shard_root = args.output_root / f"shard-{args.shard_index}"
    shard_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "baseline": shard_root / "baseline.jsonl",
        "line_mask_epoch8_step3456": shard_root / "line_mask_epoch8_step3456.jsonl",
    }
    handles = {arm: path.open("w", encoding="utf-8") for arm, path in paths.items()}
    started = time.time()
    processed = 0
    try:
        for record in shard:
            inputs = prepare_inference_inputs(
                processor, {"image_path": record["image_path"]}, torch.device(args.device)
            )
            prompt_length = int(inputs["input_ids"].shape[1])
            for arm, enabled in (("baseline", False), ("line_mask_epoch8_step3456", True)):
                runtime.enabled = enabled
                runtime.set_page(inputs, str(record["page_id"]))
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=int(protocol["generation"]["max_new_tokens"]),
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=eos,
                    )
                tokens = generated[0, prompt_length:]
                token_ids = [int(token) for token in tokens.tolist()]
                prediction = processor.tokenizer.decode(tokens, skip_special_tokens=True)
                row = {
                    "page_id": str(record["page_id"]),
                    "reference": record.get("page_text"),
                    "reference_available": bool(record.get("reference_available")),
                    "sample_source": record.get("sample_source"),
                    "domain": record.get("domain"),
                    "source_group": record.get("source_group"),
                    "prediction": prediction,
                    "generation_tokens": len(token_ids),
                    "generation_eos_hit": any(token in eos for token in token_ids),
                    "generation_limit_hit": not any(token in eos for token in token_ids),
                    "repetition": repetition_diagnostics(prediction),
                }
                handles[arm].write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handles[arm].flush()
            processed += 1
            write_json(
                shard_root / "progress.json",
                {
                    "status": "running",
                    "shard_index": args.shard_index,
                    "pages": processed,
                    "total": len(shard),
                    "last_page_id": str(record["page_id"]),
                    "time": time.time(),
                },
            )
            print(
                json.dumps({"shard": args.shard_index, "pages": processed, "total": len(shard)}),
                flush=True,
            )
    finally:
        for handle in handles.values():
            handle.close()
        runtime.remove()

    expected_ids = [str(record["page_id"]) for record in shard]
    for arm, path in paths.items():
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if [row["page_id"] for row in rows] != expected_ids:
            raise RuntimeError(f"{arm}: output coverage/order differs from shard manifest")
        labeled = [row for row in rows if row["reference_available"]]
        metrics = aggregate_ocr_metrics(
            ((row["reference"], row["prediction"]) for row in labeled), Counter()
        ) if labeled else {"pages": 0, "character_errors": 0}
        write_json(
            shard_root / f"summary-{arm}.json",
            {
                "status": "complete",
                "arm": arm,
                "all_prediction_pages": len(rows),
                "labeled_reference_pages": len(labeled),
                "unlabeled_prediction_pages": len(rows) - len(labeled),
                "metrics_on_labeled_pages_only": metrics,
                "test_manifest_read": True,
                "test_used_for_selection": False,
                "head_finite": True,
                "elapsed_seconds": time.time() - started,
            },
        )

    write_json(
        shard_root / "worker_status.json",
        {
            "status": "complete",
            "shard_index": args.shard_index,
            "pages": len(shard),
            "time": time.time(),
        },
    )


if __name__ == "__main__":
    main()
