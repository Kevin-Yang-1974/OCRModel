#!/usr/bin/env python3
"""Evaluate the locked full MTHv2 test split for line-mask and baseline arms.

Each worker owns one physical GPU (exposed as cuda:0) and a disjoint slice of
the manifest. It evaluates the no-routing baseline and the selected learned
line-mask checkpoint with identical image+prompt generation settings.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--mask-checkpoint", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.shard_count != 5 or not 0 <= args.shard_index < args.shard_count:
        parser.error("the locked evaluation requires five disjoint shards")
    return args


def main() -> None:
    args = parse_args()
    code_root = args.code_root.resolve()
    sys.path.insert(0, str(code_root / "src"))
    sys.path.insert(0, str(code_root / "tools" / "evaluation"))

    import torch

    from evaluate_window_mask_routing import eos_ids, load_backbone
    from layout_ocr.data import load_records, prepare_inference_inputs
    from layout_ocr.line_mask_head import LineMaskConfig, LineMaskHead
    from layout_ocr.line_mask_runtime import LineMaskRuntime
    from layout_ocr.metrics import aggregate_ocr_metrics
    from layout_ocr.stabilization import repetition_diagnostics

    records = load_records(args.test_manifest)
    if len(records) != 800:
        raise ValueError(f"expected all 800 official test pages, found {len(records)}")
    if any(r.get("split", r.get("official_split")) != "test" for r in records):
        raise ValueError("test manifest contains a non-test record")
    if len({str(r["page_id"]) for r in records}) != 800:
        raise ValueError("test manifest page IDs are not unique")

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if (
        selection.get("epoch") != 8
        or selection.get("step") != 3456
        or selection.get("test_manifest_read") is not False
        or selection.get("test_used_for_selection") is not False
        or Path(selection.get("checkpoint", "")).resolve() != args.mask_checkpoint.resolve()
    ):
        raise ValueError("selection is not the locked epoch8/step3456 checkpoint")

    checkpoint = torch.load(args.mask_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("epoch") != 8 or checkpoint.get("step") != 3456:
        raise ValueError("mask checkpoint payload is not epoch8/step3456")
    config = LineMaskConfig(**checkpoint["config"])

    model, processor = load_backbone(args.model_path, args.backbone_checkpoint, args.device)
    head = LineMaskHead(config).to(args.device)
    head.load_state_dict(checkpoint["head"], strict=True)
    if not all(bool(torch.isfinite(t).all()) for t in head.state_dict().values()):
        raise FloatingPointError("selected mask checkpoint contains non-finite tensors")
    head.eval()
    runtime = LineMaskRuntime(model, processor.tokenizer, head)
    eos = eos_ids(model, processor)

    shard = records[args.shard_index :: args.shard_count]
    if len(shard) != 160:
        raise ValueError(f"shard {args.shard_index} has {len(shard)} pages, expected 160")
    shard_root = args.output_root / f"shard-{args.shard_index}"
    shard_root.mkdir(parents=True, exist_ok=False)
    paths = {
        "baseline": shard_root / "baseline.jsonl",
        "line_mask_epoch8_step3456": shard_root / "line_mask_epoch8_step3456.jsonl",
    }
    handles = {arm: path.open("w", encoding="utf-8") for arm, path in paths.items()}
    processed = {arm: 0 for arm in paths}
    started = time.time()

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
                        max_new_tokens=1536,
                        do_sample=False,
                        use_cache=True,
                        eos_token_id=eos,
                    )
                tokens = generated[0, prompt_length:]
                token_ids = [int(token) for token in tokens.tolist()]
                prediction = processor.tokenizer.decode(tokens, skip_special_tokens=True)
                row = {
                    "page_id": str(record["page_id"]),
                    "reference": record["page_text"],
                    "prediction": prediction,
                    "generation_tokens": len(token_ids),
                    "generation_eos_hit": any(token in eos for token in token_ids),
                    "generation_limit_hit": not any(token in eos for token in token_ids),
                    "repetition": repetition_diagnostics(prediction),
                }
                handles[arm].write(
                    json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                )
                handles[arm].flush()
                processed[arm] += 1
                write_json(
                    shard_root / f"progress-{arm}.json",
                    {
                        "status": "running",
                        "shard_index": args.shard_index,
                        "pages": processed[arm],
                        "total": len(shard),
                        "last_page_id": str(record["page_id"]),
                        "time": time.time(),
                    },
                )
            print(
                json.dumps(
                    {
                        "shard": args.shard_index,
                        "pages": processed["baseline"],
                        "total": len(shard),
                    }
                ),
                flush=True,
            )
    finally:
        for handle in handles.values():
            handle.close()
        runtime.remove()

    rows_by_arm = {}
    for arm, path in paths.items():
        with path.open(encoding="utf-8") as handle:
            rows_by_arm[arm] = [json.loads(line) for line in handle if line.strip()]
    expected_ids = [str(record["page_id"]) for record in shard]
    for arm, rows in rows_by_arm.items():
        if [row["page_id"] for row in rows] != expected_ids:
            raise RuntimeError(f"{arm}: output coverage/order differs from shard manifest")
        metrics = aggregate_ocr_metrics(
            ((row["reference"], row["prediction"]) for row in rows), Counter()
        )
        metrics.update(
            {
                "eos_pages": sum(row["generation_eos_hit"] for row in rows),
                "generation_limit_hits": sum(row["generation_limit_hit"] for row in rows),
                "loop_pages": sum(
                    bool(row["repetition"].get("repeated_cycle_detected")) for row in rows
                ),
            }
        )
        metrics["loop_rate"] = metrics["loop_pages"] / max(1, metrics["pages"])
        write_json(
            shard_root / f"summary-{arm}.json",
            {
                "status": "complete",
                "arm": arm,
                "metrics": metrics,
                "page_ids": expected_ids,
                "test_manifest_sha256": sha256(args.test_manifest),
                "test_manifest_read": True,
                "test_used_for_selection": False,
                "elapsed_seconds": time.time() - started,
                "head_finite": True,
            },
        )

    write_json(
        shard_root / "worker_status.json",
        {"status": "complete", "shard_index": args.shard_index, "time": time.time()},
    )


if __name__ == "__main__":
    main()
