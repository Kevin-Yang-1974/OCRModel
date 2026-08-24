"""At-most-one-step official fine-tuning smoke.

This entrypoint intentionally supports only adapters that have a verified
official training route. It never opens the MTHv2 test manifest and writes a
new, isolated smoke run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.sota.adapters import AdapterError, build_adapter
from tools.sota.mthv2 import iter_manifest, manifest_contract
from tools.sota.registry import get_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--max-steps", type=int, default=1)
    args = parser.parse_args()
    if args.max_steps != 1:
        raise SystemExit("Current phase permits exactly one optimizer step for external fine-tune smoke.")
    spec = get_model(args.model)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    payload = {"model": spec.to_dict(), "split": args.split, "max_steps": 1, "test_used": False, "protocol": manifest_contract(args.manifest, "validation")}
    (args.output_dir / "protocol.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if spec.finetune_status != "official_entrypoint_available":
        result = {"event": "sota_finetune_smoke_blocked", "model": args.model, "status": spec.finetune_status, "reason": spec.finetune_source, "test_used": False}
        (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False))
        return 0
    try:
        import torch
        adapter = build_adapter(spec, args.model_root, device=args.device, dtype=args.dtype)
        adapter.load()
        sample = next(iter_manifest(args.manifest, args.image_root, split="train", limit=1))
        if not hasattr(adapter, "train_one_step"):
            raise AdapterError("official_adapter_has_no_train_one_step_contract")
        result = adapter.train_one_step(Path(sample["image_path"]), str(sample.get("page_text", "")), args.output_dir)
        if not result.get("loss_finite") or not result.get("gradient_finite") or not result.get("checkpoint_reload_ok"):
            raise AdapterError(f"one_step_contract_failed: {result}")
        result.update({"event": "sota_finetune_smoke_complete", "model": args.model, "train_pages": 1, "optimizer_steps": 1, "test_used": False, "torch_version": torch.__version__})
    except Exception as exc:
        result = {"event": "sota_finetune_smoke_failed", "model": args.model, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "test_used": False}
    (args.output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("event") == "sota_finetune_smoke_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
