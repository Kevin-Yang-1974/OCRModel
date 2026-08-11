from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from GOT.model import GOTConfig, GOTQwenForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a saved GOT2 VLQA checkpoint and optionally reload the model."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--stage", choices=("p1", "p2"), required=True)
    parser.add_argument("--max-regions", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-model-reload",
        action="store_true",
        help="Inspect config/safetensors only. P1 uses this because P2 performs the reload.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def select_unique_key(keys: list[str], suffix: str) -> str:
    matches = [key for key in keys if key.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one safetensors key ending in {suffix!r}, got {matches}."
        )
    return matches[0]


def inspect_checkpoint(
    model_dir: Path,
    stage: str,
    max_regions: int,
) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    metrics_path = model_dir / "layout_training_metrics.json"
    for path in (config_path, weights_path, metrics_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = load_json(config_path)
    metrics = load_json(metrics_path)
    if config.get("use_vlqa") is not True:
        raise RuntimeError("Saved config.use_vlqa is not true.")
    if int(config.get("vlqa_num_queries", -1)) != max_regions:
        raise RuntimeError(
            "Saved vlqa_num_queries does not match max_regions: "
            f"{config.get('vlqa_num_queries')} != {max_regions}."
        )
    if metrics.get("layout_stage") != stage:
        raise RuntimeError(
            f"Metrics stage mismatch: {metrics.get('layout_stage')!r} != {stage!r}."
        )
    global_step = int(metrics.get("global_step", 0))
    train_loss = float(metrics.get("train_loss", float("nan")))
    if global_step < 1 or not math.isfinite(train_loss) or train_loss <= 0.0:
        raise RuntimeError(
            f"Invalid training metrics: global_step={global_step}, train_loss={train_loss}."
        )

    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        layout_keys = [key for key in keys if "layout_adapter." in key]
        projector_keys = [key for key in keys if "mm_projector_vary." in key]
        if not layout_keys:
            raise RuntimeError("No layout_adapter tensors were saved.")
        if stage == "p2" and not projector_keys:
            raise RuntimeError("No mm_projector_vary tensors were saved for P2.")

        checked_keys = layout_keys + (projector_keys if stage == "p2" else [])
        nonfinite_keys = [
            key
            for key in checked_keys
            if not torch.isfinite(handle.get_tensor(key).float()).all().item()
        ]
        if nonfinite_keys:
            raise RuntimeError(f"Non-finite tensors in checkpoint: {nonfinite_keys}")

        gate_key = select_unique_key(keys, ".layout_adapter.residual_gate")
        gate = float(handle.get_tensor(gate_key).float().item())
        if not math.isfinite(gate):
            raise RuntimeError("residual_gate is not finite.")
        if stage == "p1" and gate != 0.0:
            raise RuntimeError(f"P1 residual_gate must remain exactly zero, got {gate}.")

    return {
        "config_use_vlqa": True,
        "global_step": global_step,
        "layout_tensor_count": len(layout_keys),
        "projector_tensor_count": len(projector_keys),
        "residual_gate": gate,
        "train_loss": train_loss,
        "weights_bytes": weights_path.stat().st_size,
    }


def reload_checkpoint(model_dir: Path, max_regions: int) -> dict[str, Any]:
    config = GOTConfig.from_pretrained(model_dir, local_files_only=True)
    model = GOTQwenForCausalLM.from_pretrained(
        model_dir,
        config=config,
        use_safetensors=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    adapter = model.get_model().layout_adapter
    if adapter is None:
        raise RuntimeError("Reloaded model does not contain layout_adapter.")
    if adapter.num_queries != max_regions:
        raise RuntimeError(
            f"Reloaded adapter query count mismatch: {adapter.num_queries} != {max_regions}."
        )
    result = {
        "model_class": type(model).__name__,
        "layout_adapter_class": type(adapter).__name__,
        "max_regions": adapter.num_queries,
    }
    del model
    gc.collect()
    return result


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.max_regions < 1:
        raise ValueError("--max-regions must be positive.")
    model_dir = args.model.resolve()
    payload: dict[str, Any] = {
        "status": "ok",
        "stage": args.stage,
        "model": str(model_dir),
        "safetensors": inspect_checkpoint(model_dir, args.stage, args.max_regions),
        "model_reload": None,
    }
    if not args.skip_model_reload:
        payload["model_reload"] = reload_checkpoint(model_dir, args.max_regions)
    write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        raise
