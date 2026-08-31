#!/usr/bin/env python3
"""Check a GOT2 checkpoint for non-finite weights without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _tensor_status(tensor: Any) -> tuple[bool, int]:
    import torch

    if not torch.is_tensor(tensor) or not tensor.is_floating_point():
        return True, 0
    finite = torch.isfinite(tensor)
    return bool(finite.all()), int((~finite).sum().item())


def inspect_checkpoint(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    files = sorted(checkpoint.glob("*.safetensors"))
    files += sorted(checkpoint.glob("pytorch_model*.bin"))
    files += sorted(checkpoint.glob("adapter_model*.bin"))
    if not files:
        raise FileNotFoundError(f"no .safetensors or .bin weights in {checkpoint}")
    nonfinite: list[dict[str, Any]] = []
    tensor_count = 0
    for path in files:
        if path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    tensor_count += 1
                    finite, count = _tensor_status(handle.get_tensor(name))
                    if not finite:
                        nonfinite.append({"file": path.name, "tensor": name, "nonfinite_values": count})
        else:
            import torch

            state = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(state, dict):
                continue
            for name, tensor in state.items():
                tensor_count += 1
                finite, count = _tensor_status(tensor)
                if not finite:
                    nonfinite.append({"file": path.name, "tensor": str(name), "nonfinite_values": count})
    health_path = checkpoint / "checkpoint_health.json"
    recorded = None
    if health_path.is_file():
        recorded = json.loads(health_path.read_text(encoding="utf-8"))
    return {
        "status": "ok" if not nonfinite else "nonfinite_weights",
        "checkpoint": str(checkpoint),
        "weight_files": [path.name for path in files],
        "tensor_count": tensor_count,
        "nonfinite_tensors": nonfinite[:32],
        "recorded_checkpoint_health": recorded,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    result = inspect_checkpoint(args.checkpoint)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
