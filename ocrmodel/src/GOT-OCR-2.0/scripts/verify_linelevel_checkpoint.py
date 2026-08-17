from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def select_key(keys: list[str], fragment: str, suffix: str = "weight") -> str:
    matches = [key for key in keys if fragment in key and key.endswith(suffix)]
    if not matches:
        raise KeyError(f"No tensor key contains {fragment!r} and ends with {suffix!r}.")
    return sorted(matches)[0]


def compare_tensor(source_file: Path, trained_file: Path, key: str) -> dict[str, object]:
    with safe_open(source_file, framework="pt", device="cpu") as handle:
        source = handle.get_tensor(key).float()
    with safe_open(trained_file, framework="pt", device="cpu") as handle:
        trained = handle.get_tensor(key).float()
    delta = trained - source
    return {
        "key": key,
        "shape": list(source.shape),
        "changed_elements": int(torch.count_nonzero(delta).item()),
        "max_abs_delta": float(delta.abs().max().item()),
        "mean_abs_delta": float(delta.abs().mean().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that a GOT continuation checkpoint changed only trainable paths.")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--trained-model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--metrics-name", default="linelevel_training_metrics.json")
    parser.add_argument("--expected-train-scope")
    parser.add_argument(
        "--allow-no-observed-trainable-delta",
        action="store_true",
        help=(
            "Allow a one-step low-learning-rate smoke to complete when sampled BF16 "
            "weights quantize back to their source values. Formal runs must not use this."
        ),
    )
    args = parser.parse_args()

    source_file = args.source_model.resolve() / "model.safetensors"
    if not source_file.is_file():
        raise FileNotFoundError(source_file)

    with safe_open(source_file, framework="pt", device="cpu") as source_handle:
        source_keys = list(source_handle.keys())
    projector_key = select_key(source_keys, "mm_projector_vary")
    decoder_key = select_key(source_keys, "model.layers.0.self_attn.q_proj")
    vision_key = select_key(source_keys, "vision_tower_high.patch_embed")
    if args.source_only:
        print("GOT_SOURCE_MODEL_KEYS_OK")
        print(f"tensor_count={len(source_keys)}")
        print(f"projector_key={projector_key}")
        print(f"decoder_key={decoder_key}")
        print(f"vision_key={vision_key}")
        return

    if args.trained_model is None or args.output is None:
        raise ValueError("--trained-model and --output are required unless --source-only is used.")
    trained_file = args.trained_model.resolve() / "model.safetensors"
    metrics_file = args.trained_model.resolve() / args.metrics_name
    for path in (trained_file, metrics_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    with safe_open(trained_file, framework="pt", device="cpu") as trained_handle:
        trained_keys = list(trained_handle.keys())
    source_key_set = set(source_keys)
    trained_key_set = set(trained_keys)
    missing_from_trained = sorted(source_key_set - trained_key_set)
    new_in_trained = sorted(trained_key_set - source_key_set)
    if missing_from_trained or new_in_trained:
        raise RuntimeError(
            "The trained checkpoint tensor-key set differs from the source model: "
            f"missing={missing_from_trained}, new={new_in_trained}."
        )

    comparisons = {
        "projector": compare_tensor(source_file, trained_file, projector_key),
        "decoder": compare_tensor(source_file, trained_file, decoder_key),
        "vision": compare_tensor(source_file, trained_file, vision_key),
    }
    if comparisons["vision"]["changed_elements"] != 0:
        raise RuntimeError("Frozen vision tensor changed unexpectedly.")

    training_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    if (
        args.expected_train_scope is not None
        and training_metrics.get("train_scope") != args.expected_train_scope
    ):
        raise RuntimeError(
            f"Unexpected train_scope={training_metrics.get('train_scope')!r}; "
            f"expected {args.expected_train_scope!r}."
        )
    train_scope = args.expected_train_scope or training_metrics.get("train_scope")
    required_trainable_components = ["projector"]
    if train_scope == "decoder_projector":
        required_trainable_components.append("decoder")
    missing_trainable_deltas = [
        name
        for name in required_trainable_components
        if comparisons[name]["changed_elements"] == 0
    ]
    if missing_trainable_deltas and not args.allow_no_observed_trainable_delta:
        raise RuntimeError(
            "Sampled trainable tensors did not change for: "
            f"{missing_trainable_deltas}; refusing a formal continuation checkpoint."
        )
    if train_scope == "projector" and comparisons["decoder"]["changed_elements"] != 0:
        raise RuntimeError("A frozen decoder tensor changed during projector-only training.")
    if train_scope not in {"projector", "decoder_projector"}:
        raise RuntimeError(f"Unsupported or missing train_scope in metrics: {train_scope!r}.")
    report = {
        "source_model": str(args.source_model.resolve()),
        "trained_model": str(args.trained_model.resolve()),
        "training_metrics": training_metrics,
        "key_set": {
            "source_count": len(source_keys),
            "trained_count": len(trained_keys),
            "missing_from_trained": missing_from_trained,
            "new_in_trained": new_in_trained,
        },
        "tensor_comparisons": comparisons,
        "required_trainable_components": required_trainable_components,
        "missing_trainable_deltas": missing_trainable_deltas,
        "no_delta_exception_used": bool(
            args.allow_no_observed_trainable_delta and missing_trainable_deltas
        ),
        "status": "checkpoint_continuation_verified",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("GOT_CHECKPOINT_CONTINUATION_OK")
    print(json.dumps(comparisons, ensure_ascii=False))


if __name__ == "__main__":
    main()
