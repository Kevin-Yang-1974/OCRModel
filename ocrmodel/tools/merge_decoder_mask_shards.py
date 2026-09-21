#!/usr/bin/env python3
"""Average the mask-head weights of page-sharded runs into one checkpoint.

The head-only regime trains five processes on disjoint page shards, one per card.
Each shard sees only its own pages, so a single shard's head is trained on a
fifth of the data; averaging the five is what turns them back into one head for
the whole page set.  This is weight averaging over models trained on *disjoint*
data, not an ensemble -- the head is small and every shard shares the same frozen
backbone, which is the regime where averaging is well behaved.

Only ``decoder_mask.safetensors`` is averaged.  The rest of a checkpoint (config,
fingerprint) is copied from the first shard, with the shard list recorded so the
merged artifact says what produced it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _safetensors():
    import safetensors.torch as st

    return st


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="average mask-head shards into one checkpoint")
    parser.add_argument("--shard-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weights", nargs="*", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    st = _safetensors()
    shards = [path.resolve() for path in args.shard_dir]
    for shard in shards:
        if not (shard / "decoder_mask.safetensors").is_file():
            raise FileNotFoundError(shard / "decoder_mask.safetensors")
    weights = args.weights or [1.0] * len(shards)
    if len(weights) != len(shards):
        raise SystemExit("--weights must match the number of shards")
    total = sum(weights)
    if total <= 0:
        raise SystemExit("weights must sum to a positive number")

    states = [st.load_file(shard / "decoder_mask.safetensors") for shard in shards]
    keys = set(states[0])
    for index, state in enumerate(states[1:], start=1):
        if set(state) != keys:
            raise SystemExit(f"shard {index} has a different parameter set; cannot average")
    merged = {}
    for key in keys:
        stacked = torch.stack([state[key].float() for state in states], dim=0)
        weight = torch.tensor(weights, dtype=stacked.dtype).view(-1, *([1] * (stacked.ndim - 1)))
        merged[key] = (stacked * weight).sum(dim=0) / total

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    st.save_file(merged, output / "decoder_mask.safetensors")
    for name in ("decoder_mask_config.json",):
        source = shards[0] / name
        if source.is_file():
            (output / name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    fingerprint = {}
    source = shards[0] / "fingerprint.json"
    if source.is_file():
        fingerprint = json.loads(source.read_text(encoding="utf-8"))
    fingerprint.update(
        {
            "shards": [str(shard) for shard in shards],
            "shard_weights": weights,
            "merge": "weighted_mean_over_disjoint_page_shards",
        }
    )
    (output / "fingerprint.json").write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"event": "shards_merged", "shards": len(shards), "output": str(output), "tensors": len(merged)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
