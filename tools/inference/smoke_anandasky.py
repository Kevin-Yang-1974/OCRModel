from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    workspace = Path(os.environ.get("OCR_WORKSPACE", default_root.parent))
    model_dir = Path(os.environ.get("ANANDASKY_MODEL", workspace / "models" / "AnandaSky"))
    parser = argparse.ArgumentParser(description="Validate AnandaSky artifacts and optional compatible-GPU loading.")
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--model-dir", type=Path, default=model_dir)
    parser.add_argument("--load-model", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def count_safetensor_parameters(path: Path) -> tuple[int, int]:
    parameter_count = 0
    tensor_count = 0
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor_count += 1
            parameter_count += math.prod(handle.get_slice(key).get_shape())
    return parameter_count, tensor_count


def main() -> None:
    args = parse_args()
    config = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8"))
    parameter_count, tensor_count = count_safetensor_parameters(args.model_dir / "model.safetensors")

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    image = Image.open(args.model_dir / "assets" / "duowendiyi.jpg").convert("RGB")
    processed = processor(images=image, return_tensors="pt")

    print("ANANDASKY_ARTIFACTS_OK")
    print(f"parameters={parameter_count}")
    print(f"tensors={tensor_count}")
    print(f"patch_size={config['patch_size']}")
    print(f"merge_factor={config['encoder_2d_merge_factor']}")
    print(f"pixel_values_shape={tuple(processed['pixel_values'].shape)}")
    print(f"valid_patch_tokens={int(processed['patch_attention_mask'].sum())}")

    if not torch.cuda.is_available():
        print("ANANDASKY_RUNTIME_BLOCKED: CUDA is unavailable.")
        return

    capability = torch.cuda.get_device_capability(args.device)
    device_name = torch.cuda.get_device_name(args.device)
    print(f"device={device_name}")
    print(f"compute_capability={capability[0]}.{capability[1]}")

    if capability[0] < 8:
        print("ANANDASKY_RUNTIME_BLOCKED: official FlashAttention path requires Ampere (SM80) or newer.")
        return
    if importlib.util.find_spec("flash_attn") is None:
        print("ANANDASKY_RUNTIME_BLOCKED: install flash-attn on the compatible target server.")
        return
    if not args.load_model:
        print("ANANDASKY_MODEL_LOAD_SKIPPED: rerun with --load-model after the artifact check.")
        return

    torch.cuda.set_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(f"cuda:{args.device}")
    print(f"ANANDASKY_MODEL_LOAD_OK parameters={sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
