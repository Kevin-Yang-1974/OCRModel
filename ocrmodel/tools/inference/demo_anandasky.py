from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    workspace = Path(os.environ.get("OCR_WORKSPACE", default_root.parent))
    model_dir = Path(os.environ.get("ANANDASKY_MODEL", workspace / "models" / "AnandaSky"))
    parser = argparse.ArgumentParser(
        description="Run local AnandaSky transcription on one cropped line-level image."
    )
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--model-dir", type=Path, default=model_dir)
    parser.add_argument(
        "--line-image",
        type=Path,
        default=model_dir / "assets" / "duowendiyi.jpg",
        help="A cropped horizontal text line or vertical text column; page images are not accepted.",
    )
    parser.add_argument(
        "--max-visual-tokens",
        type=int,
        default=2048,
        help="Reject likely page-level inputs before loading model weights.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")
    if args.max_visual_tokens < 1:
        raise ValueError("--max-visual-tokens must be positive.")
    if not args.line_image.is_file():
        raise FileNotFoundError(f"Line image does not exist: {args.line_image}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run through run_anandasky.sh with CUDA_VISIBLE_DEVICES=0.")

    torch.cuda.set_device(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    if capability[0] < 8:
        raise RuntimeError(f"AnandaSky requires SM80 or newer, found SM{capability[0]}{capability[1]}.")

    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    image = Image.open(args.line_image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    input_length = inputs["input_ids"].shape[1]
    visual_tokens = int(inputs["patch_attention_mask"].sum().item())
    padded_height, padded_width = (int(value) for value in inputs["pixel_values"].shape[-2:])
    if visual_tokens > args.max_visual_tokens:
        raise ValueError(
            "Input exceeds the line-level visual-token limit: "
            f"visual_tokens={visual_tokens}, limit={args.max_visual_tokens}. "
            "Crop the source page into one horizontal line or one vertical column first."
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    inputs["input_ids"] = inputs["input_ids"].to(device=device, non_blocking=True)
    inputs["attention_mask"] = inputs["attention_mask"].to(device=device, non_blocking=True)
    inputs["pixel_values"] = inputs["pixel_values"].to(
        device=device,
        dtype=dtype,
        non_blocking=True,
    )
    inputs["patch_attention_mask"] = inputs["patch_attention_mask"].to(
        device=device,
        non_blocking=True,
    )

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=args.max_new_tokens,
        )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    generated_ids = output[0, input_length:]
    transcription = processor.decode(generated_ids, skip_special_tokens=True).strip()
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)

    print("ANANDASKY_DEMO_OK")
    print(f"line_image={args.line_image.resolve()}")
    print(f"image_size={image.width}x{image.height}")
    print(f"processor_padded_size={padded_width}x{padded_height}")
    print(f"visual_tokens={visual_tokens}")
    print(f"device={torch.cuda.get_device_name(args.device)}")
    print(f"compute_capability={capability[0]}.{capability[1]}")
    print(f"generated_tokens={generated_ids.numel()}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"peak_allocated_mib={peak_mib:.1f}")
    print("transcription:")
    print(transcription)


if __name__ == "__main__":
    main()
