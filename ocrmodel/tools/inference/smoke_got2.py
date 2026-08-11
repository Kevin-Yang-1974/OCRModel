from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    ocrmodel_root = Path(os.environ.get("OCRMODEL_ROOT", default_root))
    workspace = Path(os.environ.get("OCR_WORKSPACE", ocrmodel_root.parent))
    source_root = Path(
        os.environ.get("GOT_PROJECT_ROOT", ocrmodel_root / "src" / "GOT-OCR-2.0")
    )
    model_dir = Path(os.environ.get("GOT_SOURCE_MODEL", workspace / "models" / "GOT-OCR2_0"))
    parser = argparse.ArgumentParser(
        description="Run a short, offline GOT-OCR2.0 FP16 check on one line-level image."
    )
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--source-root", type=Path, default=source_root)
    parser.add_argument("--model-dir", type=Path, default=model_dir)
    parser.add_argument(
        "--line-image",
        type=Path,
        default=model_dir / "assets" / "train_sample.jpg",
        help="A cropped horizontal text line or vertical text column.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.line_image.is_file():
        raise FileNotFoundError(f"Line image does not exist: {args.line_image}")
    if not (args.source_root / "GOT" / "__init__.py").is_file():
        raise FileNotFoundError(f"GOT source tree does not exist: {args.source_root}")
    sys.path.insert(0, str(args.source_root))

    from GOT.model import GOTQwenForCausalLM
    from GOT.model.plug.blip_process import BlipImageEvalProcessor
    from GOT.utils.conversation import conv_templates

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the GOT2 environment.")
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = GOTQwenForCausalLM.from_pretrained(
        args.model_dir,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        local_files_only=True,
        pad_token_id=151643,
        torch_dtype=torch.float16,
    )
    model = model.to(device=device, dtype=torch.float16).eval()

    image = Image.open(args.line_image).convert("RGB")
    processor = BlipImageEvalProcessor(image_size=1024)
    image_tensor = processor(image).unsqueeze(0).to(device=device, dtype=torch.float16)
    image_tensor_high = processor(image.copy()).unsqueeze(0).to(device=device, dtype=torch.float16)

    question = "<img>" + "<imgpad>" * 256 + "</img>\nOCR: "
    conversation = conv_templates["mpt"].copy()
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    input_ids = torch.as_tensor(tokenizer([prompt]).input_ids, device=device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        output_ids = model.generate(
            input_ids,
            images=[(image_tensor, image_tensor_high)],
            do_sample=False,
            num_beams=1,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
        )

    generated_ids = output_ids[0, input_ids.shape[1] :]
    generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    peak_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    print("GOT2_SMOKE_OK")
    print(f"line_image={args.line_image.resolve()}")
    print(f"image_size={image.width}x{image.height}")
    print("processor_size=1024x1024")
    print("visual_tokens=256")
    print(f"generated_tokens={generated_ids.numel()}")
    print(f"device={torch.cuda.get_device_name(args.device)}")
    print(f"dtype={next(model.parameters()).dtype}")
    print(f"peak_allocated_mib={peak_mib:.1f}")
    print(f"generated={generated!r}")


if __name__ == "__main__":
    main()
