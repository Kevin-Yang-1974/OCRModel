from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    parser = argparse.ArgumentParser(description="Run GOT-OCR2.0 on pre-split vertical columns.")
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--source-root", type=Path, default=source_root)
    parser.add_argument("--model-dir", type=Path, default=model_dir)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_text_views(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    successful = [row for row in rows if "text" in row]
    detailed_parts = []
    for row in successful:
        detailed_parts.append(
            f"[column_{row['reading_index']:03d} source_box={row['source_box']}]\n{row['text']}"
        )
    (output_dir / "columns.txt").write_text("\n\n".join(detailed_parts) + "\n", encoding="utf-8")
    merged = "\n".join(str(row["text"]).strip() for row in successful if str(row["text"]).strip())
    (output_dir / "merged_right_to_left.txt").write_text(merged + "\n", encoding="utf-8")


def strip_stop_text(text: str, stop_str: str) -> str:
    cleaned = text.strip()
    if cleaned.endswith(stop_str):
        cleaned = cleaned[: -len(stop_str)]
    return cleaned.strip()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive when supplied.")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {args.manifest}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run through run_got2.sh with CUDA_VISIBLE_DEVICES=0.")

    if not (args.source_root / "GOT" / "__init__.py").is_file():
        raise FileNotFoundError(f"GOT source tree does not exist: {args.source_root}")
    sys.path.insert(0, str(args.source_root))
    from GOT.model import GOTQwenForCausalLM
    from GOT.model.plug.blip_process import BlipImageEvalProcessor
    from GOT.utils.conversation import SeparatorStyle, conv_templates
    from GOT.utils.utils import KeywordsStoppingCriteria, disable_torch_init

    disable_torch_init()
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    columns = manifest["columns"]
    if args.limit is not None:
        columns = columns[: args.limit]

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
    ).to(device=device, dtype=torch.float16).eval()
    processor = BlipImageEvalProcessor(image_size=1024)

    question = "<img>" + "<imgpad>" * 256 + "</img>\nOCR: "
    conversation = conv_templates["mpt"].copy()
    conversation.append_message(conversation.roles[0], question)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    input_ids = torch.as_tensor(tokenizer([prompt]).input_ids, device=device)
    stop_str = conversation.sep if conversation.sep_style != SeparatorStyle.TWO else conversation.sep2

    report: dict[str, Any] = {
        "schema_version": 1,
        "model": "GOT-OCR2.0",
        "model_dir": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "input_sha256": manifest["input"]["sha256"],
        "reading_order": manifest["reading_order"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "device": torch.cuda.get_device_name(args.device),
        "dtype": str(next(model.parameters()).dtype),
        "results": [],
    }
    result_path = args.output_dir / "results.json"
    had_errors = False

    for column_info in columns:
        reading_index = int(column_info["reading_index"])
        relative_image = column_info["files"]["got"]
        image_path = args.manifest.parent / relative_image
        row: dict[str, Any] = {
            "reading_index": reading_index,
            "source_box": column_info["source_box"],
            "image": str(image_path.resolve()),
        }
        started = time.perf_counter()
        try:
            image = Image.open(image_path).convert("RGB")
            if image.width != image.height:
                raise ValueError(f"GOT input must be a square padded canvas, got {image.size}.")
            image_tensor = processor(image).unsqueeze(0).to(device=device, dtype=torch.float16)
            image_tensor_high = processor(image.copy()).unsqueeze(0).to(device=device, dtype=torch.float16)
            stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            generation_started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                output_ids = model.generate(
                    input_ids,
                    images=[(image_tensor, image_tensor_high)],
                    do_sample=False,
                    num_beams=1,
                    no_repeat_ngram_size=20,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )
            torch.cuda.synchronize(device)
            generation_elapsed = time.perf_counter() - generation_started
            generated_ids = output_ids[0, input_ids.shape[1] :]
            raw_text = tokenizer.decode(generated_ids).strip()
            text = strip_stop_text(raw_text, stop_str)
            row.update(
                {
                    "image_size": [image.width, image.height],
                    "generated_tokens": int(generated_ids.numel()),
                    "generation_seconds": generation_elapsed,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
                    "raw_text": raw_text,
                    "text": text,
                }
            )
            print(f"GOT_COLUMN_OK index={reading_index:03d} size={image.width}x{image.height}")
            print(text, flush=True)
        except Exception as exc:
            had_errors = True
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"GOT_COLUMN_ERROR index={reading_index:03d} error={row['error']}", flush=True)
            if isinstance(exc, RuntimeError):
                torch.cuda.empty_cache()
        row["total_seconds"] = time.perf_counter() - started
        report["results"].append(row)
        atomic_write_json(result_path, report)
        write_text_views(args.output_dir, report["results"])

    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["status"] = "completed_with_errors" if had_errors else "completed"
    atomic_write_json(result_path, report)
    write_text_views(args.output_dir, report["results"])
    print("GOT_COLUMNS_COMPLETED" if not had_errors else "GOT_COLUMNS_COMPLETED_WITH_ERRORS")
    print(f"results={result_path.resolve()}")
    if had_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
