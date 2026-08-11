from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    workspace = Path(os.environ.get("OCR_WORKSPACE", default_root.parent))
    model_dir = Path(os.environ.get("ANANDASKY_MODEL", workspace / "models" / "AnandaSky"))
    parser = argparse.ArgumentParser(description="Run AnandaSky on unscaled vertical columns.")
    parser.add_argument("--workspace", type=Path, default=workspace)
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


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive when supplied.")
    if not args.manifest.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {args.manifest}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run through run_anandasky.sh with CUDA_VISIBLE_DEVICES=0.")

    torch.cuda.set_device(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    if capability[0] < 8:
        raise RuntimeError(f"AnandaSky requires SM80 or newer, found SM{capability[0]}{capability[1]}.")
    device = torch.device(f"cuda:{args.device}")
    dtype = torch.bfloat16
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    columns = manifest["columns"]
    if args.limit is not None:
        columns = columns[: args.limit]

    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    report: dict[str, Any] = {
        "schema_version": 1,
        "model": "AnandaSky",
        "model_dir": str(args.model_dir.resolve()),
        "manifest": str(args.manifest.resolve()),
        "input_sha256": manifest["input"]["sha256"],
        "reading_order": manifest["reading_order"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "device": torch.cuda.get_device_name(args.device),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "dtype": str(next(model.parameters()).dtype),
        "results": [],
    }
    result_path = args.output_dir / "results.json"
    had_errors = False

    for column_info in columns:
        reading_index = int(column_info["reading_index"])
        relative_image = column_info["files"]["anandasky"]
        image_path = args.manifest.parent / relative_image
        row: dict[str, Any] = {
            "reading_index": reading_index,
            "source_box": column_info["source_box"],
            "image": str(image_path.resolve()),
        }
        started = time.perf_counter()
        try:
            image = Image.open(image_path).convert("RGB")
            if image.width >= image.height:
                raise ValueError(f"AnandaSky column must retain a vertical aspect ratio, got {image.size}.")
            inputs = processor(images=image, return_tensors="pt")
            input_length = inputs["input_ids"].shape[1]
            patch_tokens = int(inputs["patch_attention_mask"].sum().item())
            padded_h, padded_w = (int(value) for value in inputs["pixel_values"].shape[-2:])
            inputs["input_ids"] = inputs["input_ids"].to(device=device, non_blocking=True)
            inputs["attention_mask"] = inputs["attention_mask"].to(device=device, non_blocking=True)
            inputs["pixel_values"] = inputs["pixel_values"].to(
                device=device, dtype=dtype, non_blocking=True
            )
            inputs["patch_attention_mask"] = inputs["patch_attention_mask"].to(
                device=device, non_blocking=True
            )

            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
            generation_started = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=args.max_new_tokens,
                )
            torch.cuda.synchronize(device)
            generation_elapsed = time.perf_counter() - generation_started
            generated_ids = output[0, input_length:]
            text = processor.decode(generated_ids, skip_special_tokens=True).strip()
            row.update(
                {
                    "image_size": [image.width, image.height],
                    "processor_padded_size": [padded_w, padded_h],
                    "visual_tokens": patch_tokens,
                    "generated_tokens": int(generated_ids.numel()),
                    "generation_seconds": generation_elapsed,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
                    "text": text,
                }
            )
            print(
                f"ANANDASKY_COLUMN_OK index={reading_index:03d} "
                f"size={image.width}x{image.height} padded={padded_w}x{padded_h} "
                f"visual_tokens={patch_tokens}"
            )
            print(text, flush=True)
        except Exception as exc:
            had_errors = True
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"ANANDASKY_COLUMN_ERROR index={reading_index:03d} error={row['error']}", flush=True)
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
    print(
        "ANANDASKY_COLUMNS_COMPLETED"
        if not had_errors
        else "ANANDASKY_COLUMNS_COMPLETED_WITH_ERRORS"
    )
    print(f"results={result_path.resolve()}")
    if had_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
