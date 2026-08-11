from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one reproducible GOT line-level smoke sample.")
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--transcription-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_image = args.source_image.resolve()
    transcription_file = args.transcription_file.resolve()
    output_dir = args.output_dir.resolve()
    if not source_image.is_file():
        raise FileNotFoundError(f"Source line image does not exist: {source_image}")
    if not transcription_file.is_file():
        raise FileNotFoundError(f"Transcription file does not exist: {transcription_file}")

    transcription = transcription_file.read_text(encoding="utf-8").strip()
    if not transcription or "\n" in transcription or "\r" in transcription:
        raise ValueError("Smoke transcription must contain exactly one non-empty text line.")
    with Image.open(source_image) as image:
        width, height = image.size
        image_format = image.format
    source_sha256 = sha256_file(source_image)

    images_dir = output_dir / "images"
    destination = images_dir / "column_001.png"
    annotations_path = output_dir / "annotations.json"
    metadata_path = output_dir / "metadata.json"
    if output_dir.exists() and not args.force:
        if annotations_path.is_file() and destination.is_file() and metadata_path.is_file():
            existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                existing_metadata.get("source_sha256") == source_sha256
                and existing_metadata.get("transcription") == transcription
                and sha256_file(destination) == source_sha256
            ):
                print(f"LINELEVEL_SMOKE_DATA_ALREADY_EXISTS={output_dir}")
                return
        raise FileExistsError(f"Incomplete output exists; inspect it or pass --force: {output_dir}")

    images_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_image, destination)
    copied_sha256 = sha256_file(destination)
    if source_sha256 != copied_sha256:
        raise RuntimeError("Copied smoke image checksum does not match the source.")

    records = [
        {
            "image": "images/column_001.png",
            "input_level": "line",
            "conversations": [
                {"from": "human", "value": "<image>\nOCR: "},
                {"from": "gpt", "value": transcription},
            ],
        }
    ]
    annotations_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "GOT line-level continuation-training smoke test only",
        "source_image": str(source_image),
        "source_sha256": source_sha256,
        "copied_image": str(destination),
        "image_size": [width, height],
        "image_format": image_format,
        "transcription": transcription,
        "transcription_source": "manual visual check recorded on 2026-07-30",
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"LINELEVEL_SMOKE_DATA_READY={output_dir}")


if __name__ == "__main__":
    main()
