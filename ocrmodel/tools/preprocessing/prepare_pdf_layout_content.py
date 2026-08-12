from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import fitz
except ImportError as exc:  # pragma: no cover - exercised by the CLI preflight
    raise RuntimeError(
        "PyMuPDF is required. Install tools/environment/requirements-layout-synthesis.lock.txt."
    ) from exc


SPLITS = ("train", "validation", "test")
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class PdfSource:
    path: Path
    sha256: str
    size: int
    text_fingerprint: str = ""


@dataclass(frozen=True)
class TextBlock:
    page_index: int
    block_index: int
    bbox: tuple[float, float, float, float]
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare source-isolated text and real PDF crop records for the whole-page "
            "synthetic layout generator. Labels come only from the PDF text layer."
        )
    )
    parser.add_argument("--pdf-root", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--validation-sources", type=int, default=4)
    parser.add_argument("--test-sources", type=int, default=4)
    parser.add_argument("--max-blocks-per-source", type=int, default=320)
    parser.add_argument("--min-text-chars", type=int, default=12)
    parser.add_argument("--max-text-chars", type=int, default=220)
    parser.add_argument("--render-dpi", type=int, default=180)
    parser.add_argument("--crop-padding-points", type=float, default=2.0)
    parser.add_argument(
        "--font-file",
        type=Path,
        action="append",
        default=[],
        help="Only keep text fully covered by at least one of these font files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for root in args.pdf_root:
        if not root.resolve().is_dir():
            raise NotADirectoryError(root)
    if args.output_dir.resolve().exists() and any(args.output_dir.resolve().iterdir()):
        raise FileExistsError(f"Output directory must not exist or must be empty: {args.output_dir}")
    if args.validation_sources < 1 or args.test_sources < 1:
        raise ValueError("validation/test source counts must both be positive.")
    if args.max_blocks_per_source < 1:
        raise ValueError("--max-blocks-per-source must be positive.")
    if not 1 <= args.min_text_chars <= args.max_text_chars:
        raise ValueError("Text lengths must satisfy 1 <= min <= max.")
    if not 72 <= args.render_dpi <= 600:
        raise ValueError("--render-dpi must be between 72 and 600.")
    if args.crop_padding_points < 0:
        raise ValueError("--crop-padding-points cannot be negative.")
    for font_file in args.font_file:
        if not font_file.resolve().is_file():
            raise FileNotFoundError(font_file)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    lines = [" ".join(line.split()) for line in value.replace("\r", "\n").splitlines()]
    return " ".join(line for line in lines if line).strip()


def safe_id(value: str) -> str:
    normalized = SAFE_ID_RE.sub("_", value).strip("_.-")
    return normalized or "source"


def discover_unique_pdfs(roots: Iterable[Path]) -> tuple[list[PdfSource], list[dict[str, str]]]:
    by_hash: dict[str, PdfSource] = {}
    duplicates: list[dict[str, str]] = []
    paths = sorted(
        {
            path.resolve()
            for root in roots
            for path in root.resolve().rglob("*")
            if path.is_file() and path.suffix.casefold() == ".pdf"
        },
        key=lambda path: path.as_posix().casefold(),
    )
    for path in paths:
        digest = sha256_file(path)
        source = PdfSource(path=path, sha256=digest, size=path.stat().st_size)
        if digest in by_hash:
            duplicates.append(
                {"sha256": digest, "kept": str(by_hash[digest].path), "duplicate": str(path)}
            )
        else:
            by_hash[digest] = source
    return list(by_hash.values()), duplicates


def extract_text_blocks(
    source: PdfSource,
    min_text_chars: int,
    max_text_chars: int,
    coverage_fonts: Iterable[fitz.Font] = (),
) -> tuple[list[TextBlock], dict[str, Any]]:
    blocks: list[TextBlock] = []
    seen_text: set[str] = set()
    page_count = 0
    unsupported_font_block_count = 0
    coverage_fonts = tuple(coverage_fonts)
    with fitz.open(source.path) as document:
        page_count = document.page_count
        for page_index, page in enumerate(document):
            page_rect = page.rect
            for block_index, raw in enumerate(page.get_text("blocks", sort=True)):
                if len(raw) < 7 or int(raw[6]) != 0:
                    continue
                x0, y0, x1, y1 = (float(value) for value in raw[:4])
                text = normalize_text(str(raw[4]))
                if not min_text_chars <= len(text) <= max_text_chars:
                    continue
                if coverage_fonts and not all(
                    character.isspace()
                    or (
                        not unicodedata.category(character).startswith("C")
                        and any(font.has_glyph(ord(character)) for font in coverage_fonts)
                    )
                    for character in text
                ):
                    unsupported_font_block_count += 1
                    continue
                if text in seen_text:
                    continue
                if not (page_rect.x0 <= x0 < x1 <= page_rect.x1):
                    continue
                if not (page_rect.y0 <= y0 < y1 <= page_rect.y1):
                    continue
                if x1 - x0 < 8 or y1 - y0 < 5:
                    continue
                seen_text.add(text)
                blocks.append(
                    TextBlock(
                        page_index=page_index,
                        block_index=block_index,
                        bbox=(x0, y0, x1, y1),
                        text=text,
                    )
                )
    return blocks, {
        "path": str(source.path),
        "sha256": source.sha256,
        "size": source.size,
        "page_count": page_count,
        "eligible_block_count": len(blocks),
        "unsupported_font_block_count": unsupported_font_block_count,
    }


def text_layer_fingerprint(blocks: Iterable[TextBlock]) -> str:
    digest = hashlib.sha256()
    for text in sorted({normalize_text(block.text).casefold() for block in blocks}):
        digest.update(text.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def assign_splits(
    eligible: list[tuple[PdfSource, list[TextBlock], dict[str, Any]]],
    seed: int,
    validation_sources: int,
    test_sources: int,
) -> dict[str, str]:
    if len(eligible) < validation_sources + test_sources + 1:
        raise ValueError(
            "Not enough eligible unique PDF sources for isolated splits: "
            f"eligible={len(eligible)}, validation={validation_sources}, test={test_sources}."
        )
    ordered = sorted(eligible, key=lambda item: item[0].sha256)
    random.Random(seed).shuffle(ordered)
    assignments: dict[str, str] = {}
    for index, (source, _, _) in enumerate(ordered):
        if index < test_sources:
            split = "test"
        elif index < test_sources + validation_sources:
            split = "validation"
        else:
            split = "train"
        assignments[source.sha256] = split
    return assignments


def select_blocks(blocks: list[TextBlock], limit: int, seed: int, source_hash: str) -> list[TextBlock]:
    ordered = sorted(blocks, key=lambda block: (block.page_index, block.block_index, block.text))
    rng_seed = int(hashlib.sha256(f"{seed}:{source_hash}".encode()).hexdigest()[:16], 16)
    random.Random(rng_seed).shuffle(ordered)
    selected = ordered[:limit]
    return sorted(selected, key=lambda block: (block.page_index, block.block_index, block.text))


def render_crop(
    document: fitz.Document,
    block: TextBlock,
    output_path: Path,
    render_dpi: int,
    padding_points: float,
) -> None:
    page = document[block.page_index]
    clip = fitz.Rect(block.bbox)
    clip = fitz.Rect(
        max(page.rect.x0, clip.x0 - padding_points),
        max(page.rect.y0, clip.y0 - padding_points),
        min(page.rect.x1, clip.x1 + padding_points),
        min(page.rect.y1, clip.y1 + padding_points),
    )
    pixmap = page.get_pixmap(clip=clip, dpi=render_dpi, alpha=False, colorspace=fitz.csRGB)
    if pixmap.width < 2 or pixmap.height < 2:
        raise ValueError(f"Rendered crop is too small: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixmap.save(output_path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    args = parse_args()
    validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources, duplicates = discover_unique_pdfs(args.pdf_root)
    font_paths = [path.resolve() for path in args.font_file]
    coverage_fonts = [fitz.Font(fontfile=str(path)) for path in font_paths]
    extracted: list[tuple[PdfSource, list[TextBlock], dict[str, Any]]] = []
    source_reports: list[dict[str, Any]] = []
    for source in sources:
        try:
            blocks, report = extract_text_blocks(
                source,
                min_text_chars=args.min_text_chars,
                max_text_chars=args.max_text_chars,
                coverage_fonts=coverage_fonts,
            )
            report["status"] = "eligible" if blocks else "no_eligible_text_blocks"
        except Exception as exc:
            blocks = []
            report = {
                "path": str(source.path),
                "sha256": source.sha256,
                "size": source.size,
                "status": "read_error",
                "error": f"{type(exc).__name__}: {exc}",
                "eligible_block_count": 0,
            }
        source_reports.append(report)
        if blocks:
            extracted.append((source, blocks, report))

    fingerprint_unique: dict[str, tuple[PdfSource, list[TextBlock], dict[str, Any]]] = {}
    near_duplicates: list[dict[str, str]] = []
    for source, blocks, report in extracted:
        fingerprint = text_layer_fingerprint(blocks)
        report["text_layer_sha256"] = fingerprint
        source = PdfSource(
            path=source.path,
            sha256=source.sha256,
            size=source.size,
            text_fingerprint=fingerprint,
        )
        if fingerprint in fingerprint_unique:
            near_duplicates.append(
                {
                    "text_layer_sha256": fingerprint,
                    "kept": str(fingerprint_unique[fingerprint][0].path),
                    "duplicate": str(source.path),
                }
            )
            report["status"] = "duplicate_text_layer"
            continue
        fingerprint_unique[fingerprint] = (source, blocks, report)
    fingerprint_eligible = list(fingerprint_unique.values())
    text_sources: dict[str, set[str]] = {}
    for source, blocks, _ in fingerprint_eligible:
        for block in blocks:
            text_sources.setdefault(normalize_text(block.text).casefold(), set()).add(source.sha256)
    split_eligible = []
    duplicate_blocks_removed = 0
    for source, blocks, report in fingerprint_eligible:
        unique_blocks = [
            block
            for block in blocks
            if len(text_sources[normalize_text(block.text).casefold()]) == 1
        ]
        removed = len(blocks) - len(unique_blocks)
        duplicate_blocks_removed += removed
        report["cross_source_duplicate_block_count_removed"] = removed
        report["unique_block_count"] = len(unique_blocks)
        if unique_blocks:
            split_eligible.append((source, unique_blocks, report))
        else:
            report["status"] = "no_unique_text_blocks"

    assignments = assign_splits(
        split_eligible,
        seed=args.seed,
        validation_sources=args.validation_sources,
        test_sources=args.test_sources,
    )
    records: list[dict[str, Any]] = []
    split_source_counts: Counter[str] = Counter()
    split_record_counts: Counter[str] = Counter()
    for source, blocks, report in split_eligible:
        split = assignments[source.sha256]
        source_group_id = f"pdf_{source.sha256[:20]}"
        split_source_counts[split] += 1
        selected = select_blocks(
            blocks,
            limit=args.max_blocks_per_source,
            seed=args.seed,
            source_hash=source.sha256,
        )
        report["split"] = split
        report["selected_block_count"] = len(selected)
        report["source_group_id"] = source_group_id
        with fitz.open(source.path) as document:
            for ordinal, block in enumerate(selected):
                base_id = safe_id(
                    f"{source_group_id}_p{block.page_index:05d}_b{block.block_index:04d}_n{ordinal:04d}"
                )
                crop_relative = Path("crops") / split / source_group_id / f"{base_id}.png"
                crop_path = output_dir / crop_relative
                render_crop(
                    document,
                    block=block,
                    output_path=crop_path,
                    render_dpi=args.render_dpi,
                    padding_points=args.crop_padding_points,
                )
                records.extend(
                    [
                        {
                            "content_id": f"text_{base_id}",
                            "source_group_id": source_group_id,
                            "split": split,
                            "kind": "text",
                            "orientation": "any",
                            "text": block.text,
                        },
                        {
                            "content_id": f"crop_{base_id}",
                            "source_group_id": source_group_id,
                            "split": split,
                            "kind": "image",
                            "orientation": "horizontal_ltr",
                            "image": crop_relative.as_posix(),
                            "text": block.text,
                        },
                    ]
                )
                split_record_counts[split] += 2

    write_jsonl(output_dir / "content.jsonl", records)
    write_json(
        output_dir / "source_report.json",
        {
            "schema_version": 1,
            "status": "ok",
            "seed": args.seed,
            "pdf_roots": [str(root.resolve()) for root in args.pdf_root],
            "unique_pdf_count": len(sources),
            "duplicate_pdf_count": len(duplicates),
            "eligible_pdf_count": len(extracted),
            "split_eligible_pdf_count": len(split_eligible),
            "duplicate_text_layer_count": len(near_duplicates),
            "cross_source_duplicate_block_count_removed": duplicate_blocks_removed,
            "split_source_counts": dict(sorted(split_source_counts.items())),
            "split_record_counts": dict(sorted(split_record_counts.items())),
            "content_record_count": len(records),
            "render_dpi": args.render_dpi,
            "crop_padding_points": args.crop_padding_points,
            "min_text_chars": args.min_text_chars,
            "max_text_chars": args.max_text_chars,
            "max_blocks_per_source": args.max_blocks_per_source,
            "pymupdf_version": fitz.VersionBind,
            "font_files": [
                {"path": str(path), "sha256": sha256_file(path)} for path in font_paths
            ],
            "duplicates": duplicates,
            "duplicate_text_layers": near_duplicates,
            "sources": source_reports,
        },
    )
    if any(split_source_counts[split] < 1 for split in SPLITS):
        raise RuntimeError(f"One or more splits contain no source: {dict(split_source_counts)}")
    if any(split_record_counts[split] < 8 for split in SPLITS):
        raise RuntimeError(f"One or more splits contain fewer than 8 records: {dict(split_record_counts)}")

    print("PDF_LAYOUT_CONTENT_OK")
    print(f"unique_pdfs={len(sources)}")
    print(f"eligible_pdfs={len(extracted)}")
    print(f"split_eligible_pdfs={len(split_eligible)}")
    print(f"records={len(records)}")
    print(f"split_sources={json.dumps(dict(sorted(split_source_counts.items())))}")
    print(f"split_records={json.dumps(dict(sorted(split_record_counts.items())))}")
    print(f"output_dir={output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"PDF_LAYOUT_CONTENT_ERROR: {exc}", file=sys.stderr)
        raise
