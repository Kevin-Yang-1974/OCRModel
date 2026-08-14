from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from synthetic_layout_common import (
    SCHEMA_VERSION,
    TIERS,
    WRITING_DIRECTIONS,
    load_json_records,
    sha256_file,
    write_json,
)


LAYOUT_ANNOTATION_STATUSES = {"complete", "partial", "none"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit rendered whole-page synthetic manifests, including paths, bbox geometry, "
            "reading order, image hashes, and cross-split content/source leakage."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="Manifest JSONL/JSON path. Repeat to audit train/validation/test together.",
    )
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--skip-image-hash", action="store_true")
    parser.add_argument("--skip-html-check", action="store_true")
    parser.add_argument("--max-errors", type=int, default=50)
    return parser.parse_args()


def resolve_dataset_file(dataset_root: Path, value: Any, field_name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty relative path.")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{field_name} is unsafe: {value!r}")
    resolved = dataset_root / Path(*relative.parts)
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def expect_string(record: dict[str, Any], key: str, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string.")
    return value


def expect_page_size(record: dict[str, Any], context: str) -> tuple[int, int]:
    page_size = record.get("page_size")
    if (
        not isinstance(page_size, list)
        or len(page_size) != 2
        or any(not isinstance(value, int) or value <= 0 for value in page_size)
    ):
        raise ValueError(f"{context}.page_size must contain two positive integers.")
    return page_size[0], page_size[1]


def expect_bbox(value: Any, context: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{context} must contain four numbers.")
    if any(not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        raise ValueError(f"{context} contains a non-finite or non-numeric value.")
    x0, y0, x1, y1 = (float(item) for item in value)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"{context} is not an in-page normalized xyxy box: {value}")
    return x0, y0, x1, y1


def validate_rendered_fonts(
    value: Any,
    source_kind: str,
    allowed_fonts: set[str],
    context: str,
) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list.")
    total_glyphs = 0
    used_families: set[str] = set()
    for font_index, font in enumerate(value):
        font_context = f"{context}[{font_index}]"
        if not isinstance(font, dict):
            raise TypeError(f"{font_context} must be a JSON object.")
        family_name = font.get("family_name")
        postscript_name = font.get("postscript_name")
        is_custom_font = font.get("is_custom_font")
        glyph_count = font.get("glyph_count")
        if not isinstance(family_name, str) or not family_name:
            raise ValueError(f"{font_context}.family_name must be non-empty.")
        if not isinstance(postscript_name, str):
            raise ValueError(f"{font_context}.postscript_name must be a string.")
        if not isinstance(is_custom_font, bool):
            raise ValueError(f"{font_context}.is_custom_font must be boolean.")
        if not isinstance(glyph_count, int) or isinstance(glyph_count, bool) or glyph_count < 0:
            raise ValueError(f"{font_context}.glyph_count must be a non-negative integer.")
        used_families.add(family_name)
        total_glyphs += glyph_count

    if source_kind == "image":
        if value:
            raise ValueError(f"{context} must be empty for image crop content.")
        return
    if not value or total_glyphs < 1:
        raise ValueError(f"{context} contains no rendered text glyphs.")
    unexpected = sorted(used_families - allowed_fonts)
    if unexpected:
        raise ValueError(
            f"{context} contains unexpected fallback fonts {unexpected}; "
            f"allowed={sorted(allowed_fonts)}."
        )


def audit_record(
    record: dict[str, Any],
    dataset_root: Path,
    context: str,
    skip_image_hash: bool,
    skip_html_check: bool,
) -> dict[str, Any]:
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{context}.schema_version must be {SCHEMA_VERSION}, got "
            f"{record.get('schema_version')!r}."
        )
    if record.get("input_level") != "page":
        raise ValueError(f"{context}.input_level must be 'page'.")
    layout_source = expect_string(record, "layout_source", context)
    annotation_status = record.get("layout_annotation_status", "none")
    if annotation_status not in LAYOUT_ANNOTATION_STATUSES:
        raise ValueError(
            f"{context}.layout_annotation_status must be one of "
            f"{sorted(LAYOUT_ANNOTATION_STATUSES)}."
        )
    is_synthetic = layout_source == "html_synthetic"
    if is_synthetic and annotation_status != "complete":
        raise ValueError(f"{context}.layout_annotation_status must be 'complete'.")
    if record.get("bbox_format") != "xyxy_normalized":
        raise ValueError(f"{context}.bbox_format must be 'xyxy_normalized'.")

    page_id = expect_string(record, "page_id", context)
    split = expect_string(record, "split", context)
    tier = expect_string(record, "tier", context)
    if is_synthetic and tier not in TIERS:
        raise ValueError(f"{context}.tier must be one of {TIERS}, got {tier!r}.")
    if is_synthetic or "page_size" in record:
        width, height = expect_page_size(record, context)
    else:
        width = height = None
    page_text = expect_string(record, "page_text", context)
    separator = record.get("page_text_separator")
    if not isinstance(separator, str):
        raise ValueError(f"{context}.page_text_separator must be a string.")
    allowed_font_set: set[str] = set()
    if is_synthetic:
        generator = record.get("generator")
        if not isinstance(generator, dict):
            raise ValueError(f"{context}.generator must be a JSON object.")
        allowed_rendered_fonts = generator.get("allowed_rendered_fonts")
        if (
            not isinstance(allowed_rendered_fonts, list)
            or not allowed_rendered_fonts
            or any(not isinstance(font, str) or not font for font in allowed_rendered_fonts)
            or len(set(allowed_rendered_fonts)) != len(allowed_rendered_fonts)
        ):
            raise ValueError(
                f"{context}.generator.allowed_rendered_fonts must be a non-empty unique string list."
            )
        allowed_font_set = set(allowed_rendered_fonts)

    image_path = resolve_dataset_file(dataset_root, record.get("image"), f"{context}.image")
    with Image.open(image_path) as image:
        actual_size = image.size
    if width is not None and actual_size != (width, height):
        raise ValueError(
            f"{context} image size mismatch: manifest={(width, height)}, file={actual_size}."
        )
    expected_image_hash = record.get("image_sha256")
    if is_synthetic and not isinstance(expected_image_hash, str):
        raise ValueError(f"{context}.image_sha256 must be a string.")
    if not skip_image_hash and isinstance(expected_image_hash, str):
        actual_image_hash = sha256_file(image_path)
        if actual_image_hash != expected_image_hash:
            raise ValueError(
                f"{context} image SHA-256 mismatch: expected={expected_image_hash}, "
                f"actual={actual_image_hash}."
            )

    if is_synthetic and not skip_html_check:
        html_path = resolve_dataset_file(dataset_root, record.get("html"), f"{context}.html")
        if f'data-page-id="{page_id}"' not in html_path.read_text(encoding="utf-8"):
            raise ValueError(f"{context} HTML does not declare its page_id.")

    regions = record.get("regions", [])
    if not isinstance(regions, list):
        raise ValueError(f"{context}.regions must be a list.")
    if annotation_status == "complete" and not regions:
        raise ValueError(f"{context}.regions must be a non-empty list.")
    if annotation_status == "none" and regions:
        raise ValueError(f"{context}.regions must be empty when layout_annotation_status is none.")
    expected_orders = list(range(len(regions)))
    actual_orders = [region.get("reading_order") if isinstance(region, dict) else None for region in regions]
    if actual_orders != expected_orders:
        raise ValueError(
            f"{context} regions must be stored in contiguous reading order: "
            f"expected={expected_orders}, actual={actual_orders}."
        )

    region_ids: set[str] = set()
    page_content_ids: set[str] = set()
    region_texts: list[str] = []
    source_groups: set[str] = set()
    region_summaries: list[dict[str, Any]] = []
    for region_index, region in enumerate(regions):
        region_context = f"{context}.regions[{region_index}]"
        if not isinstance(region, dict):
            raise TypeError(f"{region_context} is not a JSON object.")
        if is_synthetic and region.get("valid") is not True:
            raise ValueError(f"{region_context}.valid must be true.")
        region_id = expect_string(region, "region_id", region_context)
        content_id = expect_string(region, "content_id", region_context)
        source_group_id = expect_string(region, "source_group_id", region_context)
        source_kind = expect_string(region, "source_kind", region_context)
        region_text = expect_string(region, "text", region_context)
        direction = expect_string(region, "writing_direction", region_context)
        if direction not in WRITING_DIRECTIONS:
            raise ValueError(
                f"{region_context}.writing_direction is unsupported: {direction!r}."
            )
        if source_kind not in {"text", "image"}:
            raise ValueError(f"{region_context}.source_kind is invalid: {source_kind!r}.")
        if is_synthetic:
            validate_rendered_fonts(
                region.get("rendered_fonts"),
                source_kind=source_kind,
                allowed_fonts=allowed_font_set,
                context=f"{region_context}.rendered_fonts",
            )
            visibility = region.get("content_visibility")
            if not isinstance(visibility, dict):
                raise ValueError(f"{region_context}.content_visibility must be an object.")
            client_size = visibility.get("client_size")
            scroll_size = visibility.get("scroll_size")
            if (
                not isinstance(client_size, list)
                or len(client_size) != 2
                or any(not isinstance(value, int) or value <= 0 for value in client_size)
            ):
                raise ValueError(f"{region_context}.content_visibility.client_size is invalid.")
            if (
                not isinstance(scroll_size, list)
                or len(scroll_size) != 2
                or any(not isinstance(value, int) or value <= 0 for value in scroll_size)
            ):
                raise ValueError(f"{region_context}.content_visibility.scroll_size is invalid.")
            if visibility.get("overflow") is not False:
                raise ValueError(f"{region_context} must declare content overflow=false.")
            if scroll_size[0] > client_size[0] + 1 or scroll_size[1] > client_size[1] + 1:
                raise ValueError(f"{region_context} visibility sizes prove content overflow.")
            if source_kind == "text":
                if visibility.get("dom_text_matches_label") is not True:
                    raise ValueError(f"{region_context} text must match the DOM content.")
                if visibility.get("image_natural_size") is not None:
                    raise ValueError(f"{region_context} text must not declare image dimensions.")
            else:
                natural_size = visibility.get("image_natural_size")
                if (
                    not isinstance(natural_size, list)
                    or len(natural_size) != 2
                    or any(not isinstance(value, int) or value <= 0 for value in natural_size)
                ):
                    raise ValueError(f"{region_context} image dimensions are invalid.")
                if visibility.get("dom_text_matches_label") is not None:
                    raise ValueError(f"{region_context} image must not declare DOM text equality.")
        if region_id in region_ids:
            raise ValueError(f"{context} contains duplicate region_id={region_id!r}.")
        if content_id in page_content_ids:
            raise ValueError(f"{context} repeats content_id={content_id!r} within one page.")
        region_ids.add(region_id)
        page_content_ids.add(content_id)
        source_groups.add(source_group_id)
        region_texts.append(region_text)

        bbox = expect_bbox(region.get("bbox"), f"{region_context}.bbox")
        if is_synthetic:
            bbox_px = region.get("bbox_px")
            if (
                not isinstance(bbox_px, list)
                or len(bbox_px) != 4
                or any(not isinstance(value, (int, float)) for value in bbox_px)
            ):
                raise ValueError(f"{region_context}.bbox_px must contain four numbers.")
            assert width is not None and height is not None
            expected_px = (bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height)
            max_delta = max(abs(float(actual) - expected) for actual, expected in zip(bbox_px, expected_px))
            if max_delta > 0.02:
                raise ValueError(
                    f"{region_context} normalized/pixel bbox mismatch: max_delta={max_delta:.6f}."
                )

        source_sha256 = region.get("source_sha256")
        if source_kind == "image":
            if not isinstance(source_sha256, str) or len(source_sha256) != 64:
                raise ValueError(f"{region_context}.source_sha256 is required for image content.")
            if not isinstance(region.get("source_image"), str):
                raise ValueError(f"{region_context}.source_image is required for image content.")
        elif region.get("source_image") is not None or source_sha256 is not None:
            raise ValueError(f"{region_context} text content must not declare a source image.")

        region_summaries.append(
            {
                "content_id": content_id,
                "source_group_id": source_group_id,
                "source_sha256": source_sha256,
                "source_kind": source_kind,
                "text": region_text,
                "direction": direction,
            }
        )

    if regions and separator.join(region_texts) != page_text:
        raise ValueError(f"{context}.page_text does not match ordered region text concatenation.")
    declared_groups = record.get("source_group_ids")
    if regions and (not isinstance(declared_groups, list) or declared_groups != sorted(source_groups)):
        raise ValueError(
            f"{context}.source_group_ids must equal sorted region groups: {sorted(source_groups)}."
        )
    if not regions:
        source_group_id = expect_string(record, "source_group_id", context)
        source_groups.add(source_group_id)
    conversations = record.get("conversations")
    expected_conversations = [
        {"from": "human", "value": "<image>\nOCR: "},
        {"from": "gpt", "value": page_text},
    ]
    if conversations != expected_conversations:
        raise ValueError(f"{context}.conversations does not match the page OCR target.")

    return {
        "page_id": page_id,
        "split": split,
        "tier": tier,
        "layout_source": layout_source,
        "layout_annotation_status": annotation_status,
        "template_id": record.get("template_id", layout_source),
        "image_sha256": expected_image_hash,
        "regions": region_summaries,
        "source_group_id": sorted(source_groups)[0] if source_groups else None,
    }


def main() -> int:
    args = parse_args()
    if args.max_errors < 1:
        raise ValueError("--max-errors must be positive.")

    errors: list[str] = []
    audited: list[dict[str, Any]] = []
    page_ids: set[str] = set()
    for manifest_path_input in args.manifest:
        manifest_path = manifest_path_input.resolve()
        dataset_root = manifest_path.parent
        try:
            records = load_json_records(manifest_path)
        except Exception as exc:
            errors.append(f"{manifest_path}: {exc}")
            continue
        for record_index, record in enumerate(records):
            context = f"{manifest_path.name}[{record_index}]"
            if len(errors) >= args.max_errors:
                break
            try:
                summary = audit_record(
                    record=record,
                    dataset_root=dataset_root,
                    context=context,
                    skip_image_hash=args.skip_image_hash,
                    skip_html_check=args.skip_html_check,
                )
                if summary["page_id"] in page_ids:
                    raise ValueError(f"Duplicate page_id across manifests: {summary['page_id']!r}.")
                page_ids.add(summary["page_id"])
                audited.append(summary)
            except Exception as exc:
                errors.append(f"{context}: {exc}")

    content_splits: dict[str, str] = {}
    content_fingerprints: dict[str, tuple[str, str]] = {}
    group_splits: dict[str, str] = {}
    crop_hash_splits: dict[str, str] = {}
    page_hash_splits: dict[str, str] = {}
    text_hash_splits: dict[str, str] = {}

    def enforce_single_split(mapping: dict[str, str], key: str, split: str, label: str) -> None:
        previous = mapping.setdefault(key, split)
        if previous != split:
            errors.append(f"{label} occurs in multiple splits: {key!r} -> {previous!r}, {split!r}")

    for page in audited:
        split = page["split"]
        image_sha256 = page["image_sha256"]
        if image_sha256:
            enforce_single_split(page_hash_splits, image_sha256, split, "page image hash")
        for region in page["regions"]:
            content_id = region["content_id"]
            enforce_single_split(content_splits, content_id, split, "content_id")
            enforce_single_split(
                group_splits, region["source_group_id"], split, "source_group_id"
            )
            fingerprint = (region["text"], region["source_kind"])
            normalized_text = " ".join(region["text"].casefold().split())
            enforce_single_split(text_hash_splits, normalized_text, split, "normalized text")
            previous_fingerprint = content_fingerprints.setdefault(content_id, fingerprint)
            if previous_fingerprint != fingerprint:
                errors.append(
                    f"content_id has inconsistent text/kind: {content_id!r} -> "
                    f"{previous_fingerprint!r}, {fingerprint!r}"
                )
            if region["source_sha256"]:
                enforce_single_split(
                    crop_hash_splits,
                    region["source_sha256"],
                    split,
                    "source crop hash",
                )
            if len(errors) >= args.max_errors:
                break
        if len(errors) >= args.max_errors:
            break

    split_counts = Counter(page["split"] for page in audited)
    tier_counts = Counter(page["tier"] for page in audited)
    template_counts = Counter(page["template_id"] for page in audited)
    direction_counts = Counter(
        region["direction"] for page in audited for region in page["regions"]
    )
    region_count = sum(len(page["regions"]) for page in audited)
    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "error" if errors else "ok",
        "manifest_count": len(args.manifest),
        "page_count": len(audited),
        "region_count": region_count,
        "unique_content_count": len(content_splits),
        "unique_source_group_count": len(group_splits),
        "split_counts": dict(sorted(split_counts.items())),
        "tier_counts": dict(sorted(tier_counts.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "direction_counts": dict(sorted(direction_counts.items())),
        "image_hash_checked": not args.skip_image_hash,
        "html_checked": not args.skip_html_check,
        "errors": errors[: args.max_errors],
    }
    if args.summary_json:
        summary_path = args.summary_json.resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(summary_path, summary_payload)

    if errors:
        print("SYNTHETIC_LAYOUT_AUDIT_ERROR", file=sys.stderr)
        print(f"errors={len(errors)}", file=sys.stderr)
        for error in errors[: args.max_errors]:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("SYNTHETIC_LAYOUT_AUDIT_OK")
    print(f"pages={len(audited)}")
    print(f"regions={region_count}")
    print(f"splits={json.dumps(dict(sorted(split_counts.items())), ensure_ascii=False)}")
    print(f"tiers={json.dumps(dict(sorted(tier_counts.items())), ensure_ascii=False)}")
    if args.summary_json:
        print(f"summary={args.summary_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
