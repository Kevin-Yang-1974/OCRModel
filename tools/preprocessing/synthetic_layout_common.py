from __future__ import annotations

import hashlib
import html
import json
import random
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 2
TIERS = ("s0-html-text", "s1-html-crop", "s2-hard")
WRITING_DIRECTIONS = (
    "horizontal_ltr",
    "horizontal_rtl",
    "vertical_rtl",
    "vertical_ltr",
)
VALID_ORIENTATIONS = ("horizontal", "vertical", "any")
SAFE_ID_RE = re.compile(r'^[^\s<>:"/\\|?*\x00-\x1f]{1,128}$')
HEX_COLOR_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    return normalized.strip()


def ensure_safe_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not SAFE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a non-empty path-safe identifier of at most 128 "
            f"characters, got {value!r}."
        )
    return value


def resolve_relative_file(root: Path, value: Any, field_name: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty relative path.")
    posix_path = PurePosixPath(value)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"{field_name} must stay below its declared root: {value!r}.")
    relative = Path(*posix_path.parts)
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field_name} escapes its declared root: {value!r}.") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {resolved}")
    return posix_path.as_posix(), resolved


def load_json_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        raise ValueError(f"Content manifest is empty: {path}")

    records: Any
    parsed: Any = None
    parsed_as_single_json = False
    if path.suffix.lower() != ".jsonl":
        try:
            parsed = json.loads(text)
            parsed_as_single_json = True
        except json.JSONDecodeError:
            parsed_as_single_json = False

    if parsed_as_single_json:
        if isinstance(parsed, dict):
            records = parsed.get("records")
            if records is None:
                raise ValueError(
                    f"JSON object manifest must contain a 'records' list: {path}"
                )
        else:
            records = parsed
    else:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

    if not isinstance(records, list) or not records:
        raise ValueError(f"Manifest must contain a non-empty record list: {path}")
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"Manifest record {index} is not a JSON object.")
    return records


@dataclass(frozen=True)
class ContentItem:
    content_id: str
    source_group_id: str
    split: str
    text: str
    kind: str
    orientation: str
    source_image: str | None = None
    source_image_path: Path | None = None
    source_sha256: str | None = None

    def supports_direction(self, writing_direction: str) -> bool:
        orientation = writing_direction.split("_", maxsplit=1)[0]
        return self.orientation in ("any", orientation)


def load_content_items(manifest: Path, content_root: Path | None = None) -> list[ContentItem]:
    root = (content_root or manifest.parent).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Content root does not exist: {root}")

    items: list[ContentItem] = []
    seen_content: set[str] = set()
    group_splits: dict[str, str] = {}
    for index, record in enumerate(load_json_records(manifest)):
        prefix = f"content record {index}"
        content_id = ensure_safe_id(record.get("content_id"), f"{prefix}.content_id")
        source_group_id = ensure_safe_id(
            record.get("source_group_id"), f"{prefix}.source_group_id"
        )
        split = ensure_safe_id(record.get("split"), f"{prefix}.split")
        if content_id in seen_content:
            raise ValueError(f"Duplicate content_id: {content_id}")
        seen_content.add(content_id)
        previous_split = group_splits.setdefault(source_group_id, split)
        if previous_split != split:
            raise ValueError(
                "source_group_id occurs in multiple splits: "
                f"{source_group_id!r} -> {previous_split!r}, {split!r}"
            )

        text_value = record.get("text")
        if not isinstance(text_value, str) or not normalize_text(text_value):
            raise ValueError(f"{prefix}.text must be a non-empty string.")
        text = normalize_text(text_value)

        image_value = record.get("image")
        inferred_kind = "image" if image_value is not None else "text"
        kind = str(record.get("kind", inferred_kind)).strip().lower()
        if kind not in {"text", "image"}:
            raise ValueError(f"{prefix}.kind must be 'text' or 'image', got {kind!r}.")

        orientation = str(record.get("orientation", "any")).strip().lower()
        if orientation not in VALID_ORIENTATIONS:
            raise ValueError(
                f"{prefix}.orientation must be one of {VALID_ORIENTATIONS}, got {orientation!r}."
            )
        if kind == "image" and orientation == "any":
            raise ValueError(
                f"{prefix}.orientation must be horizontal or vertical for an image crop."
            )

        source_image: str | None = None
        source_image_path: Path | None = None
        source_sha256: str | None = None
        if kind == "image":
            source_image, source_image_path = resolve_relative_file(
                root, image_value, f"{prefix}.image"
            )
            source_sha256 = sha256_file(source_image_path)
        elif image_value is not None:
            raise ValueError(f"{prefix} declares kind='text' but also supplies image.")

        items.append(
            ContentItem(
                content_id=content_id,
                source_group_id=source_group_id,
                split=split,
                text=text,
                kind=kind,
                orientation=orientation,
                source_image=source_image,
                source_image_path=source_image_path,
                source_sha256=source_sha256,
            )
        )
    return items


@dataclass
class GeneratorConfig:
    page_width: int = 1024
    page_height: int = 1024
    min_regions: int = 4
    max_regions: int = 12
    margin_min: int = 72
    margin_max: int = 112
    gap_min: int = 12
    gap_max: int = 30
    region_inset_min: int = 1
    region_inset_max: int = 5
    font_size_min: int = 28
    font_size_max: int = 46
    region_padding: int = 4
    page_text_separator: str = "\n"
    font_family: str = '"Noto Serif CJK SC", "Source Han Serif SC", SimSun, serif'
    allowed_rendered_fonts: list[str] = field(
        default_factory=lambda: ["Noto Serif CJK SC", "Source Han Serif SC", "SimSun"]
    )
    directions: list[str] = field(default_factory=lambda: list(WRITING_DIRECTIONS))
    backgrounds: list[str] = field(
        default_factory=lambda: ["#f5f0e4", "#eee6d5", "#f8f4ea", "#e9dfc8"]
    )

    @classmethod
    def from_json(cls, path: Path | None) -> "GeneratorConfig":
        config = cls()
        if path is None:
            config.validate()
            return config
        if not path.is_file():
            raise FileNotFoundError(path)
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(raw, dict):
            raise TypeError("Generator config must be a JSON object.")
        allowed = set(asdict(config))
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"Unknown generator config fields: {unknown}")
        for key, value in raw.items():
            setattr(config, key, value)
        config.validate()
        return config

    def validate(self) -> None:
        integer_fields = (
            "page_width",
            "page_height",
            "min_regions",
            "max_regions",
            "margin_min",
            "margin_max",
            "gap_min",
            "gap_max",
            "region_inset_min",
            "region_inset_max",
            "font_size_min",
            "font_size_max",
            "region_padding",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
        if self.page_width < 128 or self.page_height < 128:
            raise ValueError("page_width and page_height must both be at least 128.")
        if not 1 <= self.min_regions <= self.max_regions:
            raise ValueError("Region counts must satisfy 1 <= min_regions <= max_regions.")
        if not 0 <= self.margin_min <= self.margin_max:
            raise ValueError("Margins must satisfy 0 <= margin_min <= margin_max.")
        if 2 * self.margin_max >= min(self.page_width, self.page_height):
            raise ValueError("margin_max leaves no page content area.")
        if not 0 <= self.gap_min <= self.gap_max:
            raise ValueError("Gaps must satisfy 0 <= gap_min <= gap_max.")
        if not 0 <= self.region_inset_min <= self.region_inset_max:
            raise ValueError("Region insets must be non-negative and ordered.")
        if not 4 <= self.font_size_min <= self.font_size_max:
            raise ValueError("Font sizes must satisfy 4 <= font_size_min <= font_size_max.")
        if self.region_padding < 0:
            raise ValueError("region_padding cannot be negative.")
        if not isinstance(self.page_text_separator, str):
            raise TypeError("page_text_separator must be a string.")
        if not isinstance(self.font_family, str) or not self.font_family.strip():
            raise ValueError("font_family must be a non-empty CSS font-family value.")
        if not isinstance(self.allowed_rendered_fonts, list) or not self.allowed_rendered_fonts:
            raise ValueError("allowed_rendered_fonts must be a non-empty list.")
        if any(
            not isinstance(font_name, str) or not font_name.strip()
            for font_name in self.allowed_rendered_fonts
        ):
            raise ValueError("allowed_rendered_fonts entries must be non-empty strings.")
        if len(set(self.allowed_rendered_fonts)) != len(self.allowed_rendered_fonts):
            raise ValueError("allowed_rendered_fonts must not contain duplicates.")
        if not isinstance(self.directions, list) or not self.directions:
            raise ValueError("directions must be a non-empty list.")
        if len(set(self.directions)) != len(self.directions):
            raise ValueError("directions must not contain duplicates.")
        invalid_directions = sorted(set(self.directions) - set(WRITING_DIRECTIONS))
        if invalid_directions:
            raise ValueError(f"Unsupported writing directions: {invalid_directions}")
        if not isinstance(self.backgrounds, list) or not self.backgrounds:
            raise ValueError("backgrounds must be a non-empty list.")
        invalid_colors = [color for color in self.backgrounds if not HEX_COLOR_RE.fullmatch(color)]
        if invalid_colors:
            raise ValueError(f"Background colors must use #RRGGBB: {invalid_colors}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegionPlan:
    region_id: str
    reading_order: int
    writing_direction: str
    bbox_px: tuple[float, float, float, float]
    font_size: int
    item: ContentItem


@dataclass(frozen=True)
class PagePlan:
    page_id: str
    split: str
    tier: str
    template_id: str
    page_seed: int
    page_size: tuple[int, int]
    background: str
    page_text_separator: str
    regions: tuple[RegionPlan, ...]

    @property
    def page_text(self) -> str:
        return self.page_text_separator.join(region.item.text for region in self.regions)


def derive_page_seed(base_seed: int, split: str, page_index: int) -> int:
    payload = f"{base_seed}:{split}:{page_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def tier_accepts_item(tier: str, item: ContentItem) -> bool:
    if tier == "s0-html-text":
        return item.kind == "text"
    if tier == "s1-html-crop":
        return item.kind == "image"
    if tier == "s2-hard":
        return item.kind in {"text", "image"}
    raise ValueError(f"Unknown tier: {tier!r}")


def _plan_boxes(
    config: GeneratorConfig,
    rng: random.Random,
    direction: str,
    count: int,
) -> list[tuple[float, float, float, float]]:
    width, height = config.page_width, config.page_height
    margin_x = rng.randint(config.margin_min, config.margin_max)
    margin_y = rng.randint(config.margin_min, config.margin_max)
    gap = rng.randint(config.gap_min, config.gap_max)
    boxes_visual_order: list[tuple[float, float, float, float]] = []

    if direction.startswith("vertical"):
        extent = width - 2 * margin_x - gap * (count - 1)
        cell = extent / count
        if cell <= 2 * (config.region_inset_max + config.region_padding) + 4:
            raise ValueError(
                f"{count} vertical regions do not fit page width={width}; reduce max_regions."
            )
        for visual_index in range(count):
            inset_left = rng.randint(config.region_inset_min, config.region_inset_max)
            inset_right = rng.randint(config.region_inset_min, config.region_inset_max)
            top_jitter = rng.randint(0, config.region_inset_max)
            bottom_jitter = rng.randint(0, config.region_inset_max)
            x0 = margin_x + visual_index * (cell + gap) + inset_left
            x1 = margin_x + visual_index * (cell + gap) + cell - inset_right
            boxes_visual_order.append((x0, margin_y + top_jitter, x1, height - margin_y - bottom_jitter))
        if direction == "vertical_rtl":
            boxes_visual_order.reverse()
    else:
        extent = height - 2 * margin_y - gap * (count - 1)
        cell = extent / count
        if cell <= 2 * (config.region_inset_max + config.region_padding) + 4:
            raise ValueError(
                f"{count} horizontal regions do not fit page height={height}; reduce max_regions."
            )
        for visual_index in range(count):
            inset_top = rng.randint(config.region_inset_min, config.region_inset_max)
            inset_bottom = rng.randint(config.region_inset_min, config.region_inset_max)
            left_jitter = rng.randint(0, config.region_inset_max)
            right_jitter = rng.randint(0, config.region_inset_max)
            y0 = margin_y + visual_index * (cell + gap) + inset_top
            y1 = margin_y + visual_index * (cell + gap) + cell - inset_bottom
            boxes_visual_order.append((margin_x + left_jitter, y0, width - margin_x - right_jitter, y1))
    return boxes_visual_order


def build_page_plan(
    items: Sequence[ContentItem],
    config: GeneratorConfig,
    split: str,
    tier: str,
    base_seed: int,
    page_index: int,
) -> PagePlan:
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}, got {tier!r}.")
    split = ensure_safe_id(split, "split")
    page_seed = derive_page_seed(base_seed, split, page_index)
    rng = random.Random(page_seed)

    by_direction: dict[str, list[ContentItem]] = {}
    for direction in config.directions:
        eligible = [
            item
            for item in items
            if item.split == split and tier_accepts_item(tier, item) and item.supports_direction(direction)
        ]
        if len(eligible) >= config.min_regions:
            by_direction[direction] = eligible
    if not by_direction:
        split_count = sum(item.split == split for item in items)
        raise ValueError(
            f"No direction has at least {config.min_regions} eligible {tier} records "
            f"for split={split!r}; split contains {split_count} total records."
        )

    direction = rng.choice(sorted(by_direction))
    eligible = by_direction[direction]
    count = rng.randint(config.min_regions, min(config.max_regions, len(eligible)))
    selected = rng.sample(eligible, count)
    boxes = _plan_boxes(config, rng, direction, count)

    tier_tag = tier.split("-", maxsplit=1)[0]
    page_id = f"{split}_{tier_tag}_seed{base_seed:08d}_p{page_index:06d}"
    regions = tuple(
        RegionPlan(
            region_id=f"{page_id}_r{reading_order:03d}",
            reading_order=reading_order,
            writing_direction=direction,
            bbox_px=boxes[reading_order],
            font_size=rng.randint(config.font_size_min, config.font_size_max),
            item=item,
        )
        for reading_order, item in enumerate(selected)
    )
    return PagePlan(
        page_id=page_id,
        split=split,
        tier=tier,
        template_id=f"{direction}_{count:02d}region",
        page_seed=page_seed,
        page_size=(config.page_width, config.page_height),
        background=rng.choice(config.backgrounds),
        page_text_separator=config.page_text_separator,
        regions=regions,
    )


def _css_direction(writing_direction: str) -> tuple[str, str]:
    mapping = {
        "horizontal_ltr": ("horizontal-tb", "ltr"),
        "horizontal_rtl": ("horizontal-tb", "rtl"),
        "vertical_rtl": ("vertical-rl", "ltr"),
        "vertical_ltr": ("vertical-lr", "ltr"),
    }
    return mapping[writing_direction]


def render_page_html(
    plan: PagePlan,
    config: GeneratorConfig,
    debug_outlines: bool = False,
) -> str:
    region_nodes: list[str] = []
    for region in plan.regions:
        x0, y0, x1, y1 = region.bbox_px
        writing_mode, direction = _css_direction(region.writing_direction)
        style = (
            f"left:{x0:.3f}px;top:{y0:.3f}px;width:{x1 - x0:.3f}px;"
            f"height:{y1 - y0:.3f}px;writing-mode:{writing_mode};direction:{direction};"
            f"font-size:{region.font_size}px;padding:{config.region_padding}px;"
        )
        if debug_outlines:
            style += "outline:1px solid rgba(190,0,0,.45);"

        if region.item.kind == "text":
            content = f'<div class="text-content">{html.escape(region.item.text)}</div>'
        else:
            if region.item.source_image_path is None:
                raise RuntimeError(f"Image content has no resolved path: {region.item.content_id}")
            source_uri = html.escape(region.item.source_image_path.as_uri(), quote=True)
            content = (
                f'<img class="crop-content" src="{source_uri}" '
                f'alt="{html.escape(region.item.content_id, quote=True)}">'
            )
        region_nodes.append(
            f'<section class="region" id="{html.escape(region.region_id, quote=True)}" '
            f'data-region-id="{html.escape(region.region_id, quote=True)}" '
            f'data-reading-order="{region.reading_order}" '
            f'data-writing-direction="{region.writing_direction}" style="{style}">'
            f"{content}</section>"
        )

    embedded_plan = {
        "page_id": plan.page_id,
        "split": plan.split,
        "tier": plan.tier,
        "template_id": plan.template_id,
        "page_seed": plan.page_seed,
    }
    embedded_json = json.dumps(embedded_plan, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(plan.page_id)}</title>
<style>
html, body {{
  width: {config.page_width}px;
  height: {config.page_height}px;
  margin: 0;
  padding: 0;
  overflow: hidden;
}}
body {{
  position: relative;
  box-sizing: border-box;
  background: {plan.background};
  color: #241d17;
  font-family: {config.font_family};
  text-rendering: geometricPrecision;
  -webkit-font-smoothing: antialiased;
}}
*, *::before, *::after {{ box-sizing: border-box; }}
.region {{
  position: absolute;
  overflow: hidden;
  text-orientation: mixed;
  line-height: 1.35;
}}
.text-content {{
  width: 100%;
  height: 100%;
  white-space: pre-wrap;
  overflow: hidden;
  overflow-wrap: anywhere;
}}
.crop-content {{
  display: block;
  width: 100%;
  height: 100%;
  object-fit: contain;
  object-position: center;
}}
</style>
</head>
<body data-page-id="{html.escape(plan.page_id, quote=True)}">
{''.join(region_nodes)}
<script id="page-plan" type="application/json">{embedded_json}</script>
</body>
</html>
"""


def plan_to_record(plan: PagePlan) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "page_id": plan.page_id,
        "split": plan.split,
        "tier": plan.tier,
        "template_id": plan.template_id,
        "page_seed": plan.page_seed,
        "page_size": list(plan.page_size),
        "page_text_separator": plan.page_text_separator,
        "page_text": plan.page_text,
        "regions": [
            {
                "region_id": region.region_id,
                "content_id": region.item.content_id,
                "source_group_id": region.item.source_group_id,
                "source_kind": region.item.kind,
                "source_image": region.item.source_image,
                "source_sha256": region.item.source_sha256,
                "text": region.item.text,
                "reading_order": region.reading_order,
                "writing_direction": region.writing_direction,
                "planned_bbox_px": [round(value, 4) for value in region.bbox_px],
                "valid": True,
            }
            for region in plan.regions
        ],
    }


def manifest_record_from_dom(
    plan: PagePlan,
    dom_regions: Sequence[Mapping[str, Any]],
    image_relative: str,
    html_relative: str,
    image_sha256: str,
    generator_metadata: Mapping[str, Any],
    degradation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    width, height = plan.page_size
    dom_by_id: dict[str, Mapping[str, Any]] = {}
    for dom_region in dom_regions:
        region_id = dom_region.get("region_id")
        if not isinstance(region_id, str) or not region_id:
            raise ValueError(f"DOM region has no region_id: {dom_region!r}")
        if region_id in dom_by_id:
            raise ValueError(f"Duplicate DOM region_id: {region_id}")
        dom_by_id[region_id] = dom_region
    planned_ids = {region.region_id for region in plan.regions}
    if set(dom_by_id) != planned_ids:
        raise ValueError(
            "DOM regions do not match page plan: "
            f"missing={sorted(planned_ids - set(dom_by_id))}, "
            f"extra={sorted(set(dom_by_id) - planned_ids)}"
        )

    region_records: list[dict[str, Any]] = []
    for region in plan.regions:
        dom = dom_by_id[region.region_id]
        try:
            bbox_px = tuple(float(dom[key]) for key in ("x0", "y0", "x1", "y1"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid DOM bbox for {region.region_id}: {dom!r}") from exc
        x0, y0, x1, y1 = bbox_px
        if not (x1 > x0 and y1 > y0):
            raise ValueError(f"Empty DOM bbox for {region.region_id}: {bbox_px}")
        tolerance = 0.51
        if x0 < -tolerance or y0 < -tolerance or x1 > width + tolerance or y1 > height + tolerance:
            raise ValueError(f"DOM bbox escapes page for {region.region_id}: {bbox_px}")
        planned_delta = max(abs(actual - expected) for actual, expected in zip(bbox_px, region.bbox_px))
        if planned_delta > tolerance:
            raise ValueError(
                f"DOM bbox differs from CSS plan for {region.region_id}: "
                f"planned={region.bbox_px}, actual={bbox_px}, max_delta={planned_delta:.4f}"
            )
        bbox_normalized = (x0 / width, y0 / height, x1 / width, y1 / height)
        region_records.append(
            {
                "region_id": region.region_id,
                "content_id": region.item.content_id,
                "source_group_id": region.item.source_group_id,
                "source_kind": region.item.kind,
                "source_image": region.item.source_image,
                "source_sha256": region.item.source_sha256,
                "bbox": [round(value, 8) for value in bbox_normalized],
                "bbox_px": [round(value, 4) for value in bbox_px],
                "reading_order": region.reading_order,
                "writing_direction": region.writing_direction,
                "computed_writing_mode": str(dom.get("writing_mode", "")),
                "computed_direction": str(dom.get("direction", "")),
                "computed_font_family": str(dom.get("font_family", "")),
                "rendered_fonts": list(dom.get("rendered_fonts", [])),
                "text": region.item.text,
                "valid": True,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "input_level": "page",
        "page_id": plan.page_id,
        "split": plan.split,
        "tier": plan.tier,
        "source_group_ids": sorted({region.item.source_group_id for region in plan.regions}),
        "template_id": plan.template_id,
        "image": image_relative,
        "image_sha256": image_sha256,
        "html": html_relative,
        "page_size": [width, height],
        "page_text_separator": plan.page_text_separator,
        "page_text": plan.page_text,
        "layout_source": "html_synthetic",
        "layout_annotation_status": "complete",
        "bbox_format": "xyxy_normalized",
        "regions": region_records,
        "conversations": [
            {"from": "human", "value": "<image>\nOCR: "},
            {"from": "gpt", "value": plan.page_text},
        ],
        "generator": {
            "page_seed": plan.page_seed,
            **dict(generator_metadata),
        },
        "degradation": dict(degradation or {"tier": plan.tier, "geometry_preserved": True}),
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
