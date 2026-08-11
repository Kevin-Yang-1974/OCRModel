from __future__ import annotations

import argparse
import importlib.metadata
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from synthetic_layout_common import (
    SCHEMA_VERSION,
    GeneratorConfig,
    TIERS,
    build_page_plan,
    load_content_items,
    manifest_record_from_dom,
    plan_to_record,
    render_page_html,
    sha256_file,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic whole-page HTML/CSS OCR layouts and DOM-supervised manifests. "
            "bbox/order/direction are labels, never model inputs."
        )
    )
    parser.add_argument("--content-manifest", type=Path, required=True)
    parser.add_argument(
        "--content-root",
        type=Path,
        help="Root for relative crop paths; defaults to the content manifest directory.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument(
        "--tier",
        choices=TIERS,
        action="append",
        required=True,
        help="Synthesis tier. Repeat to emit a mixed-tier manifest in one output directory.",
    )
    parser.add_argument("--num-pages", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write deterministic HTML and plans without launching Chromium or emitting a formal manifest.",
    )
    parser.add_argument("--debug-outlines", action="store_true")
    parser.add_argument("--chromium-executable", type=Path)
    parser.add_argument(
        "--browser-channel",
        help="Optional Playwright channel such as chrome or msedge; omit for bundled Chromium.",
    )
    parser.add_argument(
        "--expected-browser-version",
        help="Fail when the launched Chromium version does not exactly match this value.",
    )
    parser.add_argument("--headful", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_pages < 1:
        raise ValueError("--num-pages must be positive.")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative.")
    if len(set(args.tier)) != len(args.tier):
        raise ValueError("--tier values must not be repeated.")
    if args.chromium_executable is not None:
        args.chromium_executable = args.chromium_executable.resolve()
        if not args.chromium_executable.is_file():
            raise FileNotFoundError(args.chromium_executable)
    if args.chromium_executable is not None and args.browser_channel:
        raise ValueError("Use either --chromium-executable or --browser-channel, not both.")
    if args.plan_only and (
        args.chromium_executable is not None
        or args.browser_channel
        or args.expected_browser_version
        or args.headful
    ):
        raise ValueError("Browser options are not used with --plan-only.")


def prepare_empty_output(output_dir: Path) -> Path:
    resolved = output_dir.resolve()
    if resolved.exists():
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        existing = list(resolved.iterdir())
        if existing:
            names = ", ".join(path.name for path in existing[:5])
            raise FileExistsError(
                f"Output directory must be empty to prevent mixed datasets: {resolved} ({names})"
            )
    else:
        resolved.mkdir(parents=True)
    return resolved


def apply_s2_degradation(image_path: Path, seed: int) -> dict[str, Any]:
    rng = random.Random(seed ^ 0xA5A55A5A)
    np_rng = np.random.default_rng(seed ^ 0x5A5AA5A5)
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    contrast = rng.uniform(0.78, 1.08)
    image = ImageEnhance.Contrast(image).enhance(contrast)

    bleed_alpha = rng.uniform(0.025, 0.075)
    ghost = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT).convert("L")
    ghost = ImageEnhance.Contrast(ghost).enhance(rng.uniform(0.45, 0.75))
    ghost_rgb = Image.merge("RGB", (ghost, ghost, ghost))
    image = Image.blend(image, ghost_rgb, bleed_alpha)

    stain_count = rng.randint(2, 7)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for _ in range(stain_count):
        center_x = rng.randint(0, image.width)
        center_y = rng.randint(0, image.height)
        radius_x = rng.randint(max(6, image.width // 80), max(12, image.width // 16))
        radius_y = rng.randint(max(6, image.height // 80), max(12, image.height // 16))
        opacity = rng.randint(4, 16)
        draw.ellipse(
            (
                center_x - radius_x,
                center_y - radius_y,
                center_x + radius_x,
                center_y + radius_y,
            ),
            fill=(92, 61, 34, opacity),
        )
    image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    blur_radius = rng.uniform(0.15, 0.85)
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    noise_sigma = rng.uniform(1.5, 5.0)
    pixels = np.asarray(image, dtype=np.float32)
    pixels += np_rng.normal(0.0, noise_sigma, size=pixels.shape)
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    Image.fromarray(pixels, mode="RGB").save(image_path)

    return {
        "tier": "s2-hard",
        "geometry_preserved": True,
        "operations": {
            "contrast_factor": round(contrast, 6),
            "bleed_through_alpha": round(bleed_alpha, 6),
            "stain_count": stain_count,
            "gaussian_blur_radius": round(blur_radius, 6),
            "gaussian_noise_sigma": round(noise_sigma, 6),
        },
    }


def import_playwright() -> Any:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for rendering. Install the layout dependencies and browser: "
            "python -m pip install -r "
            "ocrmodel/tools/environment/requirements-layout-synthesis.lock.txt; "
            "python -m playwright install chromium. Use --plan-only for a dependency-free HTML smoke."
        ) from exc
    return sync_playwright


def collect_dom_regions(page: Any) -> list[dict[str, Any]]:
    return page.eval_on_selector_all(
        ".region",
        """
        (elements) => elements.map((element) => {
          const rect = element.getBoundingClientRect();
          const style = window.getComputedStyle(element);
          return {
            region_id: element.dataset.regionId,
            reading_order: Number(element.dataset.readingOrder),
            writing_direction: element.dataset.writingDirection,
            x0: rect.left,
            y0: rect.top,
            x1: rect.right,
            y1: rect.bottom,
            writing_mode: style.writingMode,
            direction: style.direction,
            font_family: style.fontFamily,
          };
        })
        """,
    )


def collect_rendered_font_usage(page: Any) -> dict[str, list[dict[str, Any]]]:
    session = page.context.new_cdp_session(page)
    try:
        session.send("DOM.enable")
        session.send("CSS.enable")
        root_node = session.send("DOM.getDocument", {"depth": -1})["root"]["nodeId"]
        region_nodes = session.send(
            "DOM.querySelectorAll",
            {"nodeId": root_node, "selector": ".region"},
        )["nodeIds"]
        usage: dict[str, list[dict[str, Any]]] = {}
        for region_node in region_nodes:
            raw_attributes = session.send(
                "DOM.getAttributes",
                {"nodeId": region_node},
            )["attributes"]
            attributes = dict(zip(raw_attributes[::2], raw_attributes[1::2]))
            region_id = attributes.get("data-region-id")
            if not region_id:
                raise RuntimeError("Rendered region has no data-region-id attribute.")
            text_node = session.send(
                "DOM.querySelector",
                {"nodeId": region_node, "selector": ".text-content"},
            )["nodeId"]
            fonts: list[dict[str, Any]] = []
            if text_node:
                raw_fonts = session.send(
                    "CSS.getPlatformFontsForNode",
                    {"nodeId": text_node},
                )["fonts"]
                fonts = [
                    {
                        "family_name": str(font.get("familyName", "")),
                        "postscript_name": str(font.get("postScriptName", "")),
                        "is_custom_font": bool(font.get("isCustomFont", False)),
                        "glyph_count": int(font.get("glyphCount", 0)),
                    }
                    for font in raw_fonts
                ]
            usage[region_id] = fonts
        return usage
    finally:
        session.detach()


def attach_and_validate_rendered_fonts(
    plan: Any,
    dom_regions: list[dict[str, Any]],
    font_usage: dict[str, list[dict[str, Any]]],
    allowed_fonts: list[str],
) -> None:
    dom_by_id = {str(region.get("region_id", "")): region for region in dom_regions}
    if set(dom_by_id) != set(font_usage):
        raise RuntimeError(
            f"DOM/font region mismatch for {plan.page_id}: "
            f"dom={sorted(dom_by_id)}, fonts={sorted(font_usage)}"
        )
    allowed = set(allowed_fonts)
    for region in plan.regions:
        fonts = font_usage.get(region.region_id, [])
        dom_by_id[region.region_id]["rendered_fonts"] = fonts
        if region.item.kind != "text":
            continue
        if not fonts or sum(font["glyph_count"] for font in fonts) < 1:
            raise RuntimeError(
                f"No rendered glyph font was reported for text region {region.region_id}."
            )
        unexpected = sorted(
            {
                font["family_name"]
                for font in fonts
                if font["family_name"] not in allowed
            }
        )
        if unexpected:
            raise RuntimeError(
                f"Unexpected font fallback in {region.region_id}: {unexpected}; "
                f"allowed={allowed_fonts}"
            )


def validate_page_metrics(metrics: dict[str, Any], config: GeneratorConfig, page_id: str) -> None:
    expected = {
        "inner_width": config.page_width,
        "inner_height": config.page_height,
        "scroll_width": config.page_width,
        "scroll_height": config.page_height,
        "device_pixel_ratio": 1,
    }
    mismatches = {
        key: (expected_value, metrics.get(key))
        for key, expected_value in expected.items()
        if metrics.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Viewport/DOM mismatch for {page_id}: {mismatches}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    output_dir = prepare_empty_output(args.output_dir)
    config = GeneratorConfig.from_json(args.config)
    content_manifest = args.content_manifest.resolve()
    content_root = args.content_root.resolve() if args.content_root else None
    items = load_content_items(content_manifest, content_root)

    html_dir = output_dir / "html"
    html_dir.mkdir()
    plans = [
        build_page_plan(
            items=items,
            config=config,
            split=args.split,
            tier=tier,
            base_seed=args.seed,
            page_index=page_index,
        )
        for tier in args.tier
        for page_index in range(args.num_pages)
    ]
    html_paths: list[Path] = []
    for plan in plans:
        html_path = html_dir / f"{plan.page_id}.html"
        html_path.write_text(
            render_page_html(plan, config, debug_outlines=args.debug_outlines),
            encoding="utf-8",
            newline="\n",
        )
        html_paths.append(html_path)

    common_metadata = {
        "schema_version": SCHEMA_VERSION,
        "content_manifest": str(content_manifest),
        "content_manifest_sha256": sha256_file(content_manifest),
        "content_root": str((content_root or content_manifest.parent).resolve()),
        "split": args.split,
        "tiers": args.tier,
        "num_pages_per_tier": args.num_pages,
        "num_pages": len(plans),
        "base_seed": args.seed,
        "generator_config": config.to_dict(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }

    if args.plan_only:
        write_jsonl(output_dir / "plans.jsonl", (plan_to_record(plan) for plan in plans))
        write_json(
            output_dir / "dataset_meta.json",
            {**common_metadata, "status": "plan_only", "formal_manifest_emitted": False},
        )
        print("SYNTHETIC_LAYOUT_PLAN_OK")
        print(f"pages={len(plans)}")
        print(f"output_dir={output_dir}")
        print(f"plans={output_dir / 'plans.jsonl'}")
        return

    sync_playwright = import_playwright()
    images_dir = output_dir / "images"
    images_dir.mkdir()
    manifest_records: list[dict[str, Any]] = []
    browser_version = ""
    playwright_version = importlib.metadata.version("playwright")

    with sync_playwright() as playwright:
        launch_options: dict[str, Any] = {"headless": not args.headful}
        if args.chromium_executable is not None:
            launch_options["executable_path"] = str(args.chromium_executable)
        if args.browser_channel:
            launch_options["channel"] = args.browser_channel
        try:
            browser = playwright.chromium.launch(**launch_options)
        except Exception as exc:
            raise RuntimeError(
                "Chromium could not be launched. Install the Playwright browser with "
                "'python -m playwright install chromium' or provide --chromium-executable."
            ) from exc
        try:
            browser_version = browser.version
            if args.expected_browser_version and browser_version != args.expected_browser_version:
                raise RuntimeError(
                    "Chromium version mismatch: "
                    f"expected={args.expected_browser_version!r}, actual={browser_version!r}"
                )
            for plan, html_path in zip(plans, html_paths):
                page = browser.new_page(
                    viewport={"width": config.page_width, "height": config.page_height},
                    device_scale_factor=1,
                    locale="zh-CN",
                    color_scheme="light",
                )
                try:
                    page.goto(html_path.resolve().as_uri(), wait_until="load")
                    page.evaluate("() => document.fonts.ready")
                    page.wait_for_function(
                        "() => Array.from(document.images).every((image) => "
                        "image.complete && image.naturalWidth > 0 && image.naturalHeight > 0)"
                    )
                    metrics = page.evaluate(
                        """
                        () => ({
                          inner_width: window.innerWidth,
                          inner_height: window.innerHeight,
                          scroll_width: document.documentElement.scrollWidth,
                          scroll_height: document.documentElement.scrollHeight,
                          device_pixel_ratio: window.devicePixelRatio,
                        })
                        """
                    )
                    validate_page_metrics(metrics, config, plan.page_id)
                    dom_regions = collect_dom_regions(page)
                    font_usage = collect_rendered_font_usage(page)
                    attach_and_validate_rendered_fonts(
                        plan=plan,
                        dom_regions=dom_regions,
                        font_usage=font_usage,
                        allowed_fonts=config.allowed_rendered_fonts,
                    )
                    image_path = images_dir / f"{plan.page_id}.png"
                    page.screenshot(path=str(image_path), full_page=False, animations="disabled")
                finally:
                    page.close()

                with Image.open(image_path) as rendered:
                    if rendered.size != plan.page_size:
                        raise RuntimeError(
                            f"Screenshot size mismatch for {plan.page_id}: "
                            f"expected={plan.page_size}, actual={rendered.size}"
                        )
                degradation: dict[str, Any]
                if plan.tier == "s2-hard":
                    degradation = apply_s2_degradation(image_path, plan.page_seed)
                else:
                    degradation = {
                        "tier": args.tier,
                        "geometry_preserved": True,
                        "operations": {},
                    }
                manifest_records.append(
                    manifest_record_from_dom(
                        plan=plan,
                        dom_regions=dom_regions,
                        image_relative=image_path.relative_to(output_dir).as_posix(),
                        html_relative=html_path.relative_to(output_dir).as_posix(),
                        image_sha256=sha256_file(image_path),
                        generator_metadata={
                            "base_seed": args.seed,
                            "playwright_version": playwright_version,
                            "chromium_version": browser_version,
                            "device_scale_factor": 1,
                            "viewport": [config.page_width, config.page_height],
                            "font_family": config.font_family,
                            "allowed_rendered_fonts": config.allowed_rendered_fonts,
                        },
                        degradation=degradation,
                    )
                )
        finally:
            browser.close()

    write_jsonl(output_dir / "manifest.jsonl", manifest_records)
    write_json(
        output_dir / "dataset_meta.json",
        {
            **common_metadata,
            "status": "rendered",
            "formal_manifest_emitted": True,
            "playwright_version": playwright_version,
            "chromium_version": browser_version,
            "device_scale_factor": 1,
            "manifest": "manifest.jsonl",
        },
    )
    print("SYNTHETIC_LAYOUT_GENERATION_OK")
    print(f"pages={len(manifest_records)}")
    print(f"tiers={','.join(args.tier)}")
    print(f"chromium_version={browser_version}")
    print(f"output_dir={output_dir}")
    print(f"manifest={output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"SYNTHETIC_LAYOUT_GENERATION_ERROR: {exc}", file=sys.stderr)
        raise
