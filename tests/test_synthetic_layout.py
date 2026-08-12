from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image


PREPROCESSING_DIR = Path(__file__).resolve().parents[1] / "tools" / "preprocessing"
sys.path.insert(0, str(PREPROCESSING_DIR))

from audit_synthetic_layout import audit_record  # noqa: E402
from prepare_pdf_layout_content import (  # noqa: E402
    PdfSource,
    assign_splits,
    discover_unique_pdfs,
    normalize_text,
    text_layer_fingerprint,
)
from synthetic_layout_common import (  # noqa: E402
    GeneratorConfig,
    build_page_plan,
    load_content_items,
    load_json_records,
    manifest_record_from_dom,
    plan_to_record,
    render_page_html,
    sha256_file,
)


def write_text_content_manifest(path: Path, split: str = "train", count: int = 8) -> None:
    records = [
        {
            "content_id": f"content_{index:03d}",
            "source_group_id": f"book_{index // 2:03d}",
            "split": split,
            "kind": "text",
            "orientation": "any",
            "text": f"第{index}行：甲乙丙丁＆<符号>",
        }
        for index in range(count)
    ]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def small_vertical_config() -> GeneratorConfig:
    config = GeneratorConfig(
        page_width=256,
        page_height=256,
        min_regions=4,
        max_regions=4,
        margin_min=20,
        margin_max=20,
        gap_min=6,
        gap_max=6,
        region_inset_min=1,
        region_inset_max=1,
        font_size_min=14,
        font_size_max=14,
        region_padding=2,
        directions=["vertical_rtl"],
    )
    config.validate()
    return config


def fake_dom(plan: object) -> list[dict[str, object]]:
    regions = []
    for region in plan.regions:
        x0, y0, x1, y1 = region.bbox_px
        regions.append(
            {
                "region_id": region.region_id,
                "reading_order": region.reading_order,
                "writing_direction": region.writing_direction,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "writing_mode": "vertical-rl",
                "direction": "ltr",
                "font_family": "SimSun",
                "content_client_width": max(1, round(x1 - x0)),
                "content_client_height": max(1, round(y1 - y0)),
                "content_scroll_width": max(1, round(x1 - x0)),
                "content_scroll_height": max(1, round(y1 - y0)),
                "content_text": region.item.text if region.item.kind == "text" else None,
                "image_complete": True if region.item.kind == "image" else None,
                "image_natural_width": 24 if region.item.kind == "image" else None,
                "image_natural_height": 96 if region.item.kind == "image" else None,
                "rendered_fonts": (
                    [
                        {
                            "family_name": "SimSun",
                            "postscript_name": "SimSun",
                            "is_custom_font": False,
                            "glyph_count": len(region.item.text),
                        }
                    ]
                    if region.item.kind == "text"
                    else []
                ),
            }
        )
    return regions


def test_jsonl_loading_and_deterministic_rtl_plan(tmp_path: Path) -> None:
    content_manifest = tmp_path / "content.jsonl"
    write_text_content_manifest(content_manifest)
    assert len(load_json_records(content_manifest)) == 8
    items = load_content_items(content_manifest)
    config = small_vertical_config()

    first = build_page_plan(items, config, "train", "s0-html-text", 17, 0)
    second = build_page_plan(items, config, "train", "s0-html-text", 17, 0)
    assert plan_to_record(first) == plan_to_record(second)
    assert [region.reading_order for region in first.regions] == [0, 1, 2, 3]
    x_positions = [region.bbox_px[0] for region in first.regions]
    assert x_positions == sorted(x_positions, reverse=True)
    assert first.page_text == "\n".join(region.item.text for region in first.regions)


def test_pdf_content_normalization_and_source_split_are_deterministic(tmp_path: Path) -> None:
    assert normalize_text("  第一行\n 第二   行 \r\n") == "第一行 第二 行"
    extracted = []
    for index in range(9):
        source = PdfSource(
            path=tmp_path / f"source_{index}.pdf",
            sha256=f"{index:064x}",
            size=index + 1,
        )
        extracted.append((source, [], {}))
    first = assign_splits(extracted, seed=17, validation_sources=2, test_sources=2)
    second = assign_splits(extracted, seed=17, validation_sources=2, test_sources=2)
    assert first == second
    assert list(first.values()).count("train") == 5
    assert list(first.values()).count("validation") == 2
    assert list(first.values()).count("test") == 2


def test_pdf_discovery_deduplicates_identical_files(tmp_path: Path) -> None:
    (tmp_path / "a.pdf").write_bytes(b"same pdf bytes")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "b.PDF").write_bytes(b"same pdf bytes")
    (tmp_path / "c.pdf").write_bytes(b"different pdf bytes")
    sources, duplicates = discover_unique_pdfs([tmp_path])
    assert len(sources) == 2
    assert len(duplicates) == 1


def test_pdf_text_fingerprint_ignores_block_order() -> None:
    from prepare_pdf_layout_content import TextBlock

    first = TextBlock(0, 0, (0, 0, 10, 10), "同一正文")
    second = TextBlock(1, 0, (0, 0, 10, 10), "另一段落")
    assert text_layer_fingerprint([first, second]) == text_layer_fingerprint([second, first])


def test_page_ids_are_unique_across_synthesis_tiers(tmp_path: Path) -> None:
    text_manifest = tmp_path / "text.jsonl"
    write_text_content_manifest(text_manifest)
    text_items = load_content_items(text_manifest)

    crop_records = []
    for index in range(4):
        crop_path = tmp_path / f"tier_crop_{index}.png"
        Image.new("RGB", (24, 96), (245, 240, 225)).save(crop_path)
        crop_records.append(
            {
                "content_id": f"tier_crop_{index}",
                "source_group_id": f"tier_source_{index // 2}",
                "split": "train",
                "kind": "image",
                "orientation": "vertical",
                "image": crop_path.name,
                "text": f"第{index}列",
            }
        )
    crop_manifest = tmp_path / "crops.jsonl"
    crop_manifest.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in crop_records) + "\n",
        encoding="utf-8",
    )
    crop_items = load_content_items(crop_manifest)

    config = small_vertical_config()
    plans = [
        build_page_plan(text_items, config, "train", "s0-html-text", 17, 0),
        build_page_plan(crop_items, config, "train", "s1-html-crop", 17, 0),
        build_page_plan(text_items, config, "train", "s2-hard", 17, 0),
    ]
    assert len({plan.page_id for plan in plans}) == 3
    assert [plan.page_id.split("_")[1] for plan in plans] == ["s0", "s1", "s2"]


def test_single_record_jsonl_is_not_misread_as_wrapped_json(tmp_path: Path) -> None:
    manifest = tmp_path / "single.jsonl"
    record = {
        "content_id": "single",
        "source_group_id": "book",
        "split": "train",
        "text": "单条 JSONL 记录",
    }
    manifest.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_json_records(manifest) == [record]


def test_html_escapes_content_and_declares_vertical_writing(tmp_path: Path) -> None:
    content_manifest = tmp_path / "content.jsonl"
    write_text_content_manifest(content_manifest)
    plan = build_page_plan(
        load_content_items(content_manifest),
        small_vertical_config(),
        "train",
        "s0-html-text",
        23,
        1,
    )
    rendered = render_page_html(plan, small_vertical_config())
    assert "writing-mode:vertical-rl" in rendered
    assert "＆&lt;符号&gt;" in rendered
    assert f'data-page-id="{plan.page_id}"' in rendered
    assert rendered.count('class="region"') == 4


def test_manifest_from_dom_passes_formal_audit(tmp_path: Path) -> None:
    content_manifest = tmp_path / "content.jsonl"
    write_text_content_manifest(content_manifest)
    config = small_vertical_config()
    plan = build_page_plan(
        load_content_items(content_manifest),
        config,
        "train",
        "s0-html-text",
        31,
        2,
    )

    images_dir = tmp_path / "images"
    html_dir = tmp_path / "html"
    images_dir.mkdir()
    html_dir.mkdir()
    image_path = images_dir / f"{plan.page_id}.png"
    html_path = html_dir / f"{plan.page_id}.html"
    Image.new("RGB", plan.page_size, (246, 241, 226)).save(image_path)
    html_path.write_text(render_page_html(plan, config), encoding="utf-8")

    record = manifest_record_from_dom(
        plan=plan,
        dom_regions=fake_dom(plan),
        image_relative=image_path.relative_to(tmp_path).as_posix(),
        html_relative=html_path.relative_to(tmp_path).as_posix(),
        image_sha256=sha256_file(image_path),
        generator_metadata={"test": True, "allowed_rendered_fonts": ["SimSun"]},
    )
    audited = audit_record(
        record=record,
        dataset_root=tmp_path,
        context="manifest[0]",
        skip_image_hash=False,
        skip_html_check=False,
    )
    assert record["schema_version"] == 2
    assert audited["page_id"] == plan.page_id
    assert len(audited["regions"]) == 4

    record["regions"][0]["rendered_fonts"][0]["family_name"] = "Unexpected Font"
    with pytest.raises(ValueError, match="unexpected fallback fonts"):
        audit_record(
            record=record,
            dataset_root=tmp_path,
            context="manifest[0]",
            skip_image_hash=False,
            skip_html_check=False,
        )


def test_manifest_rejects_dom_geometry_drift(tmp_path: Path) -> None:
    content_manifest = tmp_path / "content.jsonl"
    write_text_content_manifest(content_manifest)
    config = small_vertical_config()
    plan = build_page_plan(
        load_content_items(content_manifest),
        config,
        "train",
        "s0-html-text",
        41,
        3,
    )
    dom = fake_dom(plan)
    dom[0]["x0"] = float(dom[0]["x0"]) + 2.0
    with pytest.raises(ValueError, match="differs from CSS plan"):
        manifest_record_from_dom(
            plan=plan,
            dom_regions=dom,
            image_relative="images/page.png",
            html_relative="html/page.html",
            image_sha256="0" * 64,
            generator_metadata={},
        )


def test_source_group_cannot_cross_content_splits(tmp_path: Path) -> None:
    manifest = tmp_path / "leak.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "content_id": "a",
                    "source_group_id": "同一版本",
                    "split": "train",
                    "text": "甲",
                },
                {
                    "content_id": "b",
                    "source_group_id": "同一版本",
                    "split": "validation",
                    "text": "乙",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="multiple splits"):
        load_content_items(manifest)


def test_s1_crop_html_uses_resolved_file_uris(tmp_path: Path) -> None:
    records = []
    for index in range(4):
        crop_path = tmp_path / f"crop_{index}.png"
        Image.new("RGB", (24, 96), (245, 240, 225)).save(crop_path)
        records.append(
            {
                "content_id": f"crop_{index}",
                "source_group_id": f"source_{index // 2}",
                "split": "train",
                "kind": "image",
                "orientation": "vertical",
                "image": crop_path.name,
                "text": f"第{index}列",
            }
        )
    manifest = tmp_path / "crops.jsonl"
    manifest.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )
    config = small_vertical_config()
    plan = build_page_plan(
        load_content_items(manifest),
        config,
        "train",
        "s1-html-crop",
        51,
        0,
    )
    rendered = render_page_html(plan, config)
    assert rendered.count('class="crop-content"') == 4
    assert rendered.count("file:///") == 4
    assert all(region.item.source_sha256 for region in plan.regions)


def test_exact_crop_direction_does_not_match_opposite_direction(tmp_path: Path) -> None:
    crop_path = tmp_path / "ltr.png"
    Image.new("RGB", (96, 24), (245, 240, 225)).save(crop_path)
    manifest = tmp_path / "directional.jsonl"
    records = [
        {
            "content_id": f"ltr_{index}",
            "source_group_id": "source_ltr",
            "split": "train",
            "kind": "image",
            "orientation": "horizontal_ltr",
            "image": crop_path.name,
            "text": f"LTR {index}",
        }
        for index in range(4)
    ]
    manifest.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )
    items = load_content_items(manifest)
    assert all(item.supports_direction("horizontal_ltr") for item in items)
    assert not any(item.supports_direction("horizontal_rtl") for item in items)


def test_plan_only_cli_requires_no_playwright(tmp_path: Path) -> None:
    content_manifest = tmp_path / "content.jsonl"
    write_text_content_manifest(content_manifest)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(small_vertical_config().to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    output_dir = tmp_path / "planned"
    command = [
        sys.executable,
        str(PREPROCESSING_DIR / "generate_synthetic_layout.py"),
        "--content-manifest",
        str(content_manifest),
        "--output-dir",
        str(output_dir),
        "--split",
        "train",
        "--tier",
        "s0-html-text",
        "--num-pages",
        "2",
        "--seed",
        "123",
        "--config",
        str(config_path),
        "--plan-only",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "SYNTHETIC_LAYOUT_PLAN_OK" in completed.stdout
    assert len(load_json_records(output_dir / "plans.jsonl")) == 2
    metadata = json.loads((output_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 2
    assert metadata["formal_manifest_emitted"] is False
    assert metadata["tiers"] == ["s0-html-text"]
    assert metadata["num_pages_per_tier"] == 2
    assert len(list((output_dir / "html").glob("*.html"))) == 2


def test_plan_only_cli_can_emit_a_mixed_tier_manifest(tmp_path: Path) -> None:
    content_manifest = tmp_path / "content.jsonl"
    write_text_content_manifest(content_manifest)
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(small_vertical_config().to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    output_dir = tmp_path / "mixed"
    command = [
        sys.executable,
        str(PREPROCESSING_DIR / "generate_synthetic_layout.py"),
        "--content-manifest",
        str(content_manifest),
        "--output-dir",
        str(output_dir),
        "--split",
        "train",
        "--tier",
        "s0-html-text",
        "--tier",
        "s2-hard",
        "--num-pages",
        "1",
        "--seed",
        "123",
        "--config",
        str(config_path),
        "--plan-only",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "SYNTHETIC_LAYOUT_PLAN_OK" in completed.stdout
    plans = load_json_records(output_dir / "plans.jsonl")
    assert [plan["tier"] for plan in plans] == ["s0-html-text", "s2-hard"]
    assert len({plan["page_id"] for plan in plans}) == 2
    metadata = json.loads((output_dir / "dataset_meta.json").read_text(encoding="utf-8"))
    assert metadata["tiers"] == ["s0-html-text", "s2-hard"]
    assert metadata["num_pages_per_tier"] == 1
    assert metadata["num_pages"] == 2
