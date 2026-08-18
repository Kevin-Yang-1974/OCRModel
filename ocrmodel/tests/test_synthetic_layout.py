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
from generate_synthetic_layout import (  # noqa: E402
    apply_effective_font_sizes,
    apply_s2_degradation,
)
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


def test_quoted_font_stack_is_escaped_and_text_font_is_fitted(tmp_path: Path) -> None:
    content_manifest = tmp_path / "content.jsonl"
    write_text_content_manifest(content_manifest, count=4)
    config = GeneratorConfig(
        page_width=256,
        page_height=256,
        min_regions=4,
        max_regions=4,
        margin_min=16,
        margin_max=16,
        gap_min=2,
        gap_max=2,
        font_size_min=12,
        font_size_max=40,
        region_padding=1,
        line_height_min=1.4,
        line_height_max=1.4,
        font_family='"Noto Serif CJK SC", serif',
        font_families=['"Noto Serif CJK SC", serif'],
        directions=["horizontal_ltr"],
    )
    config.validate()
    plan = build_page_plan(
        load_content_items(content_manifest), config, "train", "s0-html-text", 20260817, 0
    )
    rendered = render_page_html(plan, config)
    assert "font-family:&quot;Noto Serif CJK SC&quot;, serif" in rendered
    assert 'font-family:"Noto Serif CJK SC", serif' not in rendered
    assert all(config.font_size_min <= region.font_size <= config.font_size_max for region in plan.regions)
    assert any(region.font_size < config.font_size_max for region in plan.regions)


def test_effective_browser_font_sizes_are_written_back_to_plan(tmp_path: Path) -> None:
    content_manifest = tmp_path / "content.jsonl"
    write_text_content_manifest(content_manifest)
    plan = build_page_plan(
        load_content_items(content_manifest),
        small_vertical_config(),
        "train",
        "s0-html-text",
        20260817,
        0,
    )
    first = plan.regions[0]
    fitted = apply_effective_font_sizes(
        plan,
        [
            {
                "region_id": first.region_id,
                "effective_font_size": first.font_size - 1,
            }
        ],
    )
    assert fitted.regions[0].font_size == first.font_size - 1
    assert fitted.regions[1:] == plan.regions[1:]
    assert plan.regions[0].font_size == first.font_size


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


def test_diverse_dense_layout_records_language_font_and_nonuniform_widths(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "multilingual.jsonl"
    languages = ("zh-Hant", "ja", "ko", "en", "mixed-symbols")
    records = [
        {
            "content_id": f"dense_{index:03d}",
            "source_group_id": f"source_{index:03d}",
            "split": "train",
            "kind": "text",
            "orientation": "vertical",
            "language": languages[index % len(languages)],
            "text": f"密集古籍文字{index}ABC※",
        }
        for index in range(20)
    ]
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    config = GeneratorConfig(
        page_width=512,
        page_height=512,
        min_regions=8,
        max_regions=8,
        margin_min=16,
        margin_max=16,
        gap_min=12,
        gap_max=12,
        dense_gap_probability=1.0,
        dense_gap_min=0,
        dense_gap_max=2,
        region_inset_min=0,
        region_inset_max=1,
        region_extent_weight_min=0.7,
        region_extent_weight_max=1.5,
        font_size_min=18,
        font_size_max=34,
        region_padding=0,
        line_height_min=0.9,
        line_height_max=1.1,
        letter_spacing_min=-1.0,
        letter_spacing_max=0.25,
        ink_opacity_min=0.7,
        ink_opacity_max=0.95,
        font_families=["Dense Serif A, serif", "Dense Serif B, serif"],
        font_weights=[400, 600],
        text_colors=["#17110e", "#49382c"],
        allowed_rendered_fonts=["Dense Serif A", "Dense Serif B"],
        directions=["vertical_rtl"],
    )
    config.validate()
    plan = build_page_plan(
        load_content_items(manifest), config, "train", "s0-html-text", 29, 0
    )
    visual = sorted(plan.regions, key=lambda region: region.bbox_px[0])
    widths = [region.bbox_px[2] - region.bbox_px[0] for region in visual]
    gaps = [visual[index + 1].bbox_px[0] - visual[index].bbox_px[2] for index in range(7)]
    assert max(widths) - min(widths) > 8
    assert max(gaps) <= 4
    assert {region.item.language for region in plan.regions}.issubset(set(languages))
    assert all(0.9 <= region.line_height <= 1.1 for region in plan.regions)
    assert all(-1.0 <= region.letter_spacing <= 0.25 for region in plan.regions)
    rendered = render_page_html(plan, config)
    assert "letter-spacing:" in rendered
    assert "data-language=" in rendered
    record = plan_to_record(plan)
    assert all(region["language"] for region in record["regions"])
    assert all("font_family" in region for region in record["regions"])


def test_s2_diverse_degradation_is_deterministic_and_preserves_geometry(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (128, 128), (226, 210, 172)).save(first)
    Image.new("RGB", (128, 128), (226, 210, 172)).save(second)
    config = GeneratorConfig(
        s2_stain_count_min=3,
        s2_stain_count_max=3,
        s2_speckle_density_min=0.01,
        s2_speckle_density_max=0.01,
    )
    config.validate()
    first_meta = apply_s2_degradation(first, 20260817, config)
    second_meta = apply_s2_degradation(second, 20260817, config)
    assert sha256_file(first) == sha256_file(second)
    assert first_meta == second_meta
    assert first_meta["geometry_preserved"] is True
    assert first_meta["operations"]["speckle_count"] > 0
    with Image.open(first) as degraded:
        assert degraded.size == (128, 128)


def test_language_tag_and_diverse_range_validation_are_strict(tmp_path: Path) -> None:
    manifest = tmp_path / "bad_language.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "content_id": "bad",
                "source_group_id": "source",
                "split": "train",
                "text": "文本",
                "language": "zh Hant",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="language"):
        load_content_items(manifest)
    with pytest.raises(ValueError, match="line heights"):
        GeneratorConfig(line_height_min=0.2).validate()
    with pytest.raises(ValueError, match="glyph_extent_safety_factor"):
        GeneratorConfig(glyph_extent_safety_factor=0.9).validate()
    with pytest.raises(ValueError, match="speckle density"):
        GeneratorConfig(s2_speckle_density_max=0.5).validate()


def test_diverse_split_launcher_writes_reproducible_plan_protocol(tmp_path: Path) -> None:
    manifest = tmp_path / "all_splits.jsonl"
    records = [
        {
            "content_id": f"{split}_{index}",
            "source_group_id": f"{split}_source_{index}",
            "split": split,
            "kind": "text",
            "orientation": "any",
            "language": "zh-Hant" if index % 2 == 0 else "ja",
            "text": f"{split} 多語言內容 {index}",
        }
        for split in ("train", "validation", "test")
        for index in range(6)
    ]
    manifest.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    output = tmp_path / "diverse_plan"
    completed = subprocess.run(
        [
            sys.executable,
            str(PREPROCESSING_DIR / "prepare_diverse_synthetic_layout.py"),
            "--content-manifest",
            str(manifest),
            "--output-root",
            str(output),
            "--tier",
            "s0-html-text",
            "--tier",
            "s2-hard",
            "--train-pages-per-tier",
            "2",
            "--validation-pages-per-tier",
            "1",
            "--test-pages-per-tier",
            "1",
            "--plan-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    protocol = json.loads(
        (output / "dataset_protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["status"] == "plan_only"
    assert protocol["total_pages"] == 8
    assert protocol["input_level"] == "whole_page_image"
    assert protocol["layout_metadata_as_model_input"] is False
    assert len(load_json_records(output / "train" / "plans.jsonl")) == 4
    assert "diverse_synthetic_dataset_prepared" in completed.stdout


def test_direction_weights_make_ancient_vertical_dominant(tmp_path: Path) -> None:
    manifest = tmp_path / "directions.jsonl"
    write_text_content_manifest(manifest, count=12)
    config = small_vertical_config()
    config.directions = ["vertical_rtl", "horizontal_ltr"]
    config.direction_weights = {"vertical_rtl": 0.95, "horizontal_ltr": 0.05}
    config.validate()
    items = load_content_items(manifest)
    directions = [
        build_page_plan(items, config, "train", "s0-html-text", 91, index)
        .regions[0]
        .writing_direction
        for index in range(500)
    ]
    assert directions.count("vertical_rtl") > 440
    with pytest.raises(ValueError, match="every enabled direction"):
        config.direction_weights = {"vertical_rtl": 1.0}
        config.validate()
