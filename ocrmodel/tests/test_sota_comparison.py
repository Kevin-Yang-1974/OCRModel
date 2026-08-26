from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.sota.mthv2 import iter_manifest
from tools.sota.registry import EXTERNAL_MODELS, INTERNAL_BASELINES
from tools.sota.schema import normalize_text
from tools.sota.summarize_metrics import summarize


def test_external_registry_has_verified_official_ids_and_revisions() -> None:
    assert set(EXTERNAL_MODELS) == {"paddleocr_vl_1_6", "mineru2_5_pro", "glm_ocr", "opendoc_0_1b"}
    for spec in EXTERNAL_MODELS.values():
        assert spec.checkpoint_id and spec.revision and spec.official_repo and spec.license
        assert spec.input_format == "whole_page_image"
        assert spec.parameter_count is None or spec.parameter_count > 0


def test_internal_baselines_register_b0_to_b6_and_c5_budget() -> None:
    assert set(INTERNAL_BASELINES) == {f"B{i}" for i in range(7)}
    assert INTERNAL_BASELINES["B6"]["extra_p1_budget"] is True
    assert INTERNAL_BASELINES["B3"]["capacity_limit"] == 32


def test_normalization_is_deterministic_and_does_not_use_reference() -> None:
    raw = {"res": {"rec_texts": ["甲", "乙"], "boxes": [[0, 0, 1, 1]]}}
    assert normalize_text(raw) == "甲\n乙"
    assert normalize_text("```markdown\n甲\n\n\n乙\n```") == "甲\n\n乙"


def test_normalization_model_fixtures() -> None:
    assert normalize_text({"markdown": "# title\n\nbody"}) == "# title\n\nbody"
    assert normalize_text({"content": "<p>甲</p><p>乙</p>"}) == "甲\n乙"
    assert normalize_text({"results": [{"text": "a"}, {"text": "b"}]}) == "a\nb"
    assert normalize_text({"recognition_results": [{"text": "甲"}, {"text": "乙"}]}) == "甲\n乙"


def test_opendoc_raw_serialization_drops_only_internal_block_images() -> None:
    from tools.sota.adapters import _json_safe

    raw = {
        "recognition_results": [{"text": "甲"}],
        "blocks": [{"img": [[0, 1], [2, 3]], "box": [0, 0, 1, 1], "text": "甲"}],
        "img": "meaningful-top-level-field",
    }
    serialized = _json_safe(raw)
    assert serialized["recognition_results"] == [{"text": "甲"}]
    assert serialized["blocks"] == [{"box": [0, 0, 1, 1], "text": "甲"}]
    assert serialized["img"] == "meaningful-top-level-field"


def test_unified_metrics_align_pages_and_count_failures(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in (
                {"page_id": "p1", "page_text": "甲 乙"},
                {"page_id": "p2", "page_text": "丙"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    predictions.write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False)
            for record in (
                {"page_id": "p2", "normalized_text": "ignored", "status": "error", "runtime": {}},
                {"page_id": "p1", "normalized_text": "甲乙", "status": "ok", "runtime": {"latency_seconds": 2.0, "peak_memory_mib": 10}},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    metrics = summarize(manifest, predictions, expected_pages=2)
    assert metrics["pages"] == 2
    assert metrics["total_edit_distance"] == 2
    assert metrics["micro_page_cer"] == pytest.approx(0.5)
    assert metrics["whitespace_stripped_micro_page_cer"] == pytest.approx(1 / 3)
    assert metrics["failed_pages"] == 1
    assert metrics["failure_rate"] == 0.5
    assert metrics["pages_per_second"] == 0.5
    assert metrics["peak_memory_mib"] == 10
    assert metrics["test_used_for_selection"] is False


def test_unified_metrics_rejects_incomplete_predictions(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    manifest.write_text(json.dumps({"page_id": "p1", "page_text": "甲"}) + "\n", encoding="utf-8")
    predictions.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete result"):
        summarize(manifest, predictions, expected_pages=1)


def test_mthv2_whole_page_contract_rejects_test_and_chunks(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    image = image_root / "page.png"
    image.write_bytes(b"not decoded in contract test")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"split": "test", "page_id": "p", "image": "page.png", "input_level": "page"}) + "\n", encoding="utf-8")
    with pytest.raises(PermissionError):
        list(iter_manifest(manifest, image_root, split="test", limit=1))
    manifest.write_text(json.dumps({"split": "validation", "page_id": "p", "image": "page.png", "input_level": "page", "chunk_index": 0}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Oracle chunks"):
        list(iter_manifest(manifest, image_root, split="validation", limit=1))


def test_prediction_schema_keeps_raw_and_normalized_fields() -> None:
    from tools.sota.schema import PredictionRecord

    record = PredictionRecord("p", "img.png", "GLM-OCR", {"markdown": "甲"}, "甲", "ok", {"latency_seconds": 0.1}, None).to_dict()
    assert record["raw_output"] == {"markdown": "甲"}
    assert record["normalized_text"] == "甲"
    assert record["layout"] is None


def test_test_runner_requires_validation_selection() -> None:
    source = (ROOT / "tools" / "sota" / "run_selection_locked_test.py").read_text(encoding="utf-8")
    assert "selection_split" in source
    assert "test_used_for_selection" in source
    assert "MTHv2 test is locked" in source


def test_finetune_smoke_is_exactly_one_step_and_never_test() -> None:
    source = (ROOT / "tools" / "sota" / "run_finetune_smoke.py").read_text(encoding="utf-8")
    assert "args.max_steps != 1" in source
    assert 'split="train"' in source
    assert "test_used" in source


def test_formal_launcher_is_locked() -> None:
    source = (ROOT / "tools" / "sota" / "run_formal_sota_suite.sh").read_text(encoding="utf-8")
    assert "ALLOW_FORMAL_SOTA" in source
    assert "formal_sota_locked" in source


def test_opendoc_formal_launcher_is_cpu_only_and_strict() -> None:
    source = (ROOT / "tools" / "sota" / "run_opendoc_formal_tmux.sh").read_text(encoding="utf-8")
    assert "nvidia-smi" not in source
    assert source.count("--device cpu --dtype fp32") == 2
    assert "official_finetuning_unavailable" in source
    assert 'validate_predictions "${base}/validation/predictions.jsonl" 240 validation' in source
    assert 'validate_predictions "${base}/test/predictions.jsonl" 800 test' in source
    assert "--allow-formal-test" in source
    assert "--expected-pages 800" in source
    assert "|| true" not in source


def test_deployment_does_not_overwrite_and_uses_personal_root() -> None:
    source = (ROOT / "tools" / "sota" / "deploy_sota_models.sh").read_text(encoding="utf-8")
    assert "data3/yky/yangky_ocr_models" in source
    assert "overwrite=false" in source
    assert "data4/hyf" not in source
