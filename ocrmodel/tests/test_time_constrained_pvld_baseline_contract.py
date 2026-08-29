from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCKER = ROOT / "tools" / "preprocessing" / "prepare_time_constrained_validation.py"
SELECTOR = ROOT / "tools" / "evaluation" / "select_layout_ablation_checkpoint.py"
RUNNER = ROOT / "tools" / "training" / "run_time_constrained_pvld_baseline.sh"


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def record(page_id: str, tier: str, regions: int) -> dict:
    return {
        "input_level": "page",
        "split": "validation",
        "page_id": page_id,
        "tier": tier,
        "regions": [
            {"writing_direction": "vertical_rtl", "bbox": [0, 0, 1, 1]}
            for _ in range(regions)
        ],
        "generator": {"column_count": 2, "row_count": 2},
        "degradation": {"operations": {"occlusion_count": 0}},
    }


def test_lock_is_exactly_400_and_balanced_by_tier(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [
        record(f"{tier}-{index:03d}", tier, 4 + index % 90)
        for tier in ("s3-ancient-hard", "s4-mixed")
        for index in range(205)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "validation.jsonl"
    metadata = tmp_path / "lock.json"
    completed = subprocess.run(
        [
            sys.executable, str(LOCKER), "--source-manifest", str(source),
            "--output-manifest", str(output), "--metadata", str(metadata),
        ], check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    locked = [json.loads(line) for line in output.read_text().splitlines() if line]
    assert len(locked) == 400
    assert {row["tier"] for row in locked} == {"s3-ancient-hard", "s4-mixed"}
    assert sum(row["tier"] == "s3-ancient-hard" for row in locked) == 200
    assert sum(row["tier"] == "s4-mixed" for row in locked) == 200
    payload = json.loads(metadata.read_text())
    assert payload["protocol_version"] == "time_constrained_freeze_strategy_v1"
    assert payload["variant"] == "original_pvld_freeze_strategy"
    assert payload["validation_page_count"] == 400
    assert payload["test_used_for_selection"] is False


def test_selector_rejects_missing_requested_checkpoint_step(tmp_path: Path) -> None:
    module = load_module(SELECTOR)
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "model.safetensors").write_bytes(b"weights")
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "layout_training_metrics.json").write_text(
        json.dumps({"global_step": 12000, "ablation_id": "vlqa_layout_p1_p2"}),
        encoding="utf-8",
    )
    (model_root / "checkpoint-6000").mkdir()
    (model_root / "checkpoint-6000" / "model.safetensors").write_bytes(b"x")
    (model_root / "checkpoint-6000" / "config.json").write_text("{}", encoding="utf-8")
    try:
        module.discover_candidates(
            model_root,
            expected_ablation="vlqa_layout_p1_p2",
            candidate_steps={6000, 9000, 12000},
            prefer_periodic_checkpoint=True,
        )
    except FileNotFoundError as exc:
        assert "9000" in str(exc)
    else:
        raise AssertionError("missing checkpoint-9000 must fail closed")


def test_launcher_registers_original_strategy_and_fixed_candidates() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "time_constrained_freeze_strategy_v1" in source
    assert "original_pvld_freeze_strategy" in source
    assert 'p1_candidate_steps="6000,9000,12000"' in source
    assert "--candidate-steps \"${p1_candidate_steps}\"" in source
    assert '--p1-candidate-steps "${p1_candidate_steps}"' in source
    assert "--candidate-steps 10000,20000,30000" in source
    assert "--candidate-steps 8000" in source
    assert "--stages p2 --p2-max-steps 30000" in source
    assert "--stages p3 --p3-max-steps 8000" in source
    assert "test_used_for_selection" in source
    assert "--test-manifest \"${test_manifest}\"" in source
