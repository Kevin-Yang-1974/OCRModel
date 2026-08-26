from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "tools" / "evaluation" / "pvld_diagnostic_protocol.py"
SCRIPT_PATH = ROOT / "tools" / "evaluation" / "diagnose_pvld_bounded.py"
LAUNCHER_PATH = ROOT / "tools" / "evaluation" / "run_pvld_bounded_diagnostics.sh"
SPEC = importlib.util.spec_from_file_location("pvld_diagnostic_protocol_under_test", PROTOCOL_PATH)
assert SPEC is not None and SPEC.loader is not None
protocol = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = protocol
SPEC.loader.exec_module(protocol)


def test_bucket_selection_is_deterministic_and_covers_complexity() -> None:
    counts = [3, 12, 24, 60, 7, 15, 28, 80]
    records = [{"regions": [{}] * count} for count in counts]
    assert protocol.select_bucket_indices(records, 2) == {
        "0-8": [0, 4],
        "9-16": [1, 5],
        "17-32": [2, 6],
        ">32": [3, 7],
    }


def test_duplicate_diagnostic_reports_first_repeated_box() -> None:
    boxes = [
        [0.0, 0.0, 0.2, 0.2],
        [0.4, 0.4, 0.6, 0.6],
        [0.0, 0.0, 0.2, 0.2],
    ]
    result = protocol.duplicate_diagnostics(boxes)
    assert result["duplicate_after_first_count"] == 1
    assert result["duplicate_after_first_rate"] == 1 / 3
    assert result["first_duplicate_region_index"] == 2


def test_diagnostic_contract_is_train_validation_only_and_has_four_audits() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"splits_read": ["train", "validation"]' in source
    assert '"test_read": False' in source
    assert '"optimizer_steps": 0' in source
    assert "oracle_and_free_page" in source
    assert "gradient_audit" in source
    assert '"alpha_zero"' in source
    assert '"shuffled_evidence"' in source
    assert "optimizer.step" not in source


def test_gradient_groups_separate_evidence_decoder_and_record_heads() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert '"layout_evidence": list(adapter.decoder.prompt_attention.parameters())' in source
    assert '"causal_decoder": [' in source
    assert 'if not name.startswith("prompt_attention.")' in source
    assert '"record_heads": list(adapter.record_heads.parameters())' in source


def test_launcher_queries_only_explicit_non_reserved_gpu() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert 'nvidia-smi -i "${gpu_id}"' in source
    assert '[[ "${gpu_id}" != "2" ]]' in source
    assert "CUDA_VISIBLE_DEVICES=\"${gpu_id}\"" in source
    assert "test/manifest.jsonl" not in source
    assert "timeout --signal=TERM" in source
