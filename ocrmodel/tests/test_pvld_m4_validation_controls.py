from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src" / "GOT-OCR-2.0" / "GOT" / "model" / "GOT_ocr_2_0.py"
EVALUATOR = ROOT / "src" / "GOT-OCR-2.0" / "scripts" / "evaluate_GOT_layout.py"
LAUNCHER = ROOT / "tools" / "evaluation" / "run_bscc_pvld_m4_validation_controls.sbatch"


def test_model_propagates_shuffled_predicted_layout_through_generation() -> None:
    source = MODEL.read_text(encoding="utf-8")
    assert source.count("shuffle_predicted_layout: bool = False") == 2
    assert "shuffle_predicted_layout=shuffle_predicted_layout" in source
    assert '"shuffle_predicted_layout": kwargs.get("shuffle_predicted_layout", False)' in source


def test_evaluator_records_three_pvld_routing_controls() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    assert 'choices=("normal", "alpha_zero", "shuffled_evidence")' in source
    assert 'variable_adapter.residual_gate.zero_()' in source
    assert '"checkpoint_residual_gate": checkpoint_residual_gate' in source
    assert '"effective_residual_gate": effective_residual_gate' in source
    assert '"cyclic_within_batch" if shuffle_predicted_layout else "none"' in source


def test_bscc_launcher_is_validation_only_and_runs_all_controls() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:3" in source
    assert "controls=(normal alpha_zero shuffled_evidence)" in source
    assert "--layout-split validation" in source
    assert "--batch-size 2" in source
    assert 'payload.get("selection_split") != "validation"' in source
    assert 'payload.get("test_used_for_selection") is not False' in source
    assert '"test_manifest_read": False' in source
    assert "test/manifest.jsonl" not in source
