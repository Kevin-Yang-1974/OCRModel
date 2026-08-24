from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "training"
    / "run_variable_layout_a100.py"
)
SPEC = importlib.util.spec_from_file_location("variable_layout_a100_runner_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_launcher_source_keeps_recovery_log_separate() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'parser.add_argument("--resume-existing-run", action="store_true")' in source
    assert '"--save_total_limit", str(checkpoint_retention)' in source
    assert '"train.recovery.log" if args.resume_existing_run else "train.log"' in source
    assert "output.mkdir(parents=True, exist_ok=args.resume_existing_run)" in source


def test_p1_checkpoints_are_queued_every_2000_steps_and_selected_on_validation() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'parser.add_argument("--checkpoint-steps", type=int, default=2000)' in source
    assert '"--selection-purpose", "p1_layout"' in source
    assert 'selection.get("selection_split") != "validation"' in source
    assert 'selection.get(\n                "test_used_for_selection"\n            ) is not False' in source
    assert 'source = Path(selection["selected"]["model_path"]).resolve()' in source
    assert 'p1_selection_path = selection_path.resolve()' in source
    assert '"--source_validation_selection", str(source_validation_selection)' in source


def test_c5_recovery_reuses_selection_without_modifying_checkpoint() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'candidate_selection = run_root / "p1" / "validation_selection" / "selection.json"' in source
    assert 'PVLD C5 P2 requires its validation-only P1 selection.' in source

    trainer_source = (
        Path(__file__).resolve().parents[1]
        / "src" / "GOT-OCR-2.0" / "scripts" / "train_GOT_layout.py"
    ).read_text(encoding="utf-8")
    assert 'selection.get("selection_purpose") != "p1_layout"' in trainer_source
    assert 'selection.get("test_used_for_selection") is not False' in trainer_source
    assert 'selected_model != source_model.resolve()' in trainer_source
    assert 'selected.get("weights_sha256") != file_sha256(weights_path)' in trainer_source
