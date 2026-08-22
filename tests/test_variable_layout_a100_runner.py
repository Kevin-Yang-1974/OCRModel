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
