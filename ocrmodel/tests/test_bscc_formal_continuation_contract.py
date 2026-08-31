from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCKER = ROOT / "tools/preprocessing/prepare_random_validation_subset.py"
LAUNCHER = ROOT / "tools/training/run_bscc_p1_continue_p2_formal.sbatch"


def validation_record(index: int) -> dict:
    return {
        "input_level": "page",
        "split": "validation",
        "page_id": f"page-{index:04d}",
        "tier": "s3-ancient-hard" if index % 2 else "s4-mixed",
        "regions": [],
    }


def test_seeded_validation_subset_is_locked_and_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps(validation_record(index)) + "\n" for index in range(500)),
        encoding="utf-8",
    )
    outputs = []
    for suffix in ("a", "b"):
        manifest = tmp_path / f"validation-{suffix}.jsonl"
        lock = tmp_path / f"validation-{suffix}.lock.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(LOCKER),
                "--source-manifest",
                str(source),
                "--output-manifest",
                str(manifest),
                "--metadata",
                str(lock),
                "--page-count",
                "400",
                "--seed",
                "42",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        payload = json.loads(lock.read_text(encoding="utf-8"))
        assert payload["status"] == "locked"
        assert payload["seed"] == 42
        assert payload["validation_page_count"] == 400
        assert payload["test_used_for_selection"] is False
        outputs.append(manifest.read_bytes())
    assert outputs[0] == outputs[1]
    assert len(outputs[0].splitlines()) == 400


def test_bscc_launcher_registers_requested_continuation_and_p2_protocol() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "--p1-max-steps 250000" in source
    assert "--p1-candidate-steps 100000,150000,200000,250000" in source
    assert "--p1-checkpoint-steps 50000" in source
    assert "--resume-existing-run" in source
    assert "--page-count 400 --seed 42" in source
    assert "--p2-max-steps 200000 --p2-checkpoint-steps 50000" in source
    assert "--health-check-steps 10000" in source
    assert "--layout-module-fp32" in source
    assert "--p2-gate-learning-rate 1e-6" in source
    assert "--lr-scheduler-type cosine --warmup-ratio 0.001" in source
    assert "--source-validation-selection" in source
    assert "canary" not in source.lower()
