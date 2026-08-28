from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "tools" / "training" / "run_historical_s3s4_p1_p2_pilot_tmux.sh"
RUNNER = ROOT / "tools" / "training" / "run_variable_layout_a100.py"


def test_historical_pilot_isolated_and_has_registered_budget() -> None:
    source = PILOT.read_text(encoding="utf-8")
    assert "ancient_photo_diverse_formal_s3s4_20260826_v1" in source
    assert "p1-max-steps 2000" in source
    assert "p2-max-steps 5000" in source
    assert "checkpoint-steps 1000" in source
    assert "primary:replay=7:1" not in source  # protocol is passed as explicit flags
    assert "--replay-manifest" in source
    assert "--replay-ocr-loss-weight" not in source  # runner registers the fixed default
    assert "--source-validation-selection" in source
    assert '"test_run":false' in source
    assert '"p3_run":false' in source
    assert "--validation-image-root" in source
    assert "--layout-memory-resolution 64" in source
    assert 'distributed_strategy="deepspeed_zero2"' in source
    assert '--distributed-strategy) distributed_strategy="$2"' in source
    assert '--distributed-strategy "${distributed_strategy}"' in source
    assert "nccl_p2p_disable=1" in source
    assert "--nccl-p2p-disable" in source


def test_runner_exposes_separate_validation_image_root() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert '"--validation-image-root"' in source
    assert "args.validation_image_root or args.validation_manifest.parent" in source
