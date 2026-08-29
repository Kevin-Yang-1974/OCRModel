from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from unittest.mock import patch
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


def test_launcher_source_keeps_recovery_log_separate(tmp_path: Path) -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'parser.add_argument("--resume-existing-run", action="store_true")' in source
    assert '"--save_total_limit", str(checkpoint_retention)' in source
    assert "output.mkdir(parents=True, exist_ok=args.resume_existing_run)" in source
    assert runner.training_log_path(tmp_path, False) == tmp_path / "train.log"
    assert runner.training_log_path(tmp_path, True) == tmp_path / "train.recovery.log"
    (tmp_path / "train.recovery.log").touch()
    assert runner.training_log_path(tmp_path, True) == tmp_path / "train.recovery.2.log"
    (tmp_path / "train.recovery.2.log").touch()
    assert runner.training_log_path(tmp_path, True) == tmp_path / "train.recovery.3.log"


def test_p1_checkpoints_are_queued_every_2000_steps_and_selected_on_validation() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'parser.add_argument("--checkpoint-steps", type=int, default=2000)' in source
    assert '"--selection-purpose", "p1_layout"' in source
    assert 'selection.get("selection_split") != "validation"' in source
    assert 'selection.get(\n                "test_used_for_selection"\n            ) is not False' in source
    assert 'source = Path(selection["selected"]["model_path"]).resolve()' in source
    assert 'source_selection_path = selection_path.resolve()' in source
    assert '"--source_validation_selection", str(source_validation_selection)' in source
    assert 'selection_gpu_ids, selection_utilization = target_gpus(' in source
    assert 'p1_selection_command(args, output, selection_dir, selection_gpu_ids)' in source
    assert '"selection_physical_gpu_ids": list(selection_gpu_ids)' in source


def test_c5_recovery_reuses_selection_without_modifying_checkpoint() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'candidate_selection = run_root / "p1" / "validation_selection" / "selection.json"' in source
    assert 'PVLD C5 {stage.upper()} requires its validation-only preceding-stage selection.' in source

    trainer_source = (
        Path(__file__).resolve().parents[1]
        / "src" / "GOT-OCR-2.0" / "scripts" / "train_GOT_layout.py"
    ).read_text(encoding="utf-8")
    assert 'selection.get("selection_purpose") != expected_selection_purpose' in trainer_source
    assert 'selection.get("test_used_for_selection") is not False' in trainer_source
    assert 'selected_model != source_model.resolve()' in trainer_source
    assert 'selected.get("weights_sha256") != file_sha256(weights_path)' in trainer_source


def test_p3_requires_p2_ocr_selection_and_preserves_selection_provenance() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert 'parser.add_argument(\n        "--source-validation-selection"' in source
    assert 'if stage in {"p2", "p3"} and args.ablation == "vlqa_layout_p1_p2"' in source
    assert '"--source_validation_selection", str(source_validation_selection)' in source

    trainer_source = (
        Path(__file__).resolve().parents[1]
        / "src" / "GOT-OCR-2.0" / "scripts" / "train_GOT_layout.py"
    ).read_text(encoding="utf-8")
    assert '"p2": "p1_layout"' in trainer_source
    assert '"p3": "ocr"' in trainer_source
    assert 'def validate_pvld_source_selection(' in trainer_source
    assert '"source_validation_selection": (' in trainer_source


def test_runner_propagates_m1_boundary_configuration_and_parallel_selection() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"--layout_boundary_loss_weight"' in source
    assert '"--layout_count_condition_strength"' in source
    assert '"--parallel-gpu-ids", ",".join(gpu_ids)' in source
    assert '"--nproc_per_node",' in source
    assert 'environment["CUDA_VISIBLE_DEVICES"] = ",".join(ids)' in source
    assert 'choices=("deepspeed_zero2", "ddp")' in source
    assert 'str(deepspeed), "--num_gpus", str(len(gpu_ids)), "--master_port"' in source
    assert 'DeepSpeed launcher is missing' in source
    assert 'str(torchrun), "--standalone", "--nproc_per_node"' in source
    assert '["--ddp_find_unused_parameters", "True"]' in source
    assert '"distributed_strategy": args.distributed_strategy' in source
    assert '"--nccl-p2p-disable"' in source
    assert 'environment["NCCL_P2P_DISABLE"] = "1"' in source
    assert '"nccl_p2p_disable": args.nccl_p2p_disable' in source
    assert '"--pvld-use-spatial-memory"' in source
    assert '"--pvld-shared-gradient-scale"' in source
    assert '"--pvld-record-gradient-scale"' in source
    assert '"--pvld-predicted-layout-routing"' in source
    assert 'args.pvld_shared_gradient_scale if stage == "p2" else 1.0' in source
    assert 'args.pvld_predicted_layout_routing and stage == "p2"' in source
    assert '"p1_forces_legacy_gradient_scale_and_routing": True' in source
    assert 'args.stages != "p1" or args.p1_max_steps > 10' in source
    assert 'and not args.skip_p1_validation_selection' in source
    train_source = (
        Path(__file__).resolve().parents[1]
        / "src" / "GOT-OCR-2.0" / "scripts" / "train_GOT_layout.py"
    ).read_text(encoding="utf-8")
    assert '"boundary_loss": "layout_boundary_loss"' in train_source
    assert '"causal_transformer_fsm_boundary_count_v1"' in train_source
    nccl_source = (
        Path(__file__).resolve().parents[1] / "tools" / "training" / "smoke_nccl_a100.py"
    ).read_text(encoding="utf-8")
    assert 'dist.init_process_group("nccl")' in nccl_source
    assert 'torch.cuda.set_device(local_rank)' in nccl_source
    assert 'parser.add_argument("--tensor-mib", type=int, default=0)' in nccl_source
    assert 'dist.broadcast(payload, src=0)' in nccl_source
    assert 'torch.cuda.set_device(local_rank)' in train_source
    assert 'getattr(adapter, "visual_routing", None),' in train_source
    assert 'adapter.residual_gate.requires_grad_(False)' in train_source
    assert 'audit_ddp_frozen_parameters(model, training_args)' in train_source
    assert '"ddp_private_ignore_applied": False' in train_source
    assert '_set_params_and_buffers_to_ignore_for_model' not in train_source
    assert '"first_training_step_complete"' in train_source
    smoke_source = (
        Path(__file__).resolve().parents[1]
        / "tools" / "training" / "smoke_pvld_causal_decoder_cuda.py"
    ).read_text(encoding="utf-8")
    assert '"count_head_through_boundary_logits"' in smoke_source
    assert '"zero_strength_legacy_exact": True' in smoke_source


def test_target_gpus_auto_admits_all_cards_below_threshold() -> None:
    result = type("Result", (), {
        "returncode": 0,
        "stdout": "0, 49\n1, 50\n2, 3\n3, 0\n",
        "stderr": "",
    })()
    with patch.object(runner.subprocess, "run", return_value=result) as mocked:
        ids, observed = runner.target_gpus("", 50)
    assert ids == ("0", "2", "3")
    assert observed == {"0": 49, "1": 50, "2": 3, "3": 0}
    command = mocked.call_args.args[0]
    assert command[0] == "nvidia-smi"
    assert "-i" not in command
    assert "--query-gpu=index,utilization.gpu" in command


def test_target_gpus_explicit_queries_only_requested_cards() -> None:
    result = type("Result", (), {
        "returncode": 0,
        "stdout": "3, 1\n1, 2\n",
        "stderr": "",
    })()
    with patch.object(runner.subprocess, "run", return_value=result) as mocked:
        ids, observed = runner.target_gpus("3,1", 50)
    assert ids == ("3", "1")
    assert observed == {"3": 1, "1": 2}
    command = mocked.call_args.args[0]
    assert command[0:3] == ["nvidia-smi", "-i", "3,1"]


def test_target_gpus_auto_fails_when_all_cards_are_busy() -> None:
    result = type("Result", (), {
        "returncode": 0,
        "stdout": "0, 50\n1, 99\n",
        "stderr": "",
    })()
    with patch.object(runner.subprocess, "run", return_value=result):
        try:
            runner.target_gpus("", 50)
        except RuntimeError as exc:
            assert "no GPU" in str(exc)
        else:
            raise AssertionError("busy auto-admission unexpectedly succeeded")


def test_stage_learning_rates_are_differentiated() -> None:
    args = type(
        "Args",
        (),
        {
            "p1_vision_learning_rate": 1e-6,
            "p1_projector_learning_rate": 1e-5,
            "p1_layout_learning_rate": 1e-4,
            "p1_qwen_learning_rate": 0.0,
            "p1_gate_learning_rate": 0.0,
            "p1_lm_head_learning_rate": 0.0,
            "p2_vision_learning_rate": 5e-7,
            "p2_projector_learning_rate": 5e-6,
            "p2_layout_learning_rate": 5e-5,
            "p2_qwen_learning_rate": 1e-6,
            "p2_gate_learning_rate": 1e-5,
            "p2_lm_head_learning_rate": 0.0,
            "p3_vision_learning_rate": 2e-7,
            "p3_projector_learning_rate": 2e-6,
            "p3_layout_learning_rate": 1e-5,
            "p3_qwen_learning_rate": 5e-7,
            "p3_gate_learning_rate": 1e-6,
            "p3_lm_head_learning_rate": 0.0,
        },
    )()
    p1 = runner.stage_learning_rates(args, "p1")
    p2 = runner.stage_learning_rates(args, "p2")
    p3 = runner.stage_learning_rates(args, "p3")
    assert p1["vision"] > p2["vision"] > p3["vision"]
    assert p1["layout"] > p2["layout"] > p3["layout"]
    assert p2["qwen"] > p3["qwen"]
