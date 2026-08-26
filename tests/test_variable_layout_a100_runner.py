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


def test_runner_propagates_m1_boundary_configuration_and_parallel_selection() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert '"--layout_boundary_loss_weight"' in source
    assert '"--layout_count_condition_strength"' in source
    assert '"--parallel-gpu-ids", ",".join(gpu_ids)' in source
    assert '"--nproc_per_node",' in source
    assert 'environment["CUDA_VISIBLE_DEVICES"] = ",".join(ids)' in source
    assert 'choices=("deepspeed_zero2", "ddp")' in source
    assert '["--ddp_find_unused_parameters", "True"]' in source
    assert '"distributed_strategy": args.distributed_strategy' in source
    assert '"--nccl-p2p-disable"' in source
    assert 'environment["NCCL_P2P_DISABLE"] = "1"' in source
    assert '"nccl_p2p_disable": args.nccl_p2p_disable' in source
    assert '"--pvld-use-spatial-memory"' in source
    assert '"--pvld-shared-gradient-scale"' in source
    assert '"--pvld-record-gradient-scale"' in source
    assert '"--pvld-predicted-layout-routing"' in source
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
    assert 'adapter.visual_routing,' in train_source
    assert 'adapter.residual_gate.requires_grad_(False)' in train_source
    assert 'configure_ddp_frozen_parameter_ignores(model, training_args)' in train_source
    assert '_set_params_and_buffers_to_ignore_for_model' in train_source
    assert '"first_training_step_complete"' in train_source
    smoke_source = (
        Path(__file__).resolve().parents[1]
        / "tools" / "training" / "smoke_pvld_causal_decoder_cuda.py"
    ).read_text(encoding="utf-8")
    assert '"count_head_through_boundary_logits"' in smoke_source
    assert '"zero_strength_legacy_exact": True' in smoke_source
