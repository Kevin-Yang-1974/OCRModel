#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

TRAINING_TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TRAINING_TOOLS))
from c4_selection_contract import load_c4_selection

try:
    import fcntl
except ImportError:  # pragma: no cover - the launcher itself runs on Linux.
    fcntl = None  # type: ignore[assignment]


EXIT_FAILURE = 1
EXIT_USAGE = 64
EXIT_MISSING = 66
EXIT_LOCKED = 73
EXIT_EXISTS = 74
EXIT_GPU_BUSY = 75
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

OVERFIT_P1_STEPS = 1000
OVERFIT_RECORDS = 2
ABLATION_IDS = (
    "got2_zero_shot", "projector_only", "generic_adapter_projector",
    "vlqa_ocr_only", "vlqa_layout_direct", "vlqa_layout_p1_p2",
)
LAYOUT_LOSS_PRESETS = {
    "layout_none": (0.0, 0.0, 0.0, 0.0, 0.0),
    "object_only": (1.0, 0.0, 0.0, 0.0, 1.0),
    "object_bbox": (1.0, 5.0, 2.0, 0.0, 1.0),
    "object_direction_order": (1.0, 0.0, 0.0, 1.0, 1.0),
    "layout_full": (1.0, 5.0, 2.0, 1.0, 1.0),
}
OVERFIT_THRESHOLDS = {
    "object_loss_max": 0.05,
    "bbox_l1_loss_max": 0.05,
    "bbox_giou_loss_max": 0.15,
    "direction_loss_max": 0.05,
    "object_accuracy_min": 0.99,
    "bbox_mean_iou_min": 0.90,
    "direction_accuracy_min": 0.99,
}


class RunFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        exit_code: int = EXIT_FAILURE,
        log_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.exit_code = exit_code
        self.log_path = log_path


@dataclass(frozen=True)
class Settings:
    workspace: Path
    ocrmodel_root: Path
    project_root: Path
    dataset_root: Path
    train_manifest: Path
    audit_manifests: tuple[Path, ...]
    validation_manifest: Path | None
    test_manifest: Path | None
    validation_image_root: Path | None
    test_image_root: Path | None
    replay_manifest: Path | None
    replay_image_root: Path | None
    replay_split: str
    replay_max_records: int
    primary_per_replay: int
    source_model: Path
    tokenizer_model: Path
    runs_root: Path
    run_id: str
    gpu_id: str
    physical_gpu_ids: tuple[str, ...]
    mode: str
    ablation_id: str
    layout_loss_preset: str
    stages: tuple[str, ...]
    layout_split: str
    validation_split: str
    test_split: str
    validation_model_kind: str
    validation_required_stage: str | None
    seed: int
    max_regions: int
    model_max_length: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    p1_max_steps: int
    p2_max_steps: int
    p1_max_records: int
    p2_max_records: int
    p1_learning_rate: float
    p2_learning_rate: float
    p2_train_scope: str
    checkpoint_steps: int
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    object_loss_weight: float
    bbox_l1_loss_weight: float
    bbox_giou_loss_weight: float
    direction_loss_weight: float
    layout_loss_weight: float
    p2_ocr_loss_weight: float
    validation_max_records: int
    validation_max_new_tokens: int
    validation_no_repeat_ngram_size: int
    validation_object_threshold: float
    validation_iou_threshold: float
    skip_post_training_validation: bool
    skip_source_hash: bool
    c4_selection: Path | None
    c4_selected_step: int | None
    c4_validation_page_cer: float | None
    c4_validation_whitespace_page_cer: float | None
    c4_config_sha256: str | None
    c4_weights_sha256: str | None
    c4_run_root: Path | None

    @property
    def run_root(self) -> Path:
        return self.runs_root / self.run_id


@dataclass
class RunContext:
    settings: Settings
    status: dict[str, Any]
    summary: dict[str, Any]
    latest_log: Path | None = None

    @property
    def metadata_dir(self) -> Path:
        return self.settings.run_root / "metadata"

    def update_status(self, **updates: Any) -> None:
        self.status.update(updates)
        self.status["updated_at"] = timestamp()
        write_status(self.metadata_dir / "status.txt", self.status)

    def write_summary(self) -> None:
        write_json(self.settings.run_root / "summary.json", self.summary)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def compact_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def emit(event: str, **payload: Any) -> None:
    print(compact_json({"event": event, **payload}), flush=True)


def bounded(value: str, limit: int = 600) -> str:
    value = value.replace("\x00", "").strip()
    return value if len(value) <= limit else value[:limit] + "...[truncated]"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [0, 1]")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_root = Path(__file__).resolve().parents[2]
    ocrmodel_root = Path(os.environ.get("OCRMODEL_ROOT", script_root))
    workspace = Path(os.environ.get("OCR_WORKSPACE", ocrmodel_root.parent))
    project_root = Path(
        os.environ.get("GOT_PROJECT_ROOT", ocrmodel_root / "src" / "GOT-OCR-2.0")
    )
    source_model = Path(
        os.environ.get("GOT_SOURCE_MODEL", workspace / "models" / "GOT-OCR2_0")
    )
    tokenizer_model = Path(
        os.environ.get("GOT_TOKENIZER_MODEL", source_model)
    )
    runs_root = Path(
        os.environ.get("GOT_TRAINING_RUNS", workspace / "training_runs" / "GOT")
    )

    parser = argparse.ArgumentParser(
        description=(
            "Run bounded A100 preflight plus controlled GOT2 VLQA diagnostics/training. "
            "Trainer output is kept in the run directory."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Training manifest; defaults to DATASET_ROOT/manifest.jsonl.",
    )
    parser.add_argument(
        "--audit-manifest",
        action="append",
        type=Path,
        default=[],
        help="Additional validation/test manifest to include in leakage audit; repeat as needed.",
    )
    parser.add_argument("--source-model", type=Path, default=source_model)
    parser.add_argument("--tokenizer-model", type=Path, default=tokenizer_model)
    parser.add_argument("--runs-root", type=Path, default=runs_root)
    parser.add_argument("--workspace", type=Path, default=workspace)
    parser.add_argument("--ocrmodel-root", type=Path, default=ocrmodel_root)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--run-id")
    parser.add_argument(
        "--gpu-id",
        help="One physical GPU id (backward-compatible shorthand for --gpu-ids).",
    )
    parser.add_argument(
        "--gpu-ids",
        help=(
            "Comma-separated physical GPU ids used by one distributed DeepSpeed run, "
            "for example 0,2,3."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=(
            "smoke",
            "overfit",
            "validate",
            "pilot",
            "pretrain",
            "joint-train",
            "adapt",
            "ablation",
        ),
        default="smoke",
    )
    parser.add_argument(
        "--allow-unvalidated-pilot",
        action="store_true",
        help="Required for pilot because pilot does not run validation automatically.",
    )
    parser.add_argument("--ablation", choices=ABLATION_IDS)
    parser.add_argument("--layout-loss-preset", choices=tuple(LAYOUT_LOSS_PRESETS))
    parser.add_argument("--layout-split", default="train")
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        help="Held-out validation manifest required by formal pretrain/joint-train modes.",
    )
    parser.add_argument(
        "--test-manifest",
        type=Path,
        help="Held-out test manifest required by formal pretrain/joint-train modes and audit-only.",
    )
    parser.add_argument("--validation-image-root", type=Path)
    parser.add_argument("--test-image-root", type=Path)
    parser.add_argument("--replay-manifest", type=Path)
    parser.add_argument("--replay-image-root", type=Path)
    parser.add_argument("--replay-split", default="train")
    parser.add_argument("--replay-max-records", type=nonnegative_int, default=0)
    parser.add_argument("--primary-per-replay", type=positive_int, default=3)
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--test-split", default="test")
    parser.add_argument(
        "--validation-model-kind",
        choices=("baseline", "generic", "vlqa"),
        default="vlqa",
    )
    parser.add_argument(
        "--validation-required-stage",
        choices=("p1", "p2"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-regions", type=positive_int, default=16)
    parser.add_argument("--model-max-length", type=positive_int, default=2048)
    parser.add_argument("--per-device-batch-size", type=positive_int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=1)
    parser.add_argument("--p1-max-steps", type=positive_int)
    parser.add_argument("--p2-max-steps", type=positive_int)
    parser.add_argument("--p1-max-records", type=nonnegative_int)
    parser.add_argument("--p2-max-records", type=nonnegative_int)
    parser.add_argument("--p1-learning-rate", type=positive_float, default=1e-4)
    parser.add_argument("--p2-learning-rate", type=positive_float, default=5e-5)
    parser.add_argument(
        "--p2-train-scope",
        choices=("adapter_projector", "decoder_adapter_projector"),
        default="adapter_projector",
    )
    parser.add_argument("--checkpoint-steps", type=nonnegative_int, default=0)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("constant", "cosine"),
        default="constant",
    )
    parser.add_argument("--warmup-ratio", type=probability, default=0.0)
    parser.add_argument("--weight-decay", type=nonnegative_float, default=0.0)
    parser.add_argument("--object-loss-weight", type=nonnegative_float, default=1.0)
    parser.add_argument("--bbox-l1-loss-weight", type=nonnegative_float, default=5.0)
    parser.add_argument("--bbox-giou-loss-weight", type=nonnegative_float, default=2.0)
    parser.add_argument("--direction-loss-weight", type=nonnegative_float, default=1.0)
    parser.add_argument("--layout-loss-weight", type=nonnegative_float, default=1.0)
    parser.add_argument("--p2-ocr-loss-weight", type=positive_float, default=1.0)
    parser.add_argument("--validation-max-records", type=nonnegative_int, default=0)
    parser.add_argument("--validation-max-new-tokens", type=positive_int, default=2048)
    parser.add_argument("--validation-no-repeat-ngram-size", type=nonnegative_int, default=20)
    parser.add_argument("--validation-object-threshold", type=probability, default=0.5)
    parser.add_argument("--validation-iou-threshold", type=probability, default=0.5)
    parser.add_argument(
        "--skip-post-training-validation",
        action="store_true",
        help=(
            "For pretrain/joint-train/adapt modes, skip the automatic validation "
            "run after training. Formal validation manifests are still used for audit."
        ),
    )
    parser.add_argument(
        "--skip-source-hash",
        action="store_true",
        help="Skip hashing the large original model file; path and size are still recorded.",
    )
    parser.add_argument(
        "--c4-selection",
        type=Path,
        help=(
            "Formal validation-only C4 selection used as the immutable C5/C6 "
            "branch point. Its selected model must equal --source-model."
        ),
    )
    return parser.parse_args(argv)


def resolve_settings(args: argparse.Namespace) -> Settings:
    if args.gpu_id is not None and args.gpu_ids is not None:
        raise RunFailure(
            "--gpu-id and --gpu-ids are mutually exclusive.", exit_code=EXIT_USAGE
        )
    configured_gpu_ids = (
        args.gpu_ids
        if args.gpu_ids is not None
        else args.gpu_id
        if args.gpu_id is not None
        else os.environ.get("GOT_PHYSICAL_GPUS")
        or os.environ.get("GOT_PHYSICAL_GPU", "0")
    )
    gpu_id = str(configured_gpu_ids).strip()
    if args.gpu_id is not None and not re.fullmatch(r"[0-9]+", gpu_id):
        raise RunFailure(
            "--gpu-id must be one physical numeric GPU id, for example 0 or 3.",
            exit_code=EXIT_USAGE,
        )
    physical_gpu_ids = tuple(part.strip() for part in gpu_id.split(","))
    if (
        not physical_gpu_ids
        or any(not re.fullmatch(r"[0-9]+", part) for part in physical_gpu_ids)
        or len(set(physical_gpu_ids)) != len(physical_gpu_ids)
    ):
        raise RunFailure(
            "--gpu-ids must be unique comma-separated physical numeric GPU ids, "
            "for example 0,2,3.",
            exit_code=EXIT_USAGE,
        )
    gpu_id = ",".join(physical_gpu_ids)
    if not args.layout_split.strip():
        raise RunFailure("--layout-split must be non-empty.", exit_code=EXIT_USAGE)
    if not args.validation_split.strip() or not args.test_split.strip():
        raise RunFailure(
            "--validation-split and --test-split must be non-empty.",
            exit_code=EXIT_USAGE,
        )
    if args.validation_model_kind == "baseline" and args.validation_required_stage:
        raise RunFailure(
            "--validation-required-stage is only valid for VLQA validation.",
            exit_code=EXIT_USAGE,
        )
    if args.replay_image_root is not None and args.replay_manifest is None:
        raise RunFailure(
            "--replay-image-root requires --replay-manifest.",
            exit_code=EXIT_USAGE,
        )
    ablation_id = args.ablation or ""
    if args.mode == "ablation" and not ablation_id:
        raise RunFailure("--mode ablation requires --ablation.", exit_code=EXIT_USAGE)
    if args.mode != "ablation" and ablation_id:
        raise RunFailure("--ablation requires --mode ablation.", exit_code=EXIT_USAGE)
    if args.mode == "ablation":
        if args.p2_train_scope != "adapter_projector":
            raise RunFailure(
                "A1-A5 require frozen Qwen and --p2-train-scope adapter_projector.",
                exit_code=EXIT_USAGE,
            )
        if args.validation_manifest is None or args.test_manifest is None:
            raise RunFailure(
                "Formal ablation requires validation and test manifests.", exit_code=EXIT_USAGE
            )
        if args.allow_unvalidated_pilot:
            raise RunFailure("Formal ablation cannot use pilot unlock.", exit_code=EXIT_USAGE)
        default_preset = (
            "layout_none" if ablation_id in {
                "got2_zero_shot", "projector_only", "generic_adapter_projector",
                "vlqa_ocr_only",
            } else "layout_full"
        )
        layout_loss_preset = args.layout_loss_preset or default_preset
        if ablation_id in {"got2_zero_shot", "projector_only", "generic_adapter_projector", "vlqa_ocr_only"} and layout_loss_preset != "layout_none":
            raise RunFailure(f"{ablation_id} requires layout_none.", exit_code=EXIT_USAGE)
        if ablation_id in {"vlqa_layout_direct", "vlqa_layout_p1_p2"} and layout_loss_preset == "layout_none":
            raise RunFailure(f"{ablation_id} requires an enabled layout loss preset.", exit_code=EXIT_USAGE)
        if ablation_id == "got2_zero_shot":
            if any(value is not None for value in (
                args.p1_max_steps, args.p2_max_steps, args.p1_max_records, args.p2_max_records
            )):
                raise RunFailure("A0 does not create a training optimizer.", exit_code=EXIT_USAGE)
            p1_steps = p2_steps = p1_records = p2_records = 0
            stages = ()
        elif ablation_id == "vlqa_layout_p1_p2":
            if args.p1_max_steps is None or args.p2_max_steps is None:
                raise RunFailure("A5 requires explicit P1 and P2 steps.", exit_code=EXIT_USAGE)
            p1_steps, p2_steps = args.p1_max_steps, args.p2_max_steps
            p1_records = 0 if args.p1_max_records is None else args.p1_max_records
            p2_records = 0 if args.p2_max_records is None else args.p2_max_records
            stages = ("p1", "p2")
        else:
            if args.p2_max_steps is None or args.p1_max_steps is not None or args.p1_max_records is not None:
                raise RunFailure(f"{ablation_id} requires direct P2 only.", exit_code=EXIT_USAGE)
            p1_steps = p1_records = 0
            p2_steps = args.p2_max_steps
            p2_records = 0 if args.p2_max_records is None else args.p2_max_records
            stages = ("p2",)
    elif args.mode == "validate":
        layout_loss_preset = ""
        supplied = {
            "--p1-max-steps": args.p1_max_steps,
            "--p2-max-steps": args.p2_max_steps,
            "--p1-max-records": args.p1_max_records,
            "--p2-max-records": args.p2_max_records,
        }
        invalid = {name: value for name, value in supplied.items() if value is not None}
        if invalid:
            raise RunFailure(
                f"Validate does not train; remove training overrides: {invalid}",
                exit_code=EXIT_USAGE,
            )
        if args.allow_unvalidated_pilot:
            raise RunFailure(
                "--allow-unvalidated-pilot is only valid with --mode pilot.",
                exit_code=EXIT_USAGE,
            )
        p1_steps = p2_steps = p1_records = p2_records = 0
        stages = ()
    elif args.mode == "smoke":
        layout_loss_preset = ""
        supplied = {
            "--p1-max-steps": args.p1_max_steps,
            "--p2-max-steps": args.p2_max_steps,
            "--p1-max-records": args.p1_max_records,
            "--p2-max-records": args.p2_max_records,
        }
        invalid = {name: value for name, value in supplied.items() if value not in (None, 1)}
        if invalid:
            raise RunFailure(
                f"Smoke is fixed to one step and one record per stage; use pilot: {invalid}",
                exit_code=EXIT_USAGE,
            )
        p1_steps = p2_steps = p1_records = p2_records = 1
        stages = ("p1", "p2")
    elif args.mode == "overfit":
        layout_loss_preset = ""
        supplied = {
            "--p1-max-steps": args.p1_max_steps,
            "--p2-max-steps": args.p2_max_steps,
            "--p1-max-records": args.p1_max_records,
            "--p2-max-records": args.p2_max_records,
        }
        invalid = {name: value for name, value in supplied.items() if value is not None}
        if invalid:
            raise RunFailure(
                f"Overfit is fixed to P1, {OVERFIT_RECORDS} records, and "
                f"{OVERFIT_P1_STEPS} steps; do not override: {invalid}",
                exit_code=EXIT_USAGE,
            )
        p1_steps = OVERFIT_P1_STEPS
        p1_records = OVERFIT_RECORDS
        p2_steps = p2_records = 0
        stages = ("p1",)
    elif args.mode == "pilot":
        layout_loss_preset = ""
        if not args.allow_unvalidated_pilot:
            raise RunFailure(
                "Pilot requires --allow-unvalidated-pilot because pilot does not run validation automatically.",
                exit_code=EXIT_USAGE,
            )
        if args.p1_max_steps is None or args.p2_max_steps is None:
            raise RunFailure(
                "Pilot requires explicit --p1-max-steps and --p2-max-steps.",
                exit_code=EXIT_USAGE,
            )
        p1_steps = args.p1_max_steps
        p2_steps = args.p2_max_steps
        p1_records = 0 if args.p1_max_records is None else args.p1_max_records
        p2_records = 0 if args.p2_max_records is None else args.p2_max_records
        stages = ("p1", "p2")
    elif args.mode == "pretrain":
        layout_loss_preset = ""
        if args.allow_unvalidated_pilot:
            raise RunFailure(
                "--allow-unvalidated-pilot is only valid with --mode pilot.",
                exit_code=EXIT_USAGE,
            )
        if args.p1_max_steps is None:
            raise RunFailure(
                "Formal pretrain requires explicit --p1-max-steps.",
                exit_code=EXIT_USAGE,
            )
        if args.p2_max_steps is not None or args.p2_max_records is not None:
            raise RunFailure(
                "Formal pretrain is P1-only; remove P2 overrides.",
                exit_code=EXIT_USAGE,
            )
        if args.validation_manifest is None or args.test_manifest is None:
            raise RunFailure(
                "Formal pretrain requires --validation-manifest and --test-manifest.",
                exit_code=EXIT_USAGE,
            )
        if args.validation_model_kind != "vlqa" or args.validation_required_stage not in (
            None,
            "p1",
        ):
            raise RunFailure(
                "Formal pretrain validation must use a P1 VLQA checkpoint.",
                exit_code=EXIT_USAGE,
            )
        p1_steps = args.p1_max_steps
        p1_records = 0 if args.p1_max_records is None else args.p1_max_records
        p2_steps = p2_records = 0
        stages = ("p1",)
    elif args.mode == "joint-train":
        layout_loss_preset = ""
        if args.allow_unvalidated_pilot:
            raise RunFailure(
                "--allow-unvalidated-pilot is only valid with --mode pilot.",
                exit_code=EXIT_USAGE,
            )
        if args.p2_max_steps is None:
            raise RunFailure(
                "Formal joint-train requires explicit --p2-max-steps.",
                exit_code=EXIT_USAGE,
            )
        if args.p1_max_steps is not None or args.p1_max_records is not None:
            raise RunFailure(
                "Formal joint-train starts from a P1 checkpoint and is P2-only; "
                "remove P1 overrides.",
                exit_code=EXIT_USAGE,
            )
        if args.validation_manifest is None or args.test_manifest is None:
            raise RunFailure(
                "Formal joint-train requires --validation-manifest and --test-manifest.",
                exit_code=EXIT_USAGE,
            )
        if args.validation_model_kind != "vlqa" or args.validation_required_stage not in (
            None,
            "p2",
        ):
            raise RunFailure(
                "Formal joint-train validation must use a P2 VLQA checkpoint.",
                exit_code=EXIT_USAGE,
            )
        p1_steps = p1_records = 0
        p2_steps = args.p2_max_steps
        p2_records = 0 if args.p2_max_records is None else args.p2_max_records
        stages = ("p2",)
    else:
        layout_loss_preset = ""
        if args.allow_unvalidated_pilot:
            raise RunFailure(
                "--allow-unvalidated-pilot is only valid with --mode pilot.",
                exit_code=EXIT_USAGE,
            )
        if args.p2_max_steps is None:
            raise RunFailure(
                "Formal adaptation requires explicit --p2-max-steps.",
                exit_code=EXIT_USAGE,
            )
        if args.p1_max_steps is not None or args.p1_max_records is not None:
            raise RunFailure(
                "Formal adaptation is P2-only; remove P1 overrides.",
                exit_code=EXIT_USAGE,
            )
        if args.validation_manifest is None or args.test_manifest is None:
            raise RunFailure(
                "Formal adaptation requires --validation-manifest and --test-manifest.",
                exit_code=EXIT_USAGE,
            )
        if args.validation_model_kind != "vlqa" or args.validation_required_stage not in (
            None,
            "p2",
        ):
            raise RunFailure(
                "Formal adaptation validation must use a P2 VLQA checkpoint.",
                exit_code=EXIT_USAGE,
            )
        p1_steps = p1_records = 0
        p2_steps = args.p2_max_steps
        p2_records = 0 if args.p2_max_records is None else args.p2_max_records
        stages = ("p2",)

    if args.mode in {"pretrain", "joint-train", "ablation"}:
        formal_splits = {
            args.layout_split.strip(),
            args.validation_split.strip(),
            args.test_split.strip(),
        }
        if len(formal_splits) != 3:
            raise RunFailure(
                "Formal train, validation, and test split names must be pairwise distinct.",
                exit_code=EXIT_USAGE,
            )

    run_id = args.run_id or (
        f"layout_{args.mode}_{datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')}"
    )
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise RunFailure(
            "--run-id may contain only letters, digits, period, underscore, and hyphen "
            "and must be at most 128 characters.",
            exit_code=EXIT_USAGE,
        )

    dataset_root = args.dataset_root.expanduser().resolve()
    train_manifest = (args.manifest or dataset_root / "manifest.jsonl").expanduser().resolve()
    validation_manifest = (
        args.validation_manifest.expanduser().resolve()
        if args.validation_manifest is not None
        else None
    )
    test_manifest = (
        args.test_manifest.expanduser().resolve()
        if args.test_manifest is not None
        else None
    )
    validation_image_root = (
        args.validation_image_root.expanduser().resolve()
        if args.validation_image_root is not None
        else None
    )
    test_image_root = (
        args.test_image_root.expanduser().resolve()
        if args.test_image_root is not None
        else None
    )
    replay_manifest = (
        args.replay_manifest.expanduser().resolve()
        if args.replay_manifest is not None
        else None
    )
    replay_image_root = (
        args.replay_image_root.expanduser().resolve()
        if args.replay_image_root is not None
        else None
    )
    audit_manifests: list[Path] = [train_manifest]
    for path in (validation_manifest, test_manifest):
        if path is not None and path not in audit_manifests:
            audit_manifests.append(path)
    for path in args.audit_manifest:
        resolved = path.expanduser().resolve()
        if resolved not in audit_manifests:
            audit_manifests.append(resolved)
    if replay_manifest is not None and replay_manifest not in audit_manifests:
        audit_manifests.append(replay_manifest)

    source_model = args.source_model.expanduser().resolve()
    validation_model_kind = args.validation_model_kind
    validation_required_stage = args.validation_required_stage
    if args.mode == "ablation":
        validation_model_kind = (
            "generic" if ablation_id == "generic_adapter_projector"
            else "vlqa" if ablation_id.startswith("vlqa_")
            else "baseline"
        )
        validation_required_stage = "p2" if validation_model_kind == "vlqa" else None
    resolved_loss_weights = (
        LAYOUT_LOSS_PRESETS[layout_loss_preset]
        if args.mode == "ablation" else (
            args.object_loss_weight, args.bbox_l1_loss_weight,
            args.bbox_giou_loss_weight, args.direction_loss_weight,
            args.layout_loss_weight,
        )
    )
    selection_path = (
        args.c4_selection.expanduser().resolve()
        if args.c4_selection is not None
        else None
    )
    selection = None
    if selection_path is not None:
        try:
            selection = load_c4_selection(selection_path)
        except Exception as exc:
            raise RunFailure(
                f"Invalid --c4-selection: {exc}", exit_code=EXIT_USAGE
            ) from exc
        selected_model = Path(selection["selected_model_path"]).resolve()
        if selected_model != source_model:
            raise RunFailure(
                "--source-model must exactly match the model selected by "
                f"--c4-selection: {source_model} != {selected_model}",
                exit_code=EXIT_USAGE,
            )
        if args.mode != "adapt":
            raise RunFailure(
                "--c4-selection is only valid for formal adapt branches.",
                exit_code=EXIT_USAGE,
            )

    return Settings(
        workspace=args.workspace.expanduser().resolve(),
        ocrmodel_root=args.ocrmodel_root.expanduser().resolve(),
        project_root=args.project_root.expanduser().resolve(),
        dataset_root=dataset_root,
        train_manifest=train_manifest,
        audit_manifests=tuple(audit_manifests),
        validation_manifest=validation_manifest,
        test_manifest=test_manifest,
        validation_image_root=validation_image_root,
        test_image_root=test_image_root,
        replay_manifest=replay_manifest,
        replay_image_root=replay_image_root,
        replay_split=args.replay_split.strip(),
        replay_max_records=args.replay_max_records,
        primary_per_replay=args.primary_per_replay,
        source_model=source_model,
        tokenizer_model=args.tokenizer_model.expanduser().resolve(),
        runs_root=args.runs_root.expanduser().resolve(),
        run_id=run_id,
        gpu_id=gpu_id,
        physical_gpu_ids=physical_gpu_ids,
        mode=args.mode,
        ablation_id=ablation_id,
        layout_loss_preset=layout_loss_preset,
        stages=stages,
        layout_split=args.layout_split.strip(),
        validation_split=args.validation_split.strip(),
        test_split=args.test_split.strip(),
        validation_model_kind=validation_model_kind,
        validation_required_stage=validation_required_stage,
        seed=args.seed,
        max_regions=args.max_regions,
        model_max_length=args.model_max_length,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        p1_max_steps=p1_steps,
        p2_max_steps=p2_steps,
        p1_max_records=p1_records,
        p2_max_records=p2_records,
        p1_learning_rate=args.p1_learning_rate,
        p2_learning_rate=args.p2_learning_rate,
        p2_train_scope=args.p2_train_scope,
        checkpoint_steps=args.checkpoint_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        object_loss_weight=resolved_loss_weights[0],
        bbox_l1_loss_weight=resolved_loss_weights[1],
        bbox_giou_loss_weight=resolved_loss_weights[2],
        direction_loss_weight=resolved_loss_weights[3],
        layout_loss_weight=resolved_loss_weights[4],
        p2_ocr_loss_weight=args.p2_ocr_loss_weight,
        validation_max_records=args.validation_max_records,
        validation_max_new_tokens=args.validation_max_new_tokens,
        validation_no_repeat_ngram_size=args.validation_no_repeat_ngram_size,
        validation_object_threshold=args.validation_object_threshold,
        validation_iou_threshold=args.validation_iou_threshold,
        skip_post_training_validation=args.skip_post_training_validation,
        skip_source_hash=args.skip_source_hash,
        c4_selection=selection_path,
        c4_selected_step=(selection["selected_step"] if selection else None),
        c4_validation_page_cer=(
            selection["validation_page_cer"] if selection else None
        ),
        c4_validation_whitespace_page_cer=(
            selection["validation_whitespace_page_cer"] if selection else None
        ),
        c4_config_sha256=(selection["config_sha256"] if selection else None),
        c4_weights_sha256=(selection["weights_sha256"] if selection else None),
        c4_run_root=(Path(selection["c4_run_root"]) if selection else None),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RunFailure(f"Expected a JSON object: {path}")
    return payload


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in payload.items():
        rendered = compact_json(value) if isinstance(value, (dict, list, tuple)) else str(value)
        lines.append(f"{key}={rendered.replace(chr(10), ' ')}")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tail_lines(path: Path | None, count: int = 20) -> list[str]:
    if path is None or not path.is_file():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [bounded(line, 800) for line in deque(handle, maxlen=count)]


def validate_paths(settings: Settings) -> None:
    required_files = (
        settings.source_model / "model.safetensors",
        settings.source_model / "config.json",
        settings.project_root / "scripts" / "train_GOT_layout.py",
        settings.project_root / "scripts" / "verify_layout_checkpoint.py",
        settings.project_root / "scripts" / "evaluate_GOT_layout.py",
        settings.project_root / "scripts" / "layout_validation_metrics.py",
        settings.project_root / "scripts" / "local_tokenizer.py",
        settings.project_root / "GOT" / "model" / "layout_query.py",
        settings.project_root / "GOT" / "model" / "GOT_ocr_2_0.py",
        settings.project_root / "zero_config" / "zero2.json",
        settings.ocrmodel_root / "tools" / "preprocessing" / "audit_synthetic_layout.py",
        settings.ocrmodel_root / "tools" / "environment" / "check_server_envs.py",
    )
    if not settings.dataset_root.is_dir():
        raise RunFailure(
            f"Dataset root does not exist: {settings.dataset_root}", exit_code=EXIT_MISSING
        )
    if settings.train_manifest.parent != settings.dataset_root:
        raise RunFailure(
            "The training manifest must be directly inside --dataset-root so relative image "
            "paths have one unambiguous root.",
            exit_code=EXIT_USAGE,
        )
    if settings.mode in {"validate", "pretrain", "joint-train", "ablation"} and not settings.tokenizer_model.is_dir():
        raise RunFailure(
            f"Validation tokenizer directory does not exist: {settings.tokenizer_model}",
            exit_code=EXIT_MISSING,
        )
    for manifest, image_root, label in (
        (settings.validation_manifest, settings.validation_image_root, "validation"),
        (settings.test_manifest, settings.test_image_root, "test"),
        (settings.replay_manifest, settings.replay_image_root, "replay"),
    ):
        if manifest is None:
            continue
        resolved_root = image_root or manifest.parent
        if not resolved_root.is_dir():
            raise RunFailure(
                f"{label.title()} image root does not exist: {resolved_root}",
                exit_code=EXIT_MISSING,
            )
    missing = [str(path) for path in required_files if not path.is_file()]
    missing.extend(str(path) for path in settings.audit_manifests if not path.is_file())
    if missing:
        raise RunFailure(f"Required files are missing: {missing}", exit_code=EXIT_MISSING)
    try:
        source_config = read_json(settings.source_model / "config.json")
    except Exception as exc:
        raise RunFailure(
            f"Source model config is invalid: {settings.source_model / 'config.json'}",
            exit_code=EXIT_MISSING,
        ) from exc
    source_has_vlqa = source_config.get("use_vlqa") is True
    source_has_generic = source_config.get("use_generic_adapter") is True
    if settings.mode == "ablation" and (source_has_vlqa or source_has_generic):
        raise RunFailure(
            "A0-A5 formal launcher must receive the original GOT2 checkpoint; "
            "A5 P1-to-P2 chaining is internal to the same run.",
            exit_code=EXIT_USAGE,
        )
    if settings.mode in {"smoke", "overfit", "pilot", "pretrain"} and source_has_vlqa:
        raise RunFailure(
            f"Mode {settings.mode!r} must start P1 from original GOT2 without VLQA.",
            exit_code=EXIT_USAGE,
        )
    if not settings.replay_split:
        raise RunFailure("--replay-split must be non-empty.", exit_code=EXIT_USAGE)
    if settings.primary_per_replay < 1:
        raise RunFailure("--primary-per-replay must be positive.", exit_code=EXIT_USAGE)
    if settings.mode in {"joint-train", "adapt"}:
        metrics_path = settings.source_model / "layout_training_metrics.json"
        if not source_has_vlqa:
            raise RunFailure(
                "Formal adaptation must start from a completed VLQA checkpoint.",
                exit_code=EXIT_USAGE,
            )
        required_stage = "p1" if settings.mode == "joint-train" else "p2"
        if settings.c4_selection is not None:
            if required_stage != "p2":
                raise RunFailure(
                    "A selected C4 checkpoint is only valid as a P2 adapt source.",
                    exit_code=EXIT_USAGE,
                )
        else:
            if not metrics_path.is_file():
                raise RunFailure(
                    "Formal adaptation source has no layout_training_metrics.json; "
                    "periodic checkpoints require --c4-selection and must not "
                    "silently fall back to the final model.",
                    exit_code=EXIT_USAGE,
                )
            source_metrics = read_json(metrics_path)
            if source_metrics.get("layout_stage") != required_stage:
                raise RunFailure(
                    f"Source checkpoint must report layout_stage={required_stage!r}.",
                    exit_code=EXIT_USAGE,
                )
    if settings.mode != "validate":
        deepspeed = Path(sys.executable).with_name("deepspeed")
        if not deepspeed.is_file() or not os.access(deepspeed, os.X_OK):
            raise RunFailure(
                f"DeepSpeed executable is missing beside the active Python: {deepspeed}",
                exit_code=EXIT_MISSING,
            )


def training_environment(settings: Settings) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "OCR_WORKSPACE": str(settings.workspace),
            "OCRMODEL_ROOT": str(settings.ocrmodel_root),
            "GOT_PROJECT_ROOT": str(settings.project_root),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            # DeepSpeed launches one process per visible physical GPU in this order.
            "CUDA_VISIBLE_DEVICES": settings.gpu_id,
            "PYTHONNOUSERSITE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_DISABLED": "true",
            "HF_HOME": str(settings.workspace / "cache" / "huggingface"),
            "HF_HUB_DISABLE_XET": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    return environment


def gpu_processes(gpu_id: str) -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "-i",
        gpu_id,
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RunFailure(f"Cannot query GPU {gpu_id}: {exc}", exit_code=EXIT_MISSING) from exc
    if completed.returncode != 0:
        raise RunFailure(
            f"nvidia-smi failed: {bounded(completed.stderr)}", exit_code=EXIT_MISSING
        )
    processes: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line or "no running processes" in line.lower():
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=2)]
        processes.append(
            {
                "pid": parts[0] if parts else "unknown",
                "process_name": parts[1] if len(parts) > 1 else "unknown",
                "used_memory_mib": parts[2] if len(parts) > 2 else "unknown",
            }
        )
    return processes


def require_gpu_free(settings: Settings) -> None:
    for gpu_id in settings.physical_gpu_ids:
        processes = gpu_processes(gpu_id)
        if processes:
            raise RunFailure(
                f"GPU{gpu_id}_BUSY {compact_json(processes)}",
                exit_code=EXIT_GPU_BUSY,
            )


def acquire_gpu_locks(runs_root: Path, physical_gpu_ids: Sequence[str]) -> list[Any]:
    if fcntl is None:
        raise RunFailure("The A100 launcher requires Linux file locking.", exit_code=EXIT_MISSING)
    runs_root.mkdir(parents=True, exist_ok=True)
    handles: list[Any] = []
    try:
        for gpu_id in sorted(physical_gpu_ids, key=int):
            handle = (runs_root / f".layout_a100.gpu-{gpu_id}.lock").open(
                "a+", encoding="utf-8"
            )
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                handle.close()
                raise RunFailure(
                    f"Another layout A100 launcher holds the GPU {gpu_id} lock.",
                    exit_code=EXIT_LOCKED,
                ) from exc
            handles.append(handle)
    except Exception:
        for handle in reversed(handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        raise
    return handles


def run_environment_check(context: RunContext) -> dict[str, Any]:
    output_path = context.metadata_dir / "environment_check.json"
    log_path = context.metadata_dir / "environment_check.log"
    context.latest_log = log_path
    command = [
        sys.executable,
        str(context.settings.ocrmodel_root / "tools" / "environment" / "check_server_envs.py"),
        "--workspace",
        str(context.settings.workspace),
        "--ocrmodel-root",
        str(context.settings.ocrmodel_root),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
            env=training_environment(context.settings),
        )
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        report = json.loads(lines[-1]) if lines else {}
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise RunFailure(
            f"Environment check could not be parsed: {exc}", log_path=log_path
        ) from exc
    write_json(output_path, report)
    if completed.returncode != 0 or report.get("ok") is not True:
        raise RunFailure(
            "GOT environment check failed; inspect metadata/environment_check.json.",
            exit_code=EXIT_MISSING,
            log_path=log_path,
        )
    return report


def record_provenance(context: RunContext) -> dict[str, Any]:
    settings = context.settings
    source_weights = settings.source_model / "model.safetensors"
    payload: dict[str, Any] = {
        "source_model": str(settings.source_model),
        "tokenizer_model": str(settings.tokenizer_model),
        "source_model_bytes": source_weights.stat().st_size,
        "source_model_sha256": None,
        "train_manifest": str(settings.train_manifest),
        "train_manifest_sha256": file_sha256(settings.train_manifest),
        "validation_manifest": (
            str(settings.validation_manifest)
            if settings.validation_manifest is not None
            else None
        ),
        "validation_manifest_sha256": (
            file_sha256(settings.validation_manifest)
            if settings.validation_manifest is not None
            else None
        ),
        "test_manifest": (
            str(settings.test_manifest)
            if settings.test_manifest is not None
            else None
        ),
        "test_manifest_sha256": (
            file_sha256(settings.test_manifest)
            if settings.test_manifest is not None
            else None
        ),
        "code": {},
    }
    if not settings.skip_source_hash:
        payload["source_model_sha256"] = file_sha256(source_weights)
    code_paths = (
        settings.project_root / "scripts" / "train_GOT_layout.py",
        settings.project_root / "scripts" / "layout_page_dataset.py",
        settings.project_root / "scripts" / "layout_validation_metrics.py",
        settings.project_root / "scripts" / "evaluate_GOT_layout.py",
        settings.project_root / "scripts" / "local_tokenizer.py",
        settings.project_root / "scripts" / "verify_layout_checkpoint.py",
        settings.project_root / "GOT" / "model" / "layout_query.py",
        settings.project_root / "GOT" / "model" / "GOT_ocr_2_0.py",
        Path(__file__).resolve(),
    )
    code_hashes: dict[str, str] = {}
    for path in code_paths:
        try:
            label = str(path.relative_to(settings.ocrmodel_root))
        except ValueError:
            label = str(path)
        code_hashes[label] = file_sha256(path)
    payload["code"] = code_hashes
    write_json(context.metadata_dir / "provenance.json", payload)
    return payload


def run_audit(context: RunContext) -> dict[str, Any]:
    summary_path = context.metadata_dir / "audit_summary.json"
    log_path = context.metadata_dir / "audit.log"
    command = [
        sys.executable,
        str(
            context.settings.ocrmodel_root
            / "tools"
            / "preprocessing"
            / "audit_synthetic_layout.py"
        ),
    ]
    for manifest in context.settings.audit_manifests:
        command.extend(("--manifest", str(manifest)))
    command.extend(("--summary-json", str(summary_path)))
    context.latest_log = log_path
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=context.settings.ocrmodel_root,
            env=training_environment(context.settings),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0 or not summary_path.is_file():
        raise RunFailure("Synthetic layout audit failed.", log_path=log_path)
    summary = read_json(summary_path)
    split_count = int(summary.get("split_counts", {}).get(context.settings.layout_split, 0))
    required_records = OVERFIT_RECORDS if context.settings.mode == "overfit" else 1
    if summary.get("status") != "ok" or split_count < required_records:
        raise RunFailure(
            f"Audit has {split_count} usable split={context.settings.layout_split!r} "
            f"records; mode={context.settings.mode!r} requires {required_records}.",
            log_path=log_path,
        )
    if context.settings.mode in {"pretrain", "joint-train", "adapt"}:
        split_counts = summary.get("split_counts", {})
        for split, label in (
            (context.settings.validation_split, "validation"),
            (context.settings.test_split, "test"),
        ):
            if int(split_counts.get(split, 0)) < 1:
                raise RunFailure(
                    f"Formal {label} split {split!r} is empty after audit.",
                    log_path=log_path,
                )
    return summary


# Run the CUDA probe in a child so its context disappears before the next GPU check.
COMPONENT_SMOKE = r'''
import importlib.util
import json
import sys

import torch

module_path = sys.argv[1]
seed = int(sys.argv[2])
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"Expected exactly one visible GPU, got {torch.cuda.device_count()}")
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
spec = importlib.util.spec_from_file_location("layout_query_component_smoke", module_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
device = torch.device("cuda:0")
adapter = module.VisualLayoutQueryAdapter(
    visual_dim=64,
    layout_input_dim=64,
    adapter_dim=32,
    num_queries=4,
    num_heads=4,
    ffn_expansion=2,
).to(device=device, dtype=torch.bfloat16)
visual = torch.randn(2, 16, 64, device=device, dtype=torch.bfloat16) * 1_000_000.0
output = adapter(visual, memory_grid_size=(4, 4), return_attention=True)
if not torch.equal(output.visual_tokens, visual):
    raise RuntimeError("Zero residual gate did not preserve visual tokens")
initial_scales = {
    "query_abs_max": float(output.layout_queries.detach().float().abs().max().cpu()),
    "prediction_query_abs_max": float(
        output.prediction_queries.detach().float().abs().max().cpu()
    ),
    "object_logit_abs_max": float(
        output.object_logits.detach().float().abs().max().cpu()
    ),
    "direction_logit_abs_max": float(
        output.direction_logits.detach().float().abs().max().cpu()
    ),
    "bbox_logit_abs_max": float(
        output.bbox_logits.detach().float().abs().max().cpu()
    ),
}
for name, value in initial_scales.items():
    if not torch.isfinite(torch.tensor(value)) or value >= 10.0:
        raise RuntimeError(f"VLQA initial {name} is out of bounds: {value}")
criterion = module.VisualLayoutQueryLoss()
boxes = torch.tensor(
    [
        [[0.1, 0.1, 0.3, 0.8], [0.4, 0.1, 0.6, 0.8], [0, 0, 0, 0], [0, 0, 0, 0]],
        [[0.2, 0.2, 0.7, 0.4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    ],
    dtype=torch.float32,
    device=device,
)
bbox_mask = torch.tensor(
    [[True, True, False, False], [True, False, False, False]], device=device
)
object_targets = bbox_mask.to(torch.float32)
object_mask = torch.ones((2, 4), dtype=torch.bool, device=device)
directions = torch.tensor([[2, 2, -100, -100], [0, -100, -100, -100]], device=device)
losses = criterion(
    output=output,
    bbox_targets_xyxy=boxes,
    bbox_mask=bbox_mask,
    object_targets=object_targets,
    object_mask=object_mask,
    direction_targets=directions,
)
if not torch.isfinite(losses.loss):
    raise RuntimeError("VLQA component loss is not finite")
if losses.loss.dtype != torch.float32:
    raise RuntimeError(f"VLQA loss must use FP32, got {losses.loss.dtype}")
losses.loss.backward()
gradient = adapter.query_embeddings.grad
if gradient is None or not torch.isfinite(gradient).all():
    raise RuntimeError("VLQA query gradient is missing or non-finite")
print(json.dumps({
    "status": "ok",
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
    "capability": list(torch.cuda.get_device_capability(0)),
    "loss": float(losses.loss.detach().cpu()),
    "loss_dtype": str(losses.loss.dtype),
    "object_loss": float(losses.object_loss.detach().cpu()),
    "bbox_l1_loss": float(losses.bbox_l1_loss.detach().cpu()),
    "bbox_giou_loss": float(losses.bbox_giou_loss.detach().cpu()),
    "direction_loss": float(losses.direction_loss.detach().cpu()),
    "object_accuracy": float(losses.object_accuracy.detach().cpu()),
    "bbox_mean_iou": float(losses.bbox_mean_iou.detach().cpu()),
    "direction_accuracy": float(losses.direction_accuracy.detach().cpu()),
    **initial_scales,
    "query_gradient_norm": float(gradient.norm().detach().cpu()),
    "visual_shape": list(output.visual_tokens.shape),
    "query_shape": list(output.layout_queries.shape),
}, separators=(",", ":")))
'''


def run_component_smoke(context: RunContext) -> dict[str, Any]:
    log_path = context.metadata_dir / "component_smoke.log"
    output_path = context.metadata_dir / "component_smoke.json"
    context.latest_log = log_path
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                COMPONENT_SMOKE,
                str(context.settings.project_root / "GOT" / "model" / "layout_query.py"),
                str(context.settings.seed),
            ],
            cwd=context.settings.project_root,
            env=training_environment(context.settings),
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        raise RunFailure("VLQA CUDA component smoke could not complete.", log_path=log_path) from exc
    log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RunFailure("VLQA CUDA component smoke failed.", log_path=log_path)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        payload = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise RunFailure("VLQA component smoke returned invalid JSON.", log_path=log_path) from exc
    if payload.get("status") != "ok":
        raise RunFailure("VLQA component smoke did not report success.", log_path=log_path)
    if payload.get("loss_dtype") != "torch.float32":
        raise RunFailure("VLQA component smoke did not use FP32 loss.", log_path=log_path)
    for name in (
        "object_logit_abs_max",
        "direction_logit_abs_max",
        "bbox_logit_abs_max",
    ):
        value = float(payload.get(name, float("nan")))
        if not math.isfinite(value) or value >= 10.0:
            raise RunFailure(
                f"VLQA component smoke reported invalid initial {name}: {value}.",
                log_path=log_path,
            )
    write_json(output_path, payload)
    return payload


def free_master_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def build_training_command(
    settings: Settings,
    *,
    stage: str,
    source_model: Path,
    output_dir: Path,
    master_port: int,
) -> list[str]:
    if stage not in {"p1", "p2"}:
        raise ValueError(stage)
    steps = settings.p1_max_steps if stage == "p1" else settings.p2_max_steps
    records = settings.p1_max_records if stage == "p1" else settings.p2_max_records
    learning_rate = (
        settings.p1_learning_rate if stage == "p1" else settings.p2_learning_rate
    )
    ocr_loss_weight = "0" if stage == "p1" else f"{settings.p2_ocr_loss_weight:g}"
    deepspeed = Path(sys.executable).with_name("deepspeed")
    command = [
        str(deepspeed),
        "--master_port",
        str(master_port),
        str(settings.project_root / "scripts" / "train_GOT_layout.py"),
        "--deepspeed",
        str(settings.project_root / "zero_config" / "zero2.json"),
        "--model_name_or_path",
        str(source_model),
        "--tokenizer_name_or_path",
        str(settings.tokenizer_model),
        "--layout_manifest",
        str(settings.train_manifest),
        "--layout_image_root",
        str(settings.dataset_root),
        "--layout_split",
        settings.layout_split,
        "--layout_stage",
        stage,
        "--p2_train_scope",
        settings.p2_train_scope,
        "--max_regions",
        str(settings.max_regions),
        "--max_train_records",
        str(records),
        "--datasets",
        "layout-page-jsonl",
        "--conversation_version",
        "mpt",
        "--use_im_start_end",
        "True",
        "--bf16",
        "True",
        "--fp16",
        "False",
        "--gradient_accumulation_steps",
        str(settings.gradient_accumulation_steps),
        "--optim",
        "adamw_torch",
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        "steps" if settings.checkpoint_steps else "no",
        "--save_safetensors",
        "True",
        "--weight_decay",
        str(settings.weight_decay),
        "--warmup_ratio",
        str(settings.warmup_ratio),
        "--lr_scheduler_type",
        settings.lr_scheduler_type,
        "--logging_steps",
        "1",
        "--tf32",
        "False",
        "--model_max_length",
        str(settings.model_max_length),
        "--gradient_checkpointing",
        "True",
        "--dataloader_num_workers",
        "0",
        "--report_to",
        "none",
        "--remove_unused_columns",
        "False",
        "--per_device_train_batch_size",
        str(settings.per_device_batch_size),
        "--max_steps",
        str(steps),
        "--learning_rate",
        str(learning_rate),
        "--object_loss_weight",
        f"{settings.object_loss_weight:g}",
        "--bbox_l1_loss_weight",
        f"{settings.bbox_l1_loss_weight:g}",
        "--bbox_giou_loss_weight",
        f"{settings.bbox_giou_loss_weight:g}",
        "--direction_loss_weight",
        f"{settings.direction_loss_weight:g}",
        "--layout_loss_weight",
        f"{settings.layout_loss_weight:g}",
        "--ocr_loss_weight",
        ocr_loss_weight,
        "--seed",
        str(settings.seed),
        "--data_seed",
        str(settings.seed),
        "--output_dir",
        str(output_dir),
    ]
    if settings.ablation_id:
        command.extend(
            [
                "--ablation_id", settings.ablation_id,
                "--layout_loss_preset", settings.layout_loss_preset,
            ]
        )
    if settings.checkpoint_steps:
        command.extend(["--save_steps", str(settings.checkpoint_steps)])
    if settings.replay_manifest is not None:
        command.extend(
            [
                "--replay_layout_manifest",
                str(settings.replay_manifest),
                "--replay_layout_image_root",
                str(settings.replay_image_root or settings.replay_manifest.parent),
                "--replay_layout_split",
                settings.replay_split,
                "--replay_max_train_records",
                str(settings.replay_max_records),
                "--primary_per_replay",
                str(settings.primary_per_replay),
            ]
        )
    return command


def validate_stage_metrics(
    metrics: dict[str, Any],
    *,
    stage: str,
    expected_steps: int,
    max_regions: int,
    ablation_id: str = "",
) -> None:
    if metrics.get("layout_stage") != stage:
        raise RunFailure(f"{stage.upper()} metrics contain the wrong layout_stage.")
    if int(metrics.get("global_step", -1)) != expected_steps:
        raise RunFailure(
            f"{stage.upper()} global_step mismatch: "
            f"{metrics.get('global_step')} != {expected_steps}."
        )
    train_loss = float(metrics.get("train_loss", float("nan")))
    if not math.isfinite(train_loss) or train_loss <= 0.0:
        raise RunFailure(f"{stage.upper()} train_loss is invalid: {train_loss}.")
    uses_vlqa = not ablation_id or ablation_id.startswith("vlqa_")
    if uses_vlqa and metrics.get("first_batch_bbox_shape") != [1, max_regions, 4]:
        raise RunFailure(
            f"{stage.upper()} bbox batch shape is invalid: "
            f"{metrics.get('first_batch_bbox_shape')}."
        )
    supervised = int(metrics.get("first_batch_supervised_tokens", -1))
    if stage == "p1" and supervised != 0:
        raise RunFailure(f"P1 retained {supervised} OCR-supervised batch tokens.")
    if stage == "p2" and supervised < 1:
        raise RunFailure("P2 has no OCR-supervised batch tokens.")
    if uses_vlqa and metrics.get("layout_loss_compute_dtype") != "float32":
        raise RunFailure(
            f"{stage.upper()} did not report FP32 layout loss computation."
        )
    initialization = metrics.get("layout_adapter_initialization")
    try:
        source_layout_count = int(metrics.get("source_layout_tensor_count", -1))
        expected_layout_count = int(metrics.get("expected_layout_tensor_count", -1))
        parameter_abs_max = float(
            metrics.get("layout_adapter_parameter_abs_max", float("nan"))
        )
    except (TypeError, ValueError) as exc:
        raise RunFailure(f"{stage.upper()} reported invalid VLQA initialization metadata.") from exc
    if not uses_vlqa:
        if expected_layout_count != 0 or initialization != "disabled":
            raise RunFailure(f"{ablation_id} unexpectedly enabled VLQA.")
        if metrics.get("ablation_id") != ablation_id:
            raise RunFailure("Training metrics contain the wrong ablation_id.")
        return
    if expected_layout_count < 1:
        raise RunFailure(f"{stage.upper()} reported no expected VLQA tensors.")
    if not math.isfinite(parameter_abs_max):
        raise RunFailure(f"{stage.upper()} reported non-finite VLQA parameters.")
    if stage == "p1" and (
        initialization != "fresh_explicit_reset" or source_layout_count != 0
    ):
        raise RunFailure(
            "P1 did not explicitly initialize a fresh VLQA from the original checkpoint."
        )
    if stage == "p1" and parameter_abs_max > 2.0:
        raise RunFailure(
            f"P1 fresh VLQA parameter scale is invalid: abs_max={parameter_abs_max}."
        )
    if stage == "p2":
        expected_initialization = (
            "checkpoint_loaded" if ablation_id in {"", "vlqa_layout_p1_p2"}
            else "fresh_explicit_reset"
        )
        if initialization != expected_initialization:
            raise RunFailure(
                f"{ablation_id or 'legacy'} P2 initialization mismatch: {initialization}."
            )
        if expected_initialization == "checkpoint_loaded" and source_layout_count != expected_layout_count:
            raise RunFailure("P2 did not load a complete VLQA checkpoint from P1.")
        if expected_initialization == "fresh_explicit_reset" and source_layout_count != 0:
            raise RunFailure("Direct P2 unexpectedly loaded P1 VLQA tensors.")
    diagnostics = metrics.get("diagnostics")
    if not isinstance(diagnostics, dict) or int(diagnostics.get("log_count", 0)) < 1:
        raise RunFailure(f"{stage.upper()} did not record layout diagnostics.")
    tail_mean = diagnostics.get("tail_mean")
    if not isinstance(tail_mean, dict):
        raise RunFailure(f"{stage.upper()} diagnostics have no tail mean.")
    required = (
        "layout_loss",
        "object_loss",
        "bbox_l1_loss",
        "bbox_giou_loss",
        "direction_loss",
        "object_accuracy",
        "bbox_mean_iou",
        "direction_accuracy",
        "query_abs_max",
        "prediction_query_abs_max",
        "bbox_logit_abs_max",
    )
    for name in required:
        value = float(tail_mean.get(name, float("nan")))
        if not math.isfinite(value):
            raise RunFailure(
                f"{stage.upper()} diagnostic {name} is invalid: {value}."
            )


def assess_p1_overfit(metrics: dict[str, Any]) -> dict[str, Any]:
    diagnostics = metrics.get("diagnostics")
    tail_mean = diagnostics.get("tail_mean", {}) if isinstance(diagnostics, dict) else {}
    first = diagnostics.get("first", {}) if isinstance(diagnostics, dict) else {}
    last = diagnostics.get("last", {}) if isinstance(diagnostics, dict) else {}
    tail_min = diagnostics.get("tail_min", {}) if isinstance(diagnostics, dict) else {}
    tail_max = diagnostics.get("tail_max", {}) if isinstance(diagnostics, dict) else {}
    observed_names = (
        "layout_loss",
        "object_loss",
        "bbox_l1_loss",
        "bbox_giou_loss",
        "direction_loss",
        "object_accuracy",
        "bbox_mean_iou",
        "direction_accuracy",
        "object_logit_abs_max",
        "direction_logit_abs_max",
        "bbox_pred_min",
        "bbox_pred_max",
        "query_abs_max",
        "prediction_query_abs_max",
        "bbox_logit_abs_max",
        "query_gradient_norm",
        "residual_gate",
    )
    def select_observed(record: dict[str, Any]) -> dict[str, float]:
        return {
            name: float(record[name])
            for name in observed_names
            if isinstance(record.get(name), (int, float))
        }

    observed = select_observed(tail_mean)
    observed_first = select_observed(first)
    observed_last = select_observed(last)
    bbox_tail_range = {
        "bbox_l1_loss_min": tail_min.get("bbox_l1_loss"),
        "bbox_l1_loss_max": tail_max.get("bbox_l1_loss"),
        "bbox_giou_loss_min": tail_min.get("bbox_giou_loss"),
        "bbox_giou_loss_max": tail_max.get("bbox_giou_loss"),
        "bbox_mean_iou_min": tail_min.get("bbox_mean_iou"),
        "bbox_mean_iou_max": tail_max.get("bbox_mean_iou"),
    }
    criteria = {
        "exactly_two_records": int(metrics.get("dataset_examples", -1))
        == OVERFIT_RECORDS,
        "enough_diagnostic_steps": int(
            diagnostics.get("log_count", 0) if isinstance(diagnostics, dict) else 0
        )
        >= OVERFIT_P1_STEPS,
        "object_loss": observed.get("object_loss", math.inf)
        <= OVERFIT_THRESHOLDS["object_loss_max"],
        "bbox_l1_loss": observed.get("bbox_l1_loss", math.inf)
        <= OVERFIT_THRESHOLDS["bbox_l1_loss_max"],
        "bbox_giou_loss": observed.get("bbox_giou_loss", math.inf)
        <= OVERFIT_THRESHOLDS["bbox_giou_loss_max"],
        "direction_loss": observed.get("direction_loss", math.inf)
        <= OVERFIT_THRESHOLDS["direction_loss_max"],
        "object_accuracy": observed.get("object_accuracy", -math.inf)
        >= OVERFIT_THRESHOLDS["object_accuracy_min"],
        "bbox_mean_iou": observed.get("bbox_mean_iou", -math.inf)
        >= OVERFIT_THRESHOLDS["bbox_mean_iou_min"],
        "direction_accuracy": observed.get("direction_accuracy", -math.inf)
        >= OVERFIT_THRESHOLDS["direction_accuracy_min"],
    }
    return {
        "status": "pass" if all(criteria.values()) else "fail",
        "purpose": "two-record P1 implementation diagnostic; not validation performance",
        "tail_window": (
            int(diagnostics.get("tail_window", 0))
            if isinstance(diagnostics, dict)
            else 0
        ),
        "thresholds": OVERFIT_THRESHOLDS,
        "initialization": {
            "mode": metrics.get("layout_adapter_initialization"),
            "source_layout_tensor_count": metrics.get(
                "source_layout_tensor_count"
            ),
            "expected_layout_tensor_count": metrics.get(
                "expected_layout_tensor_count"
            ),
            "parameter_abs_max": metrics.get(
                "layout_adapter_parameter_abs_max"
            ),
        },
        "observed_first": observed_first,
        "observed_last": observed_last,
        "observed_tail_mean": observed,
        "bbox_tail_range": bbox_tail_range,
        "criteria": criteria,
    }


def run_stage(
    context: RunContext,
    *,
    stage: str,
    source_model: Path,
) -> dict[str, Any]:
    settings = context.settings
    stage_root = settings.run_root / stage
    metadata_dir = stage_root / "metadata"
    output_dir = stage_root / "model"
    log_path = stage_root / "train.log"
    metadata_dir.mkdir(parents=True, exist_ok=False)
    status_path = metadata_dir / "status.txt"
    steps = settings.p1_max_steps if stage == "p1" else settings.p2_max_steps
    records = settings.p1_max_records if stage == "p1" else settings.p2_max_records
    learning_rate = (
        settings.p1_learning_rate if stage == "p1" else settings.p2_learning_rate
    )
    status = {
        "status": "running",
        "stage": stage,
        "started_at": timestamp(),
        "source_model": str(source_model),
        "output_model": str(output_dir),
        "physical_gpus": list(settings.physical_gpu_ids),
        "world_size": len(settings.physical_gpu_ids),
        "max_steps": steps,
        "max_train_records": records,
        "learning_rate": learning_rate,
    }
    write_status(status_path, status)
    command = build_training_command(
        settings,
        stage=stage,
        source_model=source_model,
        output_dir=output_dir,
        master_port=free_master_port(),
    )
    write_json(metadata_dir / "command.json", command)
    context.latest_log = log_path
    context.update_status(status=f"running_{stage}", active_stage=stage)
    emit(
        "layout_stage_started",
        stage=stage,
        max_steps=steps,
        max_records=records,
        log=str(log_path),
    )
    require_gpu_free(settings)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=settings.project_root,
            env=training_environment(settings),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    status["finished_at"] = timestamp()
    status["exit_code"] = completed.returncode
    if completed.returncode != 0:
        status["status"] = "failed"
        write_status(status_path, status)
        raise RunFailure(
            f"{stage.upper()} training failed with exit code {completed.returncode}.",
            log_path=log_path,
        )

    metrics_path = output_dir / "layout_training_metrics.json"
    if not metrics_path.is_file():
        status["status"] = "failed"
        write_status(status_path, status)
        raise RunFailure(
            f"{stage.upper()} did not write layout_training_metrics.json.",
            log_path=log_path,
        )
    metrics = read_json(metrics_path)
    try:
        validate_stage_metrics(
            metrics,
            stage=stage,
            expected_steps=steps,
            max_regions=settings.max_regions,
            ablation_id=settings.ablation_id,
        )
    except RunFailure as exc:
        status["status"] = "metrics_validation_failed"
        status["metrics_error"] = bounded(str(exc))
        write_status(status_path, status)
        raise RunFailure(str(exc), log_path=log_path) from exc
    if stage == "p2" and settings.c4_selection is not None:
        branch_initialization = {
            "parent_label": "c4_vlqa_ocr_only",
            "selection_path": str(settings.c4_selection),
            "selected_c4_step": settings.c4_selected_step,
            "selected_c4_model_path": str(settings.source_model),
            "selected_c4_validation_page_cer": settings.c4_validation_page_cer,
            "selected_c4_validation_whitespace_page_cer": (
                settings.c4_validation_whitespace_page_cer
            ),
            "selected_c4_config_sha256": settings.c4_config_sha256,
            "selected_c4_weights_sha256": settings.c4_weights_sha256,
            "c4_run_root": str(settings.c4_run_root),
            "optimizer_state_initialization": "fresh",
            "scheduler_state_initialization": "fresh",
        }
        metrics["branch_initialization"] = branch_initialization
        metrics["selected_c4_step"] = settings.c4_selected_step
        metrics["selected_c4_model_path"] = str(settings.source_model)
        metrics["c4_selection"] = str(settings.c4_selection)
        metrics["c4_validation_page_cer"] = settings.c4_validation_page_cer
        metrics["initial_checkpoint_sha256"] = settings.c4_weights_sha256
        write_json(metrics_path, metrics)
    status.update(
        {
            "status": "trained",
            "global_step": metrics["global_step"],
            "train_loss": metrics["train_loss"],
            "peak_allocated_mib": metrics.get("peak_allocated_mib"),
        }
    )
    write_status(status_path, status)
    return {
        "status": "trained",
        "stage": stage,
        "model": str(output_dir),
        "train_log": str(log_path),
        "metrics": metrics,
    }


def verify_stage(
    context: RunContext,
    *,
    stage: str,
    model_dir: Path,
    skip_model_reload: bool,
) -> dict[str, Any]:
    stage_root = context.settings.run_root / stage
    output_path = stage_root / "metadata" / "checkpoint_verification.json"
    log_path = stage_root / "metadata" / "checkpoint_verification.log"
    command = [
        sys.executable,
        str(context.settings.project_root / "scripts" / "verify_layout_checkpoint.py"),
        "--model",
        str(model_dir),
        "--stage",
        stage,
        "--max-regions",
        str(context.settings.max_regions),
        "--output",
        str(output_path),
    ]
    if context.settings.ablation_id:
        command.extend(("--ablation-id", context.settings.ablation_id))
    if skip_model_reload:
        command.append("--skip-model-reload")
    environment = training_environment(context.settings)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    context.latest_log = log_path
    stage_status_path = stage_root / "metadata" / "status.txt"
    stage_status = read_status(stage_status_path)
    stage_status["status"] = "verifying"
    write_status(stage_status_path, stage_status)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=context.settings.project_root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if completed.returncode != 0 or not output_path.is_file():
        stage_status["status"] = "verification_failed"
        stage_status["verification_exit_code"] = completed.returncode
        write_status(stage_status_path, stage_status)
        raise RunFailure(
            f"{stage.upper()} checkpoint verification failed.", log_path=log_path
        )
    try:
        payload = read_json(output_path)
    except Exception as exc:
        stage_status["status"] = "verification_failed"
        stage_status["verification_error"] = bounded(str(exc))
        write_status(stage_status_path, stage_status)
        raise RunFailure(
            f"{stage.upper()} checkpoint verification JSON is invalid.",
            log_path=log_path,
        ) from exc
    if payload.get("status") != "ok":
        stage_status["status"] = "verification_failed"
        write_status(stage_status_path, stage_status)
        raise RunFailure(
            f"{stage.upper()} checkpoint verification did not report success.",
            log_path=log_path,
        )
    stage_status["status"] = "completed"
    stage_status["checkpoint_verified"] = True
    stage_status["model_reload"] = not skip_model_reload
    write_status(stage_status_path, stage_status)
    (stage_root / f"{stage.upper()}_FINISHED").touch(exist_ok=False)
    emit(
        "layout_stage_completed",
        stage=stage,
        global_step=payload["safetensors"]["global_step"],
        train_loss=payload["safetensors"]["train_loss"],
        checkpoint_reload=not skip_model_reload,
    )
    return payload


def run_validation(
    context: RunContext,
    *,
    model_path: Path | None = None,
    manifest: Path | None = None,
    image_root: Path | None = None,
    split: str | None = None,
    model_kind: str | None = None,
    required_vlqa_stage: str | None = None,
) -> dict[str, Any]:
    settings = context.settings
    selected_model = model_path or settings.source_model
    selected_manifest = manifest or settings.train_manifest
    selected_image_root = image_root or settings.dataset_root
    selected_split = split or settings.layout_split
    selected_model_kind = model_kind or settings.validation_model_kind
    validation_root = settings.run_root / "validation"
    metadata_dir = validation_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=False)
    log_path = metadata_dir / "evaluate.log"
    output_path = validation_root / "layout_validation_metrics.json"
    status_path = metadata_dir / "status.txt"
    status = {
        "status": "running",
        "stage": "validation",
        "started_at": timestamp(),
        "source_model": str(selected_model),
        "output_root": str(validation_root),
        "physical_gpus": list(settings.physical_gpu_ids),
        "inference_physical_gpu": settings.physical_gpu_ids[0],
        "layout_manifest": str(selected_manifest),
        "layout_image_root": str(selected_image_root),
        "layout_split": selected_split,
        "model_kind": selected_model_kind,
        "required_vlqa_stage": required_vlqa_stage,
        "max_records": settings.validation_max_records,
        "object_threshold": settings.validation_object_threshold,
        "iou_threshold": settings.validation_iou_threshold,
    }
    write_status(status_path, status)
    command = [
        sys.executable,
        str(settings.project_root / "scripts" / "evaluate_GOT_layout.py"),
        "--model-name-or-path",
        str(selected_model),
        "--model-kind",
        selected_model_kind,
        "--tokenizer-name-or-path",
        str(settings.tokenizer_model),
        "--layout-manifest",
        str(selected_manifest),
        "--layout-image-root",
        str(selected_image_root),
        "--layout-split",
        selected_split,
        "--output-dir",
        str(validation_root),
        "--max-regions",
        str(settings.max_regions),
        "--max-records",
        str(settings.validation_max_records),
        "--model-max-length",
        str(settings.model_max_length),
        "--max-new-tokens",
        str(settings.validation_max_new_tokens),
        "--no-repeat-ngram-size",
        str(settings.validation_no_repeat_ngram_size),
        "--object-threshold",
        str(settings.validation_object_threshold),
        "--iou-threshold",
        str(settings.validation_iou_threshold),
        "--dtype",
        "bfloat16",
        "--device",
        "cuda",
    ]
    if required_vlqa_stage is not None:
        command.extend(("--require-vlqa-stage", required_vlqa_stage))
    write_json(metadata_dir / "command.json", command)
    context.latest_log = log_path
    context.update_status(status="running_validation", active_stage="validation")
    emit(
        "layout_validation_started",
        split=selected_split,
        model_kind=selected_model_kind,
        max_records=settings.validation_max_records,
        log=str(log_path),
    )
    require_gpu_free(settings)
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=settings.project_root,
            env=training_environment(settings),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    status["finished_at"] = timestamp()
    status["exit_code"] = completed.returncode
    if completed.returncode != 0 or not output_path.is_file():
        status["status"] = "failed"
        write_status(status_path, status)
        raise RunFailure(
            f"Validation failed with exit code {completed.returncode}.",
            log_path=log_path,
        )
    try:
        summary = read_json(output_path)
    except Exception as exc:
        status["status"] = "summary_invalid"
        status["summary_error"] = bounded(str(exc))
        write_status(status_path, status)
        raise RunFailure("Validation summary JSON is invalid.", log_path=log_path) from exc
    if summary.get("status") != "ok":
        status["status"] = "summary_failed"
        write_status(status_path, status)
        raise RunFailure("Validation did not report status=ok.", log_path=log_path)
    if int(summary.get("pages", 0)) < 1:
        status["status"] = "summary_failed"
        write_status(status_path, status)
        raise RunFailure("Validation reported no processed pages.", log_path=log_path)
    input_protocol = summary.get("input_protocol")
    if not isinstance(input_protocol, dict):
        raise RunFailure("Validation summary has no input_protocol.", log_path=log_path)
    if input_protocol.get("model_inputs") != ["whole_page_image", "ocr_prompt"]:
        raise RunFailure("Validation model input protocol is not page-image plus OCR prompt.")
    if input_protocol.get("layout_metadata_as_model_input") is not False:
        raise RunFailure("Validation incorrectly reports layout metadata as a model input.")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("ocr"), dict):
        raise RunFailure("Validation summary has no OCR metrics.", log_path=log_path)
    if selected_model_kind == "vlqa" and not isinstance(metrics.get("layout"), dict):
        raise RunFailure("VLQA validation summary has no layout metrics.", log_path=log_path)
    if selected_model_kind == "baseline" and metrics.get("layout") is not None:
        raise RunFailure("Baseline validation unexpectedly reported layout metrics.")
    status.update(
        {
            "status": "completed",
            "pages": summary["pages"],
            "summary": str(output_path),
            "predictions": summary.get("predictions"),
        }
    )
    write_status(status_path, status)
    (validation_root / "VALIDATION_FINISHED").touch(exist_ok=False)
    emit(
        "layout_validation_completed",
        pages=summary["pages"],
        summary=str(output_path),
        metrics=metrics,
    )
    return {
        "status": "validated",
        "summary": str(output_path),
        "predictions": summary.get("predictions"),
        "pages": summary["pages"],
        "model_inputs": input_protocol["model_inputs"],
        "layout_metadata_as_model_input": input_protocol[
            "layout_metadata_as_model_input"
        ],
        "metrics": metrics,
        "runtime": summary.get("runtime"),
        "model_kind": selected_model_kind,
        "checkpoint_stage": summary.get("checkpoint_stage"),
        "parameters": summary.get("parameters"),
    }


def read_status(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", maxsplit=1)
            result[key] = value
    return result


def execute(settings: Settings) -> int:
    validate_paths(settings)
    lock_handles = acquire_gpu_locks(settings.runs_root, settings.physical_gpu_ids)
    try:
        require_gpu_free(settings)
        if settings.run_root.exists():
            raise RunFailure(
                f"Run output already exists: {settings.run_root}", exit_code=EXIT_EXISTS
            )
        (settings.run_root / "metadata").mkdir(parents=True, exist_ok=False)
        status = {
            "status": "running_preflight",
            "started_at": timestamp(),
            "run_id": settings.run_id,
            "mode": settings.mode,
            "run_root": str(settings.run_root),
            "dataset_root": str(settings.dataset_root),
            "train_manifest": str(settings.train_manifest),
            "source_model": str(settings.source_model),
            "physical_gpus": list(settings.physical_gpu_ids),
            "world_size": len(settings.physical_gpu_ids),
            "validation_loader": "implemented",
        }
        summary: dict[str, Any] = {
            "status": "running",
            "run_id": settings.run_id,
            "mode": settings.mode,
            "run_root": str(settings.run_root),
            "started_at": status["started_at"],
            "validation_loader": "implemented",
            "settings": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(settings).items()
            },
            "preflight": {},
            "p1": None,
            "p2": None,
        }
        summary["settings"]["audit_manifests"] = [str(path) for path in settings.audit_manifests]
        context = RunContext(settings=settings, status=status, summary=summary)
        context.update_status()
        write_json(context.metadata_dir / "resolved_settings.json", summary["settings"])
        context.write_summary()
        emit(
            "layout_a100_started",
            run_id=settings.run_id,
            mode=settings.mode,
            run_root=str(settings.run_root),
        )

        try:
            context.update_status(status="environment_check")
            emit("layout_environment_check_started")
            environment_report = run_environment_check(context)
            emit("layout_environment_check_completed")
            context.update_status(status="provenance")
            emit("layout_provenance_started")
            provenance = record_provenance(context)
            emit("layout_provenance_completed")
            context.update_status(status="audit")
            emit("layout_audit_started", manifest_count=len(settings.audit_manifests))
            audit = run_audit(context)
            emit(
                "layout_audit_completed",
                pages=audit["page_count"],
                regions=audit["region_count"],
            )
            if settings.mode == "ablation" and not settings.ablation_id.startswith("vlqa_"):
                component = {
                    "status": "not_applicable",
                    "reason": "No VLQA is constructed for this ablation.",
                }
            else:
                require_gpu_free(settings)
                context.update_status(status="component_smoke")
                emit("layout_component_smoke_started")
                component = run_component_smoke(context)
                emit("layout_component_smoke_completed", gpu=component["gpu"])
                require_gpu_free(settings)
            context.summary["preflight"] = {
                "environment": environment_report,
                "provenance": provenance,
                "audit": audit,
                "component_smoke": component,
            }
            context.update_status(
                status="preflight_completed",
                audited_pages=audit["page_count"],
                audited_regions=audit["region_count"],
            )
            context.write_summary()
            emit(
                "layout_preflight_completed",
                pages=audit["page_count"],
                regions=audit["region_count"],
                gpu=component.get("gpu"),
            )

            if settings.mode == "validate":
                require_gpu_free(settings)
                validation = run_validation(
                    context,
                    model_kind=settings.validation_model_kind,
                    required_vlqa_stage=settings.validation_required_stage,
                )
                context.summary["validation"] = validation
                context.summary.update({"status": "ok", "finished_at": timestamp()})
                context.update_status(
                    status="completed",
                    active_stage="none",
                    finished_at=context.summary["finished_at"],
                    validation_pages=validation["pages"],
                )
                context.write_summary()
                (settings.run_root / "LAYOUT_A100_FINISHED").touch(exist_ok=False)
                emit(
                    "layout_a100_completed",
                    run_id=settings.run_id,
                    run_root=str(settings.run_root),
                    summary=str(settings.run_root / "summary.json"),
                    model_inputs=validation["model_inputs"],
                    layout_metadata_as_model_input=validation[
                        "layout_metadata_as_model_input"
                    ],
                    validation={
                        "pages": validation["pages"],
                        "summary": validation["summary"],
                        "metrics": validation["metrics"],
                    },
                )
                return 0

            if settings.mode == "ablation" and settings.ablation_id == "got2_zero_shot":
                require_gpu_free(settings)
                assert settings.validation_manifest is not None
                validation = run_validation(
                    context,
                    model_path=settings.source_model,
                    manifest=settings.validation_manifest,
                    image_root=settings.validation_image_root or settings.validation_manifest.parent,
                    split=settings.validation_split,
                    model_kind="baseline",
                    required_vlqa_stage=None,
                )
                context.summary.update({
                    "p1": {"status": "not_applicable"},
                    "p2": {"status": "not_applicable"},
                    "validation": validation,
                    "status": "ok",
                    "finished_at": timestamp(),
                })
                context.update_status(
                    status="completed", active_stage="none",
                    finished_at=context.summary["finished_at"],
                    validation_pages=validation["pages"],
                )
                context.write_summary()
                (settings.run_root / "LAYOUT_A100_FINISHED").touch(exist_ok=False)
                emit(
                    "layout_a100_completed", run_id=settings.run_id,
                    ablation_id=settings.ablation_id,
                    summary=str(settings.run_root / "summary.json"),
                    validation={"pages": validation["pages"], "metrics": validation["metrics"]},
                )
                return 0

            p1 = None
            p1_model = settings.source_model
            assessment_marker: str | None = None
            if "p1" in settings.stages:
                p1 = run_stage(context, stage="p1", source_model=settings.source_model)
                context.summary["p1"] = p1
                context.write_summary()
                p1_model = Path(p1["model"])
                p1["verification"] = verify_stage(
                    context,
                    stage="p1",
                    model_dir=p1_model,
                    skip_model_reload="p2" in settings.stages,
                )
                if "p2" in settings.stages:
                    p1["reload_validated_by"] = "p2_source_model_load"
                else:
                    p1["reload_validated_by"] = "p1_checkpoint_verification"
                context.summary["p1"] = p1
                context.write_summary()
            else:
                context.summary["p1"] = {
                    "status": "external_checkpoint",
                    "model": str(settings.source_model),
                    "reason": "Formal P2 joint-train starts from a completed P1 checkpoint.",
                }

            if "p2" in settings.stages:
                require_gpu_free(settings)
                p2 = run_stage(context, stage="p2", source_model=p1_model)
                context.summary["p2"] = p2
                context.write_summary()
                p2_model = Path(p2["model"])
                p2["verification"] = verify_stage(
                    context,
                    stage="p2",
                    model_dir=p2_model,
                    skip_model_reload=False,
                )
                context.summary["p2"] = p2
            else:
                p2 = None
                context.summary["p2"] = {
                    "status": "skipped",
                    "reason": (
                        "P1-only formal pretraining"
                        if settings.mode == "pretrain"
                        else "P1-only overfit diagnostic"
                    ),
                }
                if settings.mode == "overfit":
                    assert p1 is not None
                    context.summary["overfit_assessment"] = assess_p1_overfit(
                        p1["metrics"]
                    )
                    assessment_marker = (
                        "OVERFIT_PASSED"
                        if context.summary["overfit_assessment"]["status"] == "pass"
                        else "OVERFIT_FAILED"
                    )

            if settings.mode == "ablation":
                p1_exposures = (
                    p1["metrics"].get("training_budget", {}).get(
                        "total_sample_exposures_estimate", 0
                    ) if p1 is not None else 0
                )
                p2_exposures = (
                    p2["metrics"].get("training_budget", {}).get(
                        "total_sample_exposures_estimate", 0
                    ) if p2 is not None else 0
                )
                context.summary["ablation_budget"] = {
                    "ablation_id": settings.ablation_id,
                    "p1_steps": settings.p1_max_steps,
                    "p2_steps": settings.p2_max_steps,
                    "total_steps": settings.p1_max_steps + settings.p2_max_steps,
                    "p1_page_exposures_estimate": p1_exposures,
                    "p2_page_exposures_estimate": p2_exposures,
                    "total_page_exposures_estimate": p1_exposures + p2_exposures,
                    "effective_batch_size": (
                        settings.per_device_batch_size
                        * settings.gradient_accumulation_steps
                        * len(settings.physical_gpu_ids)
                    ),
                }

            validation = None
            if (
                settings.mode in {"pretrain", "joint-train", "adapt", "ablation"}
                and not settings.skip_post_training_validation
            ):
                validation_model = Path(
                    p2["model"] if p2 is not None else p1["model"]  # type: ignore[index]
                )
                assert settings.validation_manifest is not None
                validation = run_validation(
                    context,
                    model_path=validation_model,
                    manifest=settings.validation_manifest,
                    image_root=(
                        settings.validation_image_root
                        or settings.validation_manifest.parent
                    ),
                    split=settings.validation_split,
                    model_kind=settings.validation_model_kind,
                    required_vlqa_stage=settings.validation_required_stage,
                )
                context.summary["validation"] = validation
            elif (
                settings.mode in {"pretrain", "joint-train", "adapt", "ablation"}
                and settings.skip_post_training_validation
            ):
                context.summary["validation"] = {
                    "status": "skipped",
                    "reason": "skip_post_training_validation",
                }

            context.summary.update({"status": "ok", "finished_at": timestamp()})
            completion_status = {
                "status": "completed",
                "active_stage": "none",
                "finished_at": context.summary["finished_at"],
            }
            if p1 is not None:
                completion_status["p1_global_step"] = p1["metrics"]["global_step"]
            if p2 is not None:
                completion_status["p2_global_step"] = p2["metrics"]["global_step"]
            if validation is not None:
                completion_status["validation_pages"] = validation["pages"]
            if "overfit_assessment" in context.summary:
                completion_status["overfit_status"] = context.summary[
                    "overfit_assessment"
                ]["status"]
            context.update_status(
                **completion_status,
            )
            context.write_summary()
            (settings.run_root / "LAYOUT_A100_FINISHED").touch(exist_ok=False)
            if assessment_marker is not None:
                (settings.run_root / assessment_marker).touch(exist_ok=False)
            emit(
                "layout_a100_completed",
                run_id=settings.run_id,
                ablation_id=settings.ablation_id or None,
                run_root=str(settings.run_root),
                summary=str(settings.run_root / "summary.json"),
                overfit_assessment=context.summary.get("overfit_assessment"),
                validation=(
                    {
                        "pages": validation["pages"],
                        "summary": validation["summary"],
                        "metrics": validation["metrics"],
                    }
                    if validation is not None
                    else None
                ),
            )
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            error_log = context.metadata_dir / "orchestrator_error.log"
            error_log.write_text(traceback.format_exc(), encoding="utf-8")
            exit_code = (
                130
                if isinstance(exc, KeyboardInterrupt)
                else exc.exit_code
                if isinstance(exc, RunFailure)
                else EXIT_FAILURE
            )
            failure_log = exc.log_path if isinstance(exc, RunFailure) else context.latest_log
            error_payload = {
                "type": type(exc).__name__,
                "message": bounded(str(exc)),
                "exit_code": exit_code,
                "log": str(failure_log) if failure_log else None,
                "tail": tail_lines(failure_log),
            }
            context.summary.update(
                {"status": "error", "finished_at": timestamp(), "error": error_payload}
            )
            context.update_status(
                status="failed",
                finished_at=context.summary["finished_at"],
                exit_code=exit_code,
                error_type=type(exc).__name__,
                error=bounded(str(exc)),
            )
            context.write_summary()
            (settings.run_root / "LAYOUT_A100_FAILED").touch(exist_ok=True)
            emit(
                "layout_a100_failed",
                run_id=settings.run_id,
                exit_code=exit_code,
                error=bounded(str(exc)),
                log=str(failure_log) if failure_log else None,
                tail=error_payload["tail"],
            )
            return exit_code
    finally:
        assert fcntl is not None
        for lock_handle in reversed(lock_handles):
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        settings = resolve_settings(parse_args(argv))
        return execute(settings)
    except RunFailure as exc:
        emit(
            "layout_a100_rejected",
            exit_code=exc.exit_code,
            error=bounded(str(exc)),
        )
        return exc.exit_code
    except KeyboardInterrupt:
        emit("layout_a100_interrupted", exit_code=130)
        return 130
    except Exception as exc:
        emit(
            "layout_a100_rejected",
            exit_code=EXIT_FAILURE,
            error_type=type(exc).__name__,
            error=bounded(str(exc)),
        )
        return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
