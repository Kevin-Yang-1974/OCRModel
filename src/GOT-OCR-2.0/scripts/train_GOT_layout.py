from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import transformers
from safetensors import safe_open

from GOT.model import GOTConfig, GOTQwenForCausalLM
from GOT.model.layout_query import VisualLayoutQueryAdapter
from GOT.train.trainer_vit_fixlr import GOTTrainer
from GOT.utils.arguments import DataArguments, ModelArguments, TrainingArguments
from GOT.utils.constants import IGNORE_INDEX
from GOT.utils.utils import smart_tokenizer_and_embedding_resize

from layout_page_dataset import make_layout_page_data_module, summarize_training_budget
from layout_ablation_contract import (
    ABLATIONS,
    ABLATION_IDS,
    LOSS_PRESETS,
    assert_source_protocol,
    assert_parameter_report,
    loss_weights_for,
)


OUTPUT_DIAGNOSTIC_FIELDS = {
    "ocr_loss": "ocr_loss",
    "layout_loss": "layout_loss",
    "object_loss": "layout_object_loss",
    "bbox_l1_loss": "layout_bbox_l1_loss",
    "bbox_giou_loss": "layout_bbox_giou_loss",
    "direction_loss": "layout_direction_loss",
    "object_accuracy": "layout_object_accuracy",
    "bbox_mean_iou": "layout_bbox_mean_iou",
    "direction_accuracy": "layout_direction_accuracy",
    "object_logit_abs_max": "layout_object_logit_abs_max",
    "direction_logit_abs_max": "layout_direction_logit_abs_max",
    "bbox_pred_min": "layout_bbox_pred_min",
    "bbox_pred_max": "layout_bbox_pred_max",
    "query_abs_max": "layout_query_abs_max",
    "prediction_query_abs_max": "layout_prediction_query_abs_max",
    "bbox_logit_abs_max": "layout_bbox_logit_abs_max",
    "sequence_loss": "layout_sequence_loss",
    "type_loss": "layout_type_loss",
    "count_loss": "layout_count_loss",
    "eos_accuracy": "layout_eos_accuracy",
    "region_count_mae": "layout_region_count_mae",
}

LAYOUT_ADAPTER_STATE_PREFIX = "model.layout_adapter."
GENERIC_ADAPTER_STATE_PREFIX = "model.generic_adapter."
VARIABLE_LAYOUT_STATE_PREFIX = "model.variable_layout_adapter."


def summarize_diagnostic_history(
    log_history: list[dict[str, Any]],
    *,
    tail_window: int = 20,
) -> dict[str, Any]:
    records = [
        entry
        for entry in log_history
        if "layout_loss" in entry and "step" in entry
    ]
    if not records:
        return {"log_count": 0, "tail_window": 0}
    tail = records[-min(tail_window, len(records)) :]
    numeric_keys = sorted(
        key
        for key in records[0]
        if key not in {"epoch", "step"}
        and all(isinstance(entry.get(key), (int, float)) for entry in tail)
    )

    def selected(entry: dict[str, Any]) -> dict[str, Any]:
        return {
            key: entry[key]
            for key in ("step", "epoch", *numeric_keys)
            if key in entry
        }

    return {
        "log_count": len(records),
        "tail_window": len(tail),
        "first": selected(records[0]),
        "last": selected(records[-1]),
        "tail_mean": {
            key: sum(float(entry[key]) for entry in tail) / len(tail)
            for key in numeric_keys
        },
        "tail_min": {
            key: min(float(entry[key]) for entry in tail)
            for key in numeric_keys
        },
        "tail_max": {
            key: max(float(entry[key]) for entry in tail)
            for key in numeric_keys
        },
    }


class LayoutDiagnosticTrainer(GOTTrainer):
    """Add bounded per-step VLQA diagnostics to Trainer state without printing batches."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._diagnostic_sums: dict[str, torch.Tensor] = {}
        self._diagnostic_count = 0
        self._latest_gradient_norms: dict[str, torch.Tensor] = {}
        model_base = self.model.get_model()
        self._layout_adapter = (
            model_base.layout_adapter or model_base.variable_layout_adapter
        )
        if self._layout_adapter is None:
            raise RuntimeError("LayoutDiagnosticTrainer requires an enabled layout adapter.")
        query_parameter = (
            self._layout_adapter.query_embeddings
            if hasattr(self._layout_adapter, "query_embeddings")
            else self._layout_adapter.decoder.prompt_attention.prompt_bank.prompts
        )
        self._register_gradient_hook(
            "query_gradient_norm",
            query_parameter,
        )
        self._register_gradient_hook(
            "residual_gate_gradient_abs",
            self._layout_adapter.residual_gate,
        )

    def _register_gradient_hook(self, name: str, parameter: torch.Tensor) -> None:
        if not parameter.requires_grad:
            return

        def capture(gradient: torch.Tensor) -> torch.Tensor:
            self._latest_gradient_norms[name] = gradient.detach().float().norm()
            return gradient

        parameter.register_hook(capture)

    def _record_outputs(self, outputs: Any) -> None:
        recorded = False
        for log_name, output_name in OUTPUT_DIAGNOSTIC_FIELDS.items():
            value = getattr(outputs, output_name, None)
            if value is None:
                continue
            scalar = value.detach().float()
            previous = self._diagnostic_sums.get(log_name)
            self._diagnostic_sums[log_name] = (
                scalar if previous is None else previous + scalar
            )
            recorded = True
        if recorded:
            self._diagnostic_count += 1

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if model.training:
            self._record_outputs(outputs)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
        enriched = dict(logs)
        if self._diagnostic_count:
            for name, total in self._diagnostic_sums.items():
                enriched[name] = round(
                    float(total.cpu()) / self._diagnostic_count,
                    8,
                )
            self._diagnostic_sums.clear()
            self._diagnostic_count = 0
        enriched["residual_gate"] = float(
            self._layout_adapter.residual_gate.detach().float().cpu()
        )
        enriched.update(
            {
                name: round(float(value.cpu()), 8)
                for name, value in self._latest_gradient_norms.items()
            }
        )
        self._latest_gradient_norms.clear()
        super().log(enriched, *args, **kwargs)


@dataclass
class LayoutTrainingArguments:
    layout_manifest: str = field(
        metadata={"help": "Rendered and audited whole-page manifest JSONL."}
    )
    tokenizer_name_or_path: str = field(
        default="",
        metadata={
            "help": "Fixed GOT tokenizer path; defaults to model_name_or_path."
        },
    )
    layout_image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset root; defaults to the manifest directory."},
    )
    layout_split: str = field(default="train")
    layout_stage: str = field(
        default="p1",
        metadata={"help": "p1 (layout-only warm-up) or p2 (joint page OCR/layout)."},
    )
    ablation_id: str = field(
        default="",
        metadata={"help": "Optional strict A0-A5 ablation preset; empty preserves legacy behavior."},
    )
    layout_loss_preset: str = field(
        default="",
        metadata={"help": "Strict layout loss preset used with --ablation_id."},
    )
    p2_train_scope: str = field(
        default="adapter_projector",
        metadata={
            "help": (
                "P2 trainable modules: adapter_projector or "
                "decoder_adapter_projector. P1 always uses its fixed layout warm-up scope."
            )
        },
    )
    max_regions: int = field(default=16)
    layout_architecture: str = field(default="fixed_slot")
    num_layout_prompt_queries: int = field(default=32)
    max_layout_tokens: int = field(default=2048)
    max_layout_records: int = field(default=512)
    layout_decoder_layers: int = field(default=2)
    layout_decoder_hidden_size: int = field(default=256)
    layout_decoder_num_heads: int = field(default=8)
    layout_bbox_loss_weight: float = field(default=5.0)
    layout_type_loss_weight: float = field(default=1.0)
    layout_direction_loss_weight: float = field(default=1.0)
    layout_count_loss_weight: float = field(default=0.1)
    layout_prompt_diversity_loss_weight: float = field(default=0.0)
    max_train_records: int = field(default=0)
    min_layout_regions: int = field(default=0)
    vlqa_adapter_dim: int = field(default=256)
    vlqa_num_heads: int = field(default=8)
    vlqa_ffn_expansion: int = field(default=4)
    vlqa_dropout: float = field(default=0.0)
    layout_writeback_mode: str = field(default="layout_value")
    layout_writeback_source: str = field(default="layout_evidence")
    layout_writeback_num_heads: int = field(default=8)
    layout_writeback_dropout: float = field(default=0.0)
    layout_writeback_gate_init: float = field(default=0.0)
    generic_adapter_dim: int = field(default=256)
    generic_adapter_num_heads: int = field(default=8)
    generic_adapter_ffn_expansion: int = field(default=8)
    generic_adapter_dropout: float = field(default=0.0)
    object_loss_weight: float = field(default=1.0)
    bbox_l1_loss_weight: float = field(default=5.0)
    bbox_giou_loss_weight: float = field(default=2.0)
    direction_loss_weight: float = field(default=1.0)
    layout_loss_weight: float = field(default=1.0)
    ocr_loss_weight: float = field(default=0.0)
    replay_layout_manifest: Optional[str] = field(default=None)
    replay_layout_image_root: Optional[str] = field(default=None)
    replay_layout_split: str = field(default="train")
    replay_max_train_records: int = field(default=0)
    primary_per_replay: int = field(default=3)


def validate_layout_args(args: LayoutTrainingArguments) -> None:
    if args.layout_architecture not in {"fixed_slot", "pvld"}:
        raise ValueError("--layout_architecture must be fixed_slot or pvld.")
    if args.layout_stage not in {"p1", "p2"}:
        raise ValueError("--layout_stage must be p1 or p2.")
    if args.p2_train_scope not in {
        "adapter_projector",
        "decoder_adapter_projector",
    }:
        raise ValueError(
            "--p2_train_scope must be adapter_projector or "
            "decoder_adapter_projector."
        )
    if args.ablation_id:
        if args.ablation_id not in ABLATION_IDS:
            raise ValueError(f"--ablation_id must be one of {ABLATION_IDS}.")
        if not args.layout_loss_preset:
            raise ValueError("Strict ablations require --layout_loss_preset.")
        resolved = loss_weights_for(
            args.ablation_id, args.layout_stage, args.layout_loss_preset,
            args.ocr_loss_weight,
        )
        observed = {
            "object": args.object_loss_weight,
            "bbox_l1": args.bbox_l1_loss_weight,
            "bbox_giou": args.bbox_giou_loss_weight,
            "direction_order": args.direction_loss_weight,
            "layout": args.layout_loss_weight,
            "ocr": args.ocr_loss_weight,
        }
        if observed != resolved:
            raise ValueError(
                "Ablation loss weights disagree with its declared preset: "
                f"observed={observed}, expected={resolved}."
            )
    elif args.layout_loss_preset:
        raise ValueError("--layout_loss_preset requires --ablation_id.")
    if not args.layout_split:
        raise ValueError("--layout_split must be non-empty.")
    if args.max_regions < 1:
        raise ValueError("--max_regions must be positive.")
    if args.layout_architecture == "pvld":
        if args.num_layout_prompt_queries != 32:
            raise ValueError("Formal PVLD-32 requires 32 global layout prompts.")
        if args.max_layout_tokens < 2 or args.max_layout_records < 1:
            raise ValueError("PVLD layout limits must be positive.")
        if (args.layout_decoder_hidden_size < 1 or args.layout_decoder_num_heads < 1
                or args.layout_decoder_hidden_size % args.layout_decoder_num_heads):
            raise ValueError("PVLD hidden size must be divisible by decoder heads.")
    if args.max_train_records < 0:
        raise ValueError("--max_train_records cannot be negative.")
    if args.min_layout_regions < 0:
        raise ValueError("--min_layout_regions cannot be negative.")
    if args.vlqa_adapter_dim < 1:
        raise ValueError("--vlqa_adapter_dim must be positive.")
    if args.vlqa_num_heads < 1 or args.vlqa_adapter_dim % args.vlqa_num_heads != 0:
        raise ValueError("--vlqa_adapter_dim must be divisible by --vlqa_num_heads.")
    if args.vlqa_ffn_expansion < 1:
        raise ValueError("--vlqa_ffn_expansion must be positive.")
    if not 0.0 <= args.vlqa_dropout < 1.0:
        raise ValueError("--vlqa_dropout must be in [0, 1).")
    if args.layout_writeback_mode not in {"layout_value", "vqlca", "visual_value_layout_routing"}:
        raise ValueError("unsupported --layout_writeback_mode.")
    if args.layout_writeback_source != "layout_evidence":
        raise ValueError("--layout_writeback_source must be layout_evidence.")
    if (args.layout_writeback_num_heads < 1 or
            args.vlqa_adapter_dim % args.layout_writeback_num_heads != 0):
        raise ValueError(
            "--vlqa_adapter_dim must be divisible by --layout_writeback_num_heads."
        )
    if not 0.0 <= args.layout_writeback_dropout < 1.0:
        raise ValueError("--layout_writeback_dropout must be in [0, 1).")
    if args.generic_adapter_dim < 1:
        raise ValueError("--generic_adapter_dim must be positive.")
    if (args.generic_adapter_num_heads < 1 or
            args.generic_adapter_dim % args.generic_adapter_num_heads != 0):
        raise ValueError("--generic_adapter_dim must be divisible by --generic_adapter_num_heads.")
    if args.generic_adapter_ffn_expansion < 1:
        raise ValueError("--generic_adapter_ffn_expansion must be positive.")
    weights = (
        args.object_loss_weight,
        args.bbox_l1_loss_weight,
        args.bbox_giou_loss_weight,
        args.direction_loss_weight,
        args.layout_loss_weight,
        args.ocr_loss_weight,
    )
    if any(weight < 0.0 for weight in weights):
        raise ValueError("All loss weights must be non-negative.")
    if args.layout_stage == "p1" and args.ocr_loss_weight != 0.0:
        raise ValueError("P1 is layout-only; set --ocr_loss_weight 0.")
    if args.layout_stage == "p2" and args.ocr_loss_weight <= 0.0:
        raise ValueError("P2 requires a positive --ocr_loss_weight.")
    if args.replay_max_train_records < 0:
        raise ValueError("--replay_max_train_records cannot be negative.")
    if args.primary_per_replay < 1:
        raise ValueError("--primary_per_replay must be positive.")
    if args.replay_layout_image_root and not args.replay_layout_manifest:
        raise ValueError("--replay_layout_image_root requires --replay_layout_manifest.")


def build_layout_config(
    source_model: Path,
    args: LayoutTrainingArguments,
) -> GOTConfig:
    config = GOTConfig.from_pretrained(source_model, local_files_only=True)
    if args.ablation_id:
        config.ablation_id = args.ablation_id
        config.layout_loss_preset = args.layout_loss_preset
    spec = ABLATIONS.get(args.ablation_id) if args.ablation_id else None
    requested_layout = spec.use_vlqa if spec is not None else True
    config.variable_layout_enabled = bool(
        requested_layout and args.layout_architecture == "pvld"
    )
    config.use_vlqa = bool(
        requested_layout and args.layout_architecture == "fixed_slot"
    )
    config.use_generic_adapter = spec.use_generic_adapter if spec is not None else False
    config.generic_adapter_dim = args.generic_adapter_dim
    config.generic_adapter_num_heads = args.generic_adapter_num_heads
    config.generic_adapter_ffn_expansion = args.generic_adapter_ffn_expansion
    config.generic_adapter_dropout = args.generic_adapter_dropout
    config.vlqa_num_queries = args.max_regions
    config.vlqa_adapter_dim = args.vlqa_adapter_dim
    config.vlqa_num_heads = args.vlqa_num_heads
    config.vlqa_ffn_expansion = args.vlqa_ffn_expansion
    config.vlqa_dropout = args.vlqa_dropout
    config.layout_writeback_mode = args.layout_writeback_mode
    config.layout_writeback_source = args.layout_writeback_source
    config.layout_writeback_num_heads = args.layout_writeback_num_heads
    config.layout_writeback_dropout = args.layout_writeback_dropout
    config.layout_writeback_gate_init = args.layout_writeback_gate_init
    config.num_layout_prompt_queries = args.num_layout_prompt_queries
    config.max_layout_tokens = args.max_layout_tokens
    config.max_layout_records = args.max_layout_records
    config.layout_decoder_layers = args.layout_decoder_layers
    config.layout_decoder_hidden_size = args.layout_decoder_hidden_size
    config.layout_decoder_num_heads = args.layout_decoder_num_heads
    config.layout_bbox_loss_weight = args.layout_bbox_loss_weight
    config.layout_bbox_giou_loss_weight = args.bbox_giou_loss_weight
    config.layout_type_loss_weight = args.layout_type_loss_weight
    config.layout_direction_loss_weight = args.layout_direction_loss_weight
    config.layout_count_loss_weight = args.layout_count_loss_weight
    config.layout_prompt_diversity_loss_weight = args.layout_prompt_diversity_loss_weight
    config.vlqa_num_direction_classes = 5
    config.vlqa_layout_input_dim = 1024
    config.vlqa_object_weight = args.object_loss_weight
    config.vlqa_bbox_l1_weight = args.bbox_l1_loss_weight
    config.vlqa_bbox_giou_weight = args.bbox_giou_loss_weight
    config.vlqa_direction_weight = args.direction_loss_weight
    config.layout_loss_weight = args.layout_loss_weight
    config.ocr_loss_weight = args.ocr_loss_weight
    config.layout_stage = args.layout_stage
    config.layout_architecture = args.layout_architecture
    config.use_cache = False
    config.return_dict = True
    return config


def initialize_layout_adapter_from_source(
    model: GOTQwenForCausalLM,
    source_weights: Path,
) -> dict[str, Any]:
    adapter = model.get_model().layout_adapter
    if adapter is None:
        raise RuntimeError("VLQA was not constructed from the enabled config.")

    expected_layout_keys = {
        f"{LAYOUT_ADAPTER_STATE_PREFIX}{name}" for name in adapter.state_dict()
    }
    if not expected_layout_keys:
        raise RuntimeError("VLQA has no state tensors to initialize.")
    with safe_open(str(source_weights), framework="pt", device="cpu") as handle:
        source_layout_keys = {
            name for name in handle.keys() if "layout_adapter." in name
        }

    missing: list[str] = []
    unexpected: list[str] = []
    if not source_layout_keys:
        adapter.reset_parameters()
        initialization = "fresh_explicit_reset"
    elif source_layout_keys == expected_layout_keys:
        initialization = "checkpoint_loaded"
    elif adapter.writeback_mode in {"vqlca", "visual_value_layout_routing"}:
        source_shared = {
            key for key in source_layout_keys
            if ".writeback_" not in key and ".vqlca_writeback." not in key
            and ".visual_value_layout_routing." not in key
            and not key.endswith(".residual_gate")
        }
        expected_shared = {
            key for key in expected_layout_keys
            if ".writeback_" not in key and ".vqlca_writeback." not in key
            and ".visual_value_layout_routing." not in key
            and not key.endswith(".residual_gate")
        }
        source_new_writeback = {
            key for key in source_layout_keys
            if ".vqlca_writeback." in key or ".visual_value_layout_routing." in key
        }
        if source_shared == expected_shared and not source_new_writeback:
            missing = sorted(expected_layout_keys - source_layout_keys)
            unexpected = sorted(source_layout_keys - expected_layout_keys)
            adapter.reset_writeback_parameters()
            initialization = (
                "legacy_shared_loaded_vqlca_writeback_reset"
                if adapter.writeback_mode == "vqlca"
                else "legacy_shared_loaded_visual_value_layout_routing_reset"
            )
        else:
            missing = sorted(expected_layout_keys - source_layout_keys)
            unexpected = sorted(source_layout_keys - expected_layout_keys)
            raise RuntimeError(
                "Source checkpoint contains a partial or incompatible VQLCA state. "
                f"missing_count={len(missing)}; unexpected_count={len(unexpected)}"
            )
    else:
        missing = sorted(expected_layout_keys - source_layout_keys)
        unexpected = sorted(source_layout_keys - expected_layout_keys)

        def sample(names: list[str]) -> str:
            shown = names[:12]
            suffix = f", ... (+{len(names) - len(shown)})" if len(names) > len(shown) else ""
            return ", ".join(shown) + suffix

        raise RuntimeError(
            "Source checkpoint contains a partial or incompatible VLQA state. "
            f"missing=[{sample(missing)}]; unexpected=[{sample(unexpected)}]"
        )

    parameter_abs_max = max(
        float(parameter.detach().float().abs().max().cpu())
        for parameter in adapter.parameters()
    )
    if not math.isfinite(parameter_abs_max):
        raise RuntimeError("VLQA contains non-finite parameters after checkpoint loading.")
    if initialization == "fresh_explicit_reset" and parameter_abs_max > 2.0:
        raise RuntimeError(
            "Fresh VLQA initialization produced an invalid parameter scale: "
            f"abs_max={parameter_abs_max}."
        )

    return {
        "layout_adapter_initialization": initialization,
        "source_layout_tensor_count": len(source_layout_keys),
        "expected_layout_tensor_count": len(expected_layout_keys),
        "layout_adapter_parameter_abs_max": parameter_abs_max,
        "layout_writeback_mode": adapter.writeback_mode,
        "layout_adapter_missing_keys": missing,
        "layout_adapter_unexpected_keys": unexpected,
    }


def initialize_generic_adapter_from_source(
    model: GOTQwenForCausalLM,
    source_weights: Path,
) -> dict[str, Any]:
    adapter = model.get_model().generic_adapter
    if adapter is None:
        return {
            "generic_adapter_initialization": "disabled",
            "source_generic_tensor_count": 0,
            "expected_generic_tensor_count": 0,
        }
    expected = {f"{GENERIC_ADAPTER_STATE_PREFIX}{name}" for name in adapter.state_dict()}
    with safe_open(str(source_weights), framework="pt", device="cpu") as handle:
        source = {name for name in handle.keys() if "generic_adapter." in name}
    if source:
        if source != expected:
            raise RuntimeError("Source checkpoint contains a partial generic adapter state.")
        initialization = "checkpoint_loaded"
    else:
        adapter.reset_parameters()
        initialization = "fresh_explicit_reset"
    return {
        "generic_adapter_initialization": initialization,
        "source_generic_tensor_count": len(source),
        "expected_generic_tensor_count": len(expected),
    }


def initialize_variable_layout_from_source(
    model: GOTQwenForCausalLM,
    source_weights: Path,
) -> dict[str, Any]:
    adapter = model.get_model().variable_layout_adapter
    if adapter is None:
        return {
            "layout_adapter_initialization": "disabled",
            "source_layout_tensor_count": 0,
            "expected_layout_tensor_count": 0,
            "layout_adapter_parameter_abs_max": 0.0,
            "layout_adapter_missing_keys": [],
            "layout_adapter_unexpected_keys": [],
        }
    adapter_state = adapter.state_dict()
    expected = {f"{VARIABLE_LAYOUT_STATE_PREFIX}{name}" for name in adapter_state}
    with safe_open(str(source_weights), framework="pt", device="cpu") as handle:
        source = {
            name for name in handle.keys() if name.startswith(VARIABLE_LAYOUT_STATE_PREFIX)
        }
        compatible = {}
        incompatible_shape = []
        for full_name in sorted(source & expected):
            local_name = full_name[len(VARIABLE_LAYOUT_STATE_PREFIX) :]
            value = handle.get_tensor(full_name)
            if tuple(value.shape) == tuple(adapter_state[local_name].shape):
                compatible[local_name] = value
            else:
                incompatible_shape.append(full_name)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        adapter.reset_parameters()
    if compatible:
        adapter.load_state_dict(compatible, strict=False)
    missing = sorted(expected - {f"{VARIABLE_LAYOUT_STATE_PREFIX}{name}" for name in compatible})
    unexpected = sorted((source - expected) | set(incompatible_shape))
    if not source:
        initialization = "pvld_fresh_deterministic_reset"
    elif not missing and not unexpected:
        initialization = "pvld_checkpoint_loaded"
    else:
        initialization = "pvld_legacy_checkpoint_migrated_deterministically"
    parameter_abs_max = max(
        float(parameter.detach().float().abs().max().cpu())
        for parameter in adapter.parameters()
    )
    return {
        "layout_adapter_initialization": initialization,
        "source_layout_tensor_count": len(source),
        "expected_layout_tensor_count": len(expected),
        "layout_adapter_parameter_abs_max": parameter_abs_max,
        "layout_writeback_mode": "visual_value_layout_routing",
        "layout_writeback_source": "layout_evidence",
        "layout_adapter_missing_keys": missing,
        "layout_adapter_unexpected_keys": unexpected,
        "layout_adapter_loaded_compatible_keys": sorted(
            f"{VARIABLE_LAYOUT_STATE_PREFIX}{name}" for name in compatible
        ),
        "layout_adapter_initialization_seed": 0,
    }


def configure_trainable_parameters(
    model: GOTQwenForCausalLM,
    stage: str,
    p2_train_scope: str = "adapter_projector",
    ablation_id: str = "",
) -> tuple[int, int, list[str]]:
    model.requires_grad_(False)
    model_base = model.get_model()
    if ablation_id in {"projector_only", "generic_adapter_projector"}:
        if stage != "p2":
            raise RuntimeError(f"{ablation_id} only supports direct P2.")
        model_base.mm_projector_vary.requires_grad_(True)
        if ablation_id == "generic_adapter_projector":
            if model_base.generic_adapter is None:
                raise RuntimeError("A2 generic adapter was not constructed.")
            model_base.generic_adapter.requires_grad_(True)
        elif model_base.generic_adapter is not None or model_base.layout_adapter is not None:
            raise RuntimeError("A1 must not construct or train an adapter.")
        adapter = None
    else:
        adapter = model_base.layout_adapter or model_base.variable_layout_adapter
        if adapter is None:
            raise RuntimeError("The requested layout architecture was not constructed.")

    if adapter is None:
        pass
    elif stage == "p1":
        if model_base.variable_layout_adapter is not None:
            adapter.requires_grad_(True)
            with torch.no_grad():
                adapter.residual_gate.zero_()
            model_base.vision_tower_high.requires_grad_(False)
            trainable_names = [
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            ]
            trainable = sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            )
            total = sum(parameter.numel() for parameter in model.parameters())
            return trainable, total, trainable_names
        adapter.requires_grad_(False)
        adapter.query_embeddings.requires_grad_(True)
        for module in (
            adapter.memory_norm,
            adapter.memory_projection,
            adapter.query_norm,
            adapter.query_cross_attention,
            adapter.query_ffn_norm,
            adapter.query_ffn,
            adapter.prediction_norm,
            adapter.object_head,
            adapter.box_head,
            adapter.direction_head,
        ):
            module.requires_grad_(True)
        with torch.no_grad():
            adapter.residual_gate.zero_()
    elif stage == "p2":
        if p2_train_scope == "adapter_projector":
            adapter.requires_grad_(True)
            model_base.mm_projector_vary.requires_grad_(True)
        elif p2_train_scope == "decoder_adapter_projector":
            model.requires_grad_(True)
        else:
            raise ValueError(p2_train_scope)
    else:
        raise ValueError(stage)
    model_base.vision_tower_high.requires_grad_(False)

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0:
        raise RuntimeError("No parameters are trainable.")
    return trainable, total, trainable_names


def module_parameter_report(model: GOTQwenForCausalLM) -> dict[str, dict[str, Any]]:
    base = model.get_model()
    modules = {
        "vary_vit": base.vision_tower_high,
        "qwen": model,
        "mm_projector_vary": base.mm_projector_vary,
        "generic_adapter": base.generic_adapter,
        "vlqa": base.layout_adapter,
        "pvld": base.variable_layout_adapter,
    }
    report: dict[str, dict[str, Any]] = {}
    for name, module in modules.items():
        if module is None:
            report[name] = {"present": False, "total": 0, "trainable": 0}
            continue
        parameters = list(module.parameters())
        report[name] = {
            "present": True,
            "total": sum(parameter.numel() for parameter in parameters),
            "trainable": sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            ),
        }
    # Qwen contains the visual modules; report only parameters outside the three visual paths.
    visual_ids = {
        id(parameter)
        for module in (base.vision_tower_high, base.mm_projector_vary,
                       base.generic_adapter, base.layout_adapter,
                       base.variable_layout_adapter)
        if module is not None
        for parameter in module.parameters()
    }
    qwen_parameters = [parameter for parameter in model.parameters() if id(parameter) not in visual_ids]
    report["qwen"] = {
        "present": True,
        "total": sum(parameter.numel() for parameter in qwen_parameters),
        "trainable": sum(parameter.numel() for parameter in qwen_parameters if parameter.requires_grad),
    }
    return report


def assert_ablation_trainable_scope(
    ablation_id: str,
    stage: str,
    model: GOTQwenForCausalLM,
) -> dict[str, dict[str, Any]]:
    report = module_parameter_report(model)
    if model.get_model().variable_layout_adapter is not None:
        expected = {"pvld"} if stage == "p1" else {"mm_projector_vary", "pvld"}
        actual = {
            name for name in ("mm_projector_vary", "generic_adapter", "vlqa", "pvld")
            if int(report[name]["trainable"]) > 0
        }
        if actual != expected or int(report["vary_vit"]["trainable"]) or int(report["qwen"]["trainable"]):
            raise RuntimeError(
                f"PVLD {stage.upper()} trainable modules mismatch: "
                f"actual={sorted(actual)}, expected={sorted(expected)}."
            )
        return report
    try:
        assert_parameter_report(ablation_id, stage, report)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return report


def vlqa_ocr_path_parameter_count(model: GOTQwenForCausalLM) -> int:
    adapter = model.get_model().layout_adapter
    if adapter is None:
        return 0
    auxiliary_prefixes = ("prediction_norm.", "object_head.", "box_head.", "direction_head.")
    return sum(
        parameter.numel()
        for name, parameter in adapter.named_parameters()
        if not name.startswith(auxiliary_prefixes)
    )


def vlqa_ocr_path_reference_parameter_count(args: LayoutTrainingArguments) -> int:
    reference = VisualLayoutQueryAdapter(
        visual_dim=1024,
        layout_input_dim=1024,
        adapter_dim=args.vlqa_adapter_dim,
        num_queries=args.max_regions,
        num_heads=args.vlqa_num_heads,
        ffn_expansion=args.vlqa_ffn_expansion,
        num_direction_classes=5,
        dropout=args.vlqa_dropout,
        writeback_mode=args.layout_writeback_mode,
        writeback_num_heads=args.layout_writeback_num_heads,
        writeback_dropout=args.layout_writeback_dropout,
        writeback_gate_init=args.layout_writeback_gate_init,
    )
    auxiliary_prefixes = ("prediction_norm.", "object_head.", "box_head.", "direction_head.")
    count = sum(
        parameter.numel()
        for name, parameter in reference.named_parameters()
        if not name.startswith(auxiliary_prefixes)
    )
    del reference
    return count


def main() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LayoutTrainingArguments)
    )
    model_args, data_args, training_args, layout_args = parser.parse_args_into_dataclasses()
    validate_layout_args(layout_args)
    if training_args.remove_unused_columns:
        raise ValueError(
            "Layout training requires --remove_unused_columns false so the custom collator "
            "can receive image and layout fields."
        )

    source_model = Path(str(model_args.model_name_or_path)).resolve()
    manifest = Path(layout_args.layout_manifest).resolve()
    image_root = (
        Path(layout_args.layout_image_root).resolve()
        if layout_args.layout_image_root
        else manifest.parent
    )
    output_dir = Path(training_args.output_dir).resolve()
    source_weights = source_model / "model.safetensors"
    if not source_weights.is_file():
        raise FileNotFoundError(f"Original model.safetensors does not exist: {source_model}")
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    source_config_payload = json.loads(
        (source_model / "config.json").read_text(encoding="utf-8")
    )
    source_metrics_path = source_model / "layout_training_metrics.json"
    source_metrics_payload = (
        json.loads(source_metrics_path.read_text(encoding="utf-8"))
        if source_metrics_path.is_file() else None
    )
    if layout_args.ablation_id and layout_args.layout_architecture == "fixed_slot":
        assert_source_protocol(
            layout_args.ablation_id,
            layout_args.layout_stage,
            source_config_payload,
            source_metrics_payload,
        )
    elif layout_args.ablation_id and layout_args.layout_architecture == "pvld":
        source_pvld = source_config_payload.get("variable_layout_enabled") is True
        expects_p1 = (
            layout_args.ablation_id == "vlqa_layout_p1_p2"
            and layout_args.layout_stage == "p2"
        )
        if expects_p1:
            if (not source_pvld or not source_metrics_payload
                    or source_metrics_payload.get("layout_stage") != "p1"
                    or source_metrics_payload.get("layout_architecture") != "pvld"):
                raise ValueError("PVLD C5 P2 must initialize from its validation-eligible P1 model.")
        elif source_pvld:
            raise ValueError("Direct PVLD stages must initialize from original GOT2.")

    tokenizer_source = (
        Path(layout_args.tokenizer_name_or_path).resolve()
        if layout_args.tokenizer_name_or_path
        else source_model
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
        model_max_length=training_args.model_max_length,
    )
    transformers.set_seed(training_args.seed)
    config = build_layout_config(source_model, layout_args)
    model = GOTQwenForCausalLM.from_pretrained(
        source_model,
        config=config,
        use_safetensors=True,
        local_files_only=True,
    )
    if model.get_model().layout_adapter is not None:
        layout_initialization = initialize_layout_adapter_from_source(model, source_weights)
    elif model.get_model().variable_layout_adapter is not None:
        layout_initialization = initialize_variable_layout_from_source(model, source_weights)
    else:
        layout_initialization = {
            "layout_adapter_initialization": "disabled",
            "source_layout_tensor_count": 0,
            "expected_layout_tensor_count": 0,
            "layout_adapter_parameter_abs_max": 0.0,
        }
    generic_initialization = initialize_generic_adapter_from_source(model, source_weights)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={"pad_token": "<|endoftext|>"},
        tokenizer=tokenizer,
        model=model,
    )

    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16
    vision = model.get_model().initialize_vision_modules(
        vision_tower=model_args.vision_tower,
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        freeze_vision_tower=True,
        use_im_start_end=model_args.use_im_start_end,
        vision_select_layer=model_args.vision_select_layer,
        dtype=dtype,
        device=training_args.device,
    )
    model.initialize_vision_tokenizer(
        tokenizer=tokenizer,
        freeze_lm_model=True,
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        device=training_args.device,
    )
    model.to(dtype=dtype, device=training_args.device)
    trainable, total, trainable_names = configure_trainable_parameters(
        model,
        layout_args.layout_stage,
        layout_args.p2_train_scope,
        layout_args.ablation_id,
    )
    module_parameters = (
        assert_ablation_trainable_scope(layout_args.ablation_id, layout_args.layout_stage, model)
        if layout_args.ablation_id else module_parameter_report(model)
    )

    data_args.image_token_len = 256
    data_args.image_processor = vision["image_processor"]
    data_args.image_processor_high = vision["image_processor_high"]
    data_args.use_im_start_end = model_args.use_im_start_end
    include_layout_targets = (
        model.get_model().layout_adapter is not None
        or model.get_model().variable_layout_adapter is not None
    )
    layout_target_mode = (
        "pvld" if model.get_model().variable_layout_adapter is not None else "fixed_slot"
    )
    data_module = make_layout_page_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        manifest=manifest,
        image_root=image_root,
        split=layout_args.layout_split,
        max_regions=layout_args.max_regions,
        max_records=layout_args.max_train_records,
        supervise_ocr=layout_args.layout_stage == "p2",
        include_layout_targets=include_layout_targets,
        layout_target_mode=layout_target_mode,
        max_layout_tokens=layout_args.max_layout_tokens,
        max_layout_records=layout_args.max_layout_records,
        replay_manifest=(
            Path(layout_args.replay_layout_manifest).resolve()
            if layout_args.replay_layout_manifest
            else None
        ),
        replay_image_root=(
            Path(layout_args.replay_layout_image_root).resolve()
            if layout_args.replay_layout_image_root
            else None
        ),
        replay_split=layout_args.replay_layout_split,
        replay_max_records=layout_args.replay_max_train_records,
        primary_per_replay=layout_args.primary_per_replay,
    )

    first_sample = data_module["train_dataset"][0]
    variable_layout = "layout_input_ids" in first_sample
    has_layout_targets = include_layout_targets and (
        "layout_bbox_mask" in first_sample or variable_layout
    )
    layout_regions = (
        int(first_sample[
            "layout_record_mask" if variable_layout else "layout_bbox_mask"
        ].sum().item()) if has_layout_targets else 0
    )
    object_slots = (
        int(first_sample["layout_object_mask"].sum().item())
        if has_layout_targets and not variable_layout else 0
    )
    supervised_tokens = int((first_sample["labels"] != IGNORE_INDEX).sum().item())
    if layout_args.layout_stage == "p1" and layout_regions < 1:
        raise RuntimeError("P1 requires at least one supervised layout region.")
    if has_layout_targets and layout_regions < layout_args.min_layout_regions:
        raise RuntimeError(
            f"First sample has {layout_regions} supervised layout regions; "
            f"minimum is {layout_args.min_layout_regions}."
        )
    if (layout_args.layout_stage == "p1" and not variable_layout
            and object_slots != layout_args.max_regions):
        raise RuntimeError(
            "Synthetic pages require complete object supervision for every query slot: "
            f"expected={layout_args.max_regions}, actual={object_slots}."
        )
    if layout_args.layout_stage == "p1" and supervised_tokens != 0:
        raise RuntimeError("P1 unexpectedly retained OCR-supervised tokens.")
    if layout_args.layout_stage == "p2" and supervised_tokens < 1:
        raise RuntimeError("P2 requires at least one OCR-supervised token.")

    first_batch = data_module["data_collator"]([first_sample])
    bbox_batch_shape = (
        list(first_batch["layout_bbox_targets"].shape) if has_layout_targets else None
    )
    expected_bbox_batch_shape = [
        1,
        layout_regions if variable_layout else layout_args.max_regions,
        4,
    ]
    if has_layout_targets and bbox_batch_shape != expected_bbox_batch_shape:
        raise RuntimeError(
            "Layout collator produced an unexpected bbox batch shape: "
            f"expected={expected_bbox_batch_shape}, actual={bbox_batch_shape}."
        )
    object_batch_shape = (
        list(first_batch["layout_object_targets"].shape) if has_layout_targets else None
    ) if not variable_layout else None
    expected_object_batch_shape = [1, layout_args.max_regions]
    if has_layout_targets and not variable_layout and object_batch_shape != expected_object_batch_shape:
        raise RuntimeError(
            "Layout collator produced an unexpected object batch shape: "
            f"expected={expected_object_batch_shape}, actual={object_batch_shape}."
        )
    batch_supervised_tokens = int(
        (first_batch["labels"] != IGNORE_INDEX).sum().item()
    )
    if layout_args.layout_stage == "p1" and batch_supervised_tokens != 0:
        raise RuntimeError("P1 collator unexpectedly retained OCR-supervised tokens.")
    if layout_args.layout_stage == "p2" and batch_supervised_tokens < 1:
        raise RuntimeError("P2 collator requires at least one OCR-supervised token.")
    del first_sample
    del first_batch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(training_args.device)

    print(f"SOURCE_MODEL={source_model}")
    print(f"TOKENIZER_MODEL={tokenizer_source}")
    print(f"LAYOUT_MANIFEST={manifest}")
    print(f"LAYOUT_IMAGE_ROOT={image_root}")
    print(f"LAYOUT_SPLIT={layout_args.layout_split}")
    print(f"LAYOUT_STAGE={layout_args.layout_stage}")
    print(f"LAYOUT_ARCHITECTURE={layout_args.layout_architecture}")
    print(f"ABLATION_ID={layout_args.ablation_id or 'legacy_default'}")
    print(f"P2_TRAIN_SCOPE={layout_args.p2_train_scope}")
    print(
        "LAYOUT_ADAPTER_INITIALIZATION="
        f"{layout_initialization['layout_adapter_initialization']}"
    )
    print(
        "SOURCE_LAYOUT_TENSOR_COUNT="
        f"{layout_initialization['source_layout_tensor_count']}"
    )
    print(
        "EXPECTED_LAYOUT_TENSOR_COUNT="
        f"{layout_initialization['expected_layout_tensor_count']}"
    )
    print(
        "LAYOUT_ADAPTER_PARAMETER_ABS_MAX="
        f"{layout_initialization['layout_adapter_parameter_abs_max']}"
    )
    print(f"MAX_REGIONS={layout_args.max_regions}")
    print(f"TRAINABLE_PARAMETERS={trainable}")
    print(f"TOTAL_PARAMETERS={total}")
    print(f"TRAINABLE_PARAMETER_PREFIXES={','.join(sorted(set(name.split('.')[0] for name in trainable_names)))}")
    print(f"FIRST_SAMPLE_LAYOUT_REGIONS={layout_regions}")
    print(f"FIRST_SAMPLE_OBJECT_SLOTS={object_slots}")
    print(f"FIRST_SAMPLE_SUPERVISED_TOKENS={supervised_tokens}")
    print(f"FIRST_BATCH_BBOX_SHAPE={bbox_batch_shape}")
    print(f"FIRST_BATCH_OBJECT_SHAPE={object_batch_shape}")
    print(f"FIRST_BATCH_SUPERVISED_TOKENS={batch_supervised_tokens}")

    trainer_class = (
        LayoutDiagnosticTrainer
        if (model.get_model().layout_adapter is not None
            or model.get_model().variable_layout_adapter is not None) else GOTTrainer
    )
    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    checkpoints = sorted(output_dir.glob("checkpoint-*"))
    train_result = trainer.train(resume_from_checkpoint=bool(checkpoints))
    trainer.save_state()
    trainer._safe_save(output_dir=str(output_dir))
    if training_args.local_rank in (-1, 0):
        tokenizer.save_pretrained(output_dir)

    metrics = dict(train_result.metrics)
    diagnostics = (
        summarize_diagnostic_history(trainer.state.log_history)
        if (model.get_model().layout_adapter is not None
            or model.get_model().variable_layout_adapter is not None)
        else {"log_count": 0}
    )
    budget = summarize_training_budget(
        data_module["train_dataset"],
        optimizer_steps=int(trainer.state.global_step),
        per_device_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        world_size=training_args.world_size,
    )
    vlqa_ocr_reference = (
        vlqa_ocr_path_reference_parameter_count(layout_args)
        if layout_args.layout_architecture == "fixed_slot" else 0
    )
    generic_count = module_parameters["generic_adapter"]["total"]
    adapter = model.get_model().layout_adapter
    variable_adapter = model.get_model().variable_layout_adapter
    vqlca_parameters = (
        sum(parameter.numel() for parameter in adapter.vqlca_writeback.parameters())
        if adapter is not None and adapter.vqlca_writeback is not None else 0
    )
    pvld_parameters = (
        sum(parameter.numel() for parameter in variable_adapter.parameters())
        if variable_adapter is not None else 0
    )
    metrics.update(
        {
            "global_step": int(trainer.state.global_step),
            "optimizer_steps": int(trainer.state.global_step),
            "dataset_examples": len(data_module["train_dataset"]),
            "first_sample_layout_regions": layout_regions,
            "first_sample_object_slots": object_slots,
            "first_sample_supervised_tokens": supervised_tokens,
            "first_batch_bbox_shape": bbox_batch_shape,
            "first_batch_object_shape": object_batch_shape,
            "first_batch_supervised_tokens": batch_supervised_tokens,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "module_parameters": module_parameters,
            "trainable_parameter_prefixes": sorted(
                {".".join(name.split(".")[:3]) for name in trainable_names}
            ),
            "frozen_modules": (
                ["model.vision_tower_high"]
                if layout_args.layout_stage == "p2"
                and layout_args.p2_train_scope == "decoder_adapter_projector"
                else ["language_model", "model.vision_tower_high"]
            ),
            "train_scope": (
                layout_args.p2_train_scope
                if layout_args.layout_stage == "p2"
                else "p1_layout_warmup"
            ),
            "optimizer": training_args.optim,
            "learning_rate": training_args.learning_rate,
            "lr_scheduler_type": str(training_args.lr_scheduler_type),
            "weight_decay": training_args.weight_decay,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "initial_checkpoint": str(source_model),
            "tokenizer_checkpoint": str(tokenizer_source),
            "upstream_training_history": (
                "synthetic_layout_p2_then_ancientdoc_c4"
                if layout_args.replay_layout_manifest
                else "synthetic_layout_p2_checkpoint"
            ),
            "strict_equal_parameter_control_vs_c1": False,
            "comparison_role": "vlqa_adaptation_route",
            "training_budget": budget,
            "effective_batch_size": (
                training_args.per_device_train_batch_size
                * training_args.gradient_accumulation_steps
                * training_args.world_size
            ),
            "layout_stage": layout_args.layout_stage,
            "layout_architecture": layout_args.layout_architecture,
            "variable_layout_enabled": variable_adapter is not None,
            "num_layout_prompt_queries": layout_args.num_layout_prompt_queries,
            "prompt_queries_are_region_slots": False if variable_adapter is not None else True,
            "max_layout_tokens": layout_args.max_layout_tokens,
            "max_layout_records": layout_args.max_layout_records,
            "pvld_decoder_version": (
                "causal_transformer_fsm_previous_region_v1"
                if variable_adapter is not None else None
            ),
            "pvld_decoder_memory": (
                "layout_evidence_only" if variable_adapter is not None else None
            ),
            "pvld_coverage": (
                {
                    "kind": "exclusive_previous_region_hidden_mean",
                    "shape": "B,T,D",
                    "detach": False,
                    "full_attention_saved": False,
                }
                if variable_adapter is not None else None
            ),
            "ablation_id": layout_args.ablation_id or "legacy_default",
            "layout_loss_preset": layout_args.layout_loss_preset or "legacy_explicit_weights",
            "loss_weights": {
                "ocr": layout_args.ocr_loss_weight,
                "object": layout_args.object_loss_weight,
                "bbox_l1": layout_args.bbox_l1_loss_weight,
                "bbox_giou": layout_args.bbox_giou_loss_weight,
                "direction_order": layout_args.direction_loss_weight,
                "layout": layout_args.layout_loss_weight,
            },
            "projector_trainable": module_parameters["mm_projector_vary"]["trainable"] > 0,
            "use_vlqa": model.get_model().layout_adapter is not None,
            "use_pvld": variable_adapter is not None,
            "use_generic_adapter": model.get_model().generic_adapter is not None,
            "residual_writeback_enabled": (
                model.get_model().layout_adapter is not None or variable_adapter is not None
            ),
            "layout_writeback_mode": layout_args.layout_writeback_mode,
            "layout_writeback_num_heads": layout_args.layout_writeback_num_heads,
            "layout_writeback_dropout": layout_args.layout_writeback_dropout,
            "layout_writeback_gate_init": layout_args.layout_writeback_gate_init,
            "vqlca_parameters": vqlca_parameters,
            "pvld_parameters": pvld_parameters,
            "passed_through_p1": (
                layout_args.ablation_id == "vlqa_layout_p1_p2"
                and layout_args.layout_stage == "p2"
            ),
            "layout_heads_expected_gradient": (
                (model.get_model().layout_adapter is not None or variable_adapter is not None)
                and layout_args.layout_loss_weight > 0.0
            ),
            "input_granularity": "whole_page_image",
            "input_protocol": {
                "model_inputs": ["whole_page_image", "ocr_prompt"],
                "bbox_as_model_input": False,
                "direction_as_model_input": False,
                "order_as_model_input": False,
                "layout_metadata_as_model_input": False,
            },
            "generic_adapter_parameter_match": {
                "generic_adapter": generic_count,
                "vlqa_ocr_path": vlqa_ocr_reference,
                "absolute_error": (
                    abs(generic_count - vlqa_ocr_reference) if generic_count else None
                ),
                "relative_error": (
                    abs(generic_count - vlqa_ocr_reference) / vlqa_ocr_reference
                    if generic_count and vlqa_ocr_reference else None
                ),
            },
            "max_regions": layout_args.max_regions,
            "source_model": str(source_model),
            "layout_manifest": str(manifest),
            "layout_image_root": str(image_root),
            "diagnostics": diagnostics,
            "layout_loss_compute_dtype": "float32",
            **layout_initialization,
            **generic_initialization,
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated(training_args.device) / (1024**2)
                if torch.cuda.is_available()
                else None
            ),
            "peak_reserved_mib": (
                torch.cuda.max_memory_reserved(training_args.device) / (1024**2)
                if torch.cuda.is_available()
                else None
            ),
        }
    )
    train_loss = float(metrics.get("train_loss", float("nan")))
    if trainer.state.global_step < 1 or not math.isfinite(train_loss) or train_loss <= 0:
        raise RuntimeError(
            f"Training did not produce a valid optimizer step: step={trainer.state.global_step}, "
            f"train_loss={train_loss}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "layout_training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("GOT_LAYOUT_TRAINING_OK")


if __name__ == "__main__":
    main()
