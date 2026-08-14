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
from GOT.train.trainer_vit_fixlr import GOTTrainer
from GOT.utils.arguments import DataArguments, ModelArguments, TrainingArguments
from GOT.utils.constants import IGNORE_INDEX
from GOT.utils.utils import smart_tokenizer_and_embedding_resize

from layout_page_dataset import make_layout_page_data_module


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
}

LAYOUT_ADAPTER_STATE_PREFIX = "model.layout_adapter."


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
        self._layout_adapter = model_base.layout_adapter
        if self._layout_adapter is None:
            raise RuntimeError("LayoutDiagnosticTrainer requires an enabled VLQA adapter.")
        self._register_gradient_hook(
            "query_gradient_norm",
            self._layout_adapter.query_embeddings,
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
    layout_image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset root; defaults to the manifest directory."},
    )
    layout_split: str = field(default="train")
    layout_stage: str = field(
        default="p1",
        metadata={"help": "p1 (layout-only warm-up) or p2 (joint page OCR/layout)."},
    )
    max_regions: int = field(default=16)
    max_train_records: int = field(default=0)
    min_layout_regions: int = field(default=0)
    vlqa_adapter_dim: int = field(default=256)
    vlqa_num_heads: int = field(default=8)
    vlqa_ffn_expansion: int = field(default=4)
    vlqa_dropout: float = field(default=0.0)
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
    if args.layout_stage not in {"p1", "p2"}:
        raise ValueError("--layout_stage must be p1 or p2.")
    if not args.layout_split:
        raise ValueError("--layout_split must be non-empty.")
    if args.max_regions < 1:
        raise ValueError("--max_regions must be positive.")
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
    config.use_vlqa = True
    config.vlqa_num_queries = args.max_regions
    config.vlqa_adapter_dim = args.vlqa_adapter_dim
    config.vlqa_num_heads = args.vlqa_num_heads
    config.vlqa_ffn_expansion = args.vlqa_ffn_expansion
    config.vlqa_dropout = args.vlqa_dropout
    config.vlqa_num_direction_classes = 5
    config.vlqa_layout_input_dim = 1024
    config.vlqa_object_weight = args.object_loss_weight
    config.vlqa_bbox_l1_weight = args.bbox_l1_loss_weight
    config.vlqa_bbox_giou_weight = args.bbox_giou_loss_weight
    config.vlqa_direction_weight = args.direction_loss_weight
    config.layout_loss_weight = args.layout_loss_weight
    config.ocr_loss_weight = args.ocr_loss_weight
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

    if not source_layout_keys:
        adapter.reset_parameters()
        initialization = "fresh_explicit_reset"
    elif source_layout_keys == expected_layout_keys:
        initialization = "checkpoint_loaded"
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
    }


def configure_trainable_parameters(
    model: GOTQwenForCausalLM,
    stage: str,
) -> tuple[int, int, list[str]]:
    model.requires_grad_(False)
    model_base = model.get_model()
    if model_base.layout_adapter is None:
        raise RuntimeError("VLQA was not constructed from the enabled config.")
    adapter = model_base.layout_adapter

    if stage == "p1":
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
        adapter.requires_grad_(True)
        model_base.mm_projector_vary.requires_grad_(True)
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        source_model,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
        model_max_length=training_args.model_max_length,
    )
    config = build_layout_config(source_model, layout_args)
    model = GOTQwenForCausalLM.from_pretrained(
        source_model,
        config=config,
        use_safetensors=True,
        local_files_only=True,
    )
    layout_initialization = initialize_layout_adapter_from_source(
        model,
        source_weights,
    )
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
    )

    data_args.image_token_len = 256
    data_args.image_processor = vision["image_processor"]
    data_args.image_processor_high = vision["image_processor_high"]
    data_args.use_im_start_end = model_args.use_im_start_end
    data_module = make_layout_page_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        manifest=manifest,
        image_root=image_root,
        split=layout_args.layout_split,
        max_regions=layout_args.max_regions,
        max_records=layout_args.max_train_records,
        supervise_ocr=layout_args.layout_stage == "p2",
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
    layout_regions = int(first_sample["layout_bbox_mask"].sum().item())
    object_slots = int(first_sample["layout_object_mask"].sum().item())
    supervised_tokens = int((first_sample["labels"] != IGNORE_INDEX).sum().item())
    if layout_args.layout_stage == "p1" and layout_regions < 1:
        raise RuntimeError("P1 requires at least one supervised layout region.")
    if layout_regions < layout_args.min_layout_regions:
        raise RuntimeError(
            f"First sample has {layout_regions} supervised layout regions; "
            f"minimum is {layout_args.min_layout_regions}."
        )
    if layout_args.layout_stage == "p1" and object_slots != layout_args.max_regions:
        raise RuntimeError(
            "Synthetic pages require complete object supervision for every query slot: "
            f"expected={layout_args.max_regions}, actual={object_slots}."
        )
    if layout_args.layout_stage == "p1" and supervised_tokens != 0:
        raise RuntimeError("P1 unexpectedly retained OCR-supervised tokens.")
    if layout_args.layout_stage == "p2" and supervised_tokens < 1:
        raise RuntimeError("P2 requires at least one OCR-supervised token.")

    first_batch = data_module["data_collator"]([first_sample])
    bbox_batch_shape = list(first_batch["layout_bbox_targets"].shape)
    expected_bbox_batch_shape = [1, layout_args.max_regions, 4]
    if bbox_batch_shape != expected_bbox_batch_shape:
        raise RuntimeError(
            "Layout collator produced an unexpected bbox batch shape: "
            f"expected={expected_bbox_batch_shape}, actual={bbox_batch_shape}."
        )
    object_batch_shape = list(first_batch["layout_object_targets"].shape)
    expected_object_batch_shape = [1, layout_args.max_regions]
    if object_batch_shape != expected_object_batch_shape:
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
    print(f"LAYOUT_MANIFEST={manifest}")
    print(f"LAYOUT_IMAGE_ROOT={image_root}")
    print(f"LAYOUT_SPLIT={layout_args.layout_split}")
    print(f"LAYOUT_STAGE={layout_args.layout_stage}")
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

    trainer = LayoutDiagnosticTrainer(
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
    diagnostics = summarize_diagnostic_history(trainer.state.log_history)
    metrics.update(
        {
            "global_step": int(trainer.state.global_step),
            "dataset_examples": len(data_module["train_dataset"]),
            "first_sample_layout_regions": layout_regions,
            "first_sample_object_slots": object_slots,
            "first_sample_supervised_tokens": supervised_tokens,
            "first_batch_bbox_shape": bbox_batch_shape,
            "first_batch_object_shape": object_batch_shape,
            "first_batch_supervised_tokens": batch_supervised_tokens,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "layout_stage": layout_args.layout_stage,
            "max_regions": layout_args.max_regions,
            "source_model": str(source_model),
            "layout_manifest": str(manifest),
            "layout_image_root": str(image_root),
            "diagnostics": diagnostics,
            "layout_loss_compute_dtype": "float32",
            **layout_initialization,
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
