from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import transformers

from GOT.model import GOTQwenForCausalLM
from GOT.train.trainer_vit_fixlr import GOTTrainer
from GOT.utils.arguments import DataArguments, ModelArguments, TrainingArguments
from GOT.utils.constants import IGNORE_INDEX
from GOT.utils.utils import smart_tokenizer_and_embedding_resize

from layout_page_dataset import make_layout_page_data_module, summarize_training_budget
from train_GOT_linelevel import configure_trainable_parameters


@dataclass
class PageOCRTrainingArguments:
    layout_manifest: str = field(metadata={"help": "Whole-page manifest JSONL."})
    layout_image_root: str = field(metadata={"help": "Image root for manifest paths."})
    layout_split: str = field(default="train")
    max_regions: int = field(default=16)
    max_train_records: int = field(default=0)
    train_scope: str = field(default="decoder_projector")
    min_supervised_tokens: int = field(default=1)


def main() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, PageOCRTrainingArguments)
    )
    model_args, data_args, training_args, page_args = parser.parse_args_into_dataclasses()
    if training_args.remove_unused_columns:
        raise ValueError("Whole-page OCR training requires --remove_unused_columns false.")
    if page_args.max_train_records < 0 or page_args.max_regions < 1:
        raise ValueError("Invalid record or region limit.")
    if page_args.min_supervised_tokens < 1:
        raise ValueError("--min_supervised_tokens must be positive.")

    source_model = Path(str(model_args.model_name_or_path)).resolve()
    manifest = Path(page_args.layout_manifest).resolve()
    image_root = Path(page_args.layout_image_root).resolve()
    output_dir = Path(training_args.output_dir).resolve()
    for path in (source_model / "model.safetensors", source_model / "config.json", manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    source_config = json.loads((source_model / "config.json").read_text(encoding="utf-8"))
    if source_config.get("use_vlqa") is True:
        raise RuntimeError("C1 must start from GOT2 without VLQA/layout queries.")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        source_model,
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
        model_max_length=training_args.model_max_length,
    )
    model = GOTQwenForCausalLM.from_pretrained(
        source_model,
        use_safetensors=True,
        local_files_only=True,
    )
    if getattr(model.config, "use_vlqa", False) is True:
        raise RuntimeError("Loaded C1 model unexpectedly enables VLQA.")
    model.config.use_cache = bool(model_args.use_cache)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={"pad_token": "<|endoftext|>"},
        tokenizer=tokenizer,
        model=model,
    )

    dtype = torch.bfloat16 if training_args.bf16 else (
        torch.float16 if training_args.fp16 else torch.float32
    )
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
        freeze_lm_model=False,
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        device=training_args.device,
    )
    model.to(dtype=dtype, device=training_args.device)
    trainable, total = configure_trainable_parameters(model, page_args.train_scope)
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]

    data_args.image_token_len = 256
    data_args.image_processor = vision["image_processor"]
    data_args.image_processor_high = vision["image_processor_high"]
    data_args.use_im_start_end = model_args.use_im_start_end
    data_module = make_layout_page_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        manifest=manifest,
        image_root=image_root,
        split=page_args.layout_split,
        max_regions=page_args.max_regions,
        max_records=page_args.max_train_records,
        supervise_ocr=True,
        include_layout_targets=False,
    )
    dataset = data_module["train_dataset"]
    first_sample = dataset[0]
    supervised_tokens = int((first_sample["labels"] != IGNORE_INDEX).sum().item())
    if supervised_tokens < page_args.min_supervised_tokens:
        raise RuntimeError(
            f"First sample has {supervised_tokens} supervised tokens; "
            f"minimum is {page_args.min_supervised_tokens}."
        )
    del first_sample
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(training_args.device)

    print(f"SOURCE_MODEL={source_model}")
    print(f"LAYOUT_MANIFEST={manifest}")
    print(f"LAYOUT_IMAGE_ROOT={image_root}")
    print(f"TRAIN_SCOPE={page_args.train_scope}")
    print(f"TRAINABLE_PARAMETERS={trainable}")
    print(f"TOTAL_PARAMETERS={total}")
    print(f"DATASET_EXAMPLES={len(dataset)}")
    print(f"FIRST_SAMPLE_SUPERVISED_TOKENS={supervised_tokens}")

    trainer = GOTTrainer(
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

    budget = summarize_training_budget(
        dataset,
        optimizer_steps=int(trainer.state.global_step),
        per_device_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        world_size=training_args.world_size,
    )
    metrics = dict(train_result.metrics)
    metrics.update(
        {
            "global_step": int(trainer.state.global_step),
            "optimizer_steps": int(trainer.state.global_step),
            "dataset_examples": len(dataset),
            "first_sample_supervised_tokens": supervised_tokens,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_parameter_prefixes": sorted(
                {".".join(name.split(".")[:3]) for name in trainable_names}
            ),
            "frozen_modules": ["model.vision_tower_high"],
            "train_scope": page_args.train_scope,
            "optimizer": training_args.optim,
            "learning_rate": training_args.learning_rate,
            "lr_scheduler_type": str(training_args.lr_scheduler_type),
            "weight_decay": training_args.weight_decay,
            "per_device_train_batch_size": training_args.per_device_train_batch_size,
            "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
            "initial_checkpoint": str(source_model),
            "upstream_training_history": "original_got2_checkpoint",
            "strict_equal_parameter_control_vs_c4": False,
            "comparison_role": "native_got2_full_adaptation_route",
            "model_kind": "baseline",
            "use_vlqa": False,
            "input_protocol": {
                "model_inputs": ["whole_page_image", "ocr_prompt"],
                "layout_metadata_as_model_input": False,
            },
            "layout_manifest": str(manifest),
            "layout_image_root": str(image_root),
            "layout_split": page_args.layout_split,
            "training_budget": budget,
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
            f"Training did not produce a valid optimizer step: "
            f"step={trainer.state.global_step}, train_loss={train_loss}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "page_ocr_training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("GOT_PAGE_OCR_TRAINING_OK")


if __name__ == "__main__":
    main()
