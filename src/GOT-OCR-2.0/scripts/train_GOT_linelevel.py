from __future__ import annotations

import gc
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

from GOT.model import GOTQwenForCausalLM
from GOT.train.trainer_vit_fixlr import GOTTrainer
from GOT.utils.arguments import DataArguments, ModelArguments, TrainingArguments
from GOT.utils.constants import IGNORE_INDEX
from GOT.utils.utils import smart_tokenizer_and_embedding_resize

from linelevel_dataset import make_linelevel_data_module


@dataclass
class LineLevelArguments:
    linelevel_annotations: str = field(metadata={"help": "Validated line-level JSON annotations."})
    linelevel_image_root: str = field(metadata={"help": "Root containing annotation image paths."})
    train_scope: str = field(
        default="decoder_projector",
        metadata={"help": "projector or decoder_projector; the vision tower remains frozen."},
    )
    min_supervised_tokens: int = field(default=1)


def configure_trainable_parameters(model: GOTQwenForCausalLM, scope: str) -> tuple[int, int]:
    model_base = model.get_model()
    if scope == "projector":
        model.requires_grad_(False)
        model_base.mm_projector_vary.requires_grad_(True)
    elif scope == "decoder_projector":
        model.requires_grad_(True)
        model_base.vision_tower_high.requires_grad_(False)
    else:
        raise ValueError(
            f"Unsupported train_scope={scope!r}. Full vision fine-tuning is disabled because "
            "the published single-image forward path wraps the vision tower in no_grad."
        )

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0:
        raise RuntimeError("No parameters are trainable.")
    return trainable, total


def main() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LineLevelArguments)
    )
    model_args, data_args, training_args, line_args = parser.parse_args_into_dataclasses()

    annotations = Path(line_args.linelevel_annotations).resolve()
    image_root = Path(line_args.linelevel_image_root).resolve()
    source_model = Path(str(model_args.model_name_or_path)).resolve()
    output_dir = Path(training_args.output_dir).resolve()
    if line_args.min_supervised_tokens < 1:
        raise ValueError("--min_supervised_tokens must be positive.")
    if not (source_model / "model.safetensors").is_file():
        raise FileNotFoundError(f"Original model.safetensors does not exist: {source_model}")

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
    model.config.use_cache = bool(model_args.use_cache)
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
        freeze_lm_model=False,
        pretrained_stage1_model=model_args.pretrained_stage1_model,
        device=training_args.device,
    )
    model.to(dtype=dtype, device=training_args.device)
    trainable, total = configure_trainable_parameters(model, line_args.train_scope)

    data_args.image_token_len = 256
    data_args.image_processor = vision["image_processor"]
    data_args.image_processor_high = vision["image_processor_high"]
    data_args.use_im_start_end = model_args.use_im_start_end
    data_module = make_linelevel_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        annotations=annotations,
        image_root=image_root,
    )

    first_sample = data_module["train_dataset"][0]
    supervised_tokens = int((first_sample["labels"] != IGNORE_INDEX).sum().item())
    if supervised_tokens < line_args.min_supervised_tokens:
        raise RuntimeError(
            f"First sample has only {supervised_tokens} supervised tokens; "
            f"minimum is {line_args.min_supervised_tokens}."
        )
    del first_sample
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(training_args.device)

    print(f"SOURCE_MODEL={source_model}")
    print(f"LINELEVEL_ANNOTATIONS={annotations}")
    print(f"LINELEVEL_IMAGE_ROOT={image_root}")
    print(f"TRAIN_SCOPE={line_args.train_scope}")
    print(f"TRAINABLE_PARAMETERS={trainable}")
    print(f"TOTAL_PARAMETERS={total}")
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

    metrics = dict(train_result.metrics)
    metrics.update(
        {
            "global_step": int(trainer.state.global_step),
            "dataset_examples": len(data_module["train_dataset"]),
            "first_sample_supervised_tokens": supervised_tokens,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "train_scope": line_args.train_scope,
            "source_model": str(source_model),
            "linelevel_annotations": str(annotations),
            "linelevel_image_root": str(image_root),
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
    (output_dir / "linelevel_training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("GOT_LINELEVEL_TRAINING_OK")


if __name__ == "__main__":
    main()
