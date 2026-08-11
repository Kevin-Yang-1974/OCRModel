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

from ancientdoc_dataset import make_ancientdoc_data_module, parse_split_ids
from train_GOT_linelevel import configure_trainable_parameters


@dataclass
class AncientDocArguments:
    ancientdoc_root: str = field(metadata={"help": "Read-only AncientDoc data root."})
    audit_report: str = field(metadata={"help": "Successful AncientDoc audit JSON."})
    train_splits: str = field(default="1,2,3,4")
    train_scope: str = field(default="decoder_projector")
    record_selection: str = field(default="all")
    max_train_records: int = field(default=0)
    min_supervised_tokens: int = field(default=1)


def validate_audit_report(path: Path, data_root: Path, model_max_length: int) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "ancientdoc_audit_ok":
        raise ValueError(f"AncientDoc audit did not pass: {path}")
    if Path(str(report.get("data_root"))).resolve() != data_root:
        raise ValueError("AncientDoc audit data root does not match --ancientdoc_root.")
    labels = report.get("labels", {})
    if labels.get("union_matches_full") is not True:
        raise ValueError("AncientDoc audit does not confirm five-split coverage.")
    targets = report.get("targets", {})
    over_limits = targets.get("over_model_max_length", {})
    if str(model_max_length) in over_limits and int(over_limits[str(model_max_length)]) != 0:
        raise ValueError(
            f"Audit reports {over_limits[str(model_max_length)]} records over "
            f"model_max_length={model_max_length}."
        )
    maximum = int(targets.get("full_prompt_tokens", {}).get("max", 0))
    if maximum <= 0 or maximum >= model_max_length:
        raise ValueError(
            f"Audited maximum prompt length {maximum} is not below {model_max_length}."
        )
    return report


def main() -> None:
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, AncientDocArguments)
    )
    model_args, data_args, training_args, ancient_args = parser.parse_args_into_dataclasses()

    data_root = Path(ancient_args.ancientdoc_root).resolve()
    source_model = Path(str(model_args.model_name_or_path)).resolve()
    output_dir = Path(training_args.output_dir).resolve()
    audit_path = Path(ancient_args.audit_report).resolve()
    split_ids = parse_split_ids(ancient_args.train_splits)
    if ancient_args.min_supervised_tokens < 1:
        raise ValueError("--min_supervised_tokens must be positive.")
    if ancient_args.max_train_records < 0:
        raise ValueError("--max_train_records cannot be negative.")
    if not (source_model / "model.safetensors").is_file():
        raise FileNotFoundError(source_model / "model.safetensors")
    audit_report = validate_audit_report(
        audit_path,
        data_root,
        training_args.model_max_length,
    )

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
    trainable, total = configure_trainable_parameters(model, ancient_args.train_scope)

    data_args.image_token_len = 256
    data_args.image_processor = vision["image_processor"]
    data_args.image_processor_high = vision["image_processor_high"]
    data_args.use_im_start_end = model_args.use_im_start_end
    data_module = make_ancientdoc_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        data_root=data_root,
        split_ids=split_ids,
        record_selection=ancient_args.record_selection,
        max_records=ancient_args.max_train_records,
    )
    dataset = data_module["train_dataset"]
    first_sample = dataset[0]
    supervised_tokens = int((first_sample["labels"] != IGNORE_INDEX).sum().item())
    first_input_tokens = int(first_sample["input_ids"].numel())
    if supervised_tokens < ancient_args.min_supervised_tokens:
        raise RuntimeError(
            f"First sample has only {supervised_tokens} supervised tokens; "
            f"minimum is {ancient_args.min_supervised_tokens}."
        )
    if first_input_tokens >= training_args.model_max_length:
        raise RuntimeError(
            f"First sample may be truncated: {first_input_tokens} >= "
            f"{training_args.model_max_length}."
        )
    del first_sample
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(training_args.device)

    print(f"SOURCE_MODEL={source_model}")
    print(f"ANCIENTDOC_ROOT={data_root}")
    print(f"ANCIENTDOC_AUDIT={audit_path}")
    print(f"TRAIN_SPLITS={','.join(map(str, split_ids))}")
    print(f"TRAIN_SCOPE={ancient_args.train_scope}")
    print(f"TRAINABLE_PARAMETERS={trainable}")
    print(f"TOTAL_PARAMETERS={total}")
    print(f"DATASET_EXAMPLES={len(dataset)}")
    print(f"FIRST_SAMPLE_INPUT_TOKENS={first_input_tokens}")
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
            "dataset_examples": len(dataset),
            "first_sample_input_tokens": first_input_tokens,
            "first_sample_supervised_tokens": supervised_tokens,
            "trainable_parameters": trainable,
            "total_parameters": total,
            "train_scope": ancient_args.train_scope,
            "source_model": str(source_model),
            "ancientdoc_root": str(data_root),
            "audit_report": str(audit_path),
            "audited_full_records": audit_report["labels"]["full"]["records"],
            "train_splits": list(split_ids),
            "record_selection": ancient_args.record_selection,
            "max_train_records": ancient_args.max_train_records,
            "input_level": "page_reference_compatibility",
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
    (output_dir / "ancientdoc_training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print("GOT_ANCIENTDOC_TRAINING_OK")


if __name__ == "__main__":
    main()
