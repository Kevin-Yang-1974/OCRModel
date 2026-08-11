from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.utils.constants import IGNORE_INDEX

from linelevel_dataset import make_linelevel_data_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate GOT line-level data without loading model weights.")
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-max-length", type=int, default=1024)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.source_model.resolve(),
        trust_remote_code=True,
        local_files_only=True,
        padding_side="right",
        model_max_length=args.model_max_length,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|endoftext|>"
    processor = BlipImageEvalProcessor(image_size=1024)
    data_args = SimpleNamespace(
        datasets="linelevel-json",
        conversation_version="mpt",
        sep_image_conv_front=False,
        image_token_len=256,
        image_aspect_ratio="square",
        use_im_start_end=True,
        image_processor=processor,
        image_processor_high=processor,
        box_limit=0,
    )
    data_module = make_linelevel_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        annotations=args.annotations,
        image_root=args.image_root,
    )
    sample = data_module["train_dataset"][0]
    supervised_tokens = int((sample["labels"] != IGNORE_INDEX).sum().item())
    if supervised_tokens < 1:
        raise RuntimeError("The first record has no supervised target tokens.")
    batch = data_module["data_collator"]([sample])
    print("GOT_LINELEVEL_DATA_PREFLIGHT_OK")
    print(f"dataset_examples={len(data_module['train_dataset'])}")
    print(f"input_tokens={sample['input_ids'].numel()}")
    print(f"supervised_tokens={supervised_tokens}")
    print(f"image_tensor_shape={tuple(sample['image'][0].shape)}")
    print(f"batch_input_shape={tuple(batch['input_ids'].shape)}")


if __name__ == "__main__":
    main()
