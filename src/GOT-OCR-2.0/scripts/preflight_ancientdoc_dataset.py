from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from transformers import AutoTokenizer

from GOT.model.plug.blip_process import BlipImageEvalProcessor
from GOT.utils.constants import IGNORE_INDEX

from ancientdoc_dataset import make_ancientdoc_data_module, parse_split_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate AncientDoc page data without loading GOT model weights."
    )
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-splits", default="1,2,3,4")
    parser.add_argument("--model-max-length", type=int, default=8192)
    parser.add_argument("--record-selection", choices=["all", "first", "longest"], default="longest")
    parser.add_argument("--max-records", type=int, default=1)
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
        datasets="ancientdoc-page",
        conversation_version="mpt",
        sep_image_conv_front=False,
        image_token_len=256,
        image_aspect_ratio="square",
        use_im_start_end=True,
        image_processor=processor,
        image_processor_high=processor,
        box_limit=0,
    )
    split_ids = parse_split_ids(args.train_splits)
    data_module = make_ancientdoc_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        data_root=args.data_root.resolve(),
        split_ids=split_ids,
        record_selection=args.record_selection,
        max_records=args.max_records,
    )
    dataset = data_module["train_dataset"]
    sample = dataset[0]
    supervised_tokens = int((sample["labels"] != IGNORE_INDEX).sum().item())
    input_tokens = int(sample["input_ids"].numel())
    if supervised_tokens < 1:
        raise RuntimeError("The selected AncientDoc record has no supervised tokens.")
    if input_tokens >= args.model_max_length:
        raise RuntimeError(
            f"Selected record may be truncated: {input_tokens} >= {args.model_max_length}."
        )
    batch = data_module["data_collator"]([sample])
    target = dataset.records[0]["conversations"][1]["value"]
    literal_backslash_n = target.count("\\n")
    actual_newline = target.count(chr(10))
    print("GOT_ANCIENTDOC_DATA_PREFLIGHT_OK")
    print(f"selected_splits={','.join(map(str, split_ids))}")
    print(f"dataset_examples={len(dataset)}")
    print(f"selected_record_split={dataset.record_splits[0]}")
    print(f"selected_image={dataset.records[0]['image']}")
    print(f"target_codepoints={len(target)}")
    print(f"literal_backslash_n={literal_backslash_n}")
    print(f"actual_newline={actual_newline}")
    print(f"input_tokens={input_tokens}")
    print(f"supervised_tokens={supervised_tokens}")
    print(f"image_tensor_shape={tuple(sample['image'][0].shape)}")
    print(f"batch_input_shape={tuple(batch['input_ids'].shape)}")


if __name__ == "__main__":
    main()
