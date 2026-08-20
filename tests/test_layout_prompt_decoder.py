from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "GOT-OCR-2.0"
    / "GOT"
    / "model"
    / "layout_prompt_decoder.py"
)
SPEC = importlib.util.spec_from_file_location("layout_prompt_decoder_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_prompt_bank_is_global_not_region_indexed() -> None:
    bank = module.LayoutPromptBank(hidden_size=16, num_prompts=32)
    prompts = bank(3)
    assert prompts.shape == (3, 32, 16)
    assert bank.prompts.shape == (1, 32, 16)


def test_decoder_supports_batch_padding_and_full_visual_memory() -> None:
    vocabulary = module.LayoutVocabulary()
    decoder = module.VariableLayoutDecoder(
        vocab_size=vocabulary.vocab_size,
        hidden_size=32,
        visual_size=16,
        num_prompts=32,
        num_layers=1,
        num_heads=4,
        max_layout_tokens=32,
    )
    visual = torch.randn(2, 25, 16)
    input_ids = torch.tensor(
        [
            vocabulary.encode(["<LAYOUT>", "<EOS>", "<PAD>"]),
            vocabulary.encode(["<LAYOUT>", "<REGION>", "<EOS>"]),
        ]
    )
    output = decoder(input_ids, visual)
    assert output.logits.shape == (2, 3, vocabulary.vocab_size)
    assert output.prompt_context.shape == (2, 32, 32)
    assert output.hidden_states.shape == (2, 3, 32)


def test_record_heads_bound_bbox_and_loss_backpropagates() -> None:
    vocabulary = module.LayoutVocabulary()
    decoder = module.VariableLayoutDecoder(
        vocab_size=vocabulary.vocab_size,
        hidden_size=32,
        visual_size=16,
        num_prompts=32,
        num_layers=1,
        num_heads=4,
        max_layout_tokens=16,
    )
    visual = torch.randn(2, 9, 16)
    target_ids = torch.tensor(
        [vocabulary.encode(["<LAYOUT>", "<REGION>", "<EOS>"]), vocabulary.encode(["<LAYOUT>", "<EOS>", "<PAD>"])]
    )
    output = decoder(target_ids, visual)
    heads = module.LayoutRecordHeads(hidden_size=32)
    records = heads(output.hidden_states[:, :2])
    assert torch.all((records.bbox >= 0) & (records.bbox <= 1))
    criterion = module.VariableLayoutLoss()
    loss = criterion(
        output,
        target_ids,
        records,
        torch.zeros_like(records.bbox),
        torch.zeros((2, 2), dtype=torch.long),
        torch.zeros((2, 2), dtype=torch.long),
        torch.ones((2, 2), dtype=torch.bool),
        torch.tensor([1.0, 0.0]),
        vocabulary.pad_id,
    )
    assert torch.isfinite(loss.loss)
    loss.loss.backward()
    assert decoder.prompt_attention.prompt_bank.prompts.grad is not None


def test_variable_generation_can_finish_with_eos_and_marks_truncation() -> None:
    vocabulary = module.LayoutVocabulary()
    decoder = module.VariableLayoutDecoder(
        vocab_size=vocabulary.vocab_size,
        hidden_size=16,
        visual_size=16,
        num_prompts=32,
        num_layers=1,
        num_heads=4,
        max_layout_tokens=8,
    )
    visual = torch.randn(2, 4, 16)
    output = decoder.generate(
        visual,
        layout_id=vocabulary.layout_id,
        region_id=vocabulary.region_id,
        eos_id=vocabulary.eos_id,
        pad_id=vocabulary.pad_id,
        max_layout_tokens=4,
    )
    assert output.hidden_states.shape[0] == 2
    assert len(output.region_positions or []) == 2
    assert output.generated_eos is not None
    assert output.truncated is not None
    assert output.generated_ids is not None
    assert output.generated_ids.shape[0] == 2
