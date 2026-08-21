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
    assert output.layout_evidence.shape == (2, 32, 32)
    assert output.prompt_context is output.layout_evidence
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


def test_integrated_pvld_routes_only_visual_values_and_blocks_teacher_forcing_leakage() -> None:
    project_root = MODULE_PATH.parents[2]
    sys.path.insert(0, str(project_root))
    try:
        adapter = module.PromptedVariableLayoutAdapter(
            visual_dim=16,
            high_resolution_dim=12,
            hidden_size=16,
            num_prompt_queries=32,
            decoder_layers=1,
            num_heads=4,
            max_layout_tokens=32,
            max_layout_records=8,
        )
        visual = torch.randn(2, 9, 16, requires_grad=True)
        high_resolution = torch.randn(2, 25, 12, requires_grad=True)
        vocabulary = module.LayoutVocabulary()
        first = torch.tensor([
            vocabulary.encode(["<LAYOUT>", "<REGION>", "<TYPE>", "REGION", "</TYPE>", "</REGION>", "<EOS>"]),
            vocabulary.encode(["<LAYOUT>", "<EOS>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>"]),
        ])
        second = first.clone()
        second[0, 2] = vocabulary.eos_id
        common = {
            "layout_attention_mask": first.ne(vocabulary.pad_id),
            "layout_region_positions": torch.tensor([[1], [0]]),
            "layout_record_mask": torch.tensor([[True], [False]]),
            "layout_bbox_targets": torch.zeros(2, 1, 4),
            "layout_type_targets": torch.tensor([[2], [-100]]),
            "layout_direction_targets": torch.tensor([[4], [-100]]),
            "layout_count_targets": torch.tensor([1.0, 0.0]),
        }
        output_a = adapter(visual, high_resolution, layout_input_ids=first, **common)
        output_b = adapter(visual, high_resolution, layout_input_ids=second, **common)
        assert output_a.layout_evidence.shape == (2, 32, 16)
        assert torch.equal(output_a.visual_tokens, visual)
        assert torch.equal(output_a.visual_tokens, output_b.visual_tokens)
        adapter.residual_gate.data.fill_(0.5)
        routed = adapter(visual, high_resolution, layout_input_ids=first, **common)
        assert routed.visual_tokens.shape == visual.shape
        assert not torch.equal(routed.visual_tokens, visual)
        assert routed.losses is not None and torch.isfinite(routed.losses.loss)
        (routed.losses.loss + routed.visual_tokens.square().mean()).backward()
        assert adapter.decoder.prompt_attention.prompt_bank.prompts.grad is not None
        assert adapter.visual_routing.visual_value.weight.grad is not None
        assert visual.grad is not None and torch.isfinite(visual.grad).all()
    finally:
        sys.path.remove(str(project_root))
