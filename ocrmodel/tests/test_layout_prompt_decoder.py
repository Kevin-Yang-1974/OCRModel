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
            vocabulary.encode(["<LAYOUT>", "<EOS>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>"]),
            vocabulary.encode(["<LAYOUT>", "<REGION>", "<TYPE>", "REGION", "</TYPE>", "</REGION>", "<EOS>"]),
        ]
    )
    output = decoder(input_ids, visual, target_padding_mask=input_ids.eq(vocabulary.pad_id))
    assert output.logits.shape == (2, 7, vocabulary.vocab_size)
    assert output.layout_evidence.shape == (2, 32, 32)
    assert output.prompt_context is output.layout_evidence
    assert output.hidden_states.shape == (2, 7, 32)
    assert output.coverage_region_counts.tolist() == [0, 1]


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
    target_ids = torch.tensor([
        vocabulary.encode(["<LAYOUT>", "<REGION>", "<TYPE>", "REGION", "</TYPE>", "</REGION>", "<EOS>"]),
        vocabulary.encode(["<LAYOUT>", "<EOS>", "<PAD>", "<PAD>", "<PAD>", "<PAD>", "<PAD>"]),
    ])
    output = decoder(target_ids, visual, target_padding_mask=target_ids.eq(vocabulary.pad_id))
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


def complete_record(vocabulary: module.LayoutVocabulary, type_name: str = "REGION") -> list[int]:
    return vocabulary.encode(
        ["<REGION>", "<TYPE>", type_name, "</TYPE>", "</REGION>"]
    )


def force_region_until_cap(decoder, vocabulary) -> None:
    with torch.no_grad():
        decoder.token_head.weight.zero_()
        decoder.token_head.bias.zero_()
        decoder.token_head.bias[vocabulary.region_id] = 4.0
        decoder.token_head.bias[vocabulary.eos_id] = 1.0
        decoder.token_head.bias[vocabulary.token_to_id["REGION"]] = 0.5


def test_fsm_allows_only_real_serializer_transitions() -> None:
    vocabulary = module.LayoutVocabulary()
    fsm = module.LayoutTokenFSM(vocabulary)
    legal = torch.tensor([
        vocabulary.encode(["<LAYOUT>", "<EOS>", "<PAD>"]),
        vocabulary.encode(["<LAYOUT>", "<REGION>", "<TYPE>"]),
    ])
    logits = torch.zeros(2, 3, vocabulary.vocab_size)
    masked = fsm.mask_logits(logits, legal)
    assert torch.isfinite(masked[0, 0, vocabulary.region_id])
    assert torch.isfinite(masked[0, 0, vocabulary.eos_id])
    assert torch.isneginf(masked[0, 0, vocabulary.type_id])
    assert torch.isfinite(masked[0, 1, vocabulary.pad_id])
    assert torch.isneginf(masked[0, 1, vocabulary.region_id])
    assert torch.isfinite(masked[1, 1, vocabulary.type_id])
    assert set(torch.isfinite(masked[1, 2]).nonzero().flatten().tolist()) == set(
        fsm.type_value_ids
    )
    with pytest.raises(ValueError, match="Illegal layout token"):
        fsm.mask_logits(
            torch.zeros(1, 2, vocabulary.vocab_size),
            torch.tensor([vocabulary.encode(["<LAYOUT>", "<TYPE>"])]),
        )


def test_causal_decoder_blocks_future_leakage_and_uses_earlier_history() -> None:
    torch.manual_seed(3)
    vocabulary = module.LayoutVocabulary()
    decoder = module.VariableLayoutDecoder(
        vocabulary.vocab_size, 24, 12, num_layers=2, num_heads=4,
        max_layout_tokens=16, vocabulary=vocabulary,
    ).eval()
    visual = torch.randn(1, 7, 12)
    first = torch.tensor([vocabulary.encode(
        ["<LAYOUT>", "<REGION>", "<TYPE>", "COLUMN", "</TYPE>", "</REGION>", "<EOS>"]
    )])
    second = first.clone()
    second[0, 3] = vocabulary.token_to_id["ROW"]
    output_a = decoder(first, visual)
    output_b = decoder(second, visual)
    assert torch.equal(output_a.logits[:, :3], output_b.logits[:, :3])
    assert not torch.equal(output_a.logits[:, 4:], output_b.logits[:, 4:])
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "running_sum" not in source
    assert "target.cumsum" not in source
    assert isinstance(decoder.decoder_blocks[0].self_attention, torch.nn.MultiheadAttention)


def test_generation_zero_one_multiple_regions_and_cap_statuses() -> None:
    vocabulary = module.LayoutVocabulary()
    decoder = module.VariableLayoutDecoder(
        vocabulary.vocab_size, 16, 16, num_layers=1, num_heads=4,
        max_layout_tokens=32, vocabulary=vocabulary,
    )
    visual = torch.randn(1, 4, 16)
    with torch.no_grad():
        decoder.token_head.weight.zero_()
        decoder.token_head.bias.zero_()
        decoder.token_head.bias[vocabulary.eos_id] = 4.0
    zero = decoder.generate(
        visual, layout_id=vocabulary.layout_id, region_id=vocabulary.region_id,
        eos_id=vocabulary.eos_id, pad_id=vocabulary.pad_id,
        max_layout_records=4,
    )
    assert zero.num_generated_regions.item() == 0
    assert zero.generated_ids[0, :2].tolist() == vocabulary.encode(["<LAYOUT>", "<EOS>"])
    force_region_until_cap(decoder, vocabulary)
    one = decoder.generate(
        visual, layout_id=vocabulary.layout_id, region_id=vocabulary.region_id,
        eos_id=vocabulary.eos_id, pad_id=vocabulary.pad_id,
        max_layout_records=1,
    )
    assert one.num_generated_regions.item() == 1
    assert one.generated_eos.item() and one.stopped_by_max_layout_records.item()
    assert not one.truncated_by_max_layout_tokens.item()
    assert one.generated_ids[0].tolist() == (
        vocabulary.encode(["<LAYOUT>"]) + complete_record(vocabulary) + vocabulary.encode(["<EOS>"])
    )
    multiple = decoder.generate(
        visual, layout_id=vocabulary.layout_id, region_id=vocabulary.region_id,
        eos_id=vocabulary.eos_id, pad_id=vocabulary.pad_id,
        max_layout_records=3,
    )
    assert multiple.num_generated_regions.item() == 3
    assert multiple.region_token_probabilities.shape == (1, 3)
    assert torch.all((multiple.region_token_probabilities >= 0) & (multiple.region_token_probabilities <= 1))
    assert multiple.coverage_region_counts.item() == 3


def test_token_cap_is_distinct_from_record_cap() -> None:
    vocabulary = module.LayoutVocabulary()
    decoder = module.VariableLayoutDecoder(
        vocabulary.vocab_size, 16, 16, num_layers=1, num_heads=4,
        max_layout_tokens=16, vocabulary=vocabulary,
    )
    force_region_until_cap(decoder, vocabulary)
    output = decoder.generate(
        torch.randn(1, 4, 16), layout_id=vocabulary.layout_id,
        region_id=vocabulary.region_id, eos_id=vocabulary.eos_id,
        pad_id=vocabulary.pad_id, max_layout_tokens=3, max_layout_records=8,
    )
    assert output.truncated_by_max_layout_tokens.item()
    assert not output.stopped_by_max_layout_records.item()
    assert not output.generated_eos.item()


def test_region_confidence_comes_from_step_logits_and_can_vary() -> None:
    torch.manual_seed(11)
    vocabulary = module.LayoutVocabulary()
    decoder = module.VariableLayoutDecoder(
        vocabulary.vocab_size, 16, 16, num_layers=1, num_heads=4,
        max_layout_tokens=24, vocabulary=vocabulary,
    )
    with torch.no_grad():
        decoder.token_head.weight.zero_()
        decoder.token_head.weight[vocabulary.region_id, 0] = 0.25
        decoder.token_head.bias.zero_()
        decoder.token_head.bias[vocabulary.region_id] = 3.0
        decoder.token_head.bias[vocabulary.eos_id] = 0.0
    output = decoder.generate(
        torch.randn(1, 6, 16), layout_id=vocabulary.layout_id,
        region_id=vocabulary.region_id, eos_id=vocabulary.eos_id,
        pad_id=vocabulary.pad_id, max_layout_records=2,
    )
    probabilities = output.region_token_probabilities[0]
    assert probabilities.shape == (2,)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    assert not torch.isclose(probabilities[0], probabilities[1])


def test_previous_region_coverage_changes_later_routing() -> None:
    torch.manual_seed(7)
    vocabulary = module.LayoutVocabulary()
    decoder = module.VariableLayoutDecoder(
        vocabulary.vocab_size, 16, 16, num_layers=1, num_heads=4,
        max_layout_tokens=16, vocabulary=vocabulary,
    ).eval()
    ids = torch.tensor([vocabulary.encode(
        ["<LAYOUT>", "<REGION>", "<TYPE>", "REGION", "</TYPE>", "</REGION>", "<EOS>"]
    )])
    visual = torch.randn(1, 5, 16)
    with_coverage = decoder(ids, visual)
    weight = decoder.decoder_blocks[0].coverage_projection.weight.detach().clone()
    with torch.no_grad():
        decoder.decoder_blocks[0].coverage_projection.weight.zero_()
    without_coverage = decoder(ids, visual)
    assert with_coverage.coverage_region_counts.item() == 1
    assert not torch.equal(with_coverage.logits[:, 2:], without_coverage.logits[:, 2:])
    with torch.no_grad():
        decoder.decoder_blocks[0].coverage_projection.weight.copy_(weight)


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
        second[0, 3] = vocabulary.token_to_id["ROW"]
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
