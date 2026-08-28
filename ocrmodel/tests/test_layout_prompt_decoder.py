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


def test_boundary_loss_is_page_balanced_and_supports_zero_regions() -> None:
    vocabulary = module.LayoutVocabulary()
    target_ids = torch.tensor([
        vocabulary.encode([
            "<LAYOUT>", "<REGION>", "<TYPE>", "REGION", "</TYPE>",
            "</REGION>", "<REGION>", "<TYPE>", "ROW", "</TYPE>",
            "</REGION>", "<EOS>",
        ]),
        vocabulary.encode(["<LAYOUT>", "<EOS>"]) + [vocabulary.pad_id] * 10,
    ])
    logits = torch.randn(2, 12, vocabulary.vocab_size, requires_grad=True)
    output = module.VariableLayoutOutput(
        logits=logits,
        hidden_states=torch.zeros(2, 12, 8),
        layout_evidence=torch.zeros(2, 4, 8),
    )
    records = module.LayoutRecordOutput(
        bbox=torch.full((2, 2, 4), 0.5),
        type_logits=torch.zeros(2, 2, 5),
        direction_logits=torch.zeros(2, 2, 5),
        count=torch.tensor([2.0, 0.0]),
    )
    criterion = module.VariableLayoutLoss(boundary_weight=1.0)
    losses = criterion(
        output, target_ids, records, torch.full((2, 2, 4), 0.5),
        torch.full((2, 2), -100, dtype=torch.long),
        torch.full((2, 2), -100, dtype=torch.long),
        torch.tensor([[True, True], [False, False]]),
        torch.tensor([2.0, 0.0]), vocabulary.pad_id,
        vocabulary.region_id, vocabulary.eos_id,
    )
    log_probabilities = logits[:, :-1].float().log_softmax(dim=-1)
    shifted = target_ids[:, 1:]
    page0_region = -log_probabilities[0, shifted[0].eq(vocabulary.region_id), vocabulary.region_id].mean()
    page0_eos = -log_probabilities[0, shifted[0].eq(vocabulary.eos_id), vocabulary.eos_id].mean()
    page1_eos = -log_probabilities[1, shifted[1].eq(vocabulary.eos_id), vocabulary.eos_id].mean()
    expected = (0.5 * (page0_region + page0_eos) + page1_eos) / 2
    assert torch.allclose(losses.boundary_loss, expected)
    assert torch.isfinite(losses.loss)
    losses.loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_count_condition_zero_is_exact_identity_and_prior_biases_boundaries() -> None:
    vocabulary = module.LayoutVocabulary()
    decoder = module.VariableLayoutDecoder(
        vocabulary.vocab_size, 16, 16, num_layers=1, num_heads=4,
        max_layout_tokens=16, vocabulary=vocabulary,
        count_condition_strength=0.0,
    )
    ids = torch.tensor([[vocabulary.layout_id]])
    logits = torch.zeros(1, 1, vocabulary.vocab_size)
    unchanged = decoder._apply_count_condition(logits, ids, torch.tensor([3.0]))
    assert unchanged is logits

    decoder.count_condition_strength = 1.0
    before_count = decoder._apply_count_condition(logits, ids, torch.tensor([2.0]))
    at_count = decoder._apply_count_condition(logits, ids, torch.tensor([0.0]))
    assert before_count[0, 0, vocabulary.region_id] > before_count[0, 0, vocabulary.eos_id]
    assert at_count[0, 0, vocabulary.eos_id] > at_count[0, 0, vocabulary.region_id]


def test_count_condition_has_finite_nonzero_count_prior_gradient() -> None:
    vocabulary = module.LayoutVocabulary()
    decoder = module.VariableLayoutDecoder(
        vocabulary.vocab_size, 16, 16, num_layers=1, num_heads=4,
        max_layout_tokens=16, vocabulary=vocabulary,
        count_condition_strength=1.0,
    )
    ids = torch.tensor([vocabulary.encode([
        "<LAYOUT>", "<REGION>", "<TYPE>", "REGION", "</TYPE>",
        "</REGION>", "<EOS>",
    ])])
    count_prior = torch.tensor([1.5], requires_grad=True)
    output = decoder(ids, torch.randn(1, 5, 16), count_prior=count_prior)
    loss = torch.nn.functional.cross_entropy(
        output.logits[:, :-1].reshape(-1, vocabulary.vocab_size),
        ids[:, 1:].reshape(-1),
    )
    loss.backward()
    assert count_prior.grad is not None and torch.isfinite(count_prior.grad).all()
    assert count_prior.grad.abs().sum() > 0


def test_generation_reuses_forward_count_condition_path() -> None:
    vocabulary = module.LayoutVocabulary()
    decoder = module.VariableLayoutDecoder(
        vocabulary.vocab_size, 16, 16, num_layers=1, num_heads=4,
        max_layout_tokens=8, vocabulary=vocabulary,
        count_condition_strength=1.0,
    )
    observed = []
    original_forward = decoder.forward

    def recording_forward(*args, **kwargs):
        observed.append(kwargs.get("count_prior"))
        return original_forward(*args, **kwargs)

    decoder.forward = recording_forward
    prior = torch.tensor([0.0])
    decoder.generate(
        torch.randn(1, 4, 16), layout_id=vocabulary.layout_id,
        region_id=vocabulary.region_id, eos_id=vocabulary.eos_id,
        pad_id=vocabulary.pad_id, count_prior=prior,
    )
    assert observed and all(value is prior for value in observed)


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


def test_m2_spatial_memory_appends_projected_visual_tokens_and_detaches_coverage() -> None:
    vocabulary = module.LayoutVocabulary()
    decoder = module.VariableLayoutDecoder(
        vocabulary.vocab_size, 16, 12, num_layers=1, num_heads=4,
        max_layout_tokens=16, vocabulary=vocabulary,
        use_spatial_memory=True, coverage_detach=True,
    )
    ids = torch.tensor([vocabulary.encode([
        "<LAYOUT>", "<REGION>", "<TYPE>", "REGION", "</TYPE>", "</REGION>", "<EOS>"
    ])])
    output = decoder(ids, torch.randn(1, 25, 12, requires_grad=True))
    assert output.spatial_coverage is not None
    assert output.spatial_coverage.shape == (1, 25)
    assert not output.spatial_coverage.requires_grad
    assert decoder.spatial_memory_projection.weight.numel() == 16 * 12 + 16


def test_m3_straight_through_scale_preserves_forward_and_scales_gradient() -> None:
    value = torch.tensor([2.0], requires_grad=True)
    scaled = module.straight_through_scale(value, 0.25)
    assert torch.equal(scaled.detach(), value.detach())
    scaled.sum().backward()
    assert torch.allclose(value.grad, torch.tensor([0.25]))


def test_m4_predicted_layout_routing_is_visual_value_only_and_supports_shuffle() -> None:
    adapter = module.PromptedVariableLayoutAdapter(
        visual_dim=16, high_resolution_dim=12, hidden_size=16,
        num_prompt_queries=4, decoder_layers=1, num_heads=4,
        max_layout_tokens=10, max_layout_records=2,
        predicted_layout_routing=True,
    ).eval()
    with torch.no_grad():
        adapter.decoder.token_head.weight.zero_()
        adapter.decoder.token_head.bias.zero_()
        adapter.decoder.token_head.bias[adapter.vocabulary.region_id] = 5.0
        adapter.decoder.token_head.bias[adapter.vocabulary.type_id] = 4.0
        adapter.decoder.token_head.bias[adapter.vocabulary.token_to_id["REGION"]] = 3.0
        adapter.decoder.token_head.bias[adapter.vocabulary.end_type_id] = 2.0
        adapter.decoder.token_head.bias[adapter.vocabulary.end_region_id] = 1.0
        adapter.decoder.token_head.bias[adapter.vocabulary.eos_id] = 0.5
        adapter.residual_gate.fill_(0.5)
    visual = torch.randn(2, 5, 16)
    high = torch.randn(2, 9, 12)
    with torch.no_grad():
        identity = adapter(visual, high, shuffle_predicted_layout=False)
        adapter.residual_gate.zero_()
        alpha_zero = adapter(visual, high, shuffle_predicted_layout=False)
        adapter.residual_gate.fill_(0.5)
    normal = adapter(visual, high)
    shuffled = adapter(visual, high, shuffle_predicted_layout=True)
    assert normal.predicted_layout_condition is not None
    assert normal.routing_reliability is not None
    assert normal.visual_tokens.shape == visual.shape
    assert torch.equal(alpha_zero.visual_tokens, visual)
    assert not torch.equal(normal.visual_tokens, visual)
    assert not torch.equal(normal.predicted_layout_condition, shuffled.predicted_layout_condition)
    assert adapter.visual_routing.visual_value.weight.grad is None


def test_m4_routing_casts_float_reliability_to_bfloat16_writeback() -> None:
    adapter = module.PromptedVariableLayoutAdapter(
        visual_dim=16, high_resolution_dim=12, hidden_size=16,
        num_prompt_queries=4, decoder_layers=1, num_heads=4,
        max_layout_tokens=10, max_layout_records=2,
        predicted_layout_routing=True,
    ).to(dtype=torch.bfloat16).eval()
    with torch.no_grad():
        adapter.decoder.token_head.weight.zero_()
        adapter.decoder.token_head.bias.zero_()
        adapter.decoder.token_head.bias[adapter.vocabulary.eos_id] = 5.0
        adapter.residual_gate.fill_(0.5)
    visual = torch.randn(2, 5, 16, dtype=torch.bfloat16)
    high = torch.randn(2, 9, 12, dtype=torch.bfloat16)
    output = adapter(visual, high)
    assert output.visual_tokens.dtype == torch.bfloat16
    assert output.routing_reliability is not None


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
