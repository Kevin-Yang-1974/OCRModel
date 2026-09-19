import pytest
import torch

from layout_ocr import LayoutAdapterConfig, PreMergeLayoutAdapter
from layout_ocr.train_screen import freeze_layout_branch


@pytest.mark.parametrize("mode", ["content_only", "attention", "geometry", "layout_ot"])
def test_adapter_forward_backward(mode: str) -> None:
    torch.manual_seed(7)
    config = LayoutAdapterConfig(hidden_size=16, num_queries=4, num_heads=4, mode=mode)
    module = PreMergeLayoutAdapter(config)
    tokens = torch.randn(2, 9, 16, requires_grad=True)
    positions = torch.rand(2, 9, 2)
    output = module(tokens, positions)

    assert output.merged_tokens.shape == tokens.shape
    assert output.layout_queries.shape == (2, 4, 16)
    assert output.boxes.shape == (2, 4, 4)
    if mode != "content_only":
        assert output.transport is not None
        expected = torch.full((2, 4), 0.25)
        torch.testing.assert_close(output.transport.sum(dim=-1), expected, atol=1e-5, rtol=1e-5)
    else:
        assert output.transport is None
    assert output.validity_logits is None
    assert output.gated_transport is None

    (output.merged_tokens.square().mean() + output.boxes.mean()).backward()
    assert module.query_seed.grad is not None


def test_geometry_requires_patch_positions() -> None:
    module = PreMergeLayoutAdapter(
        LayoutAdapterConfig(hidden_size=8, num_queries=2, num_heads=2, mode="geometry")
    )
    with pytest.raises(ValueError, match="patch_positions"):
        module(torch.randn(1, 4, 8))


def test_zero_gate_preserves_visual_tokens_exactly() -> None:
    torch.manual_seed(11)
    module = PreMergeLayoutAdapter(
        LayoutAdapterConfig(hidden_size=8, num_queries=2, num_heads=2, mode="geometry")
    )
    tokens = torch.randn(1, 4, 8)
    output = module(tokens, torch.rand(1, 4, 2))
    torch.testing.assert_close(output.merged_tokens, tokens, atol=0.0, rtol=0.0)


def test_zero_initialized_sem_adapter_matches_legacy_fusion() -> None:
    torch.manual_seed(13)
    base_config = LayoutAdapterConfig(
        hidden_size=8,
        num_queries=2,
        num_heads=2,
        mode="geometry",
        max_residual_scale=0.03,
        initial_residual_scale=0.01,
    )
    sem_config = LayoutAdapterConfig(
        hidden_size=8,
        num_queries=2,
        num_heads=2,
        mode="geometry",
        max_residual_scale=0.03,
        initial_residual_scale=0.01,
        sem_adapter_mlp=True,
    )
    base = PreMergeLayoutAdapter(base_config).eval()
    semantic = PreMergeLayoutAdapter(sem_config).eval()
    incompatible = semantic.load_state_dict(base.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "sem_adapter.0.weight",
        "sem_adapter.0.bias",
        "sem_adapter.2.weight",
        "sem_adapter.2.bias",
    }

    tokens = torch.randn(1, 5, 8)
    positions = torch.rand(1, 5, 2)
    base_output = base(tokens, positions)
    semantic_output = semantic(tokens, positions)
    torch.testing.assert_close(semantic_output.merged_tokens, base_output.merged_tokens)
    torch.testing.assert_close(semantic_output.layout_queries, base_output.layout_queries)
    torch.testing.assert_close(semantic_output.boxes, base_output.boxes)
    torch.testing.assert_close(semantic_output.transport, base_output.transport)


def test_frozen_layout_branch_updates_only_semantic_adapter_and_gate() -> None:
    torch.manual_seed(17)
    module = PreMergeLayoutAdapter(
        LayoutAdapterConfig(
            hidden_size=8,
            num_queries=2,
            num_heads=2,
            mode="geometry",
            box_head_mlp=True,
            sem_adapter_mlp=True,
            sem_adapter_hidden=4,
            max_residual_scale=0.03,
            initial_residual_scale=0.01,
        )
    )
    freeze_layout_branch(module)
    trainable = {
        name for name, parameter in module.named_parameters() if parameter.requires_grad
    }
    assert trainable == {
        "sem_adapter.0.weight",
        "sem_adapter.0.bias",
        "sem_adapter.2.weight",
        "sem_adapter.2.bias",
        "content_gate",
    }
    before = {
        name: parameter.detach().clone() for name, parameter in module.named_parameters()
    }
    optimizer = torch.optim.SGD(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=0.1,
    )
    output = module(torch.randn(1, 6, 8), torch.rand(1, 6, 2))
    output.merged_tokens.square().mean().backward()
    assert module.sem_adapter is not None
    assert module.sem_adapter[-1].weight.grad is not None
    assert module.query_seed.grad is None
    assert module.geom_adapter is not None
    assert module.geom_adapter[0].weight.grad is None
    optimizer.step()

    assert not torch.equal(module.sem_adapter[-1].weight.detach(), before["sem_adapter.2.weight"])
    for name, parameter in module.named_parameters():
        if not name.startswith("sem_adapter.") and name != "content_gate":
            torch.testing.assert_close(parameter.detach(), before[name])


def test_initial_residual_scale_warm_starts_nonzero_fusion() -> None:
    module = PreMergeLayoutAdapter(
        LayoutAdapterConfig(
            hidden_size=8,
            num_queries=2,
            num_heads=2,
            mode="geometry",
            max_residual_scale=0.03,
            initial_residual_scale=0.01,
        )
    )
    assert float(module.effective_residual_scale().detach()) == pytest.approx(0.01)
    tokens = torch.randn(1, 4, 8)
    output = module(tokens, torch.rand(1, 4, 2))
    assert not torch.equal(output.merged_tokens, tokens)


def test_effective_residual_scale_is_capped() -> None:
    module = PreMergeLayoutAdapter(
        LayoutAdapterConfig(
            hidden_size=8,
            num_queries=2,
            num_heads=2,
            mode="attention",
            max_residual_scale=0.03,
        )
    )
    for raw_gate, expected in ((100.0, 0.03), (-100.0, -0.03), (0.01, None)):
        with torch.no_grad():
            module.content_gate.fill_(raw_gate)
        effective = float(module.effective_residual_scale().detach())
        if expected is None:
            assert abs(effective) < 0.03
        else:
            assert effective == pytest.approx(expected)


def test_legacy_config_keeps_original_tanh_gate_semantics() -> None:
    module = PreMergeLayoutAdapter(
        LayoutAdapterConfig(hidden_size=8, num_queries=2, num_heads=2, mode="attention")
    )
    with torch.no_grad():
        module.content_gate.fill_(2.0)
    assert float(module.effective_residual_scale().detach()) == pytest.approx(
        float(torch.tanh(torch.tensor(2.0)))
    )


def test_validity_head_initializes_and_gates_transport() -> None:
    module = PreMergeLayoutAdapter(
        LayoutAdapterConfig(
            hidden_size=8,
            num_queries=4,
            num_heads=2,
            mode="geometry",
            use_validity_head=True,
            initial_valid_probability=0.05,
        )
    )
    output = module(torch.randn(1, 6, 8), torch.rand(1, 6, 2))
    assert output.validity_logits is not None
    assert output.validity_probs is not None
    assert output.gated_transport is not None
    assert output.valid_coverage is not None
    torch.testing.assert_close(
        output.validity_probs,
        torch.full_like(output.validity_probs, 0.05),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        output.valid_coverage,
        torch.full_like(output.valid_coverage, 0.05),
        atol=1e-5,
        rtol=1e-5,
    )


def test_raw_mass_validity_zero_probability_removes_query_contribution() -> None:
    module = PreMergeLayoutAdapter(
        LayoutAdapterConfig(
            hidden_size=8,
            num_queries=2,
            num_heads=2,
            mode="attention",
            use_validity_head=True,
            validity_gating_mode="raw_mass",
        )
    )
    transport = torch.tensor([[[0.25, 0.25], [0.25, 0.25]]])
    queries = torch.randn(1, 2, 8)
    probabilities = torch.tensor([[1.0, 0.0]])
    context, coverage = module._raw_mass_context(transport, queries, probabilities)
    query_zeroed = queries.clone()
    query_zeroed[:, 1] = 0.0
    expected, expected_coverage = module._raw_mass_context(
        transport, query_zeroed, probabilities
    )
    torch.testing.assert_close(context, expected, atol=0.0, rtol=0.0)
    torch.testing.assert_close(coverage, expected_coverage, atol=0.0, rtol=0.0)
    torch.testing.assert_close(coverage, torch.full_like(coverage, 0.5))
