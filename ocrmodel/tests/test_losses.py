import torch

from layout_ocr import (
    LayoutAdapterConfig,
    LayoutAdapterOutput,
    PreMergeLayoutAdapter,
    compute_layout_losses,
    match_layout_targets,
)
from layout_ocr.config import layout_loss_config
from layout_ocr.train_screen import transport_diagnostics


def test_auxiliary_losses_are_finite() -> None:
    module = PreMergeLayoutAdapter(
        LayoutAdapterConfig(hidden_size=8, num_queries=3, num_heads=2, mode="layout_ot")
    )
    output = module(torch.randn(2, 6, 8), torch.rand(2, 6, 2))
    losses = compute_layout_losses(
        output,
        target_boxes=torch.rand(2, 3, 4),
        target_orders=torch.rand(2, 3),
        target_directions=torch.tensor([[0, 1, 2], [1, 0, 0]]),
        query_mask=torch.tensor([[True, True, True], [True, True, False]]),
        token_owners=torch.tensor([[0, 0, 1, 1, 2, -1], [0, 1, 1, 2, -1, -1]]),
    )
    assert set(losses) == {
        "loss",
        "layout_box",
        "layout_order",
        "layout_direction",
        "layout_assignment",
        "transport_entropy",
        "layout_validity",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()


def test_transport_diagnostics_reports_unused_query_mass() -> None:
    module = PreMergeLayoutAdapter(
        LayoutAdapterConfig(hidden_size=8, num_queries=3, num_heads=2, mode="attention")
    )
    output = module(torch.randn(1, 6, 8), torch.rand(1, 6, 2))
    diagnostics = transport_diagnostics(
        output,
        torch.tensor([[True, False, False]]),
    )
    assert diagnostics["transport_query_mass"] is None
    assert diagnostics["fusion_query_mass"] is None
    assert diagnostics["invalid_query_transport_mass"] >= 0.0
    assert diagnostics["valid_query_transport_mass"] >= 0.0
    assert diagnostics["invalid_query_fusion_mass"] >= 0.0
    assert diagnostics["valid_query_fusion_mass"] >= 0.0


def test_hungarian_matching_remaps_targets_and_token_owners() -> None:
    output = LayoutAdapterOutput(
        merged_tokens=torch.zeros(1, 4, 8),
        layout_queries=torch.zeros(1, 3, 8),
        boxes=torch.tensor(
            [[[0.80, 0.80, 0.90, 0.90], [0.10, 0.10, 0.20, 0.20], [0.50, 0.50, 0.60, 0.60]]]
        ),
        order_scores=torch.tensor([[1.0, -1.0, 0.0]]),
        direction_logits=torch.zeros(1, 3, 3),
        transport=torch.full((1, 3, 4), 1.0 / 12.0),
    )
    targets = {
        "target_boxes": torch.tensor(
            [[[0.10, 0.10, 0.20, 0.20], [0.80, 0.80, 0.90, 0.90], [0.0, 0.0, 0.0, 0.0]]]
        ),
        "target_orders": torch.tensor([[0.0, 1.0, 0.0]]),
        "target_directions": torch.tensor([[1, 2, 0]]),
        "query_mask": torch.tensor([[True, True, False]]),
        "token_owners": torch.tensor([[0, 0, 1, -1]]),
    }

    matched = match_layout_targets(output, targets, assignment="hungarian")

    assert matched["query_mask"].tolist() == [[True, True, False]]
    torch.testing.assert_close(
        matched["target_boxes"][0, 0], targets["target_boxes"][0, 1]
    )
    torch.testing.assert_close(
        matched["target_boxes"][0, 1], targets["target_boxes"][0, 0]
    )
    torch.testing.assert_close(
        matched["target_boxes"][0, 2], torch.zeros(4)
    )
    assert matched["target_directions"].tolist() == [[2, 1, 0]]
    assert matched["token_owners"].tolist() == [[1, 1, 0, -1]]


def test_layout_loss_profiles_are_explicit() -> None:
    assert layout_loss_config("full").assignment == 1.0
    assert layout_loss_config("ocr_only").box == 0.0
    assert layout_loss_config("no_assignment").assignment == 0.0
    validity = layout_loss_config("no_assignment_validity")
    assert validity.assignment == 0.0
    assert validity.validity == 0.5
    assert layout_loss_config("no_geometry").box == 0.0


def test_validity_loss_uses_hungarian_query_mask() -> None:
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
    output = module(torch.randn(1, 5, 8), torch.rand(1, 5, 2))
    assert output.validity_logits is not None
    assert output.validity_probs is not None
    assert output.gated_transport is not None
    assert output.valid_coverage is not None
    assert float(output.validity_probs.mean()) == torch.sigmoid(
        torch.tensor(torch.logit(torch.tensor(0.05)))
    ).item()
    losses = compute_layout_losses(
        output,
        target_boxes=torch.rand(1, 4, 4),
        target_orders=torch.rand(1, 4),
        target_directions=torch.tensor([[0, 1, 2, 0]]),
        query_mask=torch.tensor([[True, False, False, True]]),
        token_owners=torch.full((1, 5), -1, dtype=torch.long),
        weights=layout_loss_config("no_assignment_validity"),
    )
    assert losses["layout_validity"].item() > 0.0
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert module.validity_head is not None
    assert module.validity_head.weight.grad is not None
