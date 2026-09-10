import torch

from layout_ocr import (
    LayoutAdapterConfig,
    LayoutAdapterOutput,
    PreMergeLayoutAdapter,
    compute_layout_losses,
    match_layout_targets,
)
from layout_ocr.config import layout_loss_config
from layout_ocr.losses import _validity_bce
from layout_ocr.train_screen import (
    matcher_churn_between_points,
    transport_diagnostics,
    validity_mechanism_acceptance,
)


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
        "layout_validity_bce",
        "layout_validity_cardinality",
        "layout_validity_ranking",
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
    fixed = layout_loss_config("validity_assignment")
    assert fixed.assignment == 0.25
    assert fixed.validity == 1.0
    assert fixed.validity_cardinality == 0.5
    assert fixed.validity_ranking == 0.1
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


def test_validity_bce_constant_logit_follows_page_prior() -> None:
    logits = torch.zeros(1, 8, requires_grad=True)
    query_mask = torch.tensor([[True, True, False, False, False, False, False, False]])
    loss = _validity_bce(logits, query_mask)
    loss.backward()
    # At p=0.5, positives push the logit up and the majority no-object class
    # pushes it down; the aggregate gradient is positive, so optimization
    # moves a constant predictor toward the actual valid prior.
    assert logits.grad is not None
    assert logits.grad[0, 0] < 0
    assert logits.grad[0, -1] > 0
    assert logits.grad.mean() > 0


def test_hungarian_matching_uses_detached_region_support() -> None:
    output = LayoutAdapterOutput(
        merged_tokens=torch.zeros(1, 2, 8),
        layout_queries=torch.zeros(1, 2, 8),
        boxes=torch.tensor(
            [[[0.1, 0.1, 0.9, 0.9], [0.1, 0.1, 0.9, 0.9]]]
        ),
        order_scores=torch.zeros(1, 2),
        direction_logits=torch.zeros(1, 2, 3),
        transport=torch.tensor([[[0.49, 0.01], [0.01, 0.49]]]),
    )
    targets = {
        "target_boxes": torch.tensor(
            [[[0.1, 0.1, 0.9, 0.9], [0.0, 0.0, 0.0, 0.0]]]
        ),
        "target_orders": torch.tensor([[0.0, 0.0]]),
        "target_directions": torch.tensor([[0, 0]]),
        "query_mask": torch.tensor([[True, False]]),
        "token_owners": torch.tensor([[0, -1]]),
    }
    matched, info = match_layout_targets(
        output, targets, assignment="hungarian", return_info=True
    )
    assert matched["query_mask"].tolist() == [[True, False]]
    assert info["matched_query_indices"] == [[0]]


def test_validity_assignment_injects_positive_and_negative_query_gradients() -> None:
    module = PreMergeLayoutAdapter(
        LayoutAdapterConfig(
            hidden_size=8,
            num_queries=4,
            num_heads=2,
            mode="geometry",
            use_validity_head=True,
            validity_gating_mode="raw_mass",
            validity_use_transport_evidence=True,
        )
    )
    output = module(torch.randn(1, 5, 8), torch.rand(1, 5, 2))
    losses = compute_layout_losses(
        output,
        target_boxes=torch.rand(1, 4, 4),
        target_orders=torch.rand(1, 4),
        target_directions=torch.tensor([[0, 1, 2, 0]]),
        query_mask=torch.tensor([[True, False, False, True]]),
        token_owners=torch.tensor([[0, 0, 3, -1, 3]]),
        weights=layout_loss_config("validity_assignment"),
    )
    assert torch.isfinite(losses["loss"])
    assert losses["layout_assignment"].item() > 0.0
    assert losses["layout_validity_cardinality"].item() >= 0.0
    assert losses["layout_validity_ranking"].item() > 0.0
    assert output.validity_logits is not None
    assignment_gradient = torch.autograd.grad(
        losses["layout_assignment"], output.validity_logits, retain_graph=True
    )[0]
    assert assignment_gradient[0, 0].abs() > 0
    assert assignment_gradient[0, 1].abs() > 0


def test_validity_acceptance_and_matcher_churn_are_validation_only() -> None:
    def point(step: int, signature: str) -> dict:
        return {
            "step": step,
            "validation": {
                "validity_p_gap": 0.2,
                "validity_auroc": 0.9,
                "mean_p_valid_no_object": 0.05,
                "mean_p_valid_matched": 0.5,
                "invalid_gated_context_share": 0.2,
                "residual_relative_norm": 0.005,
                "matcher_signatures": {"page": signature},
            },
        }

    points = [point(128, "a"), point(256, "b")]
    acceptance = validity_mechanism_acceptance(points)
    assert acceptance["available"] is True
    assert acceptance["passed"] is True
    assert matcher_churn_between_points(points) == 1.0
