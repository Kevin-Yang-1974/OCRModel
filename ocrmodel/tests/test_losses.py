import torch

from layout_ocr import (
    LayoutAdapterConfig,
    PreMergeLayoutAdapter,
    compute_layout_losses,
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
    }
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
