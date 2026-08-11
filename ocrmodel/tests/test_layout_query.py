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
    / "layout_query.py"
)
SPEC = importlib.util.spec_from_file_location("layout_query_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
layout_query = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = layout_query
SPEC.loader.exec_module(layout_query)


def test_zero_gate_preserves_visual_tokens_and_shapes() -> None:
    adapter = layout_query.VisualLayoutQueryAdapter(
        visual_dim=64,
        layout_input_dim=64,
        adapter_dim=32,
        num_queries=4,
        num_heads=4,
        ffn_expansion=2,
        num_direction_classes=5,
    )
    visual = torch.randn(2, 16, 64)
    output = adapter(visual, memory_grid_size=(4, 4), return_attention=True)
    assert torch.equal(output.visual_tokens, visual)
    assert output.layout_queries.shape == (2, 4, 32)
    assert output.prediction_queries.shape == (2, 4, 32)
    assert output.object_logits.shape == (2, 4)
    assert output.bbox_logits.shape == (2, 4, 4)
    assert output.bbox_cxcywh.shape == (2, 4, 4)
    assert output.direction_logits.shape == (2, 4, 5)
    assert output.query_attention.shape == (2, 4, 4, 16)
    assert output.writeback_attention.shape == (2, 4, 16, 4)


def test_explicit_reset_is_complete_and_initial_logits_are_bounded() -> None:
    adapter = layout_query.VisualLayoutQueryAdapter(
        visual_dim=64,
        layout_input_dim=64,
        adapter_dim=32,
        num_queries=4,
        num_heads=4,
        ffn_expansion=2,
        num_direction_classes=5,
    )
    with torch.no_grad():
        for parameter in adapter.parameters():
            parameter.fill_(float("nan"))
    adapter.reset_parameters()
    assert all(torch.isfinite(parameter).all() for parameter in adapter.parameters())

    visual = torch.randn(2, 16, 64) * 1_000_000.0
    output = adapter(visual, memory_grid_size=(4, 4))
    assert torch.isfinite(output.layout_queries).all()
    assert torch.isfinite(output.prediction_queries).all()
    assert output.object_logits.detach().abs().max().item() < 10.0
    assert output.direction_logits.detach().abs().max().item() < 10.0
    assert output.bbox_logits.detach().abs().max().item() < 10.0


def test_layout_loss_is_finite_and_backpropagates() -> None:
    adapter = layout_query.VisualLayoutQueryAdapter(
        visual_dim=32,
        layout_input_dim=32,
        adapter_dim=16,
        num_queries=3,
        num_heads=4,
        ffn_expansion=2,
    )
    visual = torch.randn(2, 9, 32)
    output = adapter(visual, memory_grid_size=(3, 3))
    bbox_targets = torch.tensor(
        [
            [[0.1, 0.1, 0.3, 0.8], [0.4, 0.1, 0.6, 0.8], [0.0, 0.0, 0.0, 0.0]],
            [[0.2, 0.2, 0.7, 0.4], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    bbox_mask = torch.tensor([[True, True, False], [True, False, False]])
    object_targets = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    object_mask = torch.ones((2, 3), dtype=torch.bool)
    direction_targets = torch.tensor([[2, 2, -100], [0, -100, -100]])
    criterion = layout_query.VisualLayoutQueryLoss()
    losses = criterion(
        output=output,
        bbox_targets_xyxy=bbox_targets,
        bbox_mask=bbox_mask,
        object_targets=object_targets,
        object_mask=object_mask,
        direction_targets=direction_targets,
    )
    assert torch.isfinite(losses.loss)
    losses.loss.backward()
    assert adapter.query_embeddings.grad is not None
    assert torch.isfinite(adapter.query_embeddings.grad).all()


def test_layout_loss_uses_float32_and_reports_overfit_metrics() -> None:
    object_logits = torch.tensor([[2.0]], dtype=torch.bfloat16, requires_grad=True)
    bbox_cxcywh = torch.tensor(
        [[[0.3, 0.4, 0.4, 0.4]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    direction_logits = torch.tensor(
        [[[0.0, 0.0, 3.0, 0.0, 0.0]]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = layout_query.VLQAOutput(
        visual_tokens=torch.zeros(1, 1, 2),
        layout_queries=torch.zeros(1, 1, 2),
        prediction_queries=torch.zeros(1, 1, 2),
        object_logits=object_logits,
        bbox_logits=torch.zeros_like(bbox_cxcywh),
        bbox_cxcywh=bbox_cxcywh,
        bbox_xyxy=layout_query.cxcywh_to_xyxy(bbox_cxcywh),
        direction_logits=direction_logits,
        layout_residual=torch.zeros(1, 1, 2),
    )
    criterion = layout_query.VisualLayoutQueryLoss()
    losses = criterion(
        output=output,
        bbox_targets_xyxy=torch.tensor([[[0.1, 0.2, 0.5, 0.6]]]),
        bbox_mask=torch.tensor([[True]]),
        object_targets=torch.tensor([[1.0]]),
        object_mask=torch.tensor([[True]]),
        direction_targets=torch.tensor([[2]]),
    )

    assert losses.loss.dtype == torch.float32
    assert losses.object_loss.dtype == torch.float32
    assert losses.bbox_l1_loss.dtype == torch.float32
    assert losses.bbox_giou_loss.dtype == torch.float32
    assert losses.direction_loss.dtype == torch.float32
    assert losses.object_accuracy.item() == 1.0
    assert losses.direction_accuracy.item() == 1.0
    assert losses.bbox_mean_iou.item() > 0.98
    assert losses.query_abs_max.item() == 0.0
    assert losses.prediction_query_abs_max.item() == 0.0
    assert losses.bbox_logit_abs_max.item() == 0.0
    losses.loss.backward()
    assert object_logits.grad is not None
    assert bbox_cxcywh.grad is not None
    assert direction_logits.grad is not None


def test_high_resolution_memory_is_only_used_as_query_memory() -> None:
    adapter = layout_query.VisualLayoutQueryAdapter(
        visual_dim=32,
        layout_input_dim=8,
        adapter_dim=16,
        num_queries=3,
        num_heads=4,
    )
    visual = torch.randn(1, 16, 32)
    high_resolution_memory = torch.randn(1, 8, 8, 8)
    output = adapter(visual, layout_memory=high_resolution_memory)
    assert output.visual_tokens.shape == visual.shape
    assert torch.equal(output.visual_tokens, visual)
