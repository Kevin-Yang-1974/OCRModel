"""Tests for the mask head itself (``decoder_mask_router.py``).

The head is tensor-in / tensor-out, so these pin the properties the trainer and
the wiring rely on: config validation, the grid order matching the bridge, the
shape/range of one step and one scan, and the noise channels that must corrupt
the feedback path without touching the clean mask.
"""

from __future__ import annotations

import pytest
import torch

from layout_ocr.decoder_mask_router import (
    DecoderMaskConfig,
    DecoderMaskRouter,
    _normalized_grid_xywh,
)
from layout_ocr.glm_bridge import patch_grid_positions


def _router(hidden: int = 16, dim: int = 8, **overrides) -> DecoderMaskRouter:
    config = DecoderMaskConfig(hidden_size=hidden, router_dim=dim, **overrides)
    return DecoderMaskRouter(config)


def _grid(thw=(1, 4, 4), merge: int = 2):
    grid = torch.tensor([thw])
    xywh, shape = _normalized_grid_xywh(grid, merge)
    return xywh, shape


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError):
        DecoderMaskConfig(router_dim=0)
    with pytest.raises(ValueError):
        DecoderMaskConfig(split_layer=0)
    with pytest.raises(ValueError):
        DecoderMaskConfig(mask_feedback_noise=1.0)
    with pytest.raises(ValueError):
        DecoderMaskConfig(input_noise=-0.1)
    with pytest.raises(ValueError):
        DecoderMaskConfig(detach_every=0)


def test_grid_xywh_matches_the_bridge_order():
    grid = torch.tensor([[1, 6, 8]])  # merge 2 -> 3x4 cells
    xywh, shape = _normalized_grid_xywh(grid, 2)
    positions = patch_grid_positions(grid, 2)  # [1, N, 2]
    assert shape == (3, 4)
    assert xywh.shape == (1, 12, 4)
    assert torch.allclose(xywh[0, :, :2], positions[0], atol=1e-6)


def test_step_returns_masks_in_the_unit_interval():
    router = _router()
    xywh, shape = _grid()
    hidden = torch.randn(1, 16)
    prev = torch.zeros(1, 4)
    keys = torch.randn(1, 4, 8)
    mask, stop, z = router._step(hidden, prev, keys, xywh, shape)
    assert mask.shape == (1, 4)
    assert stop.shape == (1, 1)
    assert z.shape == (1, 4)
    assert ((mask >= 0.0) & (mask <= 1.0)).all()
    assert ((stop >= 0.0) & (stop <= 1.0)).all()


def test_scan_returns_one_mask_per_query_in_order():
    router = _router()
    xywh, shape = _grid()
    hidden = torch.randn(1, 3, 16)
    keys = torch.randn(1, 4, 8)
    mask, stop, z = router.scan(hidden, keys, xywh, shape, detach_every=2)
    assert mask.shape == (1, 3, 4)
    assert stop.shape == (1, 3, 1)
    assert z.shape == (1, 3, 4)


def test_feedback_noise_flips_values_only_in_training():
    router = _router()
    prev = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
    router.train()
    router.feedback_noise = 1.0  # flip every entry
    out = router._feedback(prev)
    assert out[0, 0].item() == pytest.approx(0.8)
    assert out[0, 1].item() == pytest.approx(0.2)
    # Eval mode leaves the mask untouched.
    router.eval()
    assert torch.equal(router._feedback(prev), prev)


def test_input_noise_is_disabled_in_eval():
    router = _router()
    router.eval()
    router.input_noise = 1.0
    query = torch.randn(1, 8)
    assert torch.equal(router._noisy_query(query), query)


def test_use_prev_mask_false_feeds_a_zero_context():
    router = _router(use_prev_mask=False)
    router.train()
    prev = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    assert torch.equal(router._feedback(prev), torch.zeros_like(prev))


def test_trainable_parameter_count_is_positive_and_matches_parameters():
    router = _router()
    assert router.trainable_parameter_count() == sum(
        p.numel() for p in router.parameters() if p.requires_grad
    )
    assert router.trainable_parameter_count() > 0


def test_project_visual_returns_projected_keys():
    router = _router()
    visual = torch.randn(1, 4, 16)
    grid = torch.tensor([[1, 4, 4]])
    keys, xywh, shape = router.project_visual(visual, grid, 2)
    assert keys.shape == (1, 4, 8)
    assert xywh.shape == (1, 4, 4)
    assert shape == (2, 2)
