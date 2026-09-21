"""Fine-grid mask head: the orderings and the loss behaviour that decide the run.

The load-bearing claim of this experiment is that the pre-merge vision features
are block-major, so fine index ``p = m * ms*ms + q`` and a merged cell is a
*contiguous* run of ``ms*ms`` fine cells.  If that is wrong the mask is silently
transposed against the image and every arm trains on scrambled supervision --
which is why the ordering tests below are parametrised over non-square grids.
"""

import pytest
import torch

from layout_ocr.decoder_mask_router import (
    DecoderMaskConfig,
    fine_grid_xywh,
    pool_to_merged,
)
from layout_ocr.mask_losses import balanced_mask_bce, mask_and_dice_loss, soft_dice


def test_fine_grid_is_block_major_with_one_cell_per_patch():
    grid = torch.tensor([[1, 6, 8]])
    xywh, shape = fine_grid_xywh(grid, 2)
    assert shape == (6, 8)
    assert xywh.shape == (1, 48, 4)
    # Fine index p = m*4 + q with m row-major over the 3x4 merged grid.
    # m=0 -> block (row 0, col 0) -> patches (0,0),(0,1),(1,0),(1,1)
    expected_first_four = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for q, (row, col) in enumerate(expected_first_four):
        assert xywh[0, q, 0].item() == pytest.approx((col + 0.5) / 8)
        assert xywh[0, q, 1].item() == pytest.approx((row + 0.5) / 6)
    # m=1 is the next block *across*, i.e. columns 2..3 of rows 0..1 -- not row 1.
    assert xywh[0, 4, 0].item() == pytest.approx((2 + 0.5) / 8)
    assert xywh[0, 4, 1].item() == pytest.approx((0 + 0.5) / 6)


@pytest.mark.parametrize("height,width", [(6, 8), (8, 6), (4, 4), (10, 14)])
def test_pool_to_merged_maps_each_block_to_its_own_merged_cell(height, width):
    """The silent-transposition guard: block m must land on merged cell m."""

    block = 2
    merged_shape = (height // block, width // block)
    merged_cells = merged_shape[0] * merged_shape[1]
    for m in range(merged_cells):
        fine = torch.zeros(1, 1, height * width)
        fine[0, 0, m * block * block : (m + 1) * block * block] = 1.0
        pooled = pool_to_merged(fine, merged_shape, block, "max")
        assert pooled.shape == (1, 1, merged_cells)
        assert int(pooled.argmax(dim=-1).item()) == m


def test_max_pool_preserves_a_single_cell_peak_but_mean_does_not():
    """`beta` stays anchored only under max: a sharp peak must survive pooling."""

    merged_shape = (2, 2)
    fine = torch.zeros(1, 1, 16)
    fine[0, 0, 5] = 1.0  # one cell inside block 1
    assert pool_to_merged(fine, merged_shape, 2, "max")[0, 0, 1].item() == pytest.approx(1.0)
    assert pool_to_merged(fine, merged_shape, 2, "mean")[0, 0, 1].item() == pytest.approx(0.25)


def test_pool_to_merged_rejects_a_mismatched_cell_count():
    with pytest.raises(ValueError):
        pool_to_merged(torch.zeros(1, 1, 7), (2, 2), 2, "max")


def test_soft_dice_directions_and_exclusions():
    target = torch.zeros(1, 2, 8)
    target[0, 0, :4] = 1.0  # token 0 carries a box
    valid = torch.tensor([[True, True]])

    perfect = target.clone()
    assert soft_dice(perfect, target, valid).item() == pytest.approx(0.0, abs=1e-6)
    empty = torch.zeros_like(target)
    assert soft_dice(empty, target, valid).item() == pytest.approx(1.0, abs=1e-6)
    # Token 1 has an all-zero target: mutating its prediction must not matter.
    noisy = target.clone()
    noisy[0, 1, :] = 1.0
    assert soft_dice(noisy, target, valid).item() == pytest.approx(0.0, abs=1e-6)


def test_dice_weight_zero_reproduces_the_previous_loss():
    torch.manual_seed(0)
    mask = torch.rand(1, 4, 16).clamp(0.01, 0.99)
    target = torch.zeros(1, 4, 16)
    target[0, 0, 3:6] = 1.0
    target[0, 2, 10:12] = 1.0
    valid = torch.tensor([[True, True, True, True]])
    legacy = balanced_mask_bce(mask, target, valid)
    total, parts = mask_and_dice_loss(mask, target, valid, dice_weight=0.0)
    assert torch.allclose(total, legacy)
    assert parts["dice"] == 0.0


def _uniform_grid(rows: int, cols: int) -> torch.Tensor:
    ys = (torch.arange(rows, dtype=torch.float32) + 0.5) / rows
    xs = (torch.arange(cols, dtype=torch.float32) + 0.5) / cols
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    n = rows * cols
    return torch.stack(
        [xx.reshape(-1), yy.reshape(-1), torch.full((n,), 1.0 / cols), torch.full((n,), 1.0 / rows)],
        dim=-1,
    )


def test_rasterize_polygon_lights_only_the_polygon():
    """Regression: the inside test must be 'same side of every edge'.

    Summing |sign| instead of |sum of signs| passes whenever no probe lands
    exactly on an edge, which is almost always -- so the whole grid was marked
    inside and the window target came out all ones.  The head then learned to
    predict 1.0 everywhere, with a *rising* Dice score that hid the fault.
    """

    from layout_ocr.mask_targets import rasterize_polygon

    xywh = _uniform_grid(8, 8)
    # An axis-aligned box covering the top-left quarter of the page.
    box = torch.tensor([[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]])
    target = rasterize_polygon(box, xywh)
    coverage = float((target > 0.5).float().mean())
    assert coverage == pytest.approx(0.25, abs=0.06), f"covered {coverage:.3f} of the grid"
    assert coverage < 0.9, "the polygon must not cover the page"
    assert float(target.max()) == pytest.approx(1.0)


def test_rasterize_polygon_matches_rasterize_box_for_a_single_box():
    from layout_ocr.mask_targets import rasterize_box, rasterize_polygon

    xywh = _uniform_grid(10, 10)
    box_xyxy = [0.2, 0.3, 0.5, 0.45]
    corners = torch.tensor(
        [[box_xyxy[0], box_xyxy[1]], [box_xyxy[2], box_xyxy[1]],
         [box_xyxy[0], box_xyxy[3]], [box_xyxy[2], box_xyxy[3]]]
    )
    polygon = rasterize_polygon(corners, xywh)
    rectangle = rasterize_box(box_xyxy, xywh)
    assert torch.allclose(polygon, rectangle, atol=0.08)


def test_dice_weight_penalises_a_spread_mask_more_than_a_sharp_one():
    """The loophole the whole change exists to close."""

    target = torch.zeros(1, 1, 64)
    target[0, 0, 30:34] = 1.0
    valid = torch.ones(1, 1, dtype=torch.bool)
    sharp = torch.zeros(1, 1, 64)
    sharp[0, 0, 30:34] = 0.9
    spread = torch.full((1, 1, 64), 0.9)  # same peak-ish level, but everywhere
    sharp_loss, _ = mask_and_dice_loss(sharp, target, valid, dice_weight=1.0)
    spread_loss, _ = mask_and_dice_loss(spread, target, valid, dice_weight=1.0)
    assert spread_loss > sharp_loss
