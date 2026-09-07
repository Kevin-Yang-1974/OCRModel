import torch
from torch import nn

from layout_ocr import LayoutAdapterConfig, PreMergeLayoutAdapter
from layout_ocr.glm_bridge import LayoutAwarePatchMerger, patch_grid_positions


def test_patch_grid_and_bridge_contract() -> None:
    grid = torch.tensor([[1, 4, 6]])
    positions = patch_grid_positions(grid, spatial_merge_size=2)
    assert positions.shape == (1, 6, 2)
    adapter = PreMergeLayoutAdapter(
        LayoutAdapterConfig(hidden_size=8, num_queries=2, num_heads=2, mode="layout_ot")
    )
    bridge = LayoutAwarePatchMerger(nn.Linear(8, 8), adapter, spatial_merge_size=2)
    bridge.set_grid_thw(grid)
    merged = bridge(torch.randn(6, 8))
    assert merged.shape == (6, 8)
    assert bridge.last_output is not None
