from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .adapter import LayoutAdapterOutput, PreMergeLayoutAdapter
from .config import AdapterPrecision, LayoutAdapterConfig


def patch_grid_positions(grid_thw: Tensor, spatial_merge_size: int) -> Tensor:
    """Return normalized centers in GLM-OCR's post-downsample token order."""

    if grid_thw.shape != (1, 3):
        raise ValueError("the mechanism screen requires exactly one whole-page image per step")
    temporal, height, width = (int(value) for value in grid_thw[0].tolist())
    if temporal != 1:
        raise ValueError("the whole-page protocol does not accept video/multi-frame input")
    merged_height = height // spatial_merge_size
    merged_width = width // spatial_merge_size
    rows = (torch.arange(merged_height, device=grid_thw.device, dtype=torch.float32) + 0.5)
    cols = (torch.arange(merged_width, device=grid_thw.device, dtype=torch.float32) + 0.5)
    rows = rows / merged_height
    cols = cols / merged_width
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1).unsqueeze(0)


class LayoutAwarePatchMerger(nn.Module):
    """Insert the project adapter between GLM-OCR downsampling and patch merging."""

    def __init__(
        self,
        base_merger: nn.Module,
        adapter: PreMergeLayoutAdapter,
        spatial_merge_size: int,
        adapter_precision: AdapterPrecision = "mixed_bf16",
    ) -> None:
        super().__init__()
        if adapter_precision not in {"mixed_bf16", "fp32"}:
            raise ValueError(f"unsupported adapter precision: {adapter_precision}")
        self.base_merger = base_merger
        self.adapter = adapter
        self.spatial_merge_size = spatial_merge_size
        self.adapter_precision = adapter_precision
        self._grid_thw: Tensor | None = None
        self.last_output: LayoutAdapterOutput | None = None
        self.last_patch_positions: Tensor | None = None
        self.last_visual_tokens: Tensor | None = None
        self.last_residual: Tensor | None = None
        self.last_writeback_residual: Tensor | None = None
        self.last_input_dtype: torch.dtype | None = None
        self.last_adapter_input_dtype: torch.dtype | None = None
        self.last_adapter_output_dtype: torch.dtype | None = None
        self.last_merged_dtype: torch.dtype | None = None

    def set_grid_thw(self, grid_thw: Tensor) -> None:
        self._grid_thw = grid_thw

    def forward(self, hidden_state: Tensor) -> Tensor:
        if self._grid_thw is None:
            raise RuntimeError("image_grid_thw must be set before the GLM-OCR visual forward")
        positions = patch_grid_positions(self._grid_thw, self.spatial_merge_size)
        expected_tokens = positions.shape[1]
        if hidden_state.shape[0] != expected_tokens:
            raise RuntimeError(
                f"pre-merge token mismatch: got {hidden_state.shape[0]}, expected {expected_tokens}"
            )
        adapter_input = hidden_state.unsqueeze(0).float()
        if self.adapter_precision == "fp32":
            with torch.autocast(device_type=hidden_state.device.type, enabled=False):
                output = self.adapter(adapter_input, positions)
        else:
            output = self.adapter(adapter_input, positions)
        self.last_output = output
        self.last_patch_positions = positions
        adapted = output.merged_tokens.squeeze(0).to(hidden_state.dtype)
        self.last_visual_tokens = hidden_state.detach()
        self.last_residual = (output.merged_tokens.squeeze(0).float() - hidden_state.float()).detach()
        self.last_writeback_residual = (adapted.float() - hidden_state.float()).detach()
        self.last_input_dtype = hidden_state.dtype
        self.last_adapter_input_dtype = adapter_input.dtype
        self.last_adapter_output_dtype = output.merged_tokens.dtype
        self.last_merged_dtype = adapted.dtype
        return self.base_merger(adapted)


def install_layout_adapter(
    model: nn.Module,
    mode: str,
    num_queries: int = 32,
    max_residual_scale: float | None = None,
    adapter_precision: AdapterPrecision = "mixed_bf16",
) -> LayoutAwarePatchMerger:
    """Install the adapter at the verified Transformers GLM-OCR pre-merger seam."""

    visual: Any = model.model.visual
    config = LayoutAdapterConfig(
        hidden_size=int(visual.config.out_hidden_size),
        num_queries=num_queries,
        num_heads=8,
        mode=mode,
        ot_epsilon=0.1,
        ot_relaxation=0.5,
        ot_iterations=20,
        max_residual_scale=max_residual_scale,
    )
    adapter = PreMergeLayoutAdapter(config).to(next(model.parameters()).device)
    bridge = LayoutAwarePatchMerger(
        base_merger=visual.merger,
        adapter=adapter,
        spatial_merge_size=int(visual.spatial_merge_size),
        adapter_precision=adapter_precision,
    )
    visual.merger = bridge
    return bridge
