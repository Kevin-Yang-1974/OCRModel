from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

FusionMode = Literal["content_only", "attention", "geometry", "layout_ot"]
AdapterPrecision = Literal["mixed_bf16", "fp32"]
LayoutLossProfile = Literal[
    "full",
    "ocr_only",
    "no_assignment",
    "no_assignment_validity",
    "no_geometry",
]
QueryAssignment = Literal["fixed_order", "hungarian"]


@dataclass(frozen=True)
class LayoutAdapterConfig:
    hidden_size: int
    num_queries: int = 32
    num_heads: int = 8
    mode: FusionMode = "layout_ot"
    dropout: float = 0.0
    geometry_temperature: float = 0.2
    ot_epsilon: float = 0.1
    ot_relaxation: float = 0.5
    ot_iterations: int = 20
    num_directions: int = 3
    max_residual_scale: float | None = None
    initial_residual_scale: float = 0.0
    use_validity_head: bool = False
    initial_valid_probability: float = 0.05

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.num_queries <= 0:
            raise ValueError("hidden_size and num_queries must be positive")
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mode not in {"content_only", "attention", "geometry", "layout_ot"}:
            raise ValueError(f"unsupported fusion mode: {self.mode}")
        if self.ot_epsilon <= 0 or self.ot_relaxation < 0 or self.ot_iterations <= 0:
            raise ValueError("invalid optimal-transport parameters")
        if self.max_residual_scale is not None and self.max_residual_scale <= 0:
            raise ValueError("max_residual_scale must be positive when set")
        if not -1.0 < self.initial_residual_scale < 1.0:
            raise ValueError("initial_residual_scale must be strictly between -1 and 1")
        if not 0.0 < self.initial_valid_probability < 1.0:
            raise ValueError("initial_valid_probability must be strictly between 0 and 1")
        if (
            self.max_residual_scale is not None
            and abs(self.initial_residual_scale) > self.max_residual_scale
        ):
            raise ValueError("initial_residual_scale must not exceed max_residual_scale")


@dataclass(frozen=True)
class LayoutLossConfig:
    box: float = 1.0
    order: float = 0.5
    direction: float = 0.5
    assignment: float = 1.0
    transport_entropy: float = 0.0
    validity: float = 0.0


def layout_loss_config(profile: str) -> LayoutLossConfig:
    """Return one of the controlled auxiliary-loss ablation profiles."""

    if profile == "full":
        return LayoutLossConfig()
    if profile == "ocr_only":
        return LayoutLossConfig(box=0.0, order=0.0, direction=0.0, assignment=0.0)
    if profile == "no_assignment":
        return LayoutLossConfig(assignment=0.0)
    if profile == "no_assignment_validity":
        return LayoutLossConfig(assignment=0.0, validity=0.5)
    if profile == "no_geometry":
        return LayoutLossConfig(box=0.0, order=0.0, direction=0.0)
    raise ValueError(f"unsupported layout loss profile: {profile}")
