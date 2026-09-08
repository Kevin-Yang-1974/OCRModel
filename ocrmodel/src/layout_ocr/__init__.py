"""GLM-OCR pre-merge layout adaptation components.

Imports are lazy so the manifest-audit CLI remains usable without loading the
model runtime and its PyTorch dependency.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "LayoutAdapterConfig",
    "LayoutAdapterOutput",
    "LayoutLossConfig",
    "PreMergeLayoutAdapter",
    "compute_layout_losses",
    "match_layout_targets",
]


def __getattr__(name: str) -> Any:
    modules = {
        "LayoutAdapterConfig": ".config",
        "LayoutLossConfig": ".config",
        "LayoutAdapterOutput": ".adapter",
        "PreMergeLayoutAdapter": ".adapter",
        "compute_layout_losses": ".losses",
        "match_layout_targets": ".losses",
    }
    if name not in modules:
        raise AttributeError(name)
    return getattr(import_module(modules[name], __name__), name)
