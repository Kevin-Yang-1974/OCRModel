from __future__ import annotations

"""Small dependency-free LoRA support for the GLM-OCR decoder.

The experiment protocol intentionally keeps the vision tower and the original
decoder weights frozen.  This module only replaces the selected decoder
``Linear`` projections with a frozen base projection plus two FP32 low-rank
updates, so the attribution run does not depend on an optional PEFT install.
"""

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LoRALinear(nn.Module):
    """A frozen linear layer with a trainable low-rank residual."""

    def __init__(
        self,
        base_layer: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not isinstance(base_layer, nn.Linear):
            raise TypeError(f"LoRA can only wrap nn.Linear, got {type(base_layer)!r}")
        self.base_layer = base_layer
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.dropout = float(dropout)
        # Keep the update in FP32 even though the GLM-OCR backbone is BF16.
        # This makes the tiny decoder update less sensitive to BF16 quantization
        # while the returned residual is cast back to the base output dtype.
        self.lora_A = nn.Parameter(
            torch.empty(self.rank, base_layer.in_features, device=base_layer.weight.device)
        )
        self.lora_B = nn.Parameter(
            torch.empty(base_layer.out_features, self.rank, device=base_layer.weight.device)
        )
        self.lora_A.data = self.lora_A.data.float()
        self.lora_B.data = self.lora_B.data.float()
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad_(False)

    @property
    def scaling(self) -> float:
        return self.alpha / self.rank

    def forward(self, hidden_states: Tensor) -> Tensor:
        base_output = self.base_layer(hidden_states)
        update_input = hidden_states.float()
        if self.dropout:
            update_input = F.dropout(update_input, p=self.dropout, training=self.training)
        update = F.linear(F.linear(update_input, self.lora_A), self.lora_B)
        return base_output + update.to(dtype=base_output.dtype) * self.scaling


_DECODER_TARGETS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_up_proj",
    "mlp.down_proj",
)


def _unwrap_model(model: nn.Module) -> nn.Module:
    while hasattr(model, "module"):
        model = model.module  # type: ignore[assignment]
    return model


def _decoder_layers(model: nn.Module) -> Any:
    root = _unwrap_model(model)
    model_body = getattr(root, "model", root)
    language_model = getattr(model_body, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise RuntimeError("GLM-OCR decoder layers were not found at model.language_model.layers")
    return layers


def _replace_path(parent: nn.Module, path: str, replacement: nn.Module) -> nn.Module:
    pieces = path.split(".")
    owner = parent
    for piece in pieces[:-1]:
        owner = getattr(owner, piece)
        if not isinstance(owner, nn.Module):
            raise RuntimeError(f"decoder projection parent is not a module: {path}")
    attribute = pieces[-1]
    original = getattr(owner, attribute, None)
    if not isinstance(original, nn.Module):
        raise RuntimeError(f"decoder projection is missing: {path}")
    setattr(owner, attribute, replacement)
    return original


def inject_decoder_lora(
    model: nn.Module,
    *,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
) -> dict[str, Any]:
    """Inject LoRA into every selected projection of every GLM decoder layer."""

    layers = _decoder_layers(model)
    existing = [name for name, module in _unwrap_model(model).named_modules() if isinstance(module, LoRALinear)]
    if existing:
        raise RuntimeError(f"decoder already contains LoRA modules: {existing[:3]}")
    injected: list[str] = []
    for layer_index, layer in enumerate(layers):
        for target in _DECODER_TARGETS:
            original = getattr(layer, target.split(".")[0], None)
            if original is None:
                raise RuntimeError(f"decoder target parent is missing: layers.{layer_index}.{target}")
            owner = layer
            for piece in target.split(".")[:-1]:
                owner = getattr(owner, piece)
            attribute = target.split(".")[-1]
            current = getattr(owner, attribute, None)
            if isinstance(current, LoRALinear):
                raise RuntimeError(f"decoder target already wrapped: layers.{layer_index}.{target}")
            if not isinstance(current, nn.Linear):
                raise RuntimeError(
                    f"decoder target is not nn.Linear: layers.{layer_index}.{target} ({type(current)!r})"
                )
            wrapped = LoRALinear(current, rank=rank, alpha=alpha, dropout=dropout)
            setattr(owner, attribute, wrapped)
            injected.append(f"model.language_model.layers.{layer_index}.{target}")
    if not injected:
        raise RuntimeError("no decoder LoRA targets were injected")
    return {
        "enabled": True,
        "rank": int(rank),
        "alpha": float(alpha),
        "dropout": float(dropout),
        "target_count": len(injected),
        "targets": injected,
    }


def iter_lora_modules(model: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    root = _unwrap_model(model)
    for name, module in root.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def iter_lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for _, module in iter_lora_modules(model):
        yield module.lora_A
        yield module.lora_B


def lora_parameter_items(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    items: list[tuple[str, nn.Parameter]] = []
    for name, module in iter_lora_modules(model):
        items.append((f"{name}.lora_A", module.lora_A))
        items.append((f"{name}.lora_B", module.lora_B))
    return items


def lora_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: parameter.detach().cpu().contiguous()
        for name, parameter in lora_parameter_items(model)
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, Tensor]) -> None:
    expected = dict(lora_parameter_items(model))
    if set(expected) != set(state):
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        raise ValueError(
            f"LoRA checkpoint keys do not match model: missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    for name, parameter in expected.items():
        value = state[name]
        if value.shape != parameter.shape:
            raise ValueError(f"LoRA tensor shape mismatch for {name}: {value.shape} != {parameter.shape}")
        parameter.data.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def set_lora_modules_training(model: nn.Module, training: bool) -> None:
    """Toggle only LoRA module flags without changing frozen base dropout flags."""

    for _, module in iter_lora_modules(model):
        # Calling module.train() would recursively switch base_layer to train
        # mode.  The base decoder remains in eval mode for deterministic frozen
        # activations; only the optional LoRA dropout needs this flag.
        module.training = bool(training)


def decoder_lora_finite_report(model: nn.Module) -> dict[str, Any]:
    non_finite = [name for name, value in lora_state_dict(model).items() if not bool(torch.isfinite(value).all())]
    return {
        "enabled": bool(lora_parameter_items(model)),
        "parameters_finite": not non_finite,
        "non_finite_parameters": non_finite,
    }


def trainable_parameter_report(model: nn.Module) -> dict[str, Any]:
    root = _unwrap_model(model)
    total = sum(parameter.numel() for parameter in root.parameters())
    trainable = [(name, parameter) for name, parameter in root.named_parameters() if parameter.requires_grad]
    adapter_trainable = [
        (name, parameter)
        for name, parameter in trainable
        if ".merger.adapter." in f".{name}" or name.startswith("adapter.")
    ]
    lora_trainable = [
        (name, parameter)
        for name, parameter in trainable
        if name.endswith("lora_A") or name.endswith("lora_B")
    ]
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(sum(parameter.numel() for _, parameter in trainable)),
        "trainable_parameter_tensors": len(trainable),
        "adapter_trainable_parameters": int(sum(parameter.numel() for _, parameter in adapter_trainable)),
        "decoder_lora_trainable_parameters": int(sum(parameter.numel() for _, parameter in lora_trainable)),
        "trainable_fraction": sum(parameter.numel() for _, parameter in trainable) / max(1, total),
    }
