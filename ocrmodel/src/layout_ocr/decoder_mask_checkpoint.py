"""Checkpoint save/load for the learned mask router and its co-trained LoRA.

A checkpoint directory carries five pieces, each with one writer and one reader
so resume and independent evaluation share the same loader:

* ``decoder_mask_config.json``   the ``DecoderMaskConfig`` the head was built with;
* ``decoder_mask.safetensors``   the head's weights;
* ``lora.safetensors``           the decoder LoRA weights (may be absent for a
                                 head-only run);
* ``training_state.pt``          optimizer/scheduler/step/epoch for resume;
* ``fingerprint.json``           base model id/revision, tokenizer/template and
                                 manifest/protocol hashes.

Nothing here saves the per-page mask recurrence state: it is page-local by design
and never belongs in a checkpoint (plan section 6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .decoder_mask_router import DecoderMaskConfig

CONFIG_FILE = "decoder_mask_config.json"
ROUTER_FILE = "decoder_mask.safetensors"
LORA_FILE = "lora.safetensors"
TRAINING_FILE = "training_state.pt"
FINGERPRINT_FILE = "fingerprint.json"


def _safetensors() -> Any:
    try:
        import safetensors.torch as st
    except ImportError as error:  # pragma: no cover - exercised only without the dep
        raise RuntimeError("safetensors is required to save/load decoder-mask checkpoints") from error
    return st


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_decoder_mask_checkpoint(
    output_dir: str | Path,
    *,
    config: DecoderMaskConfig | None = None,
    router_state: dict[str, torch.Tensor] | None = None,
    lora_state: dict[str, torch.Tensor] | None = None,
    training_state: dict[str, Any] | None = None,
    fingerprint: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write one checkpoint and return a ``{name: path}`` manifest.

    ``config`` and ``router_state`` are optional so the same writer also serves
    the ``routing-mode none`` baseline, which has a co-trained LoRA but no head.
    """

    st = _safetensors()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    if config is not None:
        _write_json(output_dir / CONFIG_FILE, config.__dict__)
        manifest["config"] = str(output_dir / CONFIG_FILE)
    if router_state is not None:
        st.save_file({key: value.detach().float().cpu().contiguous() for key, value in router_state.items()}, output_dir / ROUTER_FILE)
        manifest["router"] = str(output_dir / ROUTER_FILE)
    if lora_state is not None:
        st.save_file({key: value.detach().float().cpu().contiguous() for key, value in lora_state.items()}, output_dir / LORA_FILE)
        manifest["lora"] = str(output_dir / LORA_FILE)
    if training_state is not None:
        torch.save(training_state, output_dir / TRAINING_FILE)
        manifest["training_state"] = str(output_dir / TRAINING_FILE)
    if fingerprint is not None:
        _write_json(output_dir / FINGERPRINT_FILE, fingerprint)
        manifest["fingerprint"] = str(output_dir / FINGERPRINT_FILE)
    return manifest


def load_config(output_dir: str | Path) -> DecoderMaskConfig:
    return DecoderMaskConfig(**{k: v for k, v in _read_json(Path(output_dir) / CONFIG_FILE).items() if k != "hidden_size"})


def load_router_state(output_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load the head weights as a CPU float32 state dict for a fresh model."""

    st = _safetensors()
    return st.load_file(Path(output_dir) / ROUTER_FILE)


def load_lora_state(output_dir: str | Path) -> dict[str, torch.Tensor] | None:
    path = Path(output_dir) / LORA_FILE
    if not path.is_file():
        return None
    st = _safetensors()
    return st.load_file(path)


def load_training_state(output_dir: str | Path) -> dict[str, Any] | None:
    path = Path(output_dir) / TRAINING_FILE
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def load_fingerprint(output_dir: str | Path) -> dict[str, Any]:
    return _read_json(Path(output_dir) / FINGERPRINT_FILE)


def restore_router(model: Any, runtime: Any, output_dir: str | Path) -> None:
    """Copy the saved head weights into an installed router.

    ``model`` is only used to pin the target device/dtype; ``runtime.router`` is
    the submodule installed by ``install_decoder_mask_router``.  Weights are cast
    back to the router's own dtype (the head runs in float32, so this is fp32).
    """

    state = load_router_state(output_dir)
    router = runtime.router
    missing, unexpected = _check_keys(router, state)
    if missing or unexpected:
        raise ValueError(
            f"router checkpoint keys do not match: missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    for name, parameter in router.named_parameters():
        parameter.data.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
    for name, buffer in router.named_buffers():
        if name in state:
            buffer.copy_(state[name].to(device=buffer.device, dtype=buffer.dtype))


def _check_keys(router: Any, state: dict[str, torch.Tensor]) -> tuple[list[str], list[str]]:
    expected = {name for name, _ in router.named_parameters()} | {name for name, _ in router.named_buffers()}
    missing = sorted(expected - set(state))
    unexpected = sorted(set(state) - expected)
    return missing, unexpected
