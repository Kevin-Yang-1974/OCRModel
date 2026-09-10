import torch
from torch import nn

from layout_ocr.lora import (
    LoRALinear,
    inject_decoder_lora,
    load_lora_state_dict,
    lora_state_dict,
    set_lora_modules_training,
)


class _Attention(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.o_proj = nn.Linear(hidden, hidden)


class _Mlp(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden, hidden * 2)
        self.down_proj = nn.Linear(hidden * 2, hidden)


class _Layer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.self_attn = _Attention(hidden)
        self.mlp = _Mlp(hidden)


class _FakeGlmOcr(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([_Layer(8), _Layer(8)])


def test_decoder_lora_injection_and_reload() -> None:
    torch.manual_seed(3)
    model = _FakeGlmOcr()
    original = model.model.language_model.layers[0].self_attn.q_proj.weight.detach().clone()
    report = inject_decoder_lora(model, rank=2, alpha=2.0)

    assert report["target_count"] == 12
    assert isinstance(model.model.language_model.layers[0].self_attn.q_proj, LoRALinear)
    assert not model.model.language_model.layers[0].self_attn.q_proj.base_layer.weight.requires_grad
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters() if parameter.requires_grad)
    assert torch.equal(
        original,
        model.model.language_model.layers[0].self_attn.q_proj.base_layer.weight,
    )

    state = lora_state_dict(model)
    assert len(state) == 24
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.25)
    load_lora_state_dict(model, state)
    reloaded = lora_state_dict(model)
    for name in state:
        torch.testing.assert_close(state[name], reloaded[name])


def test_lora_forward_has_gradient_without_training_frozen_base() -> None:
    base = nn.Linear(4, 4)
    wrapped = LoRALinear(base, rank=2, alpha=2.0)
    set_lora_modules_training(wrapped, True)
    output = wrapped(torch.randn(2, 4)).square().mean()
    output.backward()
    assert wrapped.lora_A.grad is not None
    assert wrapped.lora_B.grad is not None
    assert base.weight.grad is None
