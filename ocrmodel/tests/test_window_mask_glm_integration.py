"""Real tiny GLM-OCR SDPA/cache training and generation, without model downloads."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from layout_ocr.decoder_mask_router import DecoderMaskConfig
from layout_ocr.window_mask_routing import FirstLayerWindowRuntime


def make_tiny():
    pytest.importorskip("transformers")
    from transformers import GlmOcrConfig, GlmOcrForConditionalGeneration

    config = GlmOcrConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 32,
            "head_dim": 8,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "mrope_section": [1, 1, 2],
            },
            "pad_token_id": 0,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "out_hidden_size": 32,
            "num_heads": 4,
            "intermediate_size": 64,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
        },
        image_token_id=3,
        video_token_id=4,
        image_start_token_id=5,
        image_end_token_id=6,
        pad_token_id=0,
        eos_token_id=31,
    )
    config._attn_implementation = "sdpa"
    model = GlmOcrForConditionalGeneration(config).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tokenizer = SimpleNamespace(decode=lambda ids, **kwargs: "a" * len(ids))
    fusion = FirstLayerWindowRuntime(
        model,
        tokenizer,
        DecoderMaskConfig(router_dim=8, target_mode="window", mask_feedback_noise=0, input_noise=0),
    )
    ids = torch.tensor([[1, 5, 3, 3, 3, 3, 6, 2]])
    inputs = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "mm_token_type_ids": (ids == 3).long(),
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "pixel_values": torch.randn(16, 12),
    }
    return model, fusion, inputs


def test_real_glm_sdpa_generate_and_training_across_tbptt_boundaries():
    torch.manual_seed(42)
    model, fusion, inputs = make_tiny()
    fusion.set_page(inputs)
    with torch.inference_mode():
        tokens = model.generate(**inputs, max_new_tokens=3, do_sample=False, use_cache=True)
    assert tokens.shape[1] > inputs["input_ids"].shape[1]
    assert fusion.runtime.last_mask.shape == (1, 1, 4)
    assert model.config.text_config._attn_implementation == "sdpa"
    path = Path(__file__).parents[1] / "tools/training/train_window_mask_routing.py"
    spec = importlib.util.spec_from_file_location("window_train", path)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    target_ids = torch.tensor([10, 11, 12, 13, 31])
    targets = SimpleNamespace(
        mask=torch.tensor(
            [[[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0], [0, 0, 0, 0]]]
        ),
        spatial_valid=torch.tensor([[True, True, True, True, False]]),
        stop_target=torch.tensor([[0.0, 0, 0, 0, 1.0]]),
    )
    fusion.head.train()
    loss = train.teacher_forced_page(model, fusion, inputs, target_ids, targets, chunk_size=2)
    assert 0 < loss < 100
    assert fusion.head.query_proj.weight.grad.abs().sum() > 0
    assert fusion.head.visual_proj.weight.grad.abs().sum() > 0
    assert all(
        p.grad is None for name, p in model.named_parameters() if "decoder_mask_router" not in name
    )
