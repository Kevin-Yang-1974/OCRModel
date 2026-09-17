from types import SimpleNamespace

import torch

from layout_ocr.data import append_eos_label_token
from layout_ocr.train_screen import (
    build_loop_escape_inputs,
    build_mixed_prefix_inputs,
    scheduled_sampling_probability,
    token_weighted_loss_stats,
)


def test_append_eos_label_token_adds_one_token_only() -> None:
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    appended = append_eos_label_token(inputs, {99})
    assert appended["input_ids"].tolist() == [[10, 11, 12, 99]]
    assert appended["attention_mask"].tolist() == [[1, 1, 1, 1]]
    same = append_eos_label_token(appended, {99})
    assert same["input_ids"].tolist() == [[10, 11, 12, 99]]


def test_scheduled_sampling_schedule_is_bounded() -> None:
    args = SimpleNamespace(
        scheduled_sampling=True,
        scheduled_sampling_warmup_steps=64,
        scheduled_sampling_ramp_steps=64,
        scheduled_sampling_max_probability=0.1,
    )
    assert scheduled_sampling_probability(args, 64) == 0.0
    assert scheduled_sampling_probability(args, 96) == 0.05
    assert scheduled_sampling_probability(args, 128) == 0.1
    assert scheduled_sampling_probability(args, 256) == 0.1


def test_mixed_prefix_keeps_prompt_and_first_target() -> None:
    # Positions 0-2 are prompt, position 3 is the first OCR token, and
    # positions 4-5 are eligible target-prefix replacements.
    inputs = {
        "input_ids": torch.tensor([[10, 11, 12, 20, 21, 22]]),
        "labels": torch.tensor([[-100, -100, -100, 20, 21, 22]]),
        "pixel_values": torch.zeros(1, 1),
    }
    logits = torch.full((1, 6, 30), -100.0)
    logits[0, 3, 23] = 100.0
    logits[0, 4, 24] = 100.0
    outputs = SimpleNamespace(logits=logits)
    mixed, replaced, eligible = build_mixed_prefix_inputs(inputs, outputs, 1.0)
    assert replaced == 2
    assert eligible == 2
    assert mixed["input_ids"].tolist() == [[10, 11, 12, 20, 23, 24]]
    assert torch.equal(mixed["labels"], inputs["labels"])
    assert torch.equal(mixed["pixel_values"], inputs["pixel_values"])


def test_loop_escape_prefix_preserves_labels_and_marks_continuation() -> None:
    inputs = {
        "input_ids": torch.tensor(
            [[10, 11, 12, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]]
        ),
        "labels": torch.tensor(
            [[-100, -100, -100, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]]
        ),
        "pixel_values": torch.zeros(1, 1),
    }
    looped, mask, negative, active = build_loop_escape_inputs(
        inputs, cycle_length=2, escape_horizon=2
    )
    assert active == 2
    assert int(mask.sum()) == 2
    assert int((negative >= 0).sum()) == 2
    assert torch.equal(looped["labels"], inputs["labels"])
    assert any(
        looped["input_ids"][0, start : start + 6].tolist()
        == [20, 21, 20, 21, 20, 21]
        for start in range(3, 10)
    )


def test_token_weighted_loss_uses_valid_causal_targets() -> None:
    logits = torch.zeros(1, 4, 5)
    logits[0, 0, 1] = 2.0
    logits[0, 1, 2] = 2.0
    logits[0, 2, 3] = 2.0
    labels = torch.tensor([[-100, 1, 2, 3]])
    numerator, count = token_weighted_loss_stats(SimpleNamespace(logits=logits), labels)
    assert count == 3
    assert torch.isfinite(numerator)
    assert numerator.item() > 0.0
