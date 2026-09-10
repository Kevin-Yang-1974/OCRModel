import torch

from layout_ocr.stabilization import (
    AdaptiveCycleLogitsProcessor,
    RepeatSuppressionConfig,
    eos_focus_loss,
    repeated_cycle_positions,
    repetition_diagnostics,
    unlikelihood_loss,
)


def test_repeated_cycle_positions_only_marks_third_cycle() -> None:
    labels = torch.tensor([[-100, 1, 2, 1, 2, 1, 2, 9]])
    positions = repeated_cycle_positions(
        labels, min_cycle_length=2, max_cycle_length=2, cycle_repeats=3
    )
    assert positions.tolist() == [[False, False, False, False, False, True, True, False]]


def test_unlikelihood_and_eos_losses_are_finite() -> None:
    labels = torch.tensor([[-100, 1, 2, 1, 2, 1, 2, 9]])
    logits = torch.randn(1, labels.shape[1], 12, requires_grad=True)
    value = unlikelihood_loss(
        logits, labels, min_cycle_length=2, max_cycle_length=2, cycle_repeats=3
    ) + eos_focus_loss(logits, labels, {9})
    assert torch.isfinite(value)
    value.backward()
    assert logits.grad is not None


def test_cycle_processor_penalizes_and_can_force_eos() -> None:
    config = RepeatSuppressionConfig(
        enabled=True,
        recent_window=32,
        min_cycle_length=2,
        max_cycle_length=2,
        cycle_repeats=2,
        cycle_penalty=3.0,
        force_eos_steps=0,
    )
    processor = AdaptiveCycleLogitsProcessor(
        prompt_length=1, eos_token_ids={0}, config=config
    )
    input_ids = torch.tensor([[5, 1, 2, 1, 2]])
    scores = torch.zeros(1, 8)
    adjusted = processor(input_ids, scores)
    assert adjusted[0, 1] < adjusted[0, 3]
    assert torch.isfinite(adjusted[0, 0])


def test_repetition_diagnostics_distinguishes_normal_text() -> None:
    assert not repetition_diagnostics("甲乙丙丁", min_cycle_length=2, max_cycle_length=2)[
        "repeated_cycle_detected"
    ]
    assert repetition_diagnostics(
        "甲乙甲乙甲乙", min_cycle_length=2, max_cycle_length=2, cycle_repeats=3
    )["repeated_cycle_detected"]
