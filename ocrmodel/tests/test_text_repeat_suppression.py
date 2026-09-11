import torch

from layout_ocr.stabilization import (
    AdaptiveCycleLogitsProcessor,
    ContinuationStopHead,
    RepeatSuppressionConfig,
    continuation_head_loss,
    cycle_escape_losses,
    eos_focus_loss,
    generated_cycle_window,
    generate_with_loop_recovery,
    loop_continuation_diagnostics,
    natural_predicted_loop_loss,
    natural_loop_rollout_loss,
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


def _logits_from_predictions(predictions: list[int], vocab_size: int = 32) -> torch.Tensor:
    logits = torch.zeros(1, len(predictions) + 1, vocab_size)
    for index, token_id in enumerate(predictions):
        logits[0, index, token_id] = 6.0
    logits.requires_grad_()
    return logits


def test_natural_predicted_loop_loss_penalizes_wrong_cycle_tokens() -> None:
    labels = torch.tensor([[-100, 9, 9, 9, 9, 9, 9, 9, 9]])
    logits = _logits_from_predictions([1, 2, 1, 2, 1, 2, 3, 4])
    result = natural_predicted_loop_loss(
        logits,
        labels,
        min_cycle_length=2,
        max_cycle_length=2,
        cycle_repeats=3,
        recent_window=32,
    )
    assert result["active_tokens"].item() > 0
    assert result["active_pages"].item() == 1
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_natural_predicted_loop_loss_does_not_penalize_legal_repeat() -> None:
    labels = torch.tensor([[-100, 1, 2, 1, 2, 1, 2, 9, 9]])
    logits = _logits_from_predictions([1, 2, 1, 2, 1, 2, 9, 9])
    result = natural_predicted_loop_loss(
        logits,
        labels,
        min_cycle_length=2,
        max_cycle_length=2,
        cycle_repeats=3,
        recent_window=32,
    )
    assert result["active_tokens"].item() == 0
    assert result["loss"].item() == 0.0


def test_natural_predicted_loop_loss_ignores_prompt_positions() -> None:
    labels = torch.tensor([[-100, -100, 1, 2, 1, 2, 1, 2, 8]])
    logits = _logits_from_predictions([7, 7, 1, 2, 1, 2, 1, 2])
    result = natural_predicted_loop_loss(
        logits,
        labels,
        min_cycle_length=2,
        max_cycle_length=2,
        cycle_repeats=3,
        recent_window=32,
    )
    assert result["active_tokens"].item() == 0
    assert result["loss"].item() == 0.0


def test_generated_cycle_window_uses_real_rollout_suffix() -> None:
    result = generated_cycle_window(
        [7, 8, 1, 2, 1, 2, 1, 2, 1, 2, 9],
        min_cycle_length=2,
        max_cycle_length=2,
        cycle_repeats=3,
        recent_window=32,
    )
    assert result["detected"] is True
    assert result["end"] == 8
    assert result["length"] == 2
    assert result["cycle"] == [1, 2]
    assert not generated_cycle_window([1, 2, 1, 2, 3], min_cycle_length=2, max_cycle_length=2)[
        "detected"
    ]


def test_natural_loop_rollout_loss_combines_negative_and_continuation() -> None:
    labels = torch.tensor([[-100, 9, 10, 11, 12, 13]])
    logits = torch.zeros(1, labels.shape[1], 20, requires_grad=True)
    negative_mask = torch.zeros(1, labels.shape[1] - 1, dtype=torch.bool)
    continuation_mask = torch.zeros_like(negative_mask)
    negative_ids = torch.full_like(negative_mask, -1, dtype=torch.long)
    # Position 2 predicts the repeated token 3 while the target is 10.
    negative_mask[0, 2] = True
    continuation_mask[0, 2:4] = True
    negative_ids[0, 2] = 3
    result = natural_loop_rollout_loss(
        logits,
        labels,
        negative_mask,
        negative_ids,
        labels[:, 1:],
        continuation_mask,
    )
    assert result["active_tokens"].item() == 1
    assert result["continuation_tokens"].item() == 2
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


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


def test_loop_escape_loss_prefers_real_continuation_and_backpropagates() -> None:
    labels = torch.tensor([[-100, 1, 2, 1, 2, 1, 2, 9, 10, 11]])
    escape_mask = torch.zeros_like(labels, dtype=torch.bool)
    escape_mask[0, 7:10] = True
    negative = torch.full_like(labels, -1)
    negative[0, 7:10] = torch.tensor([1, 2, 1])
    logits = torch.randn(1, labels.shape[1], 16, requires_grad=True)
    losses = cycle_escape_losses(
        logits,
        labels,
        escape_mask,
        negative,
        {9},
    )
    head = ContinuationStopHead(hidden_size=8)
    head_loss = continuation_head_loss(
        head,
        logits,
        labels,
        {9},
        loop_mask=escape_mask,
        negative_token_ids=negative,
    )
    value = losses["escape"] + losses["margin"] + losses["continue"] + head_loss
    assert torch.isfinite(value)
    value.backward()
    assert logits.grad is not None
    assert all(parameter.grad is not None for parameter in head.parameters())


def test_loop_continuation_diagnostics_requires_non_cycle_tokens_before_eos() -> None:
    result = loop_continuation_diagnostics(
        [1, 2, 1, 2, 1, 2, 3, 4, 5, 6, 0],
        {0},
        min_cycle_length=2,
        max_cycle_length=2,
        cycle_repeats=3,
    )
    assert result["loop_detected"] is True
    assert result["continued_non_cycle_tokens"] == 4
    assert result["post_loop_eos"] is True
    assert result["post_loop_early_eos"] is False


def test_continuation_escape_never_hard_masks_eos() -> None:
    config = RepeatSuppressionConfig(
        enabled=True,
        continuation_escape=True,
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
    assert torch.isfinite(adjusted[0, 0])
    assert adjusted[0, 1] < adjusted[0, 3]


def test_stateful_recovery_penalty_decays_without_forcing_eos() -> None:
    config = RepeatSuppressionConfig(
        enabled=True,
        continuation_escape=True,
        recent_window=32,
        min_cycle_length=2,
        max_cycle_length=2,
        cycle_repeats=2,
        cycle_penalty=4.0,
        escape_budget=4,
        escape_clear_steps=2,
        force_eos_steps=0,
    )
    processor = AdaptiveCycleLogitsProcessor(
        prompt_length=1, eos_token_ids={0}, config=config
    )
    scores = torch.zeros(1, 8)
    scores[0, 1] = 3.0
    first = processor(torch.tensor([[5, 1, 2, 1, 2]]), scores)
    second = processor(torch.tensor([[5, 1, 2, 1, 2, 1]]), scores)
    assert first[0, 0] < scores[0, 0]
    assert second[0, 0] <= scores[0, 0]
    assert torch.isfinite(first[0, 0]) and torch.isfinite(second[0, 0])


def test_generate_helper_selects_plain_or_recovery_processor() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return torch.tensor([[5, 1]])

    config = RepeatSuppressionConfig(enabled=True, force_eos_steps=0)
    model = FakeModel()
    inputs = {"input_ids": torch.tensor([[5]])}
    generate_with_loop_recovery(
        model,
        inputs,
        prompt_length=1,
        eos_token_ids={0},
        config=config,
        max_new_tokens=2,
        mode="plain",
    )
    assert "logits_processor" not in model.calls[-1]
    generate_with_loop_recovery(
        model,
        inputs,
        prompt_length=1,
        eos_token_ids={0},
        config=config,
        max_new_tokens=2,
        mode="loop_recovery",
    )
    assert len(model.calls[-1]["logits_processor"]) == 1


def test_repetition_diagnostics_distinguishes_normal_text() -> None:
    assert not repetition_diagnostics("甲乙丙丁", min_cycle_length=2, max_cycle_length=2)[
        "repeated_cycle_detected"
    ]
    assert repetition_diagnostics(
        "甲乙甲乙甲乙", min_cycle_length=2, max_cycle_length=2, cycle_repeats=3
    )["repeated_cycle_detected"]
