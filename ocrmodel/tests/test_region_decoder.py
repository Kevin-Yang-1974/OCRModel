import torch

from layout_ocr.autoregressive_region import (
    AutoregressiveRegionDecoder,
    RegionDecoderConfig,
    compute_region_losses,
)


def _boxes(batch: int, count: int) -> torch.Tensor:
    values = torch.rand(batch, count, 4)
    return torch.cat(
        (
            torch.minimum(values[..., :2], values[..., 2:]),
            torch.maximum(values[..., :2], values[..., 2:]),
        ),
        dim=-1,
    )


def test_region_decoder_masks_repeated_pointers_and_backpropagates() -> None:
    config = RegionDecoderConfig(
        input_hidden_size=8,
        decoder_hidden_size=8,
        num_heads=2,
        num_layers=1,
        candidate_count=8,
        max_regions=4,
    )
    module = AutoregressiveRegionDecoder(config)
    candidates = torch.randn(1, 8, 8, requires_grad=True)
    candidate_boxes = _boxes(1, 8)
    targets = {
        "target_boxes": candidate_boxes[:, :2].detach().clone(),
        "target_directions": torch.zeros(1, 2, dtype=torch.long),
        "query_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    output = module(candidates, candidate_boxes, targets=targets)
    losses = compute_region_losses(output, targets)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert candidates.grad is not None
    inference = module(candidates.detach(), candidate_boxes)
    selected = inference.selected_indices[0][inference.selected_mask[0]]
    assert selected.numel() == selected.unique().numel()

