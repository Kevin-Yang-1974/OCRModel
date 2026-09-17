"""Two CPU ranks: verify global token means with one active or no active rank."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from layout_ocr.aligned_recovery import recovery_losses


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(1, 4, 100))

    def forward(self):
        return self.logits * 1.0


def main():
    dist.init_process_group("gloo")
    model = DistributedDataParallel(Toy())
    for active in (True, False):
        model.zero_grad(set_to_none=True)
        logits = model()
        rollout = {"accepted": active and dist.get_rank() == 0,
                   "detected": active and dist.get_rank() == 0,
                   "prefix_inputs": {"labels": torch.tensor([[-100, -100, 40, 99]])},
                   "boundary": 1, "negative": 10, "gold_next": 40, "suffix": [40, 99]}
        recovery_losses(logits, rollout, {99})["loss"].backward()
        expected = torch.zeros(1, 4, 100, requires_grad=True)
        if active:
            p = expected[0, 1].softmax(-1)[10]
            reference = (0.01 * -torch.log1p(-p)
                         + 0.05 * F.cross_entropy(expected[0, 1:2], torch.tensor([40]))
                         + 0.05 * F.cross_entropy(expected[0, 2:3], torch.tensor([99])))
        else:
            reference = expected.sum() * 0
        reference.backward()
        torch.testing.assert_close(model.module.logits.grad, expected.grad)
    if dist.get_rank() == 0:
        print(json.dumps({"status": "passed", "checks": ["mixed_active_ranks", "all_empty_ranks", "global_token_mean"]}))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
