import torch
from layout_ocr.line_mask_head import LineMaskConfig, LineMaskHead, line_mask_loss


def example():
    torch.manual_seed(42)
    head = LineMaskHead(LineMaskConfig(hidden_size=16, dim=8, detach_every=2))
    hidden, visual = torch.randn(1, 5, 16), torch.randn(1, 12, 16)
    xy = torch.rand(1, 12, 4)
    return head, hidden, visual, xy


def test_scan_equals_cached_steps_and_future_does_not_leak():
    head, hidden, visual, xy = example()
    full = head(hidden, visual, xy, (3, 4))[0]
    keys = head.encode(visual, xy, (3, 4))
    prev = torch.zeros(1, 12)
    masks = []
    for step in range(5):
        prev = head.step(hidden[:, step], keys, prev)[0]
        masks.append(prev)
    torch.testing.assert_close(full, torch.stack(masks, 1))
    changed = hidden.clone(); changed[:, 3:] += 100
    torch.testing.assert_close(full[:, :3], head(changed, visual, xy, (3, 4))[0][:, :3])


def test_loss_finite_for_empty_and_sparse_targets_and_all_parameters_have_gradient():
    head, hidden, visual, xy = example()
    target = torch.zeros(1, 5, 12)
    target[:, :2, :3] = 1
    target[:, 2:4, 3:6] = 1
    stop = torch.tensor([[0., 0., 0., 0., 1.]])
    loss, stats = line_mask_loss(head(hidden, visual, xy, (3, 4)), target, torch.ones(1, 5, dtype=torch.bool), stop)
    loss.backward()
    assert torch.isfinite(loss) and stats['mass_ratio'] > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    empty_loss, _ = line_mask_loss(head(hidden, visual, xy, (3, 4)), target*0, torch.zeros(1,5,dtype=torch.bool), stop)
    assert torch.isfinite(empty_loss)


def test_small_real_optimization_reduces_line_loss():
    head, hidden, visual, xy = example()
    target = torch.zeros(1, 5, 12); target[:, :, :4] = 1
    valid, stop = torch.ones(1,5,dtype=torch.bool), torch.zeros(1,5)
    optimizer = torch.optim.Adam(head.parameters(), lr=.01)
    values = []
    for _ in range(40):
        optimizer.zero_grad()
        loss, _ = line_mask_loss(head(hidden, visual, xy, (3,4)), target, valid, stop)
        values.append(float(loss.detach())); loss.backward(); optimizer.step()
    assert values[-1] < values[0]*.5
