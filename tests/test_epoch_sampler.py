import numpy as np
import torch

from cars_epoch import EpochStream, epoch_loss
from types import SimpleNamespace


def test_epoch_covers_every_target_once_including_partial_tail():
    stream = EpochStream(np.arange(1, 40), batch_size=3, context=4, seed=81)
    targets = []
    for _ in range(stream.steps):
        batch = stream.sample(3, 5, generator=None, device=torch.device('cpu'))
        targets.extend(batch[:, 1:][batch[:, 1:] != 256].tolist())
    assert sorted(targets) == list(range(2, 40))


def test_resumed_epoch_returns_identical_remaining_batches():
    values = np.arange(1, 57)
    uninterrupted = EpochStream(values, batch_size=3, context=4, seed=81)
    uninterrupted.sample(3, 5, generator=None, device=torch.device('cpu'))
    resumed = EpochStream(values, batch_size=3, context=4, seed=81, completed_steps=1)
    for _ in range(1, uninterrupted.steps):
        a = uninterrupted.sample(3, 5, generator=None, device=torch.device('cpu'))
        b = resumed.sample(3, 5, generator=None, device=torch.device('cpu'))
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_padded_tail_cannot_contribute_to_loss_or_gradient():
    logits = torch.randn(2, 4, 259, requires_grad=True)
    targets = torch.tensor([[3, 4, 5, 256], [256, 256, 256, 256]])
    loss = epoch_loss(SimpleNamespace(logits=logits, mtp_logits=[]), targets).total
    expected = torch.nn.functional.cross_entropy(logits[0, :3], targets[0, :3])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert logits.grad[0, 3].count_nonzero() == 0
    assert logits.grad[1].count_nonzero() == 0
