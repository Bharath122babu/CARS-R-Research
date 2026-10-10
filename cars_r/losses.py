from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor
import torch.nn.functional as F

from .model import ModelOutput


@dataclass
class LossOutput:
    total: Tensor
    autoregressive: Tensor
    mtp: Tensor


def causal_lm_loss(output: ModelOutput, targets: Tensor, *, mtp_weight: float = 0.0) -> LossOutput:
    """targets are next-token labels aligned with output.logits [B,T,V]."""
    if targets.shape != output.logits.shape[:2]:
        raise ValueError("targets must match logits [batch, time]")
    ar = F.cross_entropy(output.logits.reshape(-1, output.logits.shape[-1]), targets.reshape(-1))

    mtp_terms: list[Tensor] = []
    # mtp head 0 predicts one step beyond the normal next-token target, i.e.
    # token t+2 from hidden position t. Additional heads extend the horizon.
    for extra, logits in enumerate(output.mtp_logits, start=1):
        if targets.shape[1] <= extra:
            continue
        pred = logits[:, :-extra]
        label = targets[:, extra:]
        mtp_terms.append(F.cross_entropy(pred.reshape(-1, pred.shape[-1]), label.reshape(-1)))
    mtp = torch.stack(mtp_terms).mean() if mtp_terms else ar.new_zeros(())
    total = ar + mtp_weight * mtp
    return LossOutput(total=total, autoregressive=ar, mtp=mtp)
