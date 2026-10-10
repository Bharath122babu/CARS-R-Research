from __future__ import annotations

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        xf = x.float()
        y = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.out_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.dropout(self.out_proj(F.silu(gate) * value))


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: Tensor, positions: Tensor, theta: float = 10000.0) -> Tensor:
    """RoPE for x shaped [B, H, T, D], D even."""
    d = x.shape[-1]
    if d % 2:
        raise ValueError("RoPE dimension must be even")
    inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    freqs = torch.outer(positions.float(), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(x.dtype)[None, None, :, :]
    sin = emb.sin().to(x.dtype)[None, None, :, :]
    return x * cos + rotate_half(x) * sin
