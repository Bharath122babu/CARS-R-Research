# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Adapted from MIT-licensed flash-linear-attention reference layers/operators.
# See third_party_licenses/FLA_LICENSE.txt and baseline_sources.json.
"""Portable KDA / GDN-2 reference mixers; no Triton dependency.

FLA layer recipes are retained: causal SiLU short convolutions, normalized
queries/keys, channel decay, delta gates, and sigmoid-gated per-head RMSNorm.
The sequential recurrence replaces FLA's fused kernels and stays FP32.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import RMSNorm


def delta_recurrence(
    q: Tensor, k: Tensor, v: Tensor, log_decay: Tensor,
    erase: Tensor, write: Tensor, *, attention_mask: Tensor | None = None,
    initial_state: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """GDN-2 state [B,H,K,V]; broadcasting scalar beta gates recovers KDA.

    Read after the current delta write, as in the upstream operators. The
    caller normalizes q/k; scale is 1/sqrt(K). Masked tokens preserve state.
    """
    with torch.autocast(device_type=q.device.type, enabled=False):
        q, k, v, log_decay, erase, write = (
            value.float() for value in (q, k, v, log_decay, erase, write)
        )
        batch, length, heads, key_dim = q.shape
        value_dim = v.shape[-1]
        state = q.new_zeros(batch, heads, key_dim, value_dim) if initial_state is None else initial_state.float()
        outputs = []
        q = q * (key_dim ** -0.5)
        for index in range(length):
            qt, kt, vt = q[:, index], k[:, index], v[:, index]
            decayed = state * log_decay[:, index].exp().unsqueeze(-1)
            removed = ((erase[:, index] * kt).unsqueeze(-1) * decayed).sum(-2)
            new_value = write[:, index] * vt - removed
            updated = decayed + kt.unsqueeze(-1) * new_value.unsqueeze(-2)
            if attention_mask is not None:
                active = attention_mask[:, index].bool()
                state = torch.where(active[:, None, None, None], updated, state)
            else:
                state = updated
            output = (qt.unsqueeze(-1) * state).sum(-2)
            if attention_mask is not None:
                output = output * active[:, None, None]
            outputs.append(output)
        return torch.stack(outputs, dim=1), state


class DeltaMixer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.kind = config.architecture
        self.heads = config.n_heads
        self.key_dim = config.delta_key_dim
        self.value_dim = config.delta_value_dim
        self.allow_negative = config.delta_allow_negative_eigenvalues
        d, key, value = config.d_model, self.heads*self.key_dim, self.heads*self.value_dim
        self.q_proj = nn.Linear(d, key, bias=False)
        self.k_proj = nn.Linear(d, key, bias=False)
        self.v_proj = nn.Linear(d, value, bias=False)
        self.q_conv = nn.Conv1d(key, key, config.delta_conv_size, groups=key, padding=config.delta_conv_size-1, bias=False)
        self.k_conv = nn.Conv1d(key, key, config.delta_conv_size, groups=key, padding=config.delta_conv_size-1, bias=False)
        self.v_conv = nn.Conv1d(value, value, config.delta_conv_size, groups=value, padding=config.delta_conv_size-1, bias=False)
        self.f_proj = nn.Sequential(nn.Linear(d, self.value_dim, bias=False), nn.Linear(self.value_dim, key, bias=False))
        self.b_proj = nn.Linear(d, self.heads if self.kind == "kda" else key, bias=False)
        self.w_proj = None if self.kind == "kda" else nn.Linear(d, value, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(self.heads).uniform_(1, 16)))
        dt = torch.exp(torch.rand(key)*(math.log(0.1)-math.log(0.001))+math.log(0.001))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log._no_weight_decay = True
        self.dt_bias._no_weight_decay = True
        self.g_proj = nn.Sequential(nn.Linear(d, self.value_dim, bias=False), nn.Linear(self.value_dim, value, bias=True))
        self.o_norm = RMSNorm(self.value_dim, eps=1e-5)
        self.o_proj = nn.Linear(value, d, bias=False)

    def _convolve(self, x: Tensor, conv: nn.Conv1d, mask: Tensor | None) -> Tensor:
        length = x.shape[1]
        if mask is None:
            return F.silu(conv(x.transpose(1, 2))[..., :length].transpose(1, 2))
        # FLA unpads before short convolution. Compact/scatter preserves that
        # behavior even with holes in a mask, without changing traced shapes.
        valid = mask.bool()
        positions = (valid.long().cumsum(1)-1).clamp_min(0)
        indices = positions[..., None].expand_as(x)
        compact = torch.zeros_like(x).scatter_add(1, indices, x * valid[..., None])
        convolved = F.silu(conv(compact.transpose(1, 2))[..., :length].transpose(1, 2))
        return convolved.gather(1, indices) * valid[..., None]

    def forward(self, x: Tensor, *, attention_mask: Tensor | None = None) -> Tensor:
        batch, length, _ = x.shape
        q = self._convolve(self.q_proj(x), self.q_conv, attention_mask).reshape(batch, length, self.heads, self.key_dim)
        k = self._convolve(self.k_proj(x), self.k_conv, attention_mask).reshape(batch, length, self.heads, self.key_dim)
        v = self._convolve(self.v_proj(x), self.v_conv, attention_mask).reshape(batch, length, self.heads, self.value_dim)
        gate_logits = self.f_proj(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            # FLA's in-kernel L2 normalization uses rsqrt(sum(square)+eps).
            q = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True)+1e-6)
            k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True)+1e-6)
            gate = F.softplus(gate_logits.float()+self.dt_bias.float()).reshape(batch, length, self.heads, self.key_dim)
            log_decay = -self.A_log.float().exp()[None, None, :, None]*gate
        beta = self.b_proj(x).float().sigmoid()
        if self.kind == "kda":
            erase = beta[..., None]
            write = erase
            if self.allow_negative:
                erase = erase * 2.0
                write = write * 2.0
        else:
            erase = beta.reshape(batch, length, self.heads, self.key_dim)
            write = self.w_proj(x).float().sigmoid().reshape(batch, length, self.heads, self.value_dim)
            if self.allow_negative:
                erase = erase * 2.0
        output, _ = delta_recurrence(q, k, v, log_decay, erase, write, attention_mask=attention_mask)
        output_gate = self.g_proj(x).reshape(batch, length, self.heads, self.value_dim).float().sigmoid()
        output = (self.o_norm(output)*output_gate).reshape(batch, length, -1)
        return self.o_proj(output.to(x.dtype))
