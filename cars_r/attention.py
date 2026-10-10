from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import RMSNorm, apply_rope


class CausalSelfAttention(nn.Module):
    """Conventional dense multi-head causal self-attention with RoPE."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.h = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x: Tensor, *, attention_mask: Tensor | None = None) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.h, self.head_dim).unbind(2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        positions = torch.arange(t, device=x.device)
        q = apply_rope(q, positions, self.config.rope_theta)
        k = apply_rope(k, positions, self.config.rope_theta)
        allowed = None
        if attention_mask is not None:
            if attention_mask.shape != (b, t):
                raise ValueError("attention_mask must have shape [batch, time]")
            allowed = torch.ones(t, t, dtype=torch.bool, device=x.device).tril()[None, None] & attention_mask[:, None, None, :].bool()
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed,
            is_causal=allowed is None, dropout_p=self.config.dropout if self.training else 0.0)
        y = self.out_proj(y.transpose(1, 2).reshape(b, t, d))
        return y if attention_mask is None else y * attention_mask[..., None].to(y.dtype)


class CausalMLA(nn.Module):
    """Causal latent-KV attention anchor.

    This is a clean research implementation of the MLA idea: queries and KV
    content pass through low-rank latent bottlenecks, while a decoupled RoPE
    branch carries position. It is intentionally not claimed to be a drop-in
    reproduction of any vendor-specific kernel or checkpoint format.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.h = config.n_heads
        self.q_nope = config.mla_qk_nope_dim
        self.q_rope = config.mla_qk_rope_dim
        self.v_dim = config.mla_v_dim
        if self.q_rope % 2:
            raise ValueError("mla_qk_rope_dim must be even")

        self.q_a = nn.Linear(config.d_model, config.mla_q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.mla_q_lora_rank)
        self.q_b = nn.Linear(
            config.mla_q_lora_rank,
            self.h * (self.q_nope + self.q_rope),
            bias=False,
        )

        self.kv_a = nn.Linear(
            config.d_model,
            config.mla_kv_lora_rank + self.q_rope,
            bias=False,
        )
        self.kv_norm = RMSNorm(config.mla_kv_lora_rank)
        self.kv_b = nn.Linear(
            config.mla_kv_lora_rank,
            self.h * (self.q_nope + self.v_dim),
            bias=False,
        )
        self.out_proj = nn.Linear(self.h * self.v_dim, config.d_model, bias=False)
        self.dropout = config.dropout

    def forward(self, x: Tensor, *, attention_mask: Tensor | None = None) -> Tensor:
        b, t, _ = x.shape
        q_latent = self.q_norm(self.q_a(x))
        q = self.q_b(q_latent).view(b, t, self.h, self.q_nope + self.q_rope).transpose(1, 2)
        q_nope, q_pe = torch.split(q, (self.q_nope, self.q_rope), dim=-1)

        kv_raw = self.kv_a(x)
        kv_latent, k_pe = torch.split(kv_raw, (self.config.mla_kv_lora_rank, self.q_rope), dim=-1)
        kv = self.kv_b(self.kv_norm(kv_latent)).view(b, t, self.h, self.q_nope + self.v_dim).transpose(1, 2)
        k_nope, v = torch.split(kv, (self.q_nope, self.v_dim), dim=-1)

        positions = torch.arange(t, device=x.device)
        q_pe = apply_rope(q_pe, positions, self.config.rope_theta)
        # The positional key is shared across heads as in decoupled-RoPE MLA.
        k_pe = apply_rope(k_pe[:, None, :, :], positions, self.config.rope_theta).expand(-1, self.h, -1, -1)

        q_full = torch.cat((q_nope, q_pe), dim=-1)
        k_full = torch.cat((k_nope, k_pe), dim=-1)
        allowed = None
        if attention_mask is not None:
            if attention_mask.shape != (b, t):
                raise ValueError("attention_mask must have shape [batch, time]")
            key_ok = attention_mask[:, None, None, :].bool()
            causal = torch.ones(t, t, device=x.device, dtype=torch.bool).tril()
            allowed = causal[None, None, :, :] & key_ok
        # SDPA selects supported PyTorch attention kernels and avoids retaining
        # a Python-materialized score/weight tensor. MLA V width may differ.
        # A split can leave half-precision V rows unaligned for CUDA SDPA.
        # Keep the already-aligned laptop layout unchanged; copy only unsafe
        # view layouts (e.g. the 12-nope/16-value tiny config).
        if v.is_cuda and (v.storage_offset()*v.element_size() % 16 or any(stride*v.element_size() % 16 for stride in v.stride()[:-1])):
            v = v.contiguous()
        out = F.scaled_dot_product_attention(
            q_full, k_full, v, attn_mask=allowed if attention_mask is not None else None,
            is_causal=attention_mask is None,
            dropout_p=self.dropout if self.training else 0.0,
        ).transpose(1, 2).contiguous().view(b, t, self.h * self.v_dim)
        if attention_mask is not None:
            out = out * attention_mask[..., None].to(out.dtype)
        return self.out_proj(out)
