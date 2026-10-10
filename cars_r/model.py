from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .attention import CausalMLA, CausalSelfAttention
from .cars_operator import CARSDiagnostics, CARSState, CARSStateMixer
from .config import ModelConfig
from .delta_attention import DeltaMixer
from .layers import RMSNorm, SwiGLU


@dataclass
class ModelOutput:
    logits: Tensor
    mtp_logits: tuple[Tensor, ...]
    states: tuple[CARSState, ...]
    diagnostics: tuple[CARSDiagnostics | None, ...]
    hidden: Tensor | None = None


class CARSBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.mixer = CARSStateMixer(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        *,
        state: CARSState | None = None,
        attention_mask: Tensor | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[Tensor, CARSState, CARSDiagnostics | None]:
        mixed, state, diag = self.mixer(
            self.norm1(x),
            state=state,
            attention_mask=attention_mask,
            return_diagnostics=return_diagnostics,
        )
        x = x + self.dropout(mixed)
        x = x + self.dropout(self.ff(self.norm2(x)))
        if attention_mask is not None:
            x = x * attention_mask[..., None].to(x.dtype)
        return x, state, diag


class MLABlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalMLA(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, *, attention_mask: Tensor | None = None) -> Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask=attention_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        if attention_mask is not None:
            x = x * attention_mask[..., None].to(x.dtype)
        return x


class TransformerBlock(MLABlock):
    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.norm1 = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)


class DeltaBlock(MLABlock):
    def __init__(self, config: ModelConfig) -> None:
        # Keep the common residual/norm/FFN backbone, replace only the mixer.
        nn.Module.__init__(self)
        self.norm1 = RMSNorm(config.d_model)
        self.attn = DeltaMixer(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ff = SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)


class CARSRLanguageModel(nn.Module):
    """3:1 CARS:MLA hybrid by default, with no dynamic module routing."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        blocks: list[nn.Module] = []
        self._cars_block_indices: list[int] = []
        for i in range(config.n_layers):
            if config.layer_type(i) == "transformer":
                blocks.append(TransformerBlock(config))
            elif config.layer_type(i) in {"kda", "gdn2"}:
                blocks.append(DeltaBlock(config))
            elif config.layer_type(i) == "mla":
                blocks.append(MLABlock(config))
            else:
                self._cars_block_indices.append(i)
                blocks.append(CARSBlock(config))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mtp_heads = nn.ModuleList(
            nn.Linear(config.d_model, config.vocab_size, bias=False)
            for _ in range(max(config.mtp_steps - 1, 0))
        )
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.gradient_checkpointing = False
        self.apply(self._init_weights)
        # Re-apply architecture-aware CARS initialization after the generic
        # model-wide weight initialization.
        for block in self.blocks:
            if isinstance(block, CARSBlock):
                block.mixer.apply_architecture_initialization()
            elif isinstance(block, DeltaBlock):
                # Match FLA's low-rank output-gate bias initialization.
                nn.init.zeros_(block.attn.g_proj[-1].bias)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        self.gradient_checkpointing = bool(enabled)

    def num_parameters(self, *, trainable_only: bool = True) -> int:
        params = self.parameters()
        if trainable_only:
            return sum(p.numel() for p in params if p.requires_grad)
        return sum(p.numel() for p in params)

    def inference_num_parameters(self) -> int:
        """Parameter count excluding training-only MTP heads."""
        mtp = sum(p.numel() for head in self.mtp_heads for p in head.parameters())
        return self.num_parameters() - mtp

    @property
    def has_mla(self) -> bool:
        return any(isinstance(block, MLABlock) for block in self.blocks)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        states: Sequence[CARSState | None] | None = None,
        return_diagnostics: bool = False,
        return_hidden: bool = False,
    ) -> ModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, time]")
        b, t = input_ids.shape
        if t > self.config.max_seq_len:
            raise ValueError(f"sequence length {t} exceeds max_seq_len={self.config.max_seq_len}")
        if t == 0:
            raise ValueError("sequence must contain at least one token")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")

        if states is not None and self.has_mla:
            raise ValueError(
                "partial CARS-state continuation is invalid for hybrid CARS+MLA models because "
                "MLA KV/position cache continuation is not implemented. Recompute the full prefix "
                "or use a pure-CARS configuration until a unified cache object is available."
            )

        x = self.dropout(self.token_embedding(input_ids))
        state_inputs = list(states) if states is not None else [None] * len(self._cars_block_indices)
        if len(state_inputs) != len(self._cars_block_indices):
            raise ValueError("states length must match the number of CARS layers")

        out_states: list[CARSState] = []
        diagnostics: list[CARSDiagnostics | None] = []
        s_i = 0
        for block in self.blocks:
            if isinstance(block, CARSBlock):
                use_ckpt = (
                    self.training
                    and self.gradient_checkpointing
                    and states is None
                    and not return_diagnostics
                )
                if use_ckpt:
                    def cars_fn(inp: Tensor, _block: CARSBlock = block) -> tuple[Tensor, Tensor]:
                        out_x, out_s, _ = _block(
                            inp, state=None, attention_mask=attention_mask, return_diagnostics=False
                        )
                        return out_x, out_s.matrix
                    x, state_matrix = checkpoint(cars_fn, x, use_reentrant=False)
                    s = CARSState(state_matrix)
                    diag = None
                else:
                    x, s, diag = block(
                        x,
                        state=state_inputs[s_i],
                        attention_mask=attention_mask,
                        return_diagnostics=return_diagnostics,
                    )
                out_states.append(s)
                diagnostics.append(diag)
                s_i += 1
            else:
                if self.training and self.gradient_checkpointing and not return_diagnostics:
                    def mla_fn(inp: Tensor, _block: MLABlock = block) -> Tensor:
                        return _block(inp, attention_mask=attention_mask)
                    x = checkpoint(mla_fn, x, use_reentrant=False)
                else:
                    x = block(x, attention_mask=attention_mask)

        hidden = self.final_norm(x)
        logits = self.lm_head(hidden)
        mtp_logits = tuple(head(hidden) for head in self.mtp_heads)
        return ModelOutput(
            logits=logits,
            mtp_logits=mtp_logits,
            states=tuple(out_states),
            diagnostics=tuple(diagnostics),
            hidden=hidden if return_hidden else None,
        )
