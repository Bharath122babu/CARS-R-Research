from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


def _positive(name: str, value: int | float) -> None:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


@dataclass(slots=True)
class ModelConfig:
    architecture: str = "cars"  # cars/kda/gdn2 hybrid | conventional transformer
    # Input / backbone
    vocab_size: int = 259
    max_seq_len: int = 1024
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8
    d_ff: int = 1536
    dropout: float = 0.0
    tie_embeddings: bool = True

    # Static hybrid schedule. anchor_interval=4 => CARS,CARS,CARS,MLA.
    anchor_interval: int = 4

    # FLA KDA/GDN-2 reference mixers. One value head per query/key head.
    delta_key_dim: int = 32
    delta_value_dim: int = 32
    delta_conv_size: int = 4
    delta_allow_negative_eigenvalues: bool = False

    # CARS state geometry per head.
    cars_key_dim: int = 64
    cars_value_dim: int = 64
    cars_evidence_rank: int = 2
    cars_transport_max_strength: float = 0.35
    # Identity-biased initialization: only the transport-strength rows use this std.
    cars_transport_init_std: float = 1e-4
    # Static learned timescale priors are initialized log-spaced across heads.
    cars_half_life_min: float = 128.0
    cars_half_life_max: float = 8192.0
    cars_min_ridge: float = 1e-3
    cars_ridge_init: float = 1e-2
    cars_read_scale_init: float = 0.1
    cars_read_norm: bool = True
    # None preserves v0.1.1 cars_read_norm behavior. Explicit modes are ablations.
    cars_read_mode: str | None = None  # raw | rmsnorm | rmsclip
    cars_read_clip: float = 1.0  # fixed RMS bound before learned scale/projection
    # None preserves cars_correction_enabled. Additive is a live-writing control.
    cars_write_mode: str | None = None  # ridge | additive | none
    cars_transport_enabled: bool = True
    cars_transport_read_only: bool = False  # query-equivalent control; no persistent transport
    cars_correction_enabled: bool = True
    cars_read_before_write: bool = True

    # CARS reference recurrence arithmetic is intentionally FP32. This is not a
    # user-facing low-precision switch; future kernels may introduce one only
    # after numerical equivalence is qualified.
    cars_recurrence_fp32: bool = True

    # MLA anchor. This follows the low-rank latent-KV structure rather than
    # claiming bit-for-bit equivalence with any external implementation.
    mla_q_lora_rank: int = 128
    mla_kv_lora_rank: int = 128
    mla_qk_nope_dim: int = 48
    mla_qk_rope_dim: int = 16
    mla_v_dim: int = 64
    rope_theta: float = 10000.0

    # Training-only auxiliary future prediction heads.
    mtp_steps: int = 1

    pad_token_id: int = 256
    bos_token_id: int = 257
    eos_token_id: int = 258

    @property
    def read_mode(self) -> str:
        return self.cars_read_mode or ("rmsnorm" if self.cars_read_norm else "raw")

    @property
    def write_mode(self) -> str:
        return self.cars_write_mode or ("ridge" if self.cars_correction_enabled else "none")

    def validate(self) -> None:
        if self.architecture not in {"cars", "transformer", "kda", "gdn2"}:
            raise ValueError("architecture must be cars, transformer, kda, or gdn2")
        for name, value in self.to_dict().items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in (
            "vocab_size", "max_seq_len", "d_model", "n_layers", "n_heads",
            "d_ff", "anchor_interval", "cars_key_dim", "cars_value_dim",
            "cars_evidence_rank", "cars_half_life_min", "cars_half_life_max",
            "cars_ridge_init", "mla_q_lora_rank", "mla_kv_lora_rank",
            "mla_qk_nope_dim", "mla_qk_rope_dim", "mla_v_dim", "mtp_steps",
            "delta_key_dim", "delta_value_dim", "delta_conv_size",
        ):
            _positive(name, getattr(self, name))
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.architecture == "transformer" and (self.d_model // self.n_heads) % 2:
            raise ValueError("Transformer RoPE head dimension must be even")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.cars_transport_max_strength < 0.0:
            raise ValueError("cars_transport_max_strength cannot be negative")
        if self.cars_transport_init_std < 0.0:
            raise ValueError("cars_transport_init_std cannot be negative")
        if self.cars_half_life_max < self.cars_half_life_min:
            raise ValueError("cars_half_life_max must be >= cars_half_life_min")
        if self.cars_min_ridge <= 0.0:
            raise ValueError("cars_min_ridge must be positive")
        if self.cars_ridge_init <= self.cars_min_ridge:
            raise ValueError("cars_ridge_init must be > cars_min_ridge")
        if self.cars_read_scale_init <= 0.0:
            raise ValueError("cars_read_scale_init must be positive")
        if self.cars_read_mode not in {None, "raw", "rmsnorm", "rmsclip"}:
            raise ValueError("cars_read_mode must be raw, rmsnorm, or rmsclip")
        _positive("cars_read_clip", self.cars_read_clip)
        if self.cars_write_mode not in {None, "ridge", "additive", "none"}:
            raise ValueError("cars_write_mode must be ridge, additive, or none")
        if self.mla_qk_rope_dim % 2:
            raise ValueError("mla_qk_rope_dim must be even")
        _positive("rope_theta", self.rope_theta)
        if not self.cars_recurrence_fp32:
            raise ValueError(
                "the v0.1.1 reference operator only supports FP32 recurrence arithmetic; "
                "set cars_recurrence_fp32=true"
            )
        if self.mla_qk_nope_dim + self.mla_qk_rope_dim <= 0:
            raise ValueError("MLA query/key head dimensions must be positive")
        special = (self.pad_token_id, self.bos_token_id, self.eos_token_id)
        if len(set(special)) != 3:
            raise ValueError("special token IDs must be distinct")
        if any(i < 0 or i >= self.vocab_size for i in special):
            raise ValueError("special token IDs must fall inside vocab_size")

    def layer_type(self, index: int) -> str:
        if self.architecture == "transformer":
            return "transformer"
        return "mla" if (index + 1) % self.anchor_interval == 0 else self.architecture

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        cfg = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        cfg.validate()
        return cfg

    def save_json(self, path: str | Path) -> None:
        self.validate()
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass(slots=True)
class TrainConfig:
    seed: int = 1337
    batch_size: int = 8
    seq_len: int = 256
    max_steps: int = 10_000
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    state_lr_scale: float = 0.5
    min_lr_ratio: float = 0.1
    warmup_steps: int = 500
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    mtp_loss_weight: float = 0.15
    eval_interval: int = 250
    eval_batches: int = 20
    save_interval: int = 1000
    log_interval: int = 10
    precision: str = "bf16"  # fp32 | bf16 | fp16
    compile_model: bool = False
    gradient_checkpointing: bool = True
    # Self-contained byte-data policy. "lines" treats each physical line as a
    # document and inserts BOS/EOS; "file" treats the entire file as one doc.
    document_mode: str = "lines"
    health_interval: int = 250

    def validate(self) -> None:
        for name, value in self.to_dict().items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name in (
            "batch_size", "seq_len", "max_steps", "grad_accum_steps",
            "learning_rate", "warmup_steps", "grad_clip", "eval_interval",
            "eval_batches", "save_interval", "log_interval",
        ):
            value = getattr(self, name)
            if name == "warmup_steps":
                if value < 0:
                    raise ValueError("warmup_steps cannot be negative")
            else:
                _positive(name, value)
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")
        if self.document_mode not in {"lines", "file"}:
            raise ValueError("document_mode must be 'lines' or 'file'")
        if not 0.0 < self.min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in (0, 1]")
        if not 0.0 < self.state_lr_scale <= 1.0:
            raise ValueError("state_lr_scale must be in (0, 1]")
        if self.weight_decay < 0 or self.mtp_loss_weight < 0:
            raise ValueError("loss/decay weights cannot be negative")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.health_interval < 0:
            raise ValueError("health_interval cannot be negative (0 disables diagnostics)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        cfg = cls(**json.loads(Path(path).read_text(encoding="utf-8")))
        cfg.validate()
        return cfg
