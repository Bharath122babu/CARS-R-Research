from __future__ import annotations

import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch

from .config import ModelConfig, TrainConfig
from .model import CARSRLanguageModel

CHECKPOINT_VERSION = 2


def capture_runtime_state(
    *,
    train_generator: torch.Generator | None = None,
    val_generator: torch.Generator | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, Any]:
    """Capture stochastic/runtime state required for trajectory-reproducible resume."""
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state(),
        "torch_cuda_rng_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_generator": None if train_generator is None else train_generator.get_state(),
        "val_generator": None if val_generator is None else val_generator.get_state(),
        "grad_scaler": None if scaler is None else scaler.state_dict(),
    }
    return state


def restore_runtime_state(
    state: dict[str, Any],
    *,
    train_generator: torch.Generator | None = None,
    val_generator: torch.Generator | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Restore runtime state captured by :func:`capture_runtime_state`."""
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu_rng"].cpu())
    cuda_states = state.get("torch_cuda_rng_all")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])
    if train_generator is not None and state.get("train_generator") is not None:
        train_generator.set_state(state["train_generator"].cpu())
    if val_generator is not None and state.get("val_generator") is not None:
        val_generator.set_state(state["val_generator"].cpu())
    if scaler is not None and state.get("grad_scaler") is not None:
        scaler.load_state_dict(state["grad_scaler"])


def _atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temp_path = Path(temp_name)
    try:
        # Flush file contents before atomic rename. This is a local-filesystem
        # durability measure; distributed/object filesystems may have different
        # guarantees and should use their native checkpoint layer.
        # Windows requires a writable handle for fsync/_commit.
        with os.fdopen(fd, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_path, target)
        try:
            dir_fd = os.open(target.parent, os.O_DIRECTORY)
        except (AttributeError, OSError):
            dir_fd = None
        if dir_fd is not None:
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def save_checkpoint(
    path: str | Path,
    model: CARSRLanguageModel,
    optimizer: torch.optim.Optimizer | None,
    *,
    step: int,
    train_config: TrainConfig | None = None,
    runtime_states: dict[str, dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "step": int(step),
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": None if optimizer is None else optimizer.state_dict(),
        "train_config": None if train_config is None else train_config.to_dict(),
        "runtime_states": runtime_states or {},
        "extra": extra or {},
    }
    _atomic_torch_save(payload, target)


def read_checkpoint_payload(path: str | Path, *, device: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    version = payload.get("checkpoint_version")
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {version!r}; expected {CHECKPOINT_VERSION}. "
            "v0.1 checkpoints are intentionally not treated as exact-resume checkpoints."
        )
    return payload


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CARSRLanguageModel, dict[str, Any]]:
    """Load a model checkpoint for evaluation/generation.

    Optimizer/runtime restoration belongs to the training loop because an
    optimizer must be constructed over the exact model instance whose state is
    being restored. This avoids the misleading old API that could load optimizer
    state into parameters owned by a different model.
    """
    payload = read_checkpoint_payload(path, device=device)
    config = ModelConfig(**payload["model_config"])
    config.validate()
    model = CARSRLanguageModel(config).to(device)
    model.load_state_dict(payload["model_state"])
    return model, payload
