from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import hashlib
import math
import os
from pathlib import Path
import random
import platform
import time
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint import (
    capture_runtime_state,
    read_checkpoint_payload,
    restore_runtime_state,
    save_checkpoint,
)
from .config import ModelConfig, TrainConfig
from .data import TokenStream
from .losses import causal_lm_loss
from .model import CARSRLanguageModel
from .model import CARSBlock


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("multi-process training currently requires CUDA/NCCL")
        if not dist.is_nccl_available():
            raise RuntimeError("multi-GPU training requires an NCCL-enabled PyTorch build (use Linux); this build supports the single-GPU path")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world, local_rank, device


def is_main(rank: int) -> bool:
    return rank == 0


def seed_all(seed: int, rank: int) -> None:
    seed = seed + 1009 * rank
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def lr_multiplier(step: int, cfg: TrainConfig) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return max((step + 1) / cfg.warmup_steps, 1e-8)
    denom = max(cfg.max_steps - cfg.warmup_steps, 1)
    progress = min(max((step - cfg.warmup_steps) / denom, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cosine


def build_optimizer(model: CARSRLanguageModel, cfg: TrainConfig) -> torch.optim.Optimizer:
    matrix_decay: list[nn.Parameter] = []
    scalar_no_decay: list[nn.Parameter] = []
    state_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("raw_decay_rate", "raw_ridge", "raw_read_scale")):
            state_params.append(p)
        elif p.ndim >= 2 and not getattr(p, "_no_weight_decay", False):
            matrix_decay.append(p)
        else:
            scalar_no_decay.append(p)
    groups = [
        {"params": matrix_decay, "weight_decay": cfg.weight_decay, "lr": cfg.learning_rate},
        {"params": scalar_no_decay, "weight_decay": 0.0, "lr": cfg.learning_rate},
        {"params": state_params, "weight_decay": 0.0, "lr": cfg.learning_rate * cfg.state_lr_scale},
    ]
    return torch.optim.AdamW(groups, betas=(cfg.beta1, cfg.beta2), eps=1e-8)


def set_lr(optimizer: torch.optim.Optimizer, base_lr: float, state_scale: float, mult: float) -> None:
    optimizer.param_groups[0]["lr"] = base_lr * mult
    optimizer.param_groups[1]["lr"] = base_lr * mult
    optimizer.param_groups[2]["lr"] = base_lr * state_scale * mult


def reduce_mean(value: torch.Tensor, world: int) -> torch.Tensor:
    if world == 1:
        return value
    out = value.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out / world


def reduce_max(value: float, *, device: torch.device, world: int) -> float:
    if world == 1:
        return value
    t = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    source: TokenStream,
    cfg: TrainConfig,
    *,
    generator: torch.Generator,
    device: torch.device,
    world: int,
) -> dict[str, float]:
    model.eval()
    losses = []
    ar_losses = []
    for _ in range(cfg.eval_batches):
        batch = source.sample(cfg.batch_size, cfg.seq_len + 1, generator=generator, device=device)
        x, y = batch[:, :-1], batch[:, 1:]
        with autocast_context(device, cfg.precision):
            out = model(x)
            loss = causal_lm_loss(out, y, mtp_weight=cfg.mtp_loss_weight)
        if not torch.isfinite(loss.total):
            raise FloatingPointError("non-finite validation loss")
        losses.append(loss.total.detach().float())
        ar_losses.append(loss.autoregressive.detach().float())
    total = reduce_mean(torch.stack(losses).mean(), world)
    ar = reduce_mean(torch.stack(ar_losses).mean(), world)
    model.train()
    return {
        "val_loss": float(total),
        "val_ar_loss": float(ar),
        "val_ppl": float(torch.exp(ar.clamp(max=20.0))),
    }


@torch.no_grad()
def model_health(model: CARSRLanguageModel, batch: torch.Tensor, precision: str) -> dict[str, Any]:
    """Fixed held-out probe; never consumes training/validation sampler RNG."""
    training = model.training
    model.eval()
    try:
        with autocast_context(batch.device, precision):
            out = model(batch, return_diagnostics=True)
        layers = []
        for index, state, diag in zip(model._cars_block_indices, out.states, out.diagnostics):
            mixer = model.blocks[index].mixer
            assert diag is not None
            if not torch.isfinite(state.matrix).all():
                raise FloatingPointError(f"non-finite CARS state in layer {index}")
            layers.append({
                "layer": index,
                "state_norm_per_head": state.matrix.flatten(2).norm(dim=-1).mean(dim=0).tolist(),
                "half_life_per_head": mixer.state_half_life.tolist(),
                "ridge_per_head": mixer.ridge.tolist(),
                "read_scale_per_head": mixer.read_scale.tolist(),
                "mean_transport_strength": float(diag.transport_strength.mean()),
                "mean_innovation_norm": float(diag.innovation_norm.mean()),
                "mean_state_change": float(diag.state_change.mean()),
            })
        return {"cars_health": layers}
    finally:
        model.train(training)


def source_identity(args: argparse.Namespace, cfg: TrainConfig) -> dict[str, Any]:
    path = Path(args.text or args.bin)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    identity = {"kind": "text" if args.text else "bin", "sha256": digest.hexdigest(), "document_mode": cfg.document_mode}
    if args.bin:
        identity["metadata"] = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    val_path = getattr(args, "val_bin", None) or getattr(args, "val_text", None)
    if val_path:
        val_digest = hashlib.sha256(Path(val_path).read_bytes()).hexdigest()
        identity["validation_sha256"] = val_digest
        identity["validation_kind"] = "bin" if getattr(args, "val_bin", None) else "text"
        if getattr(args, "val_bin", None):
            identity["validation_metadata"] = json.loads(Path(val_path).with_suffix(Path(val_path).suffix + ".json").read_text(encoding="utf-8"))
    return identity


def _validate_bin_metadata(path: str | Path, model_cfg: ModelConfig) -> None:
    path = Path(path)
    meta_path = path.with_suffix(path.suffix + ".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = {
        "bos_token_id": model_cfg.bos_token_id,
        "eos_token_id": model_cfg.eos_token_id,
    }
    for key, value in expected.items():
        if int(meta.get(key, -1)) != int(value):
            raise ValueError(f"binary corpus {key}={meta.get(key)!r} does not match model {key}={value}")
    if int(meta.get("vocab_size", -1)) > model_cfg.vocab_size:
        raise ValueError("binary corpus vocabulary exceeds model vocab_size")


def load_sources(
    args: argparse.Namespace,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
) -> tuple[TokenStream, TokenStream]:
    if bool(args.text) == bool(args.bin):
        raise ValueError("provide exactly one of --text or --bin")
    external_val = getattr(args, "val_bin", None) or getattr(args, "val_text", None)
    if getattr(args, "val_bin", None) and getattr(args, "val_text", None):
        raise ValueError("provide only one of --val-bin or --val-text")
    if external_val:
        kwargs = dict(bos_token_id=model_cfg.bos_token_id, eos_token_id=model_cfg.eos_token_id, document_mode=train_cfg.document_mode)
        if args.bin:
            _validate_bin_metadata(args.bin, model_cfg)
            train_source = TokenStream.from_bin(args.bin, split="all")
        else:
            train_source = TokenStream.from_text(args.text, split="all", **kwargs)
        if getattr(args, "val_bin", None):
            _validate_bin_metadata(args.val_bin, model_cfg)
            val_source = TokenStream.from_bin(args.val_bin, split="all")
        else:
            val_source = TokenStream.from_text(args.val_text, split="all", **kwargs)
        for name in (args.bin, getattr(args, "val_bin", None)):
            if name:
                meta = json.loads(Path(name).with_suffix(Path(name).suffix + ".json").read_text(encoding="utf-8"))
                if meta.get("document_mode") != train_cfg.document_mode:
                    raise ValueError("binary corpus document_mode differs from train config")
        return train_source, val_source
    if args.text:
        kwargs = dict(
            bos_token_id=model_cfg.bos_token_id,
            eos_token_id=model_cfg.eos_token_id,
            document_mode=train_cfg.document_mode,
        )
        return (
            TokenStream.from_text(args.text, split="train", **kwargs),
            TokenStream.from_text(args.text, split="val", **kwargs),
        )
    _validate_bin_metadata(args.bin, model_cfg)
    meta = json.loads(Path(args.bin).with_suffix(Path(args.bin).suffix + ".json").read_text(encoding="utf-8"))
    if meta.get("document_mode") != train_cfg.document_mode:
        raise ValueError("binary corpus document_mode differs from train config")
    return TokenStream.from_bin(args.bin, split="train"), TokenStream.from_bin(args.bin, split="val")


def _configs_resume_compatible(saved: dict[str, Any] | None, current: TrainConfig) -> bool:
    if saved is None:
        return False
    # Exact trajectory continuation requires the full schedule configuration,
    # including max_steps, to remain unchanged. Deliberately changing the run
    # horizon should be treated as a new experiment rather than an exact resume.
    return TrainConfig(**saved).to_dict() == current.to_dict()


def _collect_runtime_states(
    local_state: dict[str, Any],
    *,
    rank: int,
    world: int,
) -> dict[str, dict[str, Any]] | None:
    if world == 1:
        return {"0": local_state}
    gathered: list[dict[str, Any] | None] | None = [None] * world if rank == 0 else None
    dist.gather_object(local_state, gathered, dst=0)
    if rank != 0:
        return None
    assert gathered is not None
    return {str(i): state for i, state in enumerate(gathered) if state is not None}


def _save_training_checkpoints(
    paths: list[Path],
    *,
    model: CARSRLanguageModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_gen: torch.Generator,
    val_gen: torch.Generator,
    step: int,
    train_cfg: TrainConfig,
    rank: int,
    world: int,
    run_contract: dict[str, Any],
    run_state: dict[str, Any] | None = None,
) -> None:
    local = capture_runtime_state(train_generator=train_gen, val_generator=val_gen, scaler=scaler)
    runtime_states = _collect_runtime_states(local, rank=rank, world=world)
    if rank == 0:
        assert runtime_states is not None
        for path in paths:
            save_checkpoint(
                path,
                model,
                optimizer,
                step=step,
                train_config=train_cfg,
                runtime_states=runtime_states,
                extra={"run_contract": run_contract, "run_state": run_state or {}},
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CARS-R PGR hybrid")
    parser.add_argument("--model-config", default="configs/tiny.json")
    parser.add_argument("--train-config", default="configs/train_smoke.json")
    parser.add_argument("--text", default=None, help="UTF-8 text file; BOS/EOS are inserted using document_mode")
    parser.add_argument("--bin", default=None, help="v0.1.1 document-aware pretokenized .bin with sidecar")
    parser.add_argument("--val-bin", default=None, help="separate full validation corpus; training then uses the entire supplied train corpus")
    parser.add_argument("--val-text", default=None)
    parser.add_argument("--out-dir", default="runs/cars")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--check-only", action="store_true", help="validate data and a forward/backward batch, then exit")
    parser.add_argument("--stop-after", type=int, default=None, help="stop after this many optimizer steps this invocation; schedule stays unchanged")
    args = parser.parse_args()

    rank, world, local_rank, device = setup_distributed()
    if args.device:
        if world > 1:
            raise ValueError("--device is only supported for single-process training")
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    model_cfg = ModelConfig.from_json(args.model_config)
    train_cfg = TrainConfig.from_json(args.train_config)
    if train_cfg.seq_len > model_cfg.max_seq_len:
        raise ValueError("train seq_len exceeds model max_seq_len")
    if args.stop_after is not None and args.stop_after <= 0:
        raise ValueError("--stop-after must be positive")
    if device.type == "cuda" and train_cfg.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support BF16; choose fp16 or fp32")
    if min(model_cfg.pad_token_id, model_cfg.bos_token_id, model_cfg.eos_token_id) < 256:
        raise ValueError("byte corpus special tokens must not overlap byte IDs 0..255")
    if model_cfg.mtp_steps > train_cfg.seq_len:
        raise ValueError("seq_len must be >= mtp_steps so all auxiliary heads receive supervision")
    out_dir = Path(args.out_dir)
    if not args.resume and not args.check_only and out_dir.exists() and (
        any(out_dir.glob("*.pt")) or (out_dir / "metrics.jsonl").exists()
    ):
        raise ValueError("out-dir already contains a run; use --resume or choose a new directory")

    seed_all(train_cfg.seed, rank)
    train_source, val_source = load_sources(args, model_cfg, train_cfg)
    for name, source in (("training", train_source), ("validation", val_source)):
        if len(source.tokens) < train_cfg.seq_len + 1:
            raise ValueError(f"{name} split has {len(source.tokens)} tokens; need at least {train_cfg.seq_len + 1}. Use more documents or a shorter seq_len.")
    run_contract = {
        "world_size": world, "device_type": device.type,
        "effective_precision": train_cfg.precision if device.type == "cuda" else "fp32",
        "source": source_identity(args, train_cfg),
        "torch_version": str(torch.__version__), "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_version": torch.version.cuda if device.type == "cuda" else None,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }

    base_model = CARSRLanguageModel(model_cfg).to(device)
    base_model.set_gradient_checkpointing(train_cfg.gradient_checkpointing)
    optimizer = build_optimizer(base_model, train_cfg)
    use_scaler = train_cfg.precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    train_gen = torch.Generator(device="cpu").manual_seed(train_cfg.seed + rank * 17)
    val_gen = torch.Generator(device="cpu").manual_seed(train_cfg.seed + 99991 + rank * 17)

    start_step = 0
    best_val_loss = float("inf")
    prior_training_seconds = 0.0
    if args.resume:
        payload = read_checkpoint_payload(args.resume, device="cpu")
        if ModelConfig(**payload["model_config"]).to_dict() != model_cfg.to_dict():
            raise ValueError("resume checkpoint model_config does not exactly match --model-config")
        if not _configs_resume_compatible(payload.get("train_config"), train_cfg):
            raise ValueError("resume checkpoint train_config differs; max_steps and the schedule must also remain unchanged")
        if payload.get("extra", {}).get("run_contract") != run_contract:
            raise ValueError("exact resume requires the same corpus, document policy, world size, device type, and effective precision; this checkpoint's run contract does not match")
        base_model.load_state_dict(payload["model_state"])
        if payload.get("optimizer_state") is None:
            raise ValueError("resume checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer_state"])
        runtime_states = payload.get("runtime_states") or {}
        rank_state = runtime_states.get(str(rank))
        if rank_state is None:
            raise ValueError(f"resume checkpoint has no runtime state for rank {rank}")
        restore_runtime_state(
            rank_state,
            train_generator=train_gen,
            val_generator=val_gen,
            scaler=scaler,
        )
        start_step = int(payload.get("step", 0))
        saved_run_state = payload.get("extra", {}).get("run_state", {})
        best_val_loss = float(saved_run_state.get("best_val_loss", float("inf")))
        prior_training_seconds = float(saved_run_state.get("training_seconds", 0.0))
        if not 0 <= start_step <= train_cfg.max_steps:
            raise ValueError("checkpoint step is outside the configured run horizon")

    if args.check_only:
        batch = train_source.sample(train_cfg.batch_size, train_cfg.seq_len + 1, generator=train_gen, device=device)
        with autocast_context(device, train_cfg.precision):
            output = base_model(batch[:, :-1])
            loss = causal_lm_loss(output, batch[:, 1:], mtp_weight=train_cfg.mtp_loss_weight).total
        if not torch.isfinite(loss):
            raise FloatingPointError("preflight loss is not finite")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), train_cfg.grad_clip, error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        validation_batch = val_source.sample(train_cfg.batch_size, train_cfg.seq_len + 1, generator=val_gen, device=device)
        base_model.eval()
        with torch.no_grad(), autocast_context(device, train_cfg.precision):
            val_loss = causal_lm_loss(base_model(validation_batch[:, :-1]), validation_batch[:, 1:], mtp_weight=train_cfg.mtp_loss_weight).total
        if not torch.isfinite(val_loss):
            raise FloatingPointError("preflight validation loss is not finite")
        print(json.dumps({"preflight": "PASS", "device": str(device), "effective_precision": run_contract["effective_precision"], "loss": float(loss.detach()), "val_loss": float(val_loss), "grad_norm": float(norm), "train_tokens": len(train_source.tokens), "val_tokens": len(val_source.tokens), "training_parameters": base_model.num_parameters(), "inference_parameters": base_model.inference_num_parameters(), "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None}))
        if world > 1:
            dist.destroy_process_group()
        return

    model: nn.Module = base_model
    if train_cfg.compile_model:
        model = torch.compile(model)
    if world > 1:
        unused = any(isinstance(block, CARSBlock) for block in base_model.blocks) and model_cfg.write_mode != "ridge"
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=unused)

    core_model = base_model
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "resolved_model_config.json").write_text(json.dumps(model_cfg.to_dict(), indent=2), encoding="utf-8")
        (out_dir / "resolved_train_config.json").write_text(json.dumps(train_cfg.to_dict(), indent=2), encoding="utf-8")
        (out_dir / "run_contract.json").write_text(json.dumps(run_contract, indent=2), encoding="utf-8")
        print(
            f"device={device} world={world} effective_precision={run_contract['effective_precision']} "
            f"train_parameters={core_model.num_parameters():,} "
            f"inference_parameters={core_model.inference_num_parameters():,}"
        )

    optimizer.zero_grad(set_to_none=True)
    log_path = out_dir / "metrics.jsonl"

    # Windowed metrics have exactly the same interval semantics as throughput.
    window_total = 0.0
    window_ar = 0.0
    window_mtp = 0.0
    window_grad = 0.0
    window_grad_samples = 0
    window_skipped = 0
    window_steps = 0
    window_tokens_per_rank = 0
    window_train_seconds = 0.0
    training_seconds = prior_training_seconds

    model.train()
    end_step = min(train_cfg.max_steps, start_step + args.stop_after) if args.stop_after else train_cfg.max_steps
    health_gen = torch.Generator().manual_seed(train_cfg.seed + 76543)
    health_batch = val_source.sample(1, train_cfg.seq_len, generator=health_gen, device=device) if train_cfg.health_interval else None
    for step in range(start_step, end_step):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_start = time.perf_counter()
        mult = lr_multiplier(step, train_cfg)
        set_lr(optimizer, train_cfg.learning_rate, train_cfg.state_lr_scale, mult)
        step_total = torch.zeros((), device=device)
        step_ar = torch.zeros((), device=device)
        step_mtp = torch.zeros((), device=device)

        for micro in range(train_cfg.grad_accum_steps):
            batch = train_source.sample(train_cfg.batch_size, train_cfg.seq_len + 1, generator=train_gen, device=device)
            x, y = batch[:, :-1], batch[:, 1:]
            sync_context = (
                model.no_sync() if isinstance(model, DDP) and micro < train_cfg.grad_accum_steps - 1 else nullcontext()
            )
            with sync_context:
                with autocast_context(device, train_cfg.precision):
                    out = model(x)
                    losses = causal_lm_loss(out, y, mtp_weight=train_cfg.mtp_loss_weight)
                    scaled_loss = losses.total / train_cfg.grad_accum_steps
                if not torch.isfinite(losses.total):
                    raise FloatingPointError(f"non-finite training loss at step {step}")
                scaler.scale(scaled_loss).backward()
            step_total += losses.total.detach()
            step_ar += losses.autoregressive.detach()
            step_mtp += losses.mtp.detach()
            window_tokens_per_rank += x.numel()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        finite_grad = bool(torch.isfinite(torch.as_tensor(grad_norm)))
        if not use_scaler and not finite_grad:
            raise FloatingPointError(f"non-finite gradient norm at step {step}: {grad_norm}")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_seconds = time.perf_counter() - train_start
        window_train_seconds += step_seconds
        training_seconds += step_seconds

        step_total /= train_cfg.grad_accum_steps
        step_ar /= train_cfg.grad_accum_steps
        step_mtp /= train_cfg.grad_accum_steps
        window_total += float(step_total)
        window_ar += float(step_ar)
        window_mtp += float(step_mtp)
        if finite_grad:
            window_grad += float(torch.as_tensor(grad_norm).detach().float())
            window_grad_samples += 1
        else:
            window_skipped += 1  # GradScaler skips this update and reduces scale.
        window_steps += 1

        if (step + 1) % train_cfg.log_interval == 0 or step + 1 == end_step:
            denom = max(window_steps, 1)
            avg_total = reduce_mean(torch.tensor(window_total / denom, device=device), world)
            avg_ar = reduce_mean(torch.tensor(window_ar / denom, device=device), world)
            avg_mtp = reduce_mean(torch.tensor(window_mtp / denom, device=device), world)
            avg_grad = reduce_mean(torch.tensor(window_grad / max(window_grad_samples, 1), device=device), world)
            slowest_train_seconds = reduce_max(window_train_seconds, device=device, world=world)
            global_tokens = window_tokens_per_rank * world
            metrics = {
                "step": step + 1,
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "architecture": model_cfg.architecture,
                "tokens_seen": (step + 1) * train_cfg.batch_size * train_cfg.seq_len * train_cfg.grad_accum_steps * world,
                "step_seconds": slowest_train_seconds / denom,
                "training_seconds": training_seconds,
                "estimated_training_minutes_remaining": (end_step - step - 1) * slowest_train_seconds / denom / 60,
                "peak_cuda_memory_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
                "train_loss": float(avg_total),
                "train_ar_loss": float(avg_ar),
                "train_mtp_loss": float(avg_mtp),
                "grad_norm": float(avg_grad),
                "lr": optimizer.param_groups[0]["lr"],
                "amp_skipped_updates_per_rank_mean": float(reduce_mean(torch.tensor(float(window_skipped), device=device), world)),
                "grad_scaler_scale": scaler.get_scale(),
                "tokens_per_second": global_tokens / max(slowest_train_seconds, 1e-9),
            }
            if is_main(rank):
                print(json.dumps(metrics))
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(metrics) + "\n")
            window_total = window_ar = window_mtp = window_grad = 0.0
            window_steps = 0
            window_grad_samples = window_skipped = 0
            window_tokens_per_rank = 0
            window_train_seconds = 0.0

        if (step + 1) % train_cfg.eval_interval == 0 or step + 1 == train_cfg.max_steps:
            metrics = {
                "step": step + 1,
                **evaluate(model, val_source, train_cfg, generator=val_gen, device=device, world=world),
            }
            if is_main(rank):
                print(json.dumps(metrics))
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(metrics) + "\n")
            if metrics["val_ar_loss"] < best_val_loss:
                best_val_loss = metrics["val_ar_loss"]
                _save_training_checkpoints([out_dir / "best.pt"], model=core_model, optimizer=optimizer,
                    scaler=scaler, train_gen=train_gen, val_gen=val_gen, step=step+1,
                    train_cfg=train_cfg, rank=rank, world=world, run_contract=run_contract,
                    run_state={"best_val_loss": best_val_loss, "training_seconds": training_seconds})

        if health_batch is not None and (step + 1) % train_cfg.health_interval == 0:
            health = {"step": step + 1, **model_health(core_model, health_batch, train_cfg.precision)}
            if is_main(rank):
                print(json.dumps(health))
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(health) + "\n")

        if (step + 1) % train_cfg.save_interval == 0:
            _save_training_checkpoints(
                [out_dir / f"step_{step + 1:08d}.pt", out_dir / "latest.pt"],
                model=core_model,
                optimizer=optimizer,
                scaler=scaler,
                train_gen=train_gen,
                val_gen=val_gen,
                step=step + 1,
                train_cfg=train_cfg,
                rank=rank,
                world=world,
                run_contract=run_contract,
                run_state={"best_val_loss": best_val_loss, "training_seconds": training_seconds},
            )

    _save_training_checkpoints(
        [out_dir / "latest.pt"] + ([out_dir / "final.pt"] if end_step == train_cfg.max_steps else []),
        model=core_model,
        optimizer=optimizer,
        scaler=scaler,
        train_gen=train_gen,
        val_gen=val_gen,
        step=end_step,
        train_cfg=train_cfg,
        rank=rank,
        world=world,
        run_contract=run_contract,
        run_state={"best_val_loss": best_val_loss, "training_seconds": training_seconds},
    )
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
