"""Stationary-key update probe. This is not a language-model benchmark.

Transport and decay are intentionally excluded to isolate damage from writing.
Random associations test the update geometry, not learned semantic capability.
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint
from .model import CARSBlock


@torch.no_grad()
def probe(mixer, *, memories=16, distractors=64, trials=8, correlation=0.0, seed=42):
    if min(memories, trials) <= 0 or distractors < 0 or not 0 <= correlation < 1:
        raise ValueError("memories/trials must be positive, distractors nonnegative, correlation in [0,1)")
    device = mixer.in_proj.weight.device
    gen = torch.Generator(device=device).manual_seed(seed)
    count = memories + distractors
    shape = (trials, mixer.h, mixer.dk, mixer.rank)
    common = torch.randn(shape, generator=gen, device=device)
    keys = [F.normalize(correlation**0.5 * common + (1-correlation)**0.5 * torch.randn(shape, generator=gen, device=device), dim=-2) for _ in range(count)]
    values = [torch.randn(trials, mixer.h, mixer.dv, mixer.rank, generator=gen, device=device) for _ in range(count)]
    state = mixer.initial_state(trials, device=device, dtype=torch.float32).matrix
    def scores(matrix):
        pred = torch.stack([matrix @ k for k in keys[:memories]], dim=0)
        target = torch.stack(values[:memories], dim=0)
        return {
            "relative_rmse_per_head": ((pred-target).square().mean(dim=(0, 1, 3, 4)) / target.square().mean(dim=(0, 1, 3, 4))).sqrt().tolist(),
            "mean_cosine_per_head": F.cosine_similarity(pred.flatten(3), target.flatten(3), dim=-1).mean(dim=(0, 1)).tolist(),
        }
    with torch.autocast(device_type=device.type, enabled=False):
        for i in range(memories):
            state, _ = mixer._correct(state, keys[i], values[i])
        before = scores(state)
        remembered = torch.stack([state @ k for k in keys[:memories]])
        for i in range(memories, count):
            state, _ = mixer._correct(state, keys[i], values[i])
        after = scores(state)
        changed = torch.stack([state @ k for k in keys[:memories]]) - remembered
    return {
        "probe": "stationary-key write interference; transport/decay excluded",
        "write_mode": mixer.config.write_mode,
        "memories": memories, "distractors": distractors, "trials": trials,
        "key_correlation": correlation, "seed": seed,
        "before_distractors": before, "after_distractors": after,
        "historical_read_change_rms_per_head": changed.square().mean(dim=(0, 1, 3, 4)).sqrt().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--memories", type=int, default=16)
    parser.add_argument("--distractors", type=int, default=64)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--correlation", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    model, _ = load_checkpoint(args.checkpoint, device=args.device)
    results = [{"layer": i, **probe(block.mixer, memories=args.memories, distractors=args.distractors,
               trials=args.trials, correlation=args.correlation, seed=args.seed)}
               for i, block in enumerate(model.blocks) if isinstance(block, CARSBlock)]
    if not results:
        raise ValueError("checkpoint has no CARS layers")
    print(json.dumps({"layers": results}, indent=2))


if __name__ == "__main__":
    main()
