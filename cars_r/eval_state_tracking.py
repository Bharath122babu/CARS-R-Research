from __future__ import annotations

import argparse
import json
import random

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint
from .tokenizer import ByteTokenizer


def build_case(rng: random.Random, steps: int) -> tuple[str, str]:
    people = ["Ava", "Ben", "Cora", "Dev", "Eli", "Fia"]
    objects = ["key", "book", "coin", "ring"]
    owner = rng.choice(people)
    obj = rng.choice(objects)
    lines = [f"{owner} owns the {obj}."]
    for _ in range(steps):
        nxt = rng.choice([p for p in people if p != owner])
        lines.append(f"{owner} gives the {obj} to {nxt}.")
        owner = nxt
    prompt = " ".join(lines) + f" Who owns the {obj}? Answer:"
    return prompt, " " + owner


@torch.no_grad()
def score_target(model, tok: ByteTokenizer, prompt: str, target: str, device: torch.device) -> tuple[float, bool]:
    p = tok.encode_prompt(prompt)
    target_ids = tok.encode(target, add_special_tokens=False)
    ids = torch.tensor([p + target_ids], device=device)
    if ids.shape[1] > model.config.max_seq_len:
        return float("nan"), False
    logits = model(ids[:, :-1]).logits[0]
    start = len(p) - 1
    relevant = logits[start : start + len(target_ids)]
    labels = torch.tensor(target_ids, device=device)
    nll = F.cross_entropy(relevant, labels, reduction="mean")
    exact = torch.equal(relevant.argmax(dim=-1), labels)
    return float(nll), bool(exact)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic relational state-tracking diagnostic")
    parser.add_argument("checkpoint")
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    tok = ByteTokenizer(
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
    )
    rng = random.Random(args.seed)
    nlls = []
    correct = 0
    for _ in range(args.cases):
        prompt, target = build_case(rng, args.steps)
        nll, exact = score_target(model, tok, prompt, target, device)
        if nll == nll:
            nlls.append(nll)
            correct += int(exact)
    result = {
        "cases_scored": len(nlls),
        "steps": args.steps,
        "mean_target_nll": sum(nlls) / max(len(nlls), 1),
        "greedy_target_exact_rate": correct / max(len(nlls), 1),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
