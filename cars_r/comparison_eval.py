"""Shared held-out windows and ownership cases for both trained architectures."""
from __future__ import annotations

import json
import math
from pathlib import Path
import random

import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint
from .comparison_data import PEOPLE, relational_case
from .data import TokenStream
from .tokenizer import ByteTokenizer
from .train import autocast_context


@torch.no_grad()
def evaluate_checkpoint(checkpoint, directory, *, device="cuda", windows=128, cases=128, seed=20261004):
    dev = torch.device(device)
    model, payload = load_checkpoint(checkpoint, device=dev)
    model.eval()
    cfg = payload.get("train_config") or {}
    precision = cfg.get("precision", "fp32") if dev.type == "cuda" else "fp32"
    context = cfg.get("seq_len", min(128, model.config.max_seq_len))
    directory = Path(directory)
    result = {"checkpoint_step": payload["step"], "architecture": model.config.architecture,
        "parameters": model.inference_num_parameters(), "evaluation_seed": seed,
        "eval_context": context, "precision": precision}
    for label, filename in (("wiki_test", "wiki_test.bin"), ("mixed_test", "test.bin")):
        source = TokenStream.from_bin(directory / filename, split="all")
        generator = torch.Generator().manual_seed(seed)
        total = 0.0
        count = 0
        for _ in range(windows):
            batch = source.sample(1, context + 1, generator=generator, device=dev)
            with autocast_context(dev, precision):
                logits = model(batch[:, :-1]).logits
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1), reduction="sum")
            total += float(loss)
            count += context
        nll = total / count
        result[label] = {"windows": windows, "tokens_scored": count, "mean_token_nll": nll,
            "byte_token_perplexity": math.exp(min(nll, 20)), "bits_per_stream_token": nll / math.log(2),
            "note": "Sampled byte-token windows including BOS/EOS; not word perplexity or pure bits-per-byte."}
    tok = ByteTokenizer()
    iid = [json.loads(line) for line in (directory / "relations_test.jsonl").read_text(encoding="utf-8").splitlines()]
    random.Random(seed).shuffle(iid)
    iid = iid[:cases]
    rng = random.Random(seed + 1)
    ood = [{"prompt": p, "target": t} for p, t in (relational_case(rng, 5) for _ in range(cases))]
    for label, examples in (("ownership_iid", iid), ("ownership_longer_chain", ood)):
        correct = 0
        losses = []
        for example in examples:
            prompt = tok.encode_prompt(example["prompt"])
            target = tok.encode(example["target"], add_special_tokens=False)
            ids = torch.tensor([prompt + target[:-1]], device=dev)
            if ids.shape[1] > model.config.max_seq_len:
                raise ValueError("ownership case exceeds the model context; do not silently skip it")
            with autocast_context(dev, precision):
                logits = model(ids).logits[0].float()
            relevant = logits[len(prompt)-1:len(prompt)-1+len(target)]
            losses.append(float(F.cross_entropy(relevant, torch.tensor(target, device=dev))))
            choice = logits[-1, [ord(person) for person in PEOPLE]].argmax().item()
            correct += PEOPLE[choice] == example["target"].strip()
        result[label] = {"cases": len(examples), "mean_answer_token_nll": sum(losses)/len(losses),
            "owner_candidate_accuracy": correct/len(examples), "candidate_chance_accuracy": 1/len(PEOPLE),
            "note": "Candidate scoring after teacher-forcing the answer-space token; not unrestricted free generation."}
    return result
