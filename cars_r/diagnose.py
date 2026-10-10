from __future__ import annotations

import argparse
import json
import torch

from .checkpoint import load_checkpoint
from .tokenizer import ByteTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect learned CARS state dynamics on text")
    parser.add_argument("checkpoint")
    parser.add_argument("text")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    tok = ByteTokenizer(
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
    )
    ids = tok.encode_prompt(args.text)[: model.config.max_seq_len]
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        out = model(x, return_diagnostics=True)
    report = {"layers": []}
    for layer_no, diag in zip(model._cars_block_indices, out.diagnostics):
        assert diag is not None
        score = diag.state_change[0].mean(dim=-1)
        k = min(args.top, score.numel())
        values, positions = torch.topk(score, k)
        report["layers"].append({
            "layer": layer_no,
            "mean_state_change": float(score.mean()),
            "mean_innovation": float(diag.innovation_norm[0].mean()),
            "mean_transport_strength": float(diag.transport_strength[0].mean()),
            "largest_state_changes": [
                {"position": int(p), "score": float(v), "byte": int(ids[int(p)])}
                for v, p in zip(values, positions)
            ],
        })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
