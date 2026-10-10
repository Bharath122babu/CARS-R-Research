from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from .checkpoint import load_checkpoint
from .tokenizer import ByteTokenizer


@torch.no_grad()
def generate(
    checkpoint: str,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str | None = None,
) -> str:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _ = load_checkpoint(checkpoint, device=dev)
    model.eval()
    tok = ByteTokenizer(
        pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
    )
    ids = torch.tensor([tok.encode_prompt(prompt)], device=dev, dtype=torch.long)

    for _ in range(max_new_tokens):
        context = ids[:, -model.config.max_seq_len :]
        logits = model(context).logits[:, -1]
        if temperature <= 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            scores = logits / temperature
            if 0 < top_k < scores.shape[-1]:
                values, indices = torch.topk(scores, top_k, dim=-1)
                probs = F.softmax(values.float(), dim=-1)
                choice = torch.multinomial(probs, num_samples=1)
                nxt = indices.gather(-1, choice)
            else:
                nxt = torch.multinomial(F.softmax(scores.float(), dim=-1), num_samples=1)
        ids = torch.cat((ids, nxt), dim=1)
        if int(nxt.item()) == tok.eos_token_id:
            break
    return tok.decode(ids[0].tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference generation (recomputes prefix; no fused cache yet)")
    parser.add_argument("checkpoint")
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    print(generate(args.checkpoint, args.prompt, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k, device=args.device))


if __name__ == "__main__":
    main()
