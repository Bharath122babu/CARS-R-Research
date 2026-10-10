"""Generate from a comparison export, optionally using compiled CPU TorchScript."""
import argparse
import json
from pathlib import Path

import torch

from .generate import generate
from .tokenizer import ByteTokenizer


@torch.no_grad()
def compiled_generate(bundle, prompt, max_new_tokens):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "export.json").read_text(encoding="utf-8"))
    context = manifest["compiled_context"]
    model = torch.jit.load(str(bundle / manifest["compiled"]), map_location="cpu").eval()
    meta = json.loads((bundle / "tokenizer.json").read_text(encoding="utf-8"))
    tokenizer = ByteTokenizer(pad_token_id=meta["pad"], bos_token_id=meta["bos"], eos_token_id=meta["eos"])
    tokens = tokenizer.encode_prompt(prompt)
    for _ in range(max_new_tokens):
        active = tokens[-context:]
        ids = torch.full((1, context), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        ids[0, :len(active)] = torch.tensor(active)
        mask[0, :len(active)] = True
        logits = model(ids, mask)[0, len(active)-1].clone()
        logits[tokenizer.pad_token_id] = logits[tokenizer.bos_token_id] = -float("inf")
        following = int(logits.argmax())
        tokens.append(following)
        if following == tokenizer.eos_token_id:
            break
    return tokenizer.decode(tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("prompt")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--compiled", action="store_true", help="use fixed-context CPU TorchScript")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    args = parser.parse_args()
    if args.compiled:
        print(compiled_generate(args.bundle, args.prompt, args.max_new_tokens))
    else:
        print(generate(str(Path(args.bundle)/"model.pt"), args.prompt, max_new_tokens=args.max_new_tokens, device=args.device))


if __name__ == "__main__":
    main()
