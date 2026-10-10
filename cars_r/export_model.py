"""Export reloadable weights and a fixed-context CPU TorchScript model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .comparison_data import sha256
from .tokenizer import ByteTokenizer


class LogitsOnly(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, ids, mask):
        return self.model(ids, attention_mask=mask).logits


def export_model(checkpoint, destination, *, context=128):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    model, payload = load_checkpoint(checkpoint, device="cpu")
    model.eval()
    if not 1 <= context <= model.config.max_seq_len:
        raise ValueError("export context must fit model max_seq_len")
    save_checkpoint(destination / "model.pt", model, None, step=payload["step"],
        extra={"inference_only": True, "source_checkpoint_sha256": sha256(Path(checkpoint))})
    model.config.save_json(destination / "model_config.json")
    tokenizer = ByteTokenizer(pad_token_id=model.config.pad_token_id,
        bos_token_id=model.config.bos_token_id, eos_token_id=model.config.eos_token_id)
    (destination / "tokenizer.json").write_text(json.dumps({"type": "utf8-bytes", "vocab_size": model.config.vocab_size,
        "pad": tokenizer.pad_token_id, "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id}, indent=2), encoding="utf-8")
    wrapper = LogitsOnly(model).eval()
    ids = torch.full((1, context), model.config.pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    example = tokenizer.encode_prompt("A owns x.")[:context]
    ids[0, :len(example)] = torch.tensor(example)
    mask[0, :len(example)] = True
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            compiled = torch.jit.trace(wrapper, (ids, mask), check_trace=False, strict=False)
            compiled = torch.jit.freeze(compiled.eval())
            # Compare new token content and active lengths, not just the trace example.
            for length in sorted(set((1, min(23, context), context))):
                changed = torch.randint(0, 256, (1, context))
                active = torch.arange(context)[None] < length
                torch.testing.assert_close(compiled(changed, active), wrapper(changed, active), rtol=3e-4, atol=3e-5)
            compiled.save(str(destination / "compiled_cpu.pt"))
            loaded = torch.jit.load(str(destination / "compiled_cpu.pt"), map_location="cpu")
            torch.testing.assert_close(loaded(ids, mask), wrapper(ids, mask), rtol=3e-4, atol=3e-5)
    finally:
        torch.set_num_threads(threads)
    report = {"architecture": model.config.architecture, "source_step": payload["step"],
        "weights": "model.pt", "compiled": "compiled_cpu.pt", "compiled_backend": "TorchScript CPU",
        "compiled_batch_size": 1, "compiled_context": context, "pad_and_mask_required": True,
        "equivalence": "PASS on changed inputs and active lengths; saved artifact reloaded",
        "note": "Fixed batch/context CPU compilation, not a fused CUDA recurrence or dynamic-shape serving cache."}
    (destination / "export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("destination")
    parser.add_argument("--context", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(export_model(args.checkpoint, args.destination, context=args.context), indent=2))


if __name__ == "__main__":
    main()
