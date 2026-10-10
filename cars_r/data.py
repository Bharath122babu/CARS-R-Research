from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


def document_aware_byte_tokens(
    raw: bytes,
    *,
    bos_token_id: int = 257,
    eos_token_id: int = 258,
    document_mode: str = "lines",
) -> np.ndarray:
    """Encode UTF-8 bytes with explicit document boundary tokens.

    `lines` (default): each physical line, including its newline bytes, is one
    document and becomes BOS + bytes + EOS. This makes BOS/EOS frequent enough
    to be trained rather than merely reserved vocabulary rows.

    `file`: the entire file is one document. This is useful for debugging but is
    not recommended for publication-scale corpora unless upstream data already
    contains document separators.
    """
    if document_mode not in {"lines", "file"}:
        raise ValueError("document_mode must be 'lines' or 'file'")
    if bos_token_id < 256 or eos_token_id < 256 or bos_token_id == eos_token_id:
        raise ValueError("BOS/EOS must be distinct and outside byte IDs 0..255")
    if document_mode == "file":
        docs = [raw]
    else:
        docs = raw.splitlines(keepends=True)
        if not docs:
            docs = [raw]

    total = sum(len(doc) + 2 for doc in docs)
    out = np.empty(total, dtype=np.int64)
    cursor = 0
    for doc in docs:
        out[cursor] = bos_token_id
        cursor += 1
        n = len(doc)
        if n:
            out[cursor : cursor + n] = np.frombuffer(doc, dtype=np.uint8).astype(np.int64, copy=False)
            cursor += n
        out[cursor] = eos_token_id
        cursor += 1
    return out


class TokenStream:
    """Contiguous document-aware token stream with random-window sampling.

    The included byte pipeline inserts BOS/EOS consistently during both direct
    text loading and `.bin` preparation. Windows may cross document boundaries;
    the explicit EOS/BOS pair is the boundary signal. We intentionally do not
    reset the recurrent state at boundaries in v0.1.1 so all baselines can use
    the same token stream semantics.
    """

    def __init__(
        self,
        tokens: np.ndarray | Tensor,
        *,
        split: str = "train",
        train_fraction: float = 0.98,
        eos_token_id: int | None = None,
    ) -> None:
        if not 0.5 <= train_fraction < 1.0:
            raise ValueError("train_fraction must be in [0.5, 1)")
        if isinstance(tokens, Tensor):
            arr = tokens.detach().cpu().numpy()
        else:
            arr = np.asarray(tokens)
        if arr.ndim != 1 or len(arr) < 4:
            raise ValueError("token stream must be a 1D array with at least 4 tokens")
        cut = max(2, min(len(arr) - 2, int(len(arr) * train_fraction)))
        if eos_token_id is not None:
            # Keep the train/validation split on a document boundary when the
            # stream carries EOS markers. Prefer the first EOS at/after the
            # nominal cut so the validation split starts at the next BOS.
            suffix = np.flatnonzero(arr[cut:] == eos_token_id)
            if suffix.size:
                candidate = cut + int(suffix[0]) + 1
                if 2 <= candidate <= len(arr) - 2:
                    cut = candidate
        if split == "all":
            self.tokens = arr
        elif split == "train":
            self.tokens = arr[:cut]
        elif split == "val":
            self.tokens = arr[cut:]
        else:
            raise ValueError("split must be train, val, or all")

    @classmethod
    def from_text(
        cls,
        path: str | Path,
        *,
        split: str = "train",
        train_fraction: float = 0.98,
        bos_token_id: int = 257,
        eos_token_id: int = 258,
        document_mode: str = "lines",
    ) -> "TokenStream":
        raw = Path(path).read_bytes()
        tokens = document_aware_byte_tokens(
            raw,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            document_mode=document_mode,
        )
        return cls(tokens, split=split, train_fraction=train_fraction, eos_token_id=eos_token_id)

    @classmethod
    def from_bin(cls, path: str | Path, *, split: str = "train", train_fraction: float = 0.98) -> "TokenStream":
        path = Path(path)
        meta_path = path.with_suffix(path.suffix + ".json")
        if not meta_path.exists():
            raise FileNotFoundError(f"missing metadata sidecar: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("format_version") != 2:
            raise ValueError("unsupported corpus format; prepare a format_version=2 corpus with document boundary tokens")
        if not meta.get("special_tokens_inserted", False):
            raise ValueError(
                f"{meta_path} does not declare document boundary tokens. Re-run `python -m cars_r.data` "
                "with v0.1.1 before serious training."
            )
        dtype = np.dtype(meta["dtype"])
        if dtype not in (np.dtype("uint16"), np.dtype("uint32")):
            raise ValueError("binary corpus dtype must be uint16 or uint32")
        if path.stat().st_size % dtype.itemsize or path.stat().st_size // dtype.itemsize != meta.get("tokens"):
            raise ValueError("binary corpus length does not match metadata (possibly truncated)")
        tokens = np.memmap(path, mode="r", dtype=dtype)
        return cls(
            tokens, split=split, train_fraction=train_fraction,
            eos_token_id=int(meta["eos_token_id"]),
        )

    def sample(
        self,
        batch_size: int,
        length: int,
        *,
        generator: torch.Generator,
        device: torch.device,
    ) -> Tensor:
        if len(self.tokens) < length:
            raise ValueError(f"split has {len(self.tokens)} tokens but sample length is {length}")
        max_start = len(self.tokens) - length
        starts = torch.randint(0, max_start + 1, (batch_size,), generator=generator).numpy()
        offsets = np.arange(length, dtype=np.int64)[None, :]
        indices = starts[:, None] + offsets
        batch = np.asarray(self.tokens[indices], dtype=np.int64)
        tensor = torch.from_numpy(batch)
        if device.type == "cuda":
            tensor = tensor.pin_memory()
        return tensor.to(device=device, dtype=torch.long, non_blocking=True)


def prepare_byte_corpus(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dtype: str = "uint16",
    bos_token_id: int = 257,
    eos_token_id: int = 258,
    document_mode: str = "lines",
) -> dict[str, int | str | bool]:
    raw = Path(input_path).read_bytes()
    if dtype not in {"uint16", "uint32"}:
        raise ValueError("dtype must be uint16 or uint32")
    if max(bos_token_id, eos_token_id) > np.iinfo(np.dtype(dtype)).max:
        raise ValueError("special token IDs exceed the selected corpus dtype")
    tokens = document_aware_byte_tokens(
        raw,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        document_mode=document_mode,
    )
    arr = tokens.astype(np.dtype(dtype), copy=False)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(target)
    meta: dict[str, int | str | bool] = {
        "format_version": 2,
        "dtype": dtype,
        "tokens": int(arr.size),
        "vocab_size": max(259, eos_token_id + 1, bos_token_id + 1),
        "tokenizer": "utf8-bytes+bos-eos",
        "source_bytes": len(raw),
        "document_mode": document_mode,
        "bos_token_id": int(bos_token_id),
        "eos_token_id": int(eos_token_id),
        "special_tokens_inserted": True,
        "bos_count": int(np.count_nonzero(arr == bos_token_id)),
        "eos_count": int(np.count_nonzero(arr == eos_token_id)),
    }
    target.with_suffix(target.suffix + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare document-aware self-contained byte-token corpus")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("uint16", "uint32"), default="uint16")
    parser.add_argument("--document-mode", choices=("lines", "file"), default="lines")
    parser.add_argument("--bos-token-id", type=int, default=257)
    parser.add_argument("--eos-token-id", type=int, default=258)
    args = parser.parse_args()
    meta = prepare_byte_corpus(
        args.input,
        args.output,
        dtype=args.dtype,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        document_mode=args.document_mode,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
