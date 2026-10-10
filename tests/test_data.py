from pathlib import Path
import json
import numpy as np
import torch
import pytest

from cars_r.data import TokenStream, document_aware_byte_tokens, prepare_byte_corpus


def test_document_aware_encoding_trains_bos_eos():
    raw = b"alpha\nbeta\ngamma\n"
    ids = document_aware_byte_tokens(raw, document_mode="lines")
    assert int(np.count_nonzero(ids == 257)) == 3
    assert int(np.count_nonzero(ids == 258)) == 3
    assert ids[0] == 257
    assert 258 in ids


def test_prepare_and_sample(tmp_path: Path):
    source = tmp_path / "text.txt"
    source.write_text(("abcdef0123456789\n" * 100), encoding="utf-8")
    binary = tmp_path / "tokens.bin"
    meta = prepare_byte_corpus(source, binary)
    assert meta["tokens"] > 100
    assert meta["special_tokens_inserted"] is True
    assert meta["bos_count"] == 100
    assert meta["eos_count"] == 100
    assert binary.exists()
    sidecar = binary.with_suffix(".bin.json")
    assert sidecar.exists()
    disk_meta = json.loads(sidecar.read_text())
    assert disk_meta["format_version"] == 2
    stream = TokenStream.from_bin(binary)
    g = torch.Generator().manual_seed(1)
    batch = stream.sample(3, 20, generator=g, device=torch.device("cpu"))
    assert batch.shape == (3, 20)
    assert batch.dtype == torch.long


def test_old_raw_byte_bin_is_rejected(tmp_path: Path):
    binary = tmp_path / "old.bin"
    np.arange(100, dtype=np.uint16).tofile(binary)
    binary.with_suffix(".bin.json").write_text(json.dumps({"dtype": "uint16"}))
    try:
        TokenStream.from_bin(binary)
    except ValueError as exc:
        assert "document boundary tokens" in str(exc)
    else:
        raise AssertionError("old corpus without special-token metadata must be rejected")


def test_train_val_split_respects_document_boundary(tmp_path: Path):
    source = tmp_path / "docs.txt"
    source.write_text("".join(f"doc-{i:03d}\n" for i in range(200)), encoding="utf-8")
    train = TokenStream.from_text(source, split="train")
    val = TokenStream.from_text(source, split="val")
    assert int(train.tokens[-1]) == 258
    assert int(val.tokens[0]) == 257


def test_truncated_binary_corpus_is_rejected(tmp_path):
    text = tmp_path / "docs.txt"
    text.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    binary = tmp_path / "tokens.bin"
    prepare_byte_corpus(text, binary)
    binary.write_bytes(binary.read_bytes()[:-2])
    with pytest.raises(ValueError, match="length does not match"):
        TokenStream.from_bin(binary)
