from __future__ import annotations


class ByteTokenizer:
    """UTF-8 byte tokenizer used by the self-contained research starter.

    The model itself is vocabulary-agnostic. Special-token IDs are configurable
    so generation/evaluation can use the exact IDs stored in a checkpoint.
    """

    def __init__(
        self,
        *,
        pad_token_id: int = 256,
        bos_token_id: int = 257,
        eos_token_id: int = 258,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.bos_token_id = int(bos_token_id)
        self.eos_token_id = int(eos_token_id)
        self.vocab_size = max(256, self.pad_token_id + 1, self.bos_token_id + 1, self.eos_token_id + 1)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        ids = list(text.encode("utf-8"))
        return [self.bos_token_id, *ids, self.eos_token_id] if add_special_tokens else ids

    def encode_prompt(self, text: str) -> list[int]:
        return [self.bos_token_id, *text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        raw = bytes(i for i in ids if 0 <= i < 256)
        return raw.decode("utf-8", errors="replace")
