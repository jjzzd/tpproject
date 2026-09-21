from __future__ import annotations


class ByteTokenizer:
    """Dependency-free UTF-8 byte tokenizer for smoke tests and data plumbing.

    It is intentionally simple. Replace this adapter with the tokenizer shipped
    with the target pretrained model before production training.
    """

    pad_id = 0
    bos_id = 1
    eos_id = 2
    unk_id = 3
    vocab_size = 260

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[bool]]:
        payload = list(text.encode("utf-8", errors="replace"))
        ids = [self.bos_id] + [value + 4 for value in payload] + [self.eos_id]
        ids = ids[:max_length]
        if ids and ids[-1] != self.eos_id and len(ids) == max_length:
            ids[-1] = self.eos_id
        mask = [True] * len(ids)
        padding = max_length - len(ids)
        ids.extend([self.pad_id] * padding)
        mask.extend([False] * padding)
        return ids, mask

    def decode(self, ids: list[int]) -> str:
        payload = bytes(value - 4 for value in ids if 4 <= value < 260)
        return payload.decode("utf-8", errors="replace")

