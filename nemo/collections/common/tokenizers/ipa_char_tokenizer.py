from __future__ import annotations

import unicodedata
from typing import Dict, Iterable, List, Optional


class IpaCharTokenizer:
    """
    Minimal IPA character tokenizer for G2P targets.
    Uses only vocab.txt, no extra config file.
    """

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        blank_token: Optional[str] = None,
        normalization: str = "NFC",
        collapse_whitespace: bool = True,
        strip_text: bool = True,
    ):
        self.vocab_file = str(vocab_file)
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.blank_token = blank_token
        self.normalization = normalization
        self.collapse_whitespace = collapse_whitespace
        self.strip_text = strip_text

        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = [line.rstrip("\n") for line in f]

        if len(vocab) != len(set(vocab)):
            raise ValueError("Duplicate tokens found in vocab.txt")

        if unk_token not in vocab:
            raise ValueError(f"Missing required token: {unk_token}")
        if pad_token not in vocab:
            raise ValueError(f"Missing required token: {pad_token}")

        self.vocab: List[str] = vocab
        self.token2id: Dict[str, int] = {t: i for i, t in enumerate(vocab)}
        self.id2token: Dict[int, str] = {i: t for i, t in enumerate(vocab)}

        self.unk_id = self.token2id[unk_token]
        self.pad_id = self.token2id[pad_token]
        self.blank_id = self.token2id[blank_token] if blank_token and blank_token in self.token2id else None

        self.tokenizer = self
        self.all_special_tokens = [x for x in [pad_token, unk_token, blank_token] if x is not None]

    def __len__(self) -> int:
        return self.vocab_size

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.token2id)

    def token_to_id(self, token: str) -> int:
        return self.token2id.get(token, self.unk_id)

    def id_to_token(self, idx: int) -> str:
        idx = int(idx)
        if idx not in self.id2token:
            raise ValueError(f"Unknown token id: {idx}")
        return self.id2token[idx]

    def normalize(self, text: Optional[str]) -> str:
        if text is None:
            return ""
        if self.normalization:
            text = unicodedata.normalize(self.normalization, text)
        if self.collapse_whitespace:
            text = " ".join(text.split())
        if self.strip_text:
            text = text.strip()
        return text

    def text_to_tokens(self, text: str) -> List[str]:
        return list(self.normalize(text))

    def tokens_to_text(self, tokens: Iterable[str]) -> str:
        specials = set(self.all_special_tokens)
        toks = [t for t in tokens if t not in specials]
        return "".join(toks)

    def tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
        return [self.token_to_id(t) for t in tokens]

    def text_to_ids(self, text: str) -> List[int]:
        return self.tokens_to_ids(self.text_to_tokens(text))

    def ids_to_tokens(self, ids: Iterable[int]) -> List[str]:
        return [self.id_to_token(i) for i in ids]

    def ids_to_text(self, ids: Iterable[int]) -> str:
        return self.tokens_to_text(self.ids_to_tokens(ids))