from __future__ import annotations

import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence


class IPASymbolTokenizer:
    """Lookup-table tokenizer over pre-segmented IPA units.

    该实现与 tng2p 的 tokenizer 思路一致：边界由上游 espeak-ng 预处理决定，
    tokenizer 只负责将空格分隔的 token 映射为 id。
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
        strict_inventory_check: bool = False,
    ):
        with open(vocab_file, "r", encoding="utf-8") as f:
            vocab = [line.rstrip("\n") for line in f if line.rstrip("\n") != ""]
        self._init_from_vocab(
            vocab=vocab,
            vocab_file=vocab_file,
            unk_token=unk_token,
            pad_token=pad_token,
            blank_token=blank_token,
            normalization=normalization,
            collapse_whitespace=collapse_whitespace,
            strip_text=strip_text,
            strict_inventory_check=strict_inventory_check,
        )

    @classmethod
    def from_vocab_list(
        cls,
        vocab: Sequence[str],
        *,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        blank_token: Optional[str] = None,
        normalization: str = "NFC",
        collapse_whitespace: bool = True,
        strip_text: bool = True,
        strict_inventory_check: bool = False,
    ) -> "IPASymbolTokenizer":
        obj = cls.__new__(cls)
        obj._init_from_vocab(
            vocab=list(vocab),
            vocab_file="<in_memory>",
            unk_token=unk_token,
            pad_token=pad_token,
            blank_token=blank_token,
            normalization=normalization,
            collapse_whitespace=collapse_whitespace,
            strip_text=strip_text,
            strict_inventory_check=strict_inventory_check,
        )
        return obj

    @classmethod
    def from_meta(
        cls,
        meta: Dict[str, object],
        *,
        vocab_key: str = "phoneme_labels",
        normalization: str = "NFC",
        collapse_whitespace: bool = True,
        strip_text: bool = True,
        strict_inventory_check: bool = False,
    ) -> "IPASymbolTokenizer":
        vocab = meta.get(vocab_key)
        if not isinstance(vocab, list) or not vocab:
            raise ValueError(f"meta[{vocab_key!r}] must be a non-empty list")
        if "<pad>" not in vocab:
            raise ValueError("phoneme_labels is missing <pad>")
        if "<unk>" not in vocab:
            raise ValueError("phoneme_labels is missing <unk>")

        return cls.from_vocab_list(
            vocab=vocab,
            unk_token="<unk>",
            pad_token="<pad>",
            blank_token=None,
            normalization=normalization,
            collapse_whitespace=collapse_whitespace,
            strip_text=strip_text,
            strict_inventory_check=strict_inventory_check,
        )

    def _init_from_vocab(
        self,
        *,
        vocab: List[str],
        vocab_file: str,
        unk_token: str,
        pad_token: str,
        blank_token: Optional[str],
        normalization: str,
        collapse_whitespace: bool,
        strip_text: bool,
        strict_inventory_check: bool,
    ) -> None:
        del strict_inventory_check  # kept for config backward-compat
        self.vocab_file = vocab_file
        self.unk_token = unk_token
        self.pad_token = pad_token
        # 与 tng2p/tokenizer.py 一致：blank 不作为词表类，也不参与 special token。
        # 为了兼容 NeMo 可能传入 blank_token 配置，这里仅保留属性但忽略其语义。
        self.blank_token = None
        self.normalization = normalization
        self.collapse_whitespace = collapse_whitespace
        self.strip_text = strip_text

        if len(vocab) != len(set(vocab)):
            raise ValueError("Duplicate tokens found in vocab")
        if unk_token not in vocab:
            raise ValueError(f"Missing required token: {unk_token}")
        if pad_token not in vocab:
            raise ValueError(f"Missing required token: {pad_token}")

        self.vocab: List[str] = vocab
        self.token2id: Dict[str, int] = {t: i for i, t in enumerate(vocab)}
        self.id2token: Dict[int, str] = {i: t for i, t in enumerate(vocab)}
        self.unk_id = self.token2id[unk_token]
        self.pad_id = self.token2id[pad_token]
        self.blank_id = None
        self.all_special_tokens = [pad_token, unk_token]
        self.special_token_set = {pad_token, unk_token}
        # NeMo 部分路径会访问 tokenizer.tokenizer
        self.tokenizer = self

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
        if not (0 <= idx < len(self.vocab)):
            raise ValueError(f"Unknown token id: {idx}")
        return self.vocab[idx]

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
        return text.split()

    def tokens_to_text(self, tokens: Iterable[str]) -> str:
        toks = [t for t in tokens if t not in self.special_token_set]
        return " ".join(toks)

    def tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
        t2i = self.token2id
        unk = self.unk_id
        return [t2i.get(t, unk) for t in tokens]

    def text_to_ids(self, text: str) -> List[int]:
        return self.tokens_to_ids(self.text_to_tokens(text))

    def ids_to_tokens(self, ids: Iterable[int]) -> List[str]:
        return [self.id_to_token(i) for i in ids]

    def ids_to_text(self, ids: Iterable[int]) -> str:
        return self.tokens_to_text(self.ids_to_tokens(ids))