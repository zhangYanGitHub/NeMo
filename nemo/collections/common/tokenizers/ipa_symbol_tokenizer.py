from __future__ import annotations

import unicodedata
import warnings
from typing import Dict, Iterable, List, Optional, Sequence


class _MultiTokenTrieNode:
    """Prefix tree node for longest-match over multi-character vocab tokens."""

    __slots__ = ("children", "token")

    def __init__(self) -> None:
        self.children: Dict[str, _MultiTokenTrieNode] = {}
        self.token: Optional[str] = None


def _build_multi_token_trie(multi_tokens: List[str]) -> _MultiTokenTrieNode:
    root = _MultiTokenTrieNode()
    for tok in multi_tokens:
        node = root
        for ch in tok:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = _MultiTokenTrieNode()
                node.children[ch] = nxt
            node = nxt
        node.token = tok
    return root


class IPASymbolTokenizer:
    """IPA tokenizer using vocab.txt or in-memory vocab with longest-match tokenization."""

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
            vocab = [line.rstrip("\n") for line in f]
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
            raise ValueError(f"meta[{vocab_key!r}] 必须是非空 list")

        if "<pad>" not in vocab:
            raise ValueError("phoneme_labels 中缺少 <pad>")
        if "<unk>" not in vocab:
            raise ValueError("phoneme_labels 中缺少 <unk>")

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
        self.vocab_file = vocab_file
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.blank_token = blank_token
        self.normalization = normalization
        self.collapse_whitespace = collapse_whitespace
        self.strip_text = strip_text
        self.strict_inventory_check = strict_inventory_check

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
        self.blank_id = self.token2id[blank_token] if blank_token and blank_token in self.token2id else None
        self.all_special_tokens = [x for x in (pad_token, unk_token, blank_token) if x is not None]
        self.special_token_set = set(self.all_special_tokens)

        lexical_tokens = [t for t in vocab if t not in self.special_token_set]
        self.multi_tokens = sorted([t for t in lexical_tokens if len(t) > 1], key=lambda x: (-len(x), x))
        self.single_tokens = set(t for t in lexical_tokens if len(t) == 1)
        self._multi_trie_root = _build_multi_token_trie(self.multi_tokens)

        self._check_token_inventory()
        self.tokenizer = self

    def _check_token_inventory(self) -> None:
        missing_parts = []
        for tok in self.multi_tokens:
            for ch in tok:
                if ch not in self.single_tokens and ch != " ":
                    missing_parts.append((tok, ch))

        if not missing_parts:
            return

        preview = ", ".join(f"{tok!r}->{ch!r}" for tok, ch in missing_parts[:10])
        message = (
            "Some multi-tokens contain characters missing from single-character vocab: "
            f"{preview}"
        )
        if self.strict_inventory_check:
            raise ValueError(message)
        warnings.warn(message)

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
        text = self.normalize(text)
        tokens: List[str] = []
        i = 0
        n = len(text)
        root = self._multi_trie_root

        while i < n:
            node = root
            j = i
            matched: Optional[str] = None

            while j < n:
                nxt = node.children.get(text[j])
                if nxt is None:
                    break
                node = nxt
                j += 1
                if node.token is not None:
                    matched = node.token

            if matched is not None:
                tokens.append(matched)
                i += len(matched)
                continue

            ch = text[i]
            if ch in self.single_tokens:
                tokens.append(ch)
            else:
                tokens.append(self.unk_token)
            i += 1

        return tokens

    def tokens_to_text(self, tokens: Iterable[str]) -> str:
        toks = [t for t in tokens if t not in self.special_token_set]
        return "".join(toks)

    def tokens_to_ids(self, tokens: Iterable[str]) -> List[int]:
        unk_id = self.unk_id
        t2i = self.token2id
        return [t2i.get(t, unk_id) for t in tokens]

    def text_to_ids(self, text: str) -> List[int]:
        return self.tokens_to_ids(self.text_to_tokens(text))

    def ids_to_tokens(self, ids: Iterable[int]) -> List[str]:
        return [self.id_to_token(i) for i in ids]

    def ids_to_text(self, ids: Iterable[int]) -> str:
        return self.tokens_to_text(self.ids_to_tokens(ids))