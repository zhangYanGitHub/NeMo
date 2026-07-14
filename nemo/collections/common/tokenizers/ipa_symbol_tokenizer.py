from __future__ import annotations

import unicodedata
from typing import Dict, Iterable, List, Optional, Sequence

# Word-boundary token. The manifest ``text`` field (and the model's decoded output) is
# "phonemes concatenated within a word, single space between words" — e.g. "bˈɔɪ hˈaʊs".
# A literal space is used as the boundary token so that tokens_to_text() is just a plain
# concatenation and naturally reproduces that exact format. It MUST therefore be present as
# a real entry in vocab.txt (the preprocessing script writes it). NOTE: because it is a bare
# space, be careful not to let editors/tools strip trailing whitespace from that vocab line.
SPACE_TOKEN = " "


class IPASymbolTokenizer:
    """Lookup-table tokenizer over IPA phoneme atoms with word-internal longest-match.

    Input ``text`` is word-internally concatenated with single spaces between words
    ("bˈɔɪ hˈaʊs"). text_to_tokens() splits on spaces, greedily longest-matches each word
    against the vocab into atomic phoneme tokens (diphthongs/affricates/length-marked vowels
    are single vocab entries, so they stay atomic), and inserts SPACE_TOKEN between words.
    tokens_to_text() concatenates tokens, so SPACE_TOKEN reproduces the word boundary and
    word-internal atoms are glued back together — round-tripping the original string.
    """

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        blank_token: Optional[str] = None,
        normalization: Optional[str] = None,
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
        normalization: Optional[str] = None,
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
        normalization: Optional[str] = None,
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
        normalization: Optional[str],
        collapse_whitespace: bool,
        strip_text: bool,
        strict_inventory_check: bool,
    ) -> None:
        del strict_inventory_check  # accepted for config backward-compat; no longer used
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
        self.has_space_token = SPACE_TOKEN in self.token2id
        # Longest word-internal match only needs to consider real (non-special) tokens; cap
        # the candidate length at the longest such token so text_to_tokens() stays O(len*max).
        self._max_token_len = max(
            (len(t) for t in vocab if t not in self.special_token_set and t != SPACE_TOKEN),
            default=1,
        )
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
        # Default normalization=None: the vocab and manifest keep espeak/piper's NATIVE Unicode
        # form (e.g. ç = c + combining cedilla), so any Unicode normalization here would DIVERGE
        # from the vocab (NFC would compose ç -> U+00E7, which is absent from the decomposed vocab
        # and would fall to <unk>). Leave codepoints untouched unless a caller explicitly opts in.
        if text is None:
            return ""
        if self.normalization:
            text = unicodedata.normalize(self.normalization, text)
        if self.collapse_whitespace:
            text = " ".join(text.split())
        if self.strip_text:
            text = text.strip()
        return text

    def _segment_word(self, word: str) -> List[str]:
        """Greedy longest-match a single (space-free) word into vocab phoneme atoms.
        Multi-codepoint atoms (diphthongs/affricates/length-marked vowels, stress-prefixed
        variants, ...) win over their prefixes because we try longer candidates first. A
        position that matches nothing falls back to a single character, which token_to_id()
        will map to <unk>."""
        tokens: List[str] = []
        i, n = 0, len(word)
        while i < n:
            matched: Optional[str] = None
            hi = min(self._max_token_len, n - i)
            for length in range(hi, 0, -1):
                cand = word[i : i + length]
                if cand in self.token2id and cand not in self.special_token_set:
                    matched = cand
                    break
            if matched is None:
                tokens.append(word[i])
                i += 1
            else:
                tokens.append(matched)
                i += len(matched)
        return tokens

    def text_to_tokens(self, text: str) -> List[str]:
        # Words are space-separated; within a word phonemes are concatenated (no separators),
        # so split on spaces, longest-match each word, and re-insert SPACE_TOKEN between words.
        tokens: List[str] = []
        for word_idx, word in enumerate(text.split()):
            if word_idx > 0:
                tokens.append(SPACE_TOKEN)
            tokens.extend(self._segment_word(word))
        return tokens

    def tokens_to_text(self, tokens: Iterable[str]) -> str:
        # SPACE_TOKEN carries the word boundary and word-internal atoms are glued back
        # together, so a plain concatenation reproduces "phonemes-within-word, space-between".
        return "".join(t for t in tokens if t not in self.special_token_set)

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