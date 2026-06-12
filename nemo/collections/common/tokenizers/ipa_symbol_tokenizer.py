# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import unicodedata
import warnings
from typing import Dict, Iterable, List, Optional


class IPASymbolTokenizer:
    """IPA tokenizer using vocab.txt with longest-match tokenization."""

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
        self.vocab_file = vocab_file
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.blank_token = blank_token
        self.normalization = normalization
        self.collapse_whitespace = collapse_whitespace
        self.strip_text = strip_text
        self.strict_inventory_check = strict_inventory_check

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
        self.all_special_tokens = [x for x in (pad_token, unk_token, blank_token) if x is not None]
        self.special_token_set = set(self.all_special_tokens)

        lexical_tokens = [t for t in vocab if t not in self.special_token_set]
        self.multi_tokens = sorted([t for t in lexical_tokens if len(t) > 1], key=lambda x: (-len(x), x))
        self.single_tokens = set(t for t in lexical_tokens if len(t) == 1)

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
        text = self.normalize(text)
        tokens: List[str] = []
        i = 0
        n = len(text)

        while i < n:
            matched = None
            for tok in self.multi_tokens:
                if text.startswith(tok, i):
                    matched = tok
                    break

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
        return [self.token_to_id(t) for t in tokens]

    def text_to_ids(self, text: str) -> List[int]:
        return self.tokens_to_ids(self.text_to_tokens(text))

    def ids_to_tokens(self, ids: Iterable[int]) -> List[str]:
        return [self.id_to_token(i) for i in ids]

    def ids_to_text(self, ids: Iterable[int]) -> str:
        return self.tokens_to_text(self.ids_to_tokens(ids))