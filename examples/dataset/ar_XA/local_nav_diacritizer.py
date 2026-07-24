#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local Arabic navigation diacritizer.

Builds high-confidence diacritization rules from an existing Piper dataset.jsonl
and applies them to plain or partially diacritized navigation text.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from piper_phonemize import tashkeel_run
except Exception:  # pragma: no cover - reported at runtime if fallback is needed
    tashkeel_run = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = REPO_ROOT / "train_data/output_data/ar_XA_20260609_111446/dataset.jsonl"
DEFAULT_NAV_LST_GLOB = REPO_ROOT / "text_for_voice_clone/ar_XA/final_train_data/ar_nav_*.lst"
DEFAULT_RULES = REPO_ROOT / "tools/ar_XA/resources/nav_diacritizer_rules.json"
DEFAULT_REPORT = REPO_ROOT / "tools/ar_XA/resources/nav_diacritizer_eval_report.json"
DEFAULT_SPLIT_DIR = REPO_ROOT / "tools/ar_XA/resources/nav_diacritizer_splits"
DEFAULT_LETTER_MAP = REPO_ROOT / "tools/ar_XA/latin_letter_readings_ar_XA.json"

ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
PRESENTATION_FORMS_RE = re.compile(r"[\uFB50-\uFDFF\uFE70-\uFEFF]")
TOKEN_RE = re.compile(
    r"[\u0621-\u064A][\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0621-\u064A]*"
    r"|[A-Z]"
    r"|\s+"
    r"|."
)
WORD_RE = re.compile(
    r"^[\u0621-\u064A][\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0621-\u064A]*$"
)
SHORT_VOWEL_AT_WORD_END_RE = re.compile(r"[\u064E\u064F\u0650]$")
PHRASE_BOUNDARY_PUNCTUATION = frozenset("،.؟!")
SENTENCE_END_PUNCTUATION = frozenset(".؟!")
STANDALONE_LATIN_LETTER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])")

CONTROL_CHARS = {
    "\u200b",
    "\u200c",
    "\u200d",
    "\ufeff",
    "\u2060",
}

FIXED_PHRASES: dict[tuple[str, ...], tuple[str, ...]] = {
    ("شارع", "الملك", "فهد"): ("شَارِعِ", "الْمَلِكِ", "فَهْدٍ"),
    ("إلى", "شارع", "الملك", "فهد"): ("إِلَى", "شَارِعِ", "الْمَلِكِ", "فَهْدٍ"),
    ("طريق", "الملك", "فهد"): ("طَرِيقِ", "الْمَلِكِ", "فَهْدٍ"),
    ("الزم", "المسار", "الأوسط"): ("اِلْزَمِ", "الْمَسَارَ", "الْأَوْسَطَ"),
    ("الزم", "المسار", "الأيمن"): ("اِلْزَمِ", "الْمَسَارَ", "الْأَيْمَنَ"),
    ("تجنب", "منحدر"): ("تَجَنَّبْ", "مَنْحَدَرَ"),
    ("على", "بعد"): ("عَلَى", "بُعْدِ"),
    ("ادخل", "الدوار"): ("اُدْخُلِ", "الدَّوَّارَ"),
    ("استمر", "على", "طريق"): ("اِسْتَمِرْ", "عَلَى", "طَرِيقِ"),
}


@dataclass(frozen=True, slots=True)
class Token:
    kind: str
    text: str
    plain: str


def normalize_text(text: str) -> str:
    """Normalize encoding and spacing without removing existing diacritics."""
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if ch not in CONTROL_CHARS)
    cleaned: list[str] = []
    for ch in text:
        category = unicodedata.category(ch)
        if category.startswith("C") and ch not in "\n\t":
            continue
        cleaned.append(ch)
    text = "".join(cleaned)
    text = text.replace("？", ".").replace("?", ".")
    text = text.replace("！", ".").replace("!", ".")
    text = text.replace("。", ".").replace("．", ".")
    text = text.replace(",", "،").replace("؛", "،").replace(";", "،")
    text = re.sub(r"[،]{2,}", "،", text)
    text = re.sub(r"[.]{2,}", ".", text)
    text = re.sub(r"\s+([،.؟!])", r"\1", text)
    text = re.sub(r"([،.؟!])(?=\S)", r"\1 ", text)
    text = re.sub(r"\s+", " ", text.replace("\u00a0", " ")).strip()
    return text


ARABIC_DIACRITICS = (
    "\u0610\u0611\u0612\u0613\u0614\u0615\u0616\u0617\u0618\u0619\u061A"
    "\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655"
    "\u0656\u0657\u0658\u0659\u065A\u065B\u065C\u065D\u065E\u065F\u0670"
    "\u06D6\u06D7\u06D8\u06D9\u06DA\u06DB\u06DC\u06DD\u06DE\u06DF\u06E0"
    "\u06E1\u06E2\u06E3\u06E4\u06E5\u06E6\u06E7\u06E8\u06E9\u06EA\u06EB"
    "\u06EC\u06ED\u0640"
)
_STRIP_DIACRITICS_TABLE = str.maketrans("", "", ARABIC_DIACRITICS)


def strip_diacritics(text: str) -> str:
    return text.translate(_STRIP_DIACRITICS_TABLE)


def has_diacritics(text: str) -> bool:
    return bool(ARABIC_DIACRITICS_RE.search(text))


def load_letter_map(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    letters = data.get("letters", data)
    if not isinstance(letters, dict):
        raise ValueError(f"Bad letter map format: {path}")
    return {str(k): str(v) for k, v in letters.items()}


def replace_latin_letters(
    text: str, letter_map: dict[str, str]
) -> tuple[str, set[int]]:
    """Expand standalone A-Z and return word ordinals created by expansion."""
    parts: list[str] = []
    protected_word_ordinals: set[int] = set()
    word_ordinal = 0
    cursor = 0
    for match in STANDALONE_LATIN_LETTER_RE.finditer(text):
        prefix = text[cursor : match.start()]
        parts.append(prefix)
        word_ordinal += len(word_tokens(prefix))

        letter = match.group(1)
        reading = letter_map.get(letter, letter)
        parts.append(reading)
        if reading != letter:
            reading_word_count = len(word_tokens(reading))
            protected_word_ordinals.update(
                range(word_ordinal, word_ordinal + reading_word_count)
            )
            word_ordinal += reading_word_count
        cursor = match.end()

    parts.append(text[cursor:])
    return "".join(parts), protected_word_ordinals


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    for match in TOKEN_RE.finditer(text):
        raw = match.group(0)
        if raw.isspace():
            tokens.append(Token("space", raw, raw))
        elif WORD_RE.match(raw):
            tokens.append(Token("word", raw, strip_diacritics(raw)))
        else:
            tokens.append(Token("other", raw, strip_diacritics(raw)))
    return tokens


def word_tokens(text: str) -> list[Token]:
    return [tok for tok in tokenize(text) if tok.kind == "word"]


def word_segments(tokens: list[Token]) -> list[list[int]]:
    """Return word-token indexes without crossing navigation punctuation."""
    segments: list[list[int]] = []
    current: list[int] = []
    for idx, tok in enumerate(tokens):
        if tok.kind == "word":
            current.append(idx)
        elif tok.text in PHRASE_BOUNDARY_PUNCTUATION and current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def rule_token_indexes(tokens: list[Token]) -> list[int]:
    """Return word and navigation-punctuation indexes used by n-gram rules."""
    return [
        idx
        for idx, tok in enumerate(tokens)
        if tok.kind == "word" or tok.text in PHRASE_BOUNDARY_PUNCTUATION
    ]


def grouped_chars(text: str) -> list[tuple[str, str]]:
    """Return base Arabic chars with following combining marks."""
    groups: list[tuple[str, str]] = []
    for ch in text:
        if ARABIC_DIACRITICS_RE.match(ch):
            if groups:
                base, marks = groups[-1]
                groups[-1] = (base, marks + ch)
            else:
                groups.append(("", ch))
        else:
            groups.append((ch, ""))
    return groups


def merge_preserving_existing(existing: str, proposed: str) -> str:
    """Keep existing marks; fill unmarked characters from proposed if aligned."""
    if not has_diacritics(existing):
        return proposed

    existing_groups = grouped_chars(existing)
    proposed_groups = grouped_chars(proposed)
    existing_base = "".join(base for base, _ in existing_groups)
    proposed_base = "".join(base for base, _ in proposed_groups)
    if existing_base != proposed_base:
        return existing

    out: list[str] = []
    for (base, marks), (_, proposed_marks) in zip(existing_groups, proposed_groups):
        out.append(base)
        out.append(marks if marks else proposed_marks)
    return "".join(out)


def make_key(words: Iterable[str]) -> str:
    return "\u241f".join(words)


def split_key(key: str) -> list[str]:
    return key.split("\u241f") if key else []


def split_grouped_indexes(
    texts: list[str], train_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """Split rows while keeping identical undiacritized lines in one split."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for idx, text in enumerate(texts):
        grouped[strip_diacritics(text)].append(idx)

    groups = list(grouped.values())
    random.Random(seed).shuffle(groups)

    train_indexes: list[int] = []
    test_indexes: list[int] = []
    for group in groups:
        if len(train_indexes) + len(group) <= train_size:
            train_indexes.extend(group)
        else:
            test_indexes.extend(group)

    if not train_indexes or not test_indexes:
        raise ValueError("Grouped split produced an empty train or test split")

    random.Random(seed + 1).shuffle(train_indexes)
    random.Random(seed + 2).shuffle(test_indexes)
    return train_indexes, test_indexes


def extract_rules(
    texts: list[str],
    max_ngram: int,
    min_count: int,
    min_confidence: float,
) -> dict[str, dict[str, str]]:
    counts: dict[int, dict[str, Counter[str]]] = {
        n: defaultdict(Counter) for n in range(1, max_ngram + 1)
    }

    for text in texts:
        tokens = tokenize(text)
        indexes = rule_token_indexes(tokens)
        plain = [
            tokens[idx].plain if tokens[idx].kind == "word" else tokens[idx].text
            for idx in indexes
        ]
        diac = [tokens[idx].text for idx in indexes]
        for n in range(1, max_ngram + 1):
            if len(indexes) < n:
                continue
            for i in range(0, len(indexes) - n + 1):
                plain_key = make_key(plain[i : i + n])
                diac_key = make_key(diac[i : i + n])
                counts[n][plain_key][diac_key] += 1

    rules: dict[str, dict[str, str]] = {}
    for n, table in counts.items():
        n_rules: dict[str, str] = {}
        for plain_key, variants in table.items():
            total = sum(variants.values())
            best, best_count = variants.most_common(1)[0]
            confidence = best_count / total
            if n == 1:
                n_min_count = min_count
                n_min_conf = min_confidence
            elif n == 2:
                n_min_count = max(2, min_count - 1)
                n_min_conf = max(0.80, min_confidence - 0.1)
            else:
                # Longer navigation phrases are often sparse but reliable when
                # their undiacritized skeleton has only one observed reading.
                n_min_count = 1
                n_min_conf = 1.0
            if best_count >= n_min_count and confidence >= n_min_conf:
                n_rules[plain_key] = best
        rules[str(n)] = n_rules
    return rules


def extract_line_rules(texts: list[str]) -> dict[str, str]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for text in texts:
        plain = strip_diacritics(text)
        counts[plain][text] += 1
    rules: dict[str, str] = {}
    for plain, variants in counts.items():
        if len(variants) == 1:
            rules[plain] = variants.most_common(1)[0][0]
    return rules


def load_dataset_texts(path: Path) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = str(obj.get("text", "")).strip()
            if not text:
                raise ValueError(f"line {line_no}: missing text")
            texts.append(normalize_text(text))
    return texts


def load_lst_texts(paths: list[Path]) -> list[str]:
    texts: list[str] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line_no, raw in enumerate(f, 1):
                line = raw.rstrip("\n")
                if not line.strip():
                    continue
                if "|" in line:
                    text = line.split("|", 1)[1].strip()
                elif "\t" in line:
                    text = line.split("\t", 1)[1].strip()
                else:
                    text = line.strip()
                if not text:
                    raise ValueError(f"{path}:{line_no}: missing text")
                texts.append(normalize_text(text))
    return texts


def expand_lst_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        if Path(pattern).is_absolute():
            matches = sorted(Path("/").glob(pattern.lstrip("/")))
        else:
            matches = sorted(Path().glob(pattern))
        if matches:
            paths.extend(matches)
            continue
        literal = Path(pattern)
        if literal.exists():
            paths.append(literal)
            continue
        raise FileNotFoundError(f"No .lst files matched: {pattern}")
    return paths


def fallback_tashkeel(text: str) -> str:
    if tashkeel_run is None:
        raise RuntimeError(
            "piper_phonemize.tashkeel_run is unavailable; "
            "cannot diacritize words not covered by rules"
        )
    try:
        return normalize_text(tashkeel_run(text))
    except Exception as exc:
        raise RuntimeError(f"tashkeel_run failed for text: {text[:80]!r}") from exc


class LocalNavDiacritizer:
    def __init__(self, rules: dict[str, Any], letter_map: dict[str, str]) -> None:
        self.rules = rules
        self.letter_map = letter_map
        self.fixed_rules: dict[int, dict[str, list[str]]] = defaultdict(dict)
        for plain_words, diac_words in FIXED_PHRASES.items():
            self.fixed_rules[len(plain_words)][make_key(plain_words)] = list(diac_words)
        self.line_rules: dict[str, str] = {
            str(k): str(v) for k, v in rules.get("line_rules", {}).items()
        }
        raw_ngram_rules = rules.get("ngram_rules", {})
        self.ngram_rules: dict[int, dict[str, list[str]]] = {}
        for n_str, mapping in raw_ngram_rules.items():
            n = int(n_str)
            self.ngram_rules[n] = {
                plain_key: split_key(diac_key) for plain_key, diac_key in mapping.items()
            }
        self.max_ngram = max(self.ngram_rules.keys(), default=1)

    @staticmethod
    def _next_non_space_token(tokens: list[Token], idx: int) -> Token | None:
        for token in tokens[idx + 1 :]:
            if token.kind != "space":
                return token
        return None

    @staticmethod
    def _last_base_has_existing_marks(word: str) -> bool:
        groups = [(base, marks) for base, marks in grouped_chars(word) if base]
        return bool(groups and groups[-1][1])

    def _fixed_reading(
        self,
        plain_words: list[str],
        start: int,
        length: int,
        segment_indexes: list[int],
        tokens: list[Token],
    ) -> list[str] | None:
        key = make_key(plain_words[start : start + length])
        reading = self.fixed_rules.get(length, {}).get(key)
        if not reading:
            return None

        result = list(reading)
        if plain_words[start + length - 1] != "فهد":
            return result

        token_idx = segment_indexes[start + length - 1]
        next_token = self._next_non_space_token(tokens, token_idx)
        if next_token and next_token.text in SENTENCE_END_PUNCTUATION:
            result[-1] = "فَهْدْ"
        return result

    def _render(
        self,
        tokens: list[Token],
        proposed: dict[int, str],
        protected_indexes: set[int],
    ) -> str:
        out: list[str] = []
        for idx, tok in enumerate(tokens):
            if tok.kind != "word":
                out.append(tok.text)
                continue
            if idx in protected_indexes:
                rendered = tok.text
            else:
                candidate = proposed.get(idx, tok.text)
                rendered = merge_preserving_existing(tok.text, candidate)

            next_token = self._next_non_space_token(tokens, idx)
            if (
                next_token
                and next_token.text in SENTENCE_END_PUNCTUATION
                and not self._last_base_has_existing_marks(tok.text)
            ):
                rendered = SHORT_VOWEL_AT_WORD_END_RE.sub("\u0652", rendered)
            out.append(rendered)
        return "".join(out)

    def _apply_line_rule(
        self,
        tokens: list[Token],
        line_rule: str,
        protected_indexes: set[int],
    ) -> str | None:
        rule_tokens = tokenize(line_rule)
        input_words = [tok for tok in tokens if tok.kind == "word"]
        rule_words = [tok.text for tok in rule_tokens if tok.kind == "word"]
        if len(input_words) != len(rule_words):
            return None
        if any(tok.plain != strip_diacritics(rule) for tok, rule in zip(input_words, rule_words)):
            return None

        proposed: dict[int, str] = {}
        for idx, rule_word in zip(
            (idx for idx, tok in enumerate(tokens) if tok.kind == "word"),
            rule_words,
        ):
            proposed[idx] = rule_word
        return self._render(tokens, proposed, protected_indexes)

    def diacritize_line(self, line: str) -> str:
        original = line.rstrip("\n")
        if not original.strip():
            return original

        text = normalize_text(original)
        text, protected_word_ordinals = replace_latin_letters(text, self.letter_map)
        line_rule = self.line_rules.get(strip_diacritics(text))
        tokens = tokenize(text)
        word_indexes_in_order = [
            idx for idx, tok in enumerate(tokens) if tok.kind == "word"
        ]
        protected_indexes = {
            word_indexes_in_order[ordinal]
            for ordinal in protected_word_ordinals
            if ordinal < len(word_indexes_in_order)
        }
        if line_rule:
            rendered_line_rule = self._apply_line_rule(
                tokens, line_rule, protected_indexes
            )
            if rendered_line_rule is not None:
                return rendered_line_rule

        proposed: dict[int, str] = {}
        all_word_indexes = word_indexes_in_order
        indexes = rule_token_indexes(tokens)
        # Single-word rows (general_words / POI): no cross-word context, so cap ngram
        # at 1 — unless punctuation-aware rules apply (e.g. فهد + .), which need n>=2.
        is_word_unigram = len(all_word_indexes) == 1
        ngram_cap = 1 if is_word_unigram and len(indexes) == 1 else self.max_ngram
        skip_fixed_override = is_word_unigram and len(indexes) == 1

        for word_indexes in word_segments(tokens):
            plain_words = [tokens[idx].plain for idx in word_indexes]

            i = 0
            while i < len(word_indexes):
                matched = False
                max_n = min(self.max_ngram, len(word_indexes) - i)
                for n in range(max_n, 0, -1):
                    diac_words = self._fixed_reading(
                        plain_words, i, n, word_indexes, tokens
                    )
                    if diac_words:
                        for offset, diac_word in enumerate(diac_words):
                            proposed[word_indexes[i + offset]] = diac_word
                        i += n
                        matched = True
                        break

                if matched:
                    continue

                i += 1

        rule_plain = [
            tokens[idx].plain if tokens[idx].kind == "word" else tokens[idx].text
            for idx in indexes
        ]
        i = 0
        while i < len(indexes):
            matched = False
            max_n = min(ngram_cap, len(indexes) - i)
            for n in range(max_n, 0, -1):
                table = self.ngram_rules.get(n)
                if not table:
                    continue
                key = make_key(rule_plain[i : i + n])
                diac_items = table.get(key)
                if not diac_items:
                    continue
                for offset, diac_item in enumerate(diac_items):
                    token_idx = indexes[i + offset]
                    if tokens[token_idx].kind == "word":
                        proposed[token_idx] = diac_item
                i += n
                matched = True
                break
            if not matched:
                i += 1

        if not skip_fixed_override:
            for word_indexes in word_segments(tokens):
                plain_words = [tokens[idx].plain for idx in word_indexes]
                # Fixed phrases override statistical matches, but stay within the
                # current punctuation-delimited segment.
                fixed_lengths = sorted(self.fixed_rules.keys(), reverse=True)
                for i in range(len(word_indexes)):
                    for n in fixed_lengths:
                        if i + n > len(word_indexes):
                            continue
                        diac_words = self._fixed_reading(
                            plain_words, i, n, word_indexes, tokens
                        )
                        if not diac_words:
                            continue
                        for offset, diac_word in enumerate(diac_words):
                            proposed[word_indexes[i + offset]] = diac_word
                        break

        # Run Tashkeel only when at least one word is not covered by rules.
        if any(idx not in proposed for idx in all_word_indexes):
            fallback_text = fallback_tashkeel(strip_diacritics(text))
            fallback_tokens = tokenize(fallback_text)
            fallback_words = [tok.text for tok in fallback_tokens if tok.kind == "word"]
            if len(fallback_words) == len(all_word_indexes):
                for idx, fallback_word in zip(all_word_indexes, fallback_words):
                    proposed.setdefault(idx, fallback_word)
            else:
                # If tashkeel changes the word count, we cannot safely align word-by-word
                # to apply `proposed` rules. We must return the raw tashkeel output.
                # This happens when tashkeel merges or splits words (e.g., handling
                # certain Arabic ligatures or punctuation edge cases).
                return fallback_text

        return self._render(tokens, proposed, protected_indexes)


def diacritize_texts(
    texts: list[str], rules: dict[str, Any], letter_map: dict[str, str]
) -> list[str]:
    engine = LocalNavDiacritizer(rules, letter_map)
    return [engine.diacritize_line(text) for text in texts]


def evaluate(
    gold_texts: list[str],
    pred_texts: list[str],
    letter_map: dict[str, str],
) -> dict[str, Any]:
    if len(gold_texts) != len(pred_texts):
        raise ValueError("gold/pred length mismatch")

    sentence_matches = 0
    word_matches = 0
    word_total = 0
    skeleton_matches = 0
    latin_remaining = 0
    presentation_remaining = 0

    examples: list[dict[str, Any]] = []
    for idx, (gold, pred) in enumerate(zip(gold_texts, pred_texts), 1):
        if gold == pred:
            sentence_matches += 1
        elif len(examples) < 30:
            examples.append({"line": idx, "gold": gold, "pred": pred})

        if strip_diacritics(gold) == strip_diacritics(pred):
            skeleton_matches += 1

        gold_words = [tok.text for tok in word_tokens(gold)]
        pred_words = [tok.text for tok in word_tokens(pred)]
        word_total += max(len(gold_words), len(pred_words))
        for g_word, p_word in zip(gold_words, pred_words):
            if g_word == p_word:
                word_matches += 1

        if re.search(r"(?<![A-Za-z])[A-Z](?![A-Za-z])", pred):
            latin_remaining += 1
        if PRESENTATION_FORMS_RE.search(pred):
            presentation_remaining += 1

    total = len(gold_texts)
    return {
        "total": total,
        "sentence_accuracy": sentence_matches / total if total else 0.0,
        "word_accuracy": word_matches / word_total if word_total else 0.0,
        "skeleton_match_rate": skeleton_matches / total if total else 0.0,
        "latin_remaining_lines": latin_remaining,
        "presentation_form_lines": presentation_remaining,
        "mismatch_examples": examples,
    }


def build_rules_command(args: argparse.Namespace) -> None:
    source_files: list[str]
    if args.lst_glob:
        lst_paths = expand_lst_globs(args.lst_glob)
        texts = load_lst_texts(lst_paths)
        source_files = [str(path) for path in lst_paths]
        source_kind = "lst"
    else:
        texts = load_dataset_texts(args.dataset_jsonl)
        source_files = [str(args.dataset_jsonl)]
        source_kind = "dataset_jsonl"

    if args.train_size <= 0 or args.train_size >= len(texts):
        raise ValueError(
            f"--train-size must be between 1 and {len(texts) - 1}, got {args.train_size}"
        )

    train_indexes, test_indexes = split_grouped_indexes(
        texts, train_size=args.train_size, seed=args.seed
    )
    train_texts = [texts[i] for i in train_indexes]
    test_texts = [texts[i] for i in test_indexes]

    args.split_output_dir.mkdir(parents=True, exist_ok=True)
    train_split_path = args.split_output_dir / "train_split.jsonl"
    test_split_path = args.split_output_dir / "test_split.jsonl"
    with train_split_path.open("w", encoding="utf-8") as f:
        for split_idx, original_idx in enumerate(train_indexes, 1):
            row = {
                "split_index": split_idx,
                "dataset_index": original_idx + 1,
                "text": texts[original_idx],
                "plain_text": strip_diacritics(texts[original_idx]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with test_split_path.open("w", encoding="utf-8") as f:
        for split_idx, original_idx in enumerate(test_indexes, 1):
            row = {
                "split_index": split_idx,
                "dataset_index": original_idx + 1,
                "text": texts[original_idx],
                "plain_text": strip_diacritics(texts[original_idx]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    rules = {
        "version": 1,
        "source_kind": source_kind,
        "source_files": source_files,
        "seed": args.seed,
        "train_size": len(train_texts),
        "test_size": len(test_texts),
        "max_ngram": args.max_ngram,
        "min_count": args.min_count,
        "min_confidence": args.min_confidence,
        "ngram_rules": extract_rules(
            train_texts,
            max_ngram=args.max_ngram,
            min_count=args.min_count,
            min_confidence=args.min_confidence,
        ),
        "line_rules": extract_line_rules(train_texts),
    }

    letter_map = load_letter_map(args.letter_map)
    test_inputs = [strip_diacritics(text) for text in test_texts]
    preds = diacritize_texts(test_inputs, rules, letter_map)
    report = evaluate(test_texts, preds, letter_map)
    report.update(
        {
            "source_kind": source_kind,
            "source_files": source_files,
            "rules_output": str(args.rules_output),
            "train_split": str(train_split_path),
            "test_split": str(test_split_path),
            "seed": args.seed,
            "train_size": len(train_texts),
            "test_size": len(test_texts),
            "rule_counts": {
                n: len(mapping) for n, mapping in rules["ngram_rules"].items()
            },
            "line_rule_count": len(rules["line_rules"]),
        }
    )

    args.rules_output.parent.mkdir(parents=True, exist_ok=True)
    args.rules_output.write_text(
        json.dumps(rules, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote rules: {args.rules_output}")
    print(f"Wrote report: {args.report_output}")
    print(f"Wrote train split: {train_split_path}")
    print(f"Wrote test split: {test_split_path}")
    print(f"Sentence accuracy: {report['sentence_accuracy']:.4f}")
    print(f"Word accuracy: {report['word_accuracy']:.4f}")
    print(f"Skeleton match rate: {report['skeleton_match_rate']:.4f}")


def diacritize_command(args: argparse.Namespace) -> None:
    rules = json.loads(args.rules.read_text(encoding="utf-8"))
    letter_map = load_letter_map(args.letter_map)
    engine = LocalNavDiacritizer(rules, letter_map)

    lines = args.input.read_text(encoding="utf-8").splitlines()
    out_lines = [engine.diacritize_line(line) for line in lines]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(out_lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Wrote output: {args.output}")
    print(f"Lines: {len(out_lines)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-rules", help="Build rules from dataset.jsonl")
    p_build.add_argument("--dataset-jsonl", type=Path, default=DEFAULT_DATASET)
    p_build.add_argument(
        "--lst-glob",
        action="append",
        default=[],
        help=(
            "Build from one or more .lst glob patterns instead of dataset.jsonl. "
            f"Example: {DEFAULT_NAV_LST_GLOB}"
        ),
    )
    p_build.add_argument("--rules-output", type=Path, default=DEFAULT_RULES)
    p_build.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    p_build.add_argument("--split-output-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    p_build.add_argument("--letter-map", type=Path, default=DEFAULT_LETTER_MAP)
    p_build.add_argument("--seed", type=int, default=20260623)
    p_build.add_argument("--train-size", type=int, default=6000)
    p_build.add_argument("--max-ngram", type=int, default=6)
    p_build.add_argument("--min-count", type=int, default=3)
    p_build.add_argument("--min-confidence", type=float, default=0.9)
    p_build.set_defaults(func=build_rules_command)

    p_diac = sub.add_parser("diacritize", help="Diacritize a txt file")
    p_diac.add_argument("--input", type=Path, required=True)
    p_diac.add_argument("--output", type=Path, required=True)
    p_diac.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    p_diac.add_argument("--letter-map", type=Path, default=DEFAULT_LETTER_MAP)
    p_diac.set_defaults(func=diacritize_command)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
