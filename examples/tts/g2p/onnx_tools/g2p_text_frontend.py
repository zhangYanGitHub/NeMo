#!/usr/bin/env python3
"""Shared G2P text frontend for INFERENCE — kept byte-for-byte in sync with training.

SOURCE OF TRUTH (NeMo training repo):
    examples/dataset/text_normalize.py          -> normalize_for_g2p
    examples/dataset/preprocess_ipa_childes_split.py
        -> normalize_text / strip_punctuation / split_into_segments

WHY THIS FILE EXISTS
    Training builds each manifest line as a punctuation-free, number-expanded SEGMENT:
        normalize_text -> normalize_for_g2p (digits/codes -> words)
                       -> split_into_segments (cut at pause/sentence punctuation)
    The model therefore only ever sees punctuation-free, TN'd segments. Inference MUST
    feed the model the SAME thing, or train != serve. This module reproduces that exact
    pipeline so both the model path and the espeak reference path use one frontend.

DO NOT edit the logic here in isolation: any change MUST be mirrored in the two NeMo
training scripts above (and vice versa), otherwise the model sees different inputs at
train vs serve time.
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple

_DATASET_ROOT = Path(__file__).resolve().parents[3] / "dataset"
if str(_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATASET_ROOT))

from ar_XA.diacritizer_frontend import diacritize_arabic_line, is_arabic_voice

# ── number / code TN (BYTE-FOR-BYTE mirror of examples/dataset/text_normalize.py) ─
# Any edit here MUST be mirrored in text_normalize.normalize_for_g2p (and vice versa),
# incl. the locale dispatch and the symbol-expansion table, or train != serve.
_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand"), (100, "hundred")]
_ORDINAL_SPECIAL = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}
_TENS_ORDINAL = {
    "twenty": "twentieth", "thirty": "thirtieth", "forty": "fortieth", "fifty": "fiftieth",
    "sixty": "sixtieth", "seventy": "seventieth", "eighty": "eightieth", "ninety": "ninetieth",
}


def int_to_words(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens = _TENS[n // 10]
        return tens if n % 10 == 0 else f"{tens} {_ONES[n % 10]}"
    for value, name in _SCALES:
        if n >= value:
            head = int_to_words(n // value)
            rem = n % value
            return f"{head} {name}" + (f" {int_to_words(rem)}" if rem else "")
    return _ONES[n]


def ordinal_to_words(n: int) -> str:
    words = int_to_words(n).split()
    last = words[-1]
    if last in _ORDINAL_SPECIAL:
        words[-1] = _ORDINAL_SPECIAL[last]
    elif last in _TENS_ORDINAL:
        words[-1] = _TENS_ORDINAL[last]
    elif last.endswith("y"):
        words[-1] = last[:-1] + "ieth"
    else:
        words[-1] = last + "th"
    return " ".join(words)


# ── German (de) cardinals (mirror of text_normalize.int_to_words_de) ──────────────
_DE_ONES = [
    "null", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
    "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn",
    "achtzehn", "neunzehn",
]
_DE_TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
_DE_SCALES = [(10 ** 9, "Milliarde", "Milliarden"), (10 ** 6, "Million", "Millionen")]


def _de_below_100(n: int, final: bool) -> str:
    if n < 20:
        if n == 1:
            return "eins" if final else "ein"
        return _DE_ONES[n]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _DE_TENS[tens]
    return f"{_DE_ONES[ones]}und{_DE_TENS[tens]}"


def _de_below_1000(n: int, final: bool) -> str:
    if n < 100:
        return _de_below_100(n, final)
    hundreds, rem = divmod(n, 100)
    out = _DE_ONES[hundreds] + "hundert"
    if rem:
        out += _de_below_100(rem, final)
    return out


def _de_below_million(n: int, final: bool) -> str:
    if n < 1000:
        return _de_below_1000(n, final)
    thousands, rem = divmod(n, 1000)
    out = _de_below_1000(thousands, final=False) + "tausend"
    if rem:
        out += _de_below_1000(rem, final)
    return out


def int_to_words_de(n: int) -> str:
    if n < 0:
        return "minus " + int_to_words_de(-n)
    if n == 0:
        return "null"
    parts: List[str] = []
    for value, sing, plur in _DE_SCALES:
        if n >= value:
            count, n = divmod(n, value)
            if count == 1:
                parts.append("eine " + sing)
            else:
                parts.append(_de_below_million(count, final=False) + " " + plur)
    if n > 0:
        parts.append(_de_below_million(n, final=True))
    return " ".join(parts)


# ── French (fr) cardinals (mirror of text_normalize.int_to_words_fr) ──────────────
# Standard France (NOT Belgian/Swiss septante/nonante): 70 = soixante-dix, 80 = quatre-vingts,
# 90 = quatre-vingt-dix. "et" links only 21/31/41/51/61/71; 22-29, 72-79, 81-99 use hyphens
# without "et". "quatre-vingts"/"cents" take a plural -s ONLY word-final; "mille" is invariable.
_FR_ONES = [
    "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
    "dix", "onze", "douze", "treize", "quatorze", "quinze", "seize",
    "dix-sept", "dix-huit", "dix-neuf",
]
_FR_TENS = ["", "", "vingt", "trente", "quarante", "cinquante", "soixante", "", "", ""]


def _fr_below_100(n: int) -> str:
    if n < 20:
        return _FR_ONES[n]
    t, u = divmod(n, 10)
    if t <= 6:
        base = _FR_TENS[t]
        if u == 0:
            return base
        if u == 1:
            return f"{base} et un"
        return f"{base}-{_FR_ONES[u]}"
    if t == 7:
        if u == 0:
            return "soixante-dix"
        if u == 1:
            return "soixante et onze"
        return f"soixante-{_FR_ONES[10 + u]}"
    if t == 8:
        return "quatre-vingts" if u == 0 else f"quatre-vingt-{_FR_ONES[u]}"
    return f"quatre-vingt-{_FR_ONES[10 + u]}"


def _fr_below_1000(n: int) -> str:
    if n < 100:
        return _fr_below_100(n)
    h, r = divmod(n, 100)
    head = "cent" if h == 1 else f"{_FR_ONES[h]} cent"
    if r == 0:
        return head if h == 1 else f"{_FR_ONES[h]} cents"
    return f"{head} {_fr_below_100(r)}"


def _fr_strip_plural_before_scale(s: str) -> str:
    if s.endswith("quatre-vingts"):
        return s[:-1]
    if s.endswith("cents"):
        return s[:-1]
    return s


def _fr_below_million(n: int) -> str:
    if n < 1000:
        return _fr_below_1000(n)
    th, r = divmod(n, 1000)
    head = "mille" if th == 1 else f"{_fr_strip_plural_before_scale(_fr_below_1000(th))} mille"
    return head if r == 0 else f"{head} {_fr_below_1000(r)}"


_FR_SCALES = [(10 ** 9, "milliard", "milliards"), (10 ** 6, "million", "millions")]


def int_to_words_fr(n: int) -> str:
    if n < 0:
        return "moins " + int_to_words_fr(-n)
    if n == 0:
        return "zéro"
    parts: List[str] = []
    for value, sing, plur in _FR_SCALES:
        if n >= value:
            count, n = divmod(n, value)
            parts.append(_fr_below_million(count) + " " + (sing if count == 1 else plur))
    if n > 0:
        parts.append(_fr_below_million(n))
    return " ".join(parts)


_CARDINALS = {"en": int_to_words, "de": int_to_words_de, "fr": int_to_words_fr}


def _cardinal(n: int, locale: str) -> str:
    return _CARDINALS.get(locale, int_to_words)(n)


_TOKEN_RE = re.compile(r"\S+")
_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$", re.IGNORECASE)
_RUN_RE = re.compile(r"\d+|[^\d]+")


def _transform_token(token: str, locale: str = "en") -> str:
    parts = token.split("-")
    out_parts: List[str] = []
    for part in parts:
        if not part:
            continue
        if locale == "en":
            m = _ORDINAL_RE.match(part)
            if m:
                out_parts.append(ordinal_to_words(int(m.group(1))))
                continue
        pieces: List[str] = []
        for run in _RUN_RE.findall(part):
            if run.isdecimal():
                pieces.append(_cardinal(int(run), locale))
            else:
                pieces.append(run)
        out_parts.append(" ".join(pieces))
    return " ".join(out_parts)


# Hyphens between alphabetic words are an orthographic joiner of independently-spoken
# words (compound modifiers / nouns, prefixes, spelled numbers), so render them as a word
# boundary (space). KEEP the hyphen when a part is a single letter ("A-frame", "U-turn",
# "X-ray"): English speaks it as a letter name and a space would reduce it.
_ALPHA_HYPHEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)+")


def _split_hyphen_compound(m: re.Match) -> str:
    parts = m.group(0).split("-")
    if any(len(p) == 1 for p in parts):
        return m.group(0)
    return " ".join(parts)


def normalize_hyphens(text: str) -> str:
    if "-" not in text:
        return text
    return _ALPHA_HYPHEN_RE.sub(_split_hyphen_compound, text)


# 有读法的符号 → 词（mirror of text_normalize._SYMBOLS）。% & + 等经 char_policy 放行的符号
# 在此展开成词（+ -> plus，覆盖 +49 -> „plus 49"），产出纯字母，绝不把符号漏进 grapheme。
_SYMBOLS = {
    "de": {
        "°C": " Grad Celsius ", "°F": " Grad Fahrenheit ", "°": " Grad ",
        "%": " Prozent ", "‰": " Promille ", "&": " und ", "€": " Euro ", "$": " Dollar ",
        "£": " Pfund ", "µ": " Mikro ", "×": " mal ", "±": " plus minus ", "+": " plus ",
        "½": " einhalb ", "¼": " ein viertel ", "¾": " drei viertel ", "§": " Paragraf ",
    },
    "en": {
        "°C": " degrees Celsius ", "°F": " degrees Fahrenheit ", "°": " degrees ",
        "%": " percent ", "‰": " per mille ", "&": " and ", "€": " euro ", "$": " dollars ",
        "£": " pounds ", "µ": " micro ", "×": " times ", "±": " plus minus ", "+": " plus ",
        "½": " one half ", "¼": " one quarter ", "¾": " three quarters ", "§": " section ",
    },
    "fr": {
        "°C": " degrés Celsius ", "°F": " degrés Fahrenheit ", "°": " degrés ",
        "%": " pour cent ", "‰": " pour mille ", "&": " et ", "€": " euros ", "$": " dollars ",
        "£": " livres ", "µ": " micro ", "×": " fois ", "±": " plus ou moins ", "+": " plus ",
        "½": " un demi ", "¼": " un quart ", "¾": " trois quarts ", "§": " paragraphe ",
    },
}


def _expand_symbols(text: str, locale: str) -> str:
    table = _SYMBOLS.get(locale, _SYMBOLS["en"])
    if not any(sym in text for sym in table):
        return text
    for sym in sorted(table, key=len, reverse=True):
        if sym in text:
            text = text.replace(sym, table[sym])
    return text


def normalize_for_g2p(text: str, locale: str = "en") -> str:
    """BYTE-FOR-BYTE mirror of text_normalize.normalize_for_g2p(text, locale).
    locale selects the spell-out rules ('en' cardinals+ordinals, 'de' German cardinals)
    and the symbol table; it MUST match lang_config.json's per-language number_locale."""
    if not text:
        return text

    text = normalize_hyphens(text)
    # Arabic G2P training keeps Western digits in graphemes (normalize_numbers=false).
    if locale == "ar":
        return " ".join(text.split())

    text = _expand_symbols(text, locale)

    if not any(ch.isdigit() for ch in text):
        return " ".join(text.split())

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        return _transform_token(tok, locale) if any(ch.isdigit() for ch in tok) else tok

    return " ".join(_TOKEN_RE.sub(repl, text).split())


# ── normalize / punctuation strip / segment (mirror of preprocess script) ─────────
def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


_PUNCT_STRIP = frozenset('.,!?;:"()[]{}<>…—–«»‹›„“”/\\|*&^%$#@~`')
_SEGMENT_SPLIT_RE = re.compile(r"[.,!?;:…]+|[—–]+")


def strip_punctuation(text: str) -> str:
    cleaned = "".join(" " if ch in _PUNCT_STRIP else ch for ch in text)
    return " ".join(cleaned.split())


def split_into_segments(text: str) -> List[str]:
    segments: List[str] = []
    for piece in _SEGMENT_SPLIT_RE.split(text):
        seg = strip_punctuation(piece)
        if seg:
            segments.append(seg)
    return segments


def prepare_arabic_grapheme_text(text: str, *, diacritize: bool = True) -> str:
    """Arabic train==serve: diacritizer (incl. letter→Arabic) then NFC whitespace + ar TN."""
    if diacritize:
        text = diacritize_arabic_line(text)
    text = normalize_text(text)
    return normalize_for_g2p(text, "ar")


def text_to_segments(
    text: str,
    locale: str = "en",
    *,
    diacritize_arabic: Optional[bool] = None,
) -> List[str]:
    """Full inference frontend == training data prep:
    normalize_text -> [ar: local_nav_diacritizer] -> normalize_for_g2p(locale) -> split_into_segments.
    Returns the punctuation-free, TN'd segments to send to the model one by one.
    locale ('en'/'de'/'fr'/'ar') MUST match lang_config.json's number_locale for the language."""
    if diacritize_arabic is None:
        diacritize_arabic = locale == "ar"
    if diacritize_arabic and locale == "ar":
        text = prepare_arabic_grapheme_text(text, diacritize=True)
    else:
        text = normalize_for_g2p(normalize_text(text), locale)
    return split_into_segments(text)


# locale (BCP-47-ish tag or config key) -> number_locale used by normalize_for_g2p.
# Mirrors lang_config.json's per-language number_locale (en_US->en, de_DE->de, fr_FR->fr).
_NUMBER_LOCALE_BY_LANG = {
    "en": "en", "en-us": "en", "en_us": "en", "en-gb": "en",
    "de": "de", "de-de": "de", "de_de": "de", "de-at": "de", "de-ch": "de",
    "fr": "fr", "fr-fr": "fr", "fr_fr": "fr", "fr-be": "fr", "fr-ch": "fr", "fr-ca": "fr",
    "ar": "ar", "ar-xa": "ar", "ar_xa": "ar",
}


def is_arabic_g2p_lang(lang: str) -> bool:
    """True for ar / ar_XA / ar-JO style tags (G2P uses letter-lexicon split path)."""
    if not lang:
        return False
    key = lang.strip().lower().replace("_", "-")
    if key in ("ar", "ar-xa"):
        return True
    return key.startswith("ar-")


# Latin letter runs in otherwise Arabic navigation text (إثنان A, GPS→handled elsewhere).
_LATIN_LETTER_RUN_RE = re.compile(r"[A-Za-z]+")

# Default letter-name IPA lexicon (scheme 2: letters NOT in G2P model; product frontend only).
_DEFAULT_AR_LETTER_LEXICON_PATH = (
    Path(__file__).resolve().parents[3]
    / "dataset/prepare/config/ar_XA/letter_pronunciation_lexicon.json"
)

_ar_letter_lexicon_cache: Optional[Dict[str, str]] = None


def load_ar_letter_lexicon(path: Optional[Path] = None) -> Dict[str, str]:
    """Load A–Z -> IPA string map from letter_pronunciation_lexicon.json."""
    global _ar_letter_lexicon_cache
    p = Path(path).expanduser().resolve() if path else _DEFAULT_AR_LETTER_LEXICON_PATH
    if _ar_letter_lexicon_cache is not None and (path is None):
        return _ar_letter_lexicon_cache
    raw = json.loads(p.read_text(encoding="utf-8"))
    letters = raw.get("letters") or {}
    out = {str(k).upper(): str(v).strip() for k, v in letters.items() if k and v}
    if path is None:
        _ar_letter_lexicon_cache = out
    return out


def split_arabic_latin_segments(text: str) -> List[Tuple[str, Literal["g2p", "letter"]]]:
    """Split mixed Arabic+Latin text into ordered G2P vs single-letter tokens.

    Example: ``"اتجه إلى المخرج إثنان A"`` →
      ``[("اتجه إلى المخرج إثنان ", "g2p"), ("A", "letter")]``.
    Consecutive Latin letters are split one-by-one (``AB`` → ``A``, ``B``).
    """
    text = normalize_text(text)
    if not text:
        return []
    if not _LATIN_LETTER_RUN_RE.search(text):
        return [(text, "g2p")]
    out: List[Tuple[str, Literal["g2p", "letter"]]] = []
    pos = 0
    for m in _LATIN_LETTER_RUN_RE.finditer(text):
        if m.start() > pos:
            out.append((text[pos : m.start()], "g2p"))
        for ch in m.group(0):
            out.append((ch.upper(), "letter"))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], "g2p"))
    return out


def phonemize_arabic_with_letter_lexicon(
    text: str,
    g2p_phonemize: Callable[[str], str],
    *,
    letter_lexicon: Optional[Dict[str, str]] = None,
    locale: str = "ar",
    diacritize: bool = True,
) -> str:
    """Scheme 2 inference: diacritize (incl. Latin->Arabic), G2P segments + lexicon fallback.

    *g2p_phonemize* should run the Arabic G2P model on one punctuation-free segment
    (caller typically wraps ``text_to_segments`` + ONNX client).
    """
    if diacritize and is_arabic_g2p_lang(locale):
        text = prepare_arabic_grapheme_text(text, diacritize=True)
    lex = letter_lexicon if letter_lexicon is not None else load_ar_letter_lexicon()
    parts: List[str] = []
    for piece, kind in split_arabic_latin_segments(text):
        if kind == "letter":
            ipa = lex.get(piece.upper())
            if not ipa:
                raise KeyError(f"No letter pronunciation for {piece!r} in Arabic letter lexicon")
            parts.append(ipa)
            continue
        piece = piece.strip()
        if not piece:
            continue
        for seg in text_to_segments(piece, locale=locale, diacritize_arabic=False):
            ipa = g2p_phonemize(seg)
            if ipa:
                parts.append(ipa)
    return " ".join(parts)


def number_locale_for(lang: str) -> str:
    """Map an en_US / de-DE / fr_FR style language tag to the 'en'/'de'/'fr' number_locale.
    Unknown tags fall back to 'en' (English cardinals), never crashing."""
    if not lang:
        return "en"
    key = lang.strip().lower().replace("_", "-")
    if key in _NUMBER_LOCALE_BY_LANG:
        return _NUMBER_LOCALE_BY_LANG[key]
    return _NUMBER_LOCALE_BY_LANG.get(key.split("-", 1)[0], "en")


if __name__ == "__main__":
    import sys

    for line in (sys.argv[1:] or ["In 500 meters, turn left.", "Take exit B15; then merge."]):
        print(f"{line!r} -> en={text_to_segments(line, 'en')}  de={text_to_segments(line, 'de')}")
