"""Grapheme and IPA normalization with strategy-based abbreviation expansion."""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from slot_values import ABBREVIATIONS

# Zero-width and invisible characters sometimes emitted by espeak-ng
_INVISIBLE_CHARS_RE = re.compile(
    r"[\u200b\u200c\u200d\ufeff\u2060\u00ad]"
)

# Unicode punctuation replacements (optional)
_PUNCT_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
}

# Hyphen spacing: "I - 95" -> "I-95", but keep " - " in prose minimal
_HYPHEN_SPACE_RE = re.compile(r"\s*-\s*")

_MULTI_SPACE_RE = re.compile(r"\s+")

# --- English cardinal / ordinal helpers (spoken form for TTS / G2P) ---
_UNDER_TWENTY = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_TENS_WORDS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_ORDINAL_UNDER_TWENTY = (
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
    "seventeenth", "eighteenth", "nineteenth",
)
_TENS_ORDINAL = {
    20: "twentieth", 30: "thirtieth", 40: "fortieth", 50: "fiftieth",
    60: "sixtieth", 70: "seventieth", 80: "eightieth", 90: "ninetieth",
}


def _under_100_cardinal(n: int) -> str:
    if n < 20:
        return _UNDER_TWENTY[n]
    if n < 100:
        ten, one = divmod(n, 10)
        if one == 0:
            return _TENS_WORDS[ten]
        return f"{_TENS_WORDS[ten]} {_UNDER_TWENTY[one]}"
    raise ValueError(n)


def _under_1000_cardinal(n: int) -> str:
    if n < 100:
        return _under_100_cardinal(n)
    hundreds, rem = divmod(n, 100)
    head = f"{_UNDER_TWENTY[hundreds]} hundred"
    if rem == 0:
        return head
    return f"{head} {_under_100_cardinal(rem)}"


def int_to_cardinal_words(n: int) -> str:
    """0 <= n < 1_000_000 -> spoken English cardinal."""
    if n < 0:
        return "minus " + int_to_cardinal_words(-n)
    if n < 1000:
        return _under_1000_cardinal(n)
    if n < 1_000_000:
        thousands, rem = divmod(n, 1000)
        left = _under_1000_cardinal(thousands) + " thousand"
        if rem == 0:
            return left
        return f"{left} {_under_1000_cardinal(rem)}"
    return str(n)


def _ordinal_under_100(n: int) -> str:
    if n < 1:
        return int_to_cardinal_words(n)
    if n < 20:
        return _ORDINAL_UNDER_TWENTY[n - 1]
    if n < 100:
        if n % 10 == 0:
            return _TENS_ORDINAL[n]
        return f"{_TENS_WORDS[n // 10]} {_ordinal_under_100(n % 10)}"
    raise ValueError(n)


def int_to_ordinal_words(n: int) -> str:
    """1 <= n < 1000 -> spoken English ordinal (e.g. 12 -> twelfth)."""
    if n < 1:
        return int_to_cardinal_words(n)
    if n < 100:
        return _ordinal_under_100(n)
    hundreds, rem = divmod(n, 100)
    head_word = _UNDER_TWENTY[hundreds]
    if rem == 0:
        return f"{head_word} hundredth"
    if rem < 20:
        return f"{head_word} hundred {_ORDINAL_UNDER_TWENTY[rem - 1]}"
    return f"{head_word} hundred {_ordinal_under_100(rem)}"


def _decimal_to_words(int_part: str, frac_part: str) -> str:
    left = int_to_cardinal_words(int(int_part))
    digits = " ".join(_UNDER_TWENTY[int(c)] for c in frac_part if c.isdigit())
    return f"{left} point {digits}" if digits else left


def _apply_static_abbrev_map(text: str) -> str:
    """Expand slot_values ABBREVIATIONS (longest keys first)."""
    result = text
    for abbrev in sorted(ABBREVIATIONS.keys(), key=len, reverse=True):
        if abbrev in result:
            result = result.replace(abbrev, ABBREVIATIONS[abbrev])
    return result


# Bare abbreviations often emitted by route / road builders (no trailing period)
_BARE_ABBREV_REPLACEMENTS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bHwy\b(?!\.)", re.I), "Highway"),
    (re.compile(r"\bAve\b(?!\.)", re.I), "Avenue"),
    (re.compile(r"\bBlvd\b(?!\.)", re.I), "Boulevard"),
    (re.compile(r"\bRd\b(?!\.)", re.I), "Road"),
    (re.compile(r"\bLn\b(?!\.)", re.I), "Lane"),
    (re.compile(r"\bCt\b(?!\.)", re.I), "Court"),
    (re.compile(r"\bPl\b(?!\.)", re.I), "Place"),
    (re.compile(r"\bPkwy\b(?!\.)", re.I), "Parkway"),
    (re.compile(r"\bCir\b(?!\.)", re.I), "Circle"),
    (re.compile(r"\bTer\b(?!\.)", re.I), "Terrace"),
    (re.compile(r"\bExpy\b(?!\.)", re.I), "Expressway"),
)


def _expand_bare_abbreviations(text: str) -> str:
    result = text
    for pat, repl in _BARE_ABBREV_REPLACEMENTS:
        result = pat.sub(repl, result)
    return result


def spoken_numbers_in_text(text: str) -> str:
    """Replace Arabic numerals with English spoken-word forms (navigation-friendly)."""
    if not text:
        return text

    def repl_ordinal(m: re.Match[str]) -> str:
        return int_to_ordinal_words(int(m.group(1)))

    result = re.sub(r"\b(\d{1,3})(st|nd|rd|th)\b", repl_ordinal, text, flags=re.I)

    def repl_exit(m: re.Match[str]) -> str:
        n = int(m.group(1))
        suf = (m.group(2) or "").upper()
        num_w = int_to_cardinal_words(n)
        if not suf:
            return f"Exit {num_w}"
        return f"Exit {num_w} {suf}"

    result = re.sub(r"\bExit\s+(\d{1,3})([A-Za-z])?\b", repl_exit, result, flags=re.I)

    _space_routes = (
        (re.compile(r"\bI\s+(\d{1,4})\b", re.I), "Interstate {}"),
        (re.compile(r"\bUS\s+(\d{1,4})\b", re.I), "U S {}"),
        (re.compile(r"\bSR\s+(\d{1,4})\b", re.I), "State Route {}"),
        (re.compile(r"\bFM\s+(\d{1,4})\b", re.I), "Farm to Market Road {}"),
        (re.compile(r"\bTX\s+(\d{1,4})\b", re.I), "Texas Highway {}"),
    )
    for pat, fmt in _space_routes:
        def _mk_sp(fmt_inner: str):
            def _inner(m: re.Match[str]) -> str:
                return fmt_inner.format(int_to_cardinal_words(int(m.group(1))))

            return _inner

        result = pat.sub(_mk_sp(fmt), result)

    _route_hyphen = (
        (re.compile(r"\bI-(\d{1,4})\b", re.I), "Interstate {}"),
        (re.compile(r"\bUS-(\d{1,4})\b", re.I), "U S {}"),
        (re.compile(r"\bSR-(\d{1,4})\b", re.I), "State Route {}"),
        (re.compile(r"\bFM-(\d{1,4})\b", re.I), "Farm to Market Road {}"),
        (re.compile(r"\bTX-(\d{1,4})\b", re.I), "Texas Highway {}"),
        (re.compile(r"\bHwy-(\d{1,4})\b", re.I), "Highway {}"),
    )
    for pat, fmt in _route_hyphen:
        def _mk(fmt_inner: str):
            def _inner(m: re.Match[str]) -> str:
                return fmt_inner.format(int_to_cardinal_words(int(m.group(1))))

            return _inner

        result = pat.sub(_mk(fmt), result)

    result = re.sub(
        r"\bRoute\s+(\d{1,4})\b",
        lambda m: "Route " + int_to_cardinal_words(int(m.group(1))),
        result,
        flags=re.I,
    )
    result = re.sub(
        r"\bHighway\s+(\d{1,4})\b",
        lambda m: "Highway " + int_to_cardinal_words(int(m.group(1))),
        result,
        flags=re.I,
    )
    result = re.sub(
        r"\bHwy\s+(\d{1,4})\b",
        lambda m: "Highway " + int_to_cardinal_words(int(m.group(1))),
        result,
        flags=re.I,
    )

    def repl_decimal(m: re.Match[str]) -> str:
        return _decimal_to_words(m.group(1), m.group(2))

    result = re.sub(r"\b(\d+)\.(\d+)\b", repl_decimal, result)

    def repl_int(m: re.Match[str]) -> str:
        return int_to_cardinal_words(int(m.group(0)))

    result = re.sub(r"\b\d{1,6}\b", repl_int, result)
    return _MULTI_SPACE_RE.sub(" ", result).strip()


class AbbreviationStrategy(ABC):
    """Strategy interface for context-aware abbreviation expansion."""

    @abstractmethod
    def expand(self, text: str, role: str = "default") -> str:
        raise NotImplementedError


class MultiSenseAbbreviationStrategy(AbbreviationStrategy):
    """Expand abbreviations with role-dependent interpretation."""

    # role -> token -> list of possible expansions (first is default)
    ROLE_EXPANSIONS: Dict[str, Dict[str, List[str]]] = {
        "default": {
            "St.": ["Street", "Saint"],
            "Dr.": ["Drive", "Doctor"],
            "Mt.": ["Mount"],
            "Ft.": ["Fort"],
            "Ave.": ["Avenue"],
            "Blvd.": ["Boulevard"],
            "Rd.": ["Road"],
            "Ln.": ["Lane"],
            "Ct.": ["Court"],
            "Pl.": ["Place"],
            "Pkwy.": ["Parkway"],
            "Hwy.": ["Highway"],
            "Cir.": ["Circle"],
            "Ter.": ["Terrace"],
            "Expy.": ["Expressway"],
        },
        "road_name": {
            "St.": ["Street"],
            "Dr.": ["Drive"],
            "Mt.": ["Mount"],
            "Ft.": ["Fort"],
            "Ave.": ["Avenue"],
            "Blvd.": ["Boulevard"],
            "Rd.": ["Road"],
        },
        "poi_name": {
            "St.": ["Saint", "Street"],
            "Dr.": ["Doctor", "Drive"],
            "Mt.": ["Mount"],
            "Ft.": ["Fort"],
        },
        "address_string": {
            "St.": ["Street"],
            "Ave.": ["Avenue"],
            "Blvd.": ["Boulevard"],
            "Rd.": ["Road"],
            "Dr.": ["Drive"],
        },
    }

    def __init__(self, prefer_index: int = 0, rng=None):
        self.prefer_index = prefer_index
        self.rng = rng

    def _pick(self, options: List[str]) -> str:
        if self.rng and len(options) > 1:
            return self.rng.choice(options)
        idx = min(self.prefer_index, len(options) - 1)
        return options[idx]

    def expand(self, text: str, role: str = "default") -> str:
        table = self.ROLE_EXPANSIONS.get(role, self.ROLE_EXPANSIONS["default"])
        merged = {**self.ROLE_EXPANSIONS["default"], **table}
        result = text
        # Longer tokens first to avoid partial replacements
        for abbrev in sorted(merged.keys(), key=len, reverse=True):
            if abbrev in result:
                replacement = self._pick(merged[abbrev])
                result = result.replace(abbrev, replacement)
        return result


class NoAbbreviationStrategy(AbbreviationStrategy):
    def expand(self, text: str, role: str = "default") -> str:
        return text


def normalize_graphemes(
    text: str,
    *,
    expand_abbrev: bool = True,
    abbrev_role: str = "default",
    abbrev_strategy: Optional[AbbreviationStrategy] = None,
    replace_punctuation: bool = True,
    normalize_hyphen_spacing: bool = True,
    spoken_numbers: bool = True,
) -> str:
    """Normalize navigation grapheme text."""
    if text is None:
        return ""
    result = unicodedata.normalize("NFC", text)
    result = result.strip()
    if replace_punctuation:
        for src, dst in _PUNCT_REPLACEMENTS.items():
            result = result.replace(src, dst)
    result = _INVISIBLE_CHARS_RE.sub("", result)
    if normalize_hyphen_spacing:
        # Only tighten hyphens that look like route/road tokens
        def _fix_hyphen(m: re.Match) -> str:
            return "-"

        result = _HYPHEN_SPACE_RE.sub(_fix_hyphen, result)
    result = _MULTI_SPACE_RE.sub(" ", result)
    if spoken_numbers:
        result = spoken_numbers_in_text(result)
        result = _MULTI_SPACE_RE.sub(" ", result).strip()
    if expand_abbrev:
        strategy = abbrev_strategy or MultiSenseAbbreviationStrategy()
        result = strategy.expand(result, role=abbrev_role)
        result = _apply_static_abbrev_map(result)
        result = _expand_bare_abbreviations(result)
        result = _MULTI_SPACE_RE.sub(" ", result).strip()
    return result


def normalize_ipa(text: str) -> str:
    """Normalize IPA output while preserving stress and length marks."""
    if text is None:
        return ""
    result = unicodedata.normalize("NFC", text)
    result = result.strip()
    result = _INVISIBLE_CHARS_RE.sub("", result)
    # espeak may emit newlines between clauses
    result = result.replace("\n", " ").replace("\r", " ")
    result = _MULTI_SPACE_RE.sub(" ", result)
    return result


def infer_abbrev_role(slots: Dict[str, str]) -> str:
    """Infer abbreviation expansion role from filled slots."""
    if "address_string" in slots:
        return "address_string"
    if "poi_name" in slots:
        return "poi_name"
    if any(k in slots for k in ("road_name", "road_type", "street_name")):
        return "road_name"
    return "default"


def detect_abbreviations(text: str) -> List[str]:
    found = []
    for abbrev in ABBREVIATIONS:
        if abbrev in text:
            found.append(abbrev)
    return found
