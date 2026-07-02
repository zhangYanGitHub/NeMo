#!/usr/bin/env python3
"""Shared text normalization (TN) for the G2P pipeline.

WHY THIS EXISTS
    The Conformer-CTC grapheme vocab contains NO digits (only a-z / A-Z + a few
    symbols) and ``unk_id`` is None, so any raw digit in the input is *silently
    dropped* before the model: ``"B15"`` -> ``"b"`` (the ``15`` vanishes). espeak
    does number expansion internally; since the neural model REPLACES espeak, we
    must reproduce that expansion BEFORE the model as a deterministic rule layer.

CONTRACT (do not break)
    This module MUST be used IDENTICALLY in:
      1. training data prep  (preprocess_ipa_childes_split.py --normalize-numbers)
      2. the inference frontend (call normalize_for_g2p() on the text before G2P)
    so the model sees the same spoken form at train and serve time.

SCOPE (deliberately conservative)
    * integers           -> cardinal words   ("500" -> "five hundred")
    * ordinals "<n>th"   -> ordinal words     ("45th" -> "forty fifth")
    * alphanumeric codes -> letters + number  ("B15" -> "B fifteen",
                                               "221B" -> "two hundred twenty one B",
                                               "US-101" -> "US one hundred one",
                                               "I-5" -> "I five")
    * hyphens between alphabetic words -> word boundary (space). In English a hyphen is
      an orthographic joiner of independently-spoken words (compound modifiers, prefixes,
      spelled numbers: "well-known", "non-stop", "twenty-five"), so the spoken form equals
      the same words with spaces. KEPT when a part is a single letter ("A-frame", "U-turn",
      "X-ray"), which English pronounces as a letter name; a space would reduce it.
    * Number rules ONLY touch tokens that contain a digit. Pure-letter words and
      abbreviations (NW, US, don't) are otherwise left untouched on purpose: they
      survive to the model as-is (do_lower=false keeps their case), and abbreviation
      expansion (NW->northwest, US->United States) is domain-specific TN that belongs
      upstream, not here.

FIDELITY NOTE
    The exact spoken form must match espeak (e.g. "101" -> "one hundred one" vs
    "one oh one"). Run ``python text_normalize.py --verify`` on a machine with
    piper_phonemize to compare TN+phonemize against raw phonemize and tune below.
"""
from __future__ import annotations

import re
from typing import List

_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
# en-US: no "and" before the tens/ones (that is the en-GB style).
_SCALES = [(10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand"), (100, "hundred")]

# cardinal -> ordinal overrides for the final word
_ORDINAL_SPECIAL = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}
_TENS_ORDINAL = {
    "twenty": "twentieth", "thirty": "thirtieth", "forty": "fortieth", "fifty": "fiftieth",
    "sixty": "sixtieth", "seventy": "seventieth", "eighty": "eightieth", "ninety": "ninetieth",
}


def int_to_words(n: int) -> str:
    """Non-negative integer -> en-US cardinal words. 221 -> 'two hundred twenty one'."""
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
    """Non-negative integer -> en-US ordinal words. 45 -> 'forty fifth'."""
    cardinal = int_to_words(n)
    words = cardinal.split()
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


_TOKEN_RE = re.compile(r"\S+")
_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$", re.IGNORECASE)
_RUN_RE = re.compile(r"\d+|[^\d]+")


def _transform_token(token: str, for_phonemize: bool = False) -> str:
    """Transform a single whitespace-delimited token that contains >=1 digit.

    Args:
        token: the raw token (e.g. "2A", "101A", "I-95").
        for_phonemize: when True, single-uppercase-letter suffix "A" is
            rendered as "A-" so that piper/espeak reads it as the letter name
            (ˈeɪ) rather than the indefinite article (ɐ).  All other letters
            are unaffected because only "A" is genuinely ambiguous.
    """
    # Hyphen acts as a separator inside codes: US-101 -> "US 101", I-5 -> "I 5".
    parts = token.split("-")
    out_parts: List[str] = []
    for part in parts:
        if not part:
            continue
        m = _ORDINAL_RE.match(part)
        if m:
            out_parts.append(ordinal_to_words(int(m.group(1))))
            continue
        # Split into alternating digit / non-digit runs; expand digit runs only.
        pieces: List[str] = []
        for run in _RUN_RE.findall(part):
            if run.isdigit():
                pieces.append(int_to_words(int(run)))
            else:
                # When building the phonemize text, a bare uppercase "A" that
                # immediately follows a digit expansion (e.g. "2A" -> "two A")
                # would be misread by espeak as the indefinite article.
                # Appending "-" forces letter-name pronunciation.
                if for_phonemize and run == "A":
                    pieces.append("A-")
                else:
                    pieces.append(run)
        out_parts.append(" ".join(pieces))
    return " ".join(out_parts)


# English rule for hyphens between alphabetic words.
#
# In English a hyphen joins tokens that are each pronounced as their own word
# (compound modifiers: "well-known", "high-speed"; compound nouns: "mother-in-law";
# prefixes: "non-stop", "anti-lock"; spelled numbers: "twenty-five"). The hyphen is an
# orthographic joiner, not a phonetic unit, so the spoken form equals the same words
# written with spaces. We therefore render such hyphens as a word boundary (space),
# giving the canonical multi-word grapheme form regardless of how the source spelled it.
#
# EXCEPTION — single-letter parts. When a part is one letter, English pronounces it as
# the LETTER NAME, not as a reduced word ("A-frame" = AY-, "U-turn" = YOO-, "X-ray" =
# EX-, "e-mail" = EE-). Replacing the hyphen with a space would let the part collapse to
# an unstressed word vowel (a -> schwa). Keep the hyphen so the letter-name reading holds.
#
# Digit-bearing codes (US-101, I-5) are out of scope here and handled by the number
# rules below, which already treat the hyphen as a separator.
_ALPHA_HYPHEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)+")


def _split_hyphen_compound(m: re.Match) -> str:
    parts = m.group(0).split("-")
    # A single-letter part is spoken as a letter name; keep it bound by the hyphen.
    if any(len(p) == 1 for p in parts):
        return m.group(0)
    return " ".join(parts)


def normalize_hyphens(text: str) -> str:
    """Render hyphens between alphabetic words as word boundaries (spaces), since the
    hyphen is an orthographic joiner whose spoken form equals the space-separated words.
    Keep the hyphen when a part is a single letter (letter-name reading: A-frame, U-turn,
    X-ray). Digit-bearing codes (US-101, I-5) are left for the number rules below."""
    if "-" not in text:
        return text
    return _ALPHA_HYPHEN_RE.sub(_split_hyphen_compound, text)


def normalize_for_g2p(text: str, for_phonemize: bool = False) -> str:
    """Expand digits/codes to spoken words so nothing is dropped by the digit-free
    grapheme vocab, and normalize alphabetic hyphen compounds (off-road -> off road).
    Only tokens containing a digit are number-expanded; everything else is preserved
    verbatim. Whitespace is collapsed.

    Args:
        text: raw input text.
        for_phonemize: pass True when the output will be fed to piper/espeak rather
            than used as grapheme input for the G2P model.  This adds a "-" suffix
            to uppercase "A" that originated from a digit+letter token so that
            espeak reads it as the letter name (ˈeɪ) rather than the indefinite
            article (ɐ).  Has no effect on tokens that are purely alphabetic.
    """
    if not text:
        return text

    text = normalize_hyphens(text)

    if not any(ch.isdigit() for ch in text):
        return " ".join(text.split())

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        return _transform_token(tok, for_phonemize=for_phonemize) if any(ch.isdigit() for ch in tok) else tok

    return " ".join(_TOKEN_RE.sub(repl, text).split())


def normalize_for_g2p_phonemize(text: str) -> str:
    """Like :func:`normalize_for_g2p` but for the phonemize input path.

    Identical to ``normalize_for_g2p(text, for_phonemize=True)``: digit+letter-A
    tokens (e.g. "2A") are expanded with a trailing hyphen on the letter
    ("two A-") so that piper/espeak reads "A" as a letter name, not an article.
    Use this function when building the text that will be phonemized; use the
    plain :func:`normalize_for_g2p` for the grapheme input to the G2P model.
    """
    return normalize_for_g2p(text, for_phonemize=True)


def _verify() -> None:
    """Compare TN+phonemize against raw phonemize to confirm/tune readings.
    Requires piper_phonemize (present on the training machine)."""
    try:
        from piper_phonemize import phonemize_espeak  # type: ignore[reportMissingImports]
    except Exception as exc:  # pragma: no cover - only runs where piper is installed
        print(f"piper_phonemize not available: {exc}")
        return

    def ph(t: str) -> str:
        return "".join("".join(s) for s in phonemize_espeak(t, "en-us"))

    cases = [
        # number / code rules
        "B15", "221B Baker Street", "US-101", "I-5", "M25",
        "In 500 meters turn right", "exit 45th street", "1024",
        # hyphen rule by English category: compound modifier, prefix, spelled number,
        # then single-letter (letter-name) which must KEEP the hyphen
        "a well-known high-speed route", "a non-stop anti-lock test",
        "twenty-five north-bound lanes", "take a U-turn", "an A-frame and an e-mail",
    ]
    for raw in cases:
        tn = normalize_for_g2p(raw)
        same = ph(raw) == ph(tn)
        print(f"{raw!r:28} -> TN {tn!r}")
        print(f"     espeak(raw)= {ph(raw)!r}")
        print(f"     espeak(tn) = {ph(tn)!r}   {'MATCH' if same else 'DIFF (tune rule)'}")
        print()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="G2P text normalization (numbers/codes).")
    ap.add_argument("--verify", action="store_true", help="probe against piper_phonemize")
    ap.add_argument("text", nargs="*", help="normalize the given text and print")
    args = ap.parse_args()
    if args.verify:
        _verify()
    elif args.text:
        print(normalize_for_g2p(" ".join(args.text)))
    else:
        ap.print_help()
