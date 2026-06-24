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
    * ONLY tokens that contain a digit are touched. Pure-letter words and
      abbreviations (NW, US, well-known, don't) are left untouched on purpose:
      they survive to the model as-is (do_lower=false keeps their case), and
      abbreviation expansion (NW->northwest, US->United States) is domain-specific
      TN that belongs upstream, not here.

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


def _transform_token(token: str) -> str:
    """Transform a single whitespace-delimited token that contains >=1 digit."""
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
                pieces.append(run)
        out_parts.append(" ".join(pieces))
    return " ".join(out_parts)


def normalize_for_g2p(text: str) -> str:
    """Expand digits/codes to spoken words so nothing is dropped by the digit-free
    grapheme vocab. Only tokens containing a digit are modified; everything else is
    preserved verbatim. Whitespace is collapsed."""
    if not text or not any(ch.isdigit() for ch in text):
        return text

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        return _transform_token(tok) if any(ch.isdigit() for ch in tok) else tok

    return " ".join(_TOKEN_RE.sub(repl, text).split())


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
        "B15", "221B Baker Street", "US-101", "I-5", "M25",
        "In 500 meters turn right", "exit 45th street", "1024",
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
