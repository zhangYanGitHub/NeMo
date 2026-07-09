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

LOCALE
    ``normalize_for_g2p(text, locale=...)`` selects the language's spell-out rules
    (``'en'`` cardinals+ordinals, ``'de'`` German cardinals). The locale is wired from
    lang_config.json's per-language ``number_locale`` and MUST match between train and
    serve. The SCOPE notes below describe the English (``'en'``) rules; German adds
    standard closed-form cardinals (einundzwanzig, dreihundertfünfundvierzig, ...).

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


# ---------------------------------------------------------------------------
# German (de) cardinals
# ---------------------------------------------------------------------------
# German writes numbers below a million as a SINGLE closed word ("einundzwanzig",
# "dreihundertfünfundvierzig"), units-before-tens joined by "und". We spell out the
# standard German word form; espeak-ng 'de' then phonemizes that word normally. (We do
# NOT try to reproduce espeak's own idiosyncratic digit reading, which splits numbers
# into space-separated chunks -- the SAME normalized text is fed to both the grapheme
# input and espeak, so the target stays aligned with whatever the German word is.)
#
# Irregular stems: sechzehn/sechzig (not sechs-), siebzehn/siebzig (not sieben-),
# dreißig (ß). "ein" is the combining form (einundzwanzig, einhundert, eintausend);
# a bare, sentence-final 1 is "eins" (see the *final* flag below).
_DE_ONES = [
    "null", "ein", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
    "zehn", "elf", "zwölf", "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn",
    "achtzehn", "neunzehn",
]
_DE_TENS = ["", "", "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig"]
# Scale words are separate, feminine nouns (eine Million / zwei Millionen).
_DE_SCALES = [(10 ** 9, "Milliarde", "Milliarden"), (10 ** 6, "Million", "Millionen")]


def _de_below_100(n: int, final: bool) -> str:
    """0..99 as one German word. *final* True -> a bare 1 reads 'eins', else 'ein'."""
    if n < 20:
        if n == 1:
            return "eins" if final else "ein"
        return _DE_ONES[n]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _DE_TENS[tens]
    # units-before-tens joined by "und": 21 -> einundzwanzig, 45 -> fünfundvierzig.
    return f"{_DE_ONES[ones]}und{_DE_TENS[tens]}"


def _de_below_1000(n: int, final: bool) -> str:
    if n < 100:
        return _de_below_100(n, final)
    hundreds, rem = divmod(n, 100)
    # hundreds multiplier: 1 -> "ein" (einhundert), never "eins".
    out = _DE_ONES[hundreds] + "hundert"
    if rem:
        out += _de_below_100(rem, final)
    return out


def _de_below_million(n: int, final: bool) -> str:
    if n < 1000:
        return _de_below_1000(n, final)
    thousands, rem = divmod(n, 1000)
    # thousands multiplier is never sentence-final -> 1 stays "ein" (eintausend).
    out = _de_below_1000(thousands, final=False) + "tausend"
    if rem:
        out += _de_below_1000(rem, final)
    return out


def int_to_words_de(n: int) -> str:
    """Non-negative integer -> German cardinal words. 21 -> 'einundzwanzig',
    345 -> 'dreihundertfünfundvierzig', 1000000 -> 'eine Million'."""
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


# Cardinal spell-out dispatch by number-normalization locale (see lang_config.json's
# 'number_locale'). Unknown locales fall back to English so nothing crashes.
_CARDINALS = {"en": int_to_words, "de": int_to_words_de}


def _cardinal(n: int, locale: str) -> str:
    return _CARDINALS.get(locale, int_to_words)(n)


_TOKEN_RE = re.compile(r"\S+")
_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$", re.IGNORECASE)
_RUN_RE = re.compile(r"\d+|[^\d]+")


def _transform_token(token: str, locale: str = "en") -> str:
    """Transform a single whitespace-delimited token that contains >=1 digit.

    Tokens are only ever rewritten into their genuine spoken form (digits/ordinals
    expanded, code hyphens treated as separators). We never inject characters to
    steer espeak toward a different phoneme (e.g. no "A" -> "A-" letter-name trick),
    so espeak's phoneme output is whatever it produces for the real text.

    The English "<n>th/st/nd/rd" ordinal suffix is unambiguous and expanded here; German
    ordinals are written "<n>." (a trailing period indistinguishable from a sentence dot,
    and stripped by the punctuation pipeline anyway), so for locale != 'en' we only expand
    cardinals -- the safe, unambiguous transformation.

    Args:
        token: the raw token (e.g. "2A", "101A", "I-95").
        locale: number-normalization locale ('en', 'de', ...).
    """
    # Hyphen acts as a separator inside codes: US-101 -> "US 101", I-5 -> "I 5".
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
        # Split into alternating digit / non-digit runs; expand digit runs only.
        pieces: List[str] = []
        for run in _RUN_RE.findall(part):
            if run.isdigit():
                pieces.append(_cardinal(int(run), locale))
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


def normalize_for_g2p(text: str, locale: str = "en") -> str:
    """Expand digits/codes to spoken words so nothing is dropped by the digit-free
    grapheme vocab, and normalize alphabetic hyphen compounds (off-road -> off road).
    Only tokens containing a digit are number-expanded; everything else is preserved
    verbatim. Whitespace is collapsed.

    The SAME normalized text is used for both the grapheme (model) input and the
    espeak phonemize input. This is pure front-end text processing: it only rewrites
    text into its genuine spoken form and never injects characters to change espeak's
    phoneme output (no "A" -> "A-" letter-name trick), so espeak's phonemes are left
    exactly as produced for the real text.

    Args:
        text: raw input text.
        locale: number-normalization locale selecting the spell-out rules ('en', 'de').
            Wired from lang_config.json's per-language 'number_locale'. MUST match the
            inference frontend for the language being served.
    """
    if not text:
        return text

    text = normalize_hyphens(text)

    if not any(ch.isdigit() for ch in text):
        return " ".join(text.split())

    def repl(m: re.Match) -> str:
        tok = m.group(0)
        return _transform_token(tok, locale) if any(ch.isdigit() for ch in tok) else tok

    return " ".join(_TOKEN_RE.sub(repl, text).split())


_VERIFY_CASES = {
    "en": (
        "en-us",
        [
            # number / code rules
            "B15", "221B Baker Street", "US-101", "I-5", "M25",
            "In 500 meters turn right", "exit 45th street", "1024",
            # hyphen rule by English category: compound modifier, prefix, spelled number,
            # then single-letter (letter-name) which must KEEP the hyphen
            "a well-known high-speed route", "a non-stop anti-lock test",
            "twenty-five north-bound lanes", "take a U-turn", "an A-frame and an e-mail",
        ],
    ),
    "de": (
        "de",
        [
            # cardinals across every scale boundary (units-und-tens, hundreds, thousands, Mio.)
            "0", "1", "7", "16", "21", "45", "100", "101", "345", "500", "1024",
            "In 500 Metern rechts abbiegen", "B15", "US-101",
        ],
    ),
}


def _verify(locale: str = "en") -> None:
    """Spell numbers out and phonemize both the raw and normalized text so a human can
    confirm/tune the readings for *locale*. Requires piper_phonemize (present on the
    training machine). Note: German numbers deliberately do NOT byte-match espeak's own
    digit reading (espeak splits them into space-separated chunks); the pipeline feeds the
    spelled-out word to both grapheme input and espeak, so alignment is preserved."""
    try:
        from piper_phonemize import phonemize_espeak  # type: ignore[reportMissingImports]
    except Exception as exc:  # pragma: no cover - only runs where piper is installed
        print(f"piper_phonemize not available: {exc}")
        return

    voice, cases = _VERIFY_CASES.get(locale, _VERIFY_CASES["en"])

    def ph(t: str) -> str:
        return "".join("".join(s) for s in phonemize_espeak(t, voice))

    for raw in cases:
        tn = normalize_for_g2p(raw, locale=locale)
        same = ph(raw) == ph(tn)
        print(f"{raw!r:28} -> TN {tn!r}")
        print(f"     espeak(raw)= {ph(raw)!r}")
        print(f"     espeak(tn) = {ph(tn)!r}   {'MATCH' if same else 'DIFF (expected for de numbers)'}")
        print()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="G2P text normalization (numbers/codes).")
    ap.add_argument("--verify", action="store_true", help="probe against piper_phonemize")
    ap.add_argument("--locale", default="en", help="number-normalization locale ('en', 'de')")
    ap.add_argument("text", nargs="*", help="normalize the given text and print")
    args = ap.parse_args()
    if args.verify:
        _verify(args.locale)
    elif args.text:
        print(normalize_for_g2p(" ".join(args.text), locale=args.locale))
    else:
        ap.print_help()
