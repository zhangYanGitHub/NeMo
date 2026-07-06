#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Executor, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

from piper_phonemize import phonemize_espeak  # type: ignore[reportMissingImports]
from text_normalize import normalize_for_g2p, normalize_for_g2p_phonemize
from tqdm import tqdm

ZWJ = "\u200d"
TIES = {ZWJ, "\u0361", "\u035c"}
PRIMARY_STRESS = "\u02c8"  # ˈ
SECONDARY_STRESS = "\u02cc"  # ˌ
STRESS = {PRIMARY_STRESS, SECONDARY_STRESS}
LENGTH = "\u02d0"  # ː
SPECIAL_TOKENS = ("<pad>", "<unk>")
MANIFEST_WRITE_BUFFER_LINES = 1024

# Per-voice multi-character phoneme atom whitelists (diphthongs, affricates, ...) live in
# phoneme_inventories.json, NOT here, so adding a new language never requires touching this
# script -- only that data file. Long vowels (iː, uː, ...), nasalization, syllabicity etc. do
# NOT need to be listed there at all: they're handled generically by is_attaching() below for
# ANY language, since they're always "base char + combining mark" rather than two independent
# base characters glued together (which is what genuinely needs a whitelist to disambiguate).
DEFAULT_PHONEME_INVENTORIES_PATH = Path(__file__).with_name("phoneme_inventories.json")


@functools.lru_cache(maxsize=None)
def load_multi_char_phoneme_inventories(path: str) -> Dict[str, Tuple[str, ...]]:
    """Load {voice_code: (atoms sorted longest-first)} from a phoneme_inventories.json-style
    file. Cached per path so every worker process/thread only pays the (tiny) parse cost once.
    Missing file -> empty mapping (every voice falls back to generic-only merging), so this is
    never a hard requirement for the script to run."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    out: Dict[str, Tuple[str, ...]] = {}
    for voice, atoms in raw.items():
        if voice.startswith("_") or not isinstance(atoms, list):
            continue  # e.g. the "_comment" key
        out[normalize_voice(voice)] = tuple(sorted(set(atoms), key=len, reverse=True))
    return out


def get_multi_char_phonemes_for_voice(voice: str, inventories: Dict[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    """Exact voice match first (e.g. 'en-us'), else fall back to the primary-language subtag
    (e.g. an unlisted 'en-gb' reuses 'en'), else () -- generic combining-mark merging only."""
    voice = normalize_voice(voice)
    if voice in inventories:
        return inventories[voice]
    lang = voice.split("-", 1)[0]
    return inventories.get(lang, ())


# Word-boundary token. The manifest ``text`` field stores phonemes concatenated within a
# word and a single space between words (e.g. "bˈɔɪ hˈaʊs"), which is exactly espeak-ng's
# native output shape and also the model's decoded output shape. A literal space is the
# boundary token, so it must be present as its own entry in vocab.txt. This mirrors
# SPACE_TOKEN in nemo/collections/common/tokenizers/ipa_symbol_tokenizer.py, whose
# longest-match text_to_tokens() re-derives the atomic tokens from this string at train time.
SPACE_TOKEN = " "


def is_attaching(ch: str) -> bool:
    """True for codepoints that always attach to the PRECEDING symbol: the length
    mark 'ː' and any true Unicode combining mark (nasalization ̃, syllabic ̩, ...)."""
    if ch == LENGTH:
        return True
    return unicodedata.combining(ch) != 0


def tokenize_phoneme_word(
    ipa_word: str, stress_mode: str = "attach", multi_char_phonemes: Sequence[str] = ()
) -> List[str]:
    """Longest-match (maximal munch) tokenizer for a SINGLE word's espeak-ng IPA string
    against *multi_char_phonemes* (a language-specific whitelist, longest-first -- see
    get_multi_char_phonemes_for_voice()/phoneme_inventories.json) plus generic
    combining-mark attachment (is_attaching(), language-agnostic). Returns one atomic
    phoneme unit per token, with stress marks folded into the unit they belong to
    (mode-dependent)."""
    units: List[str] = []
    pending_stress = ""
    join_next = False
    i, n = 0, len(ipa_word)

    while i < n:
        ch = ipa_word[i]

        if ch in TIES:
            # Explicit tie bar (rare for en-us, kept for other espeak locales): merge
            # the next matched symbol into the previous unit instead of starting a new one.
            join_next = True
            i += 1
            continue

        if ch in STRESS:
            if stress_mode == "separate":
                units.append(ch)
            elif stress_mode == "attach":
                pending_stress += ch
            # stress_mode == "drop": stress marks are already stripped by the caller.
            i += 1
            continue

        matched = ch
        for cand in multi_char_phonemes:
            if ipa_word.startswith(cand, i):
                matched = cand
                break
        i += len(matched)

        while i < n and is_attaching(ipa_word[i]):
            matched += ipa_word[i]
            i += 1

        if join_next and units:
            units[-1] += matched
            join_next = False
        else:
            units.append(pending_stress + matched)
            pending_stress = ""

    return units


def phonemes_to_text_and_atoms(
    phoneme_str: str, stress_mode: str = "attach", multi_char_phonemes: Sequence[str] = ()
) -> Tuple[str, List[str]]:
    """Turn a full (multi-word) espeak-ng phoneme string into:
      1. the manifest ``text`` value: phonemes concatenated WITHIN each word, single space
         BETWEEN words (e.g. "bˈɔɪ hˈaʊs") — the model's target/decoded shape; and
      2. the flat list of atomic phoneme units (no spaces) for building the phoneme vocab,
         so vocab.txt covers exactly the tokens the tokenizer's longest-match will produce.
    Word-internal concatenation is loss-free: "".join(units) reproduces the original word,
    so tokenize + concat round-trips (verified in run_self_check())."""
    word_strs: List[str] = []
    atoms: List[str] = []
    for word in phoneme_str.split():
        units = tokenize_phoneme_word(word, stress_mode, multi_char_phonemes)
        word_strs.append("".join(units))
        atoms.extend(units)
    return " ".join(word_strs), atoms


# Regression cases pinned against real piper_phonemize/espeak-ng en-us output. Guards against
# the exact incident that once happened here: the (then hardcoded) multi-char atom whitelist
# got silently overwritten with a flat single-character inventory, which decomposed "aɪ" ->
# "a" + "ɪ" (and similarly for other diphthongs/affricates) and inflated phoneme token counts
# enough to silently drop ~560k valid rows via the CTC T>=U length filter. run_self_check()
# below turns any regression here -- whether in the code or in phoneme_inventories.json --
# into an immediate, loud failure instead of a silent data loss only noticed after a full
# preprocessing + training run.
_SELF_CHECK_CASES = {
    # word -> raw espeak-ng IPA -> expected tokenize_phoneme_word() output (stress_mode="attach")
    "boy": ("bˈɔɪ", ["b", "ˈɔɪ"]),
    "house": ("hˈaʊs", ["h", "ˈaʊ", "s"]),
    "face": ("fˈeɪs", ["f", "ˈeɪ", "s"]),
    "home": ("hˈoʊm", ["h", "ˈoʊ", "m"]),
    "time": ("tˈaɪm", ["t", "ˈaɪ", "m"]),
    "judge": ("dʒˈʌdʒ", ["dʒ", "ˈʌ", "dʒ"]),
    "church": ("tʃˈɜːtʃ", ["tʃ", "ˈɜː", "tʃ"]),
    "palmyra": ("pˈɑːmˈaɪɹə", ["p", "ˈɑː", "m", "ˈaɪ", "ɹ", "ə"]),
}


def run_self_check(inventories: Dict[str, Tuple[str, ...]]) -> None:
    """Assert tokenize_phoneme_word() still treats every known en-us diphthong/affricate
    (loaded from phoneme_inventories.json, plus length-mark attachment) as a single atomic
    unit, that word-internal concatenation round-trips the original word, and that
    phonemes_to_text_and_atoms() emits the "concat-within-word, space-between-words" shape.
    Fails loudly if phoneme_inventories.json no longer has a usable 'en-us' entry."""
    en_us_atoms = get_multi_char_phonemes_for_voice("en-us", inventories)
    if not en_us_atoms:
        raise AssertionError(
            "run_self_check: phoneme_inventories.json has no usable 'en-us' entry (got "
            f"{en_us_atoms!r}). Check that the file exists next to this script and wasn't "
            "accidentally emptied/renamed -- without it, diphthongs/affricates silently stop "
            "being merged into atomic tokens for en-us data."
        )
    for word, (ipa, expected) in _SELF_CHECK_CASES.items():
        got = tokenize_phoneme_word(ipa, multi_char_phonemes=en_us_atoms)
        if got != expected:
            raise AssertionError(
                f"tokenize_phoneme_word regression for {word!r} (ipa={ipa!r}): "
                f"got {got}, expected {expected}. phoneme_inventories.json's 'en-us' entry or "
                f"is_attaching() was likely modified in a way that stops merging diphthongs/"
                f"affricates/length marks into single atomic tokens -- this silently inflates "
                f"phoneme token counts and can drop huge amounts of training data via the CTC "
                f"T>=U filter. See the comment above _SELF_CHECK_CASES for the historical incident."
            )
        # Word-internal concatenation must be loss-free (attach mode keeps stress in-string).
        if "".join(got) != ipa:
            raise AssertionError(
                f"tokenize_phoneme_word is not round-trippable for {word!r}: "
                f"''.join({got}) = {''.join(got)!r} != original {ipa!r}"
            )
    # Full string: phonemes concatenated within a word, single space between words.
    text, atoms = phonemes_to_text_and_atoms("bˈɔɪ hˈaʊs", multi_char_phonemes=en_us_atoms)
    if text != "bˈɔɪ hˈaʊs":
        raise AssertionError(f"phonemes_to_text_and_atoms text regression: got {text!r}, expected 'bˈɔɪ hˈaʊs'")
    if atoms != ["b", "ˈɔɪ", "h", "ˈaʊ", "s"]:
        raise AssertionError(f"phonemes_to_text_and_atoms atoms regression: got {atoms}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess ipa-childes-split CSV into NeMo G2P manifest and vocab files. "
            "Only reads language code + raw text columns, then regenerates phonemes "
            "with piper-phonemize (espeak backend)."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="ipa-childes-split style CSV path")
    parser.add_argument("--output-dir", type=Path, required=True, help="output directory")
    parser.add_argument("--text-field", type=str, default="sentence", help="raw text column name")
    parser.add_argument("--lang-field", type=str, default="espeak_lang_code", help="espeak language code column name")
    parser.add_argument("--default-voice", type=str, default="en-us", help="fallback voice when lang field is empty")
    parser.add_argument(
        "--phoneme-inventories",
        type=Path,
        default=DEFAULT_PHONEME_INVENTORIES_PATH,
        help=(
            "JSON file mapping voice code -> list of multi-character IPA phoneme atoms "
            "(diphthongs, affricates, ...) that must be kept atomic by the longest-match "
            "tokenizer. Add a new language by adding an entry here, not by editing this "
            "script. Defaults to phoneme_inventories.json next to this script."
        ),
    )
    parser.add_argument("--workers", type=int, default=0, help="parallel workers; 0 means auto by CPU count")
    parser.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="process",
        help="parallel backend; process is usually faster for very large datasets",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="rows per worker batch; 0 means auto by CPU count")
    parser.add_argument("--limit", type=int, default=0, help="optional row limit")
    parser.add_argument("--stress", choices=["attach", "separate", "drop"], default="attach")
    parser.add_argument(
        "--strip-punctuation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Strip pause/sentence punctuation from text BEFORE phonemizing so neither "
            "text_graphemes nor the phoneme target contains punctuation tokens "
            "(pure G2P; pauses are inserted by the frontend). Keeps apostrophes/hyphens."
        ),
    )
    parser.add_argument(
        "--split-on-punctuation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Split each sentence at pause/sentence punctuation into segments and emit "
            "ONE manifest line per segment, matching the frontend that sends each "
            "segment to the G2P model separately (train==serve). Segments are always "
            "punctuation-free. Use the shared split_into_segments() in the frontend."
        ),
    )
    parser.add_argument(
        "--normalize-numbers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Expand digits/codes to spoken words (text_normalize.normalize_for_g2p) "
            "BEFORE phonemizing, because the grapheme vocab has no digits (raw digits "
            "would be dropped). MUST be applied identically in the inference frontend."
        ),
    )
    parser.add_argument("--manifest-name", type=str, default="train.json")
    parser.add_argument("--phoneme-vocab-name", type=str, default="phoneme_vocab.txt")
    parser.add_argument("--grapheme-vocab-name", type=str, default="grapheme_vocab.txt")
    parser.add_argument("--merged-vocab-name", type=str, default="vocab.txt")
    parser.add_argument("--write-vocab", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


# Pause/sentence punctuation to strip for the "pure G2P + frontend-inserted pause"
# pipeline. We deliberately KEEP characters that change pronunciation rather than
# act as pauses: apostrophes (contractions: don't, it's) and hyphen (well-known).
_PUNCT_STRIP = frozenset('.,!?;:"()[]{}<>…—–«»‹›„“”/\\|*&^%$#@~`')


def strip_punctuation(text: str) -> str:
    """Remove pause/sentence punctuation so the espeak target carries no
    punctuation tokens. Punct is replaced by a space (not deleted) to avoid
    merging adjacent words, then whitespace is collapsed."""
    cleaned = "".join(" " if ch in _PUNCT_STRIP else ch for ch in text)
    return " ".join(cleaned.split())


def is_punct_unit(unit: str) -> bool:
    """A segmented phoneme unit that is purely strip-punctuation (defensive)."""
    return bool(unit) and all(ch in _PUNCT_STRIP for ch in unit)


# Pause/sentence boundaries the frontend splits on before sending each segment to
# the G2P model. THIS RULE MUST BE IDENTICAL in training data prep and inference
# frontend, otherwise the model sees different inputs at train vs serve time.
_SEGMENT_SPLIT_RE = re.compile(r"[.,!?;:…]+|[—–]+")


def split_into_segments(text: str) -> List[str]:
    """Split a (normalized) sentence at pause/sentence punctuation into the exact
    punctuation-free segments the frontend will send to the G2P model. Each segment
    is additionally cleaned of any residual (non-splitting) punctuation."""
    segments: List[str] = []
    for piece in _SEGMENT_SPLIT_RE.split(text):
        seg = strip_punctuation(piece)
        if seg:
            segments.append(seg)
    return segments


def normalize_voice(voice: str) -> str:
    return voice.strip().lower().replace("_", "-")


def run_phonemize(text: str, voice: str) -> str:
    """Return piper/espeak IPA in word-level format: each word's phonemes
    concatenated, words separated by a single space — identical to the
    ``phoneme_lists_to_ipa`` helper in generate_g2p_manifest_espeak.py."""
    phoneme_lists = phonemize_espeak(text, voice)
    return "".join("".join(sentence) for sentence in phoneme_lists)


def run_phonemize_batch(texts: Sequence[str], voice: str) -> List[str]:
    if not texts:
        return []
    out_lines: List[str] = []
    for text in texts:
        out_lines.append(run_phonemize(text, voice=voice).strip().replace("\n", " "))
    return out_lines


def iter_csv_rows(
    csv_path: Path,
    text_field: str,
    lang_field: str,
    limit: int,
    strip_punct: bool = False,
    split_punct: bool = False,
    normalize_nums: bool = False,
) -> Iterator[Tuple[str, str, str]]:
    """Yield ``(lang, grapheme_text, phonemize_text)`` triples.

    *grapheme_text* is the model's input (clean TN, no digits).
    *phonemize_text* is fed to piper/espeak; for digit+letter-A tokens it uses
    the "A-" form (via :func:`normalize_for_g2p_phonemize`) so that espeak reads
    the letter "A" as a letter name (ˈeɪ) instead of the indefinite article (ɐ).
    For all other tokens the two texts are identical.
    """
    seen = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            text_idx = header.index(text_field)
        except ValueError as exc:
            raise ValueError(f"text field {text_field!r} not found in header: {header}") from exc
        try:
            lang_idx = header.index(lang_field)
        except ValueError as exc:
            raise ValueError(f"lang field {lang_field!r} not found in header: {header}") from exc

        for row in reader:
            if text_idx >= len(row):
                continue
            raw = normalize_text(row[text_idx])
            if not raw:
                continue
            lang = row[lang_idx].strip() if lang_idx < len(row) else ""

            if normalize_nums:
                grapheme = normalize_for_g2p(raw)
                ph_text = normalize_for_g2p_phonemize(raw)
            else:
                grapheme = raw
                ph_text = raw

            if split_punct:
                grapheme_segs = split_into_segments(grapheme)
                # phonemize_text may differ only in "A-" vs "A"; punctuation
                # positions are identical, so we can segment both the same way.
                ph_segs = split_into_segments(ph_text)
                segment_pairs = list(zip(grapheme_segs, ph_segs))
            elif strip_punct:
                g_seg = strip_punctuation(grapheme)
                p_seg = strip_punctuation(ph_text)
                segment_pairs = [(g_seg, p_seg)] if g_seg else []
            else:
                segment_pairs = [(grapheme, ph_text)]

            for g_seg, p_seg in segment_pairs:
                if g_seg:
                    yield (lang, g_seg, p_seg)
                    seen += 1
                    if limit > 0 and seen >= limit:
                        return


def iter_batches(rows: Iterator[Tuple[str, str, str]], batch_size: int) -> Iterator[List[Tuple[str, str, str]]]:
    batch: List[Tuple[str, str, str]] = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def approx_csv_body_line_count(csv_path: Path) -> int:
    """Fast (data lines ≈ newlines − header) for tqdm total; assumes no multiline CSV fields."""
    try:
        with csv_path.open("rb") as f:
            n_newlines = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))
    except OSError:
        return 0
    return max(0, n_newlines - 1)


def ordered_process_batches(
    executor: Executor,
    max_in_flight: int,
    batches_iter: Iterator[List[Tuple[str, str, str]]],
    default_voice: str,
    stress_mode: str,
    strip_punct: bool = False,
    inventories: Dict[str, Tuple[str, ...]] = None,
) -> Iterator[List[Tuple[str, str, List[str]]]]:
    """Like executor.map(process_batch, ...) but submit only max_in_flight tasks ahead; preserve order."""
    inventories = inventories or {}
    it = enumerate(batches_iter)
    pending: dict = {}
    saved: dict[int, List[Tuple[str, str, List[str]]]] = {}
    next_emit = 0
    exhausted = False

    def fill_pending() -> None:
        nonlocal exhausted
        while len(pending) < max_in_flight and not exhausted:
            try:
                idx, batch = next(it)
            except StopIteration:
                exhausted = True
                return
            fut = executor.submit(process_batch, batch, default_voice, stress_mode, strip_punct, inventories)
            pending[fut] = idx

    def emit_ready() -> Iterator[List[Tuple[str, str, List[str]]]]:
        nonlocal next_emit
        while next_emit in saved:
            yield saved.pop(next_emit)
            next_emit += 1

    fill_pending()
    while pending or saved:
        if pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                idx = pending.pop(fut)
                saved[idx] = fut.result()
        yield from emit_ready()
        if exhausted and not pending and not saved:
            break
        fill_pending()


def process_batch(
    batch_rows: Sequence[Tuple[str, str, str]],
    default_voice: str,
    stress_mode: str,
    strip_punct: bool = False,
    inventories: Dict[str, Tuple[str, ...]] = None,
) -> List[Tuple[str, str, List[str]]]:
    """Phonemize a batch of (lang, grapheme_text, phonemize_text) rows.

    Returns (grapheme_text, phoneme_text, atom_tokens) triples, where phoneme_text is the
    manifest ``text`` value (phonemes concatenated within a word, single space between words)
    and atom_tokens is the flat list of atomic phoneme units for building the phoneme vocab.
    """
    inventories = inventories or {}
    results: List[Tuple[str, str, List[str]]] = [("", "", [])] * len(batch_rows)
    by_voice: dict[str, List[Tuple[int, str, str]]] = {}
    for i, (lang, grapheme, ph_text) in enumerate(batch_rows):
        voice = normalize_voice(lang) if lang else default_voice
        if not voice:
            voice = default_voice
        by_voice.setdefault(voice, []).append((i, grapheme, ph_text))

    for voice, indexed_items in by_voice.items():
        # Language-specific multi-char atom whitelist (diphthongs/affricates/...), looked up
        # per voice from phoneme_inventories.json -- adding a new language never touches this
        # function. Unlisted voices still tokenize correctly via the generic combining-mark
        # rule alone (is_attaching()), just without any language-specific atom merging.
        multi_char_phonemes = get_multi_char_phonemes_for_voice(voice, inventories)
        phon_texts = [pt for _, _, pt in indexed_items]
        ipa_lines = run_phonemize_batch(phon_texts, voice=voice)
        for (src_idx, src_grapheme, _), ipa_line in zip(indexed_items, ipa_lines):
            phoneme_str = ipa_line.strip()
            if stress_mode == "drop":
                phoneme_str = phoneme_str.replace(PRIMARY_STRESS, "").replace(SECONDARY_STRESS, "")

            # text = 词内音素拼接、词间单空格（espeak 原生形态，也是模型输出形态）；
            # atoms = 用完整 IPA 音素表最长匹配切出的原子单元（双元音/塞擦音/长音符等保持
            # 不可再分），仅用于统计音素词表，保证 tokenizer 最长匹配产出的 token 都在表内。
            phoneme_text, atoms = phonemes_to_text_and_atoms(
                phoneme_str, stress_mode=stress_mode, multi_char_phonemes=multi_char_phonemes
            )
            results[src_idx] = (src_grapheme, phoneme_text, atoms)

    return results


def write_vocab(path: Path, tokens: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for tok in SPECIAL_TOKENS:
            f.write(tok + "\n")
        for tok in tokens:
            if tok and tok not in SPECIAL_TOKENS:
                f.write(tok + "\n")


def recommend_runtime(cpu_count: int) -> tuple[int, int]:
    """Return (workers, batch_size) tuned for large CSV preprocessing."""
    cpu_count = max(1, cpu_count)
    # Leave a small CPU headroom for OS / IO threads. espeak phonemize is CPU-bound,
    # so scale with the full core count (no artificial 24-worker cap) for large datasets.
    workers = max(1, cpu_count - 2)

    if cpu_count <= 8:
        batch_size = 2048
    elif cpu_count <= 16:
        batch_size = 4096
    else:
        batch_size = 8192
    return workers, batch_size


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inventories_path = args.phoneme_inventories.expanduser().resolve()
    inventories = load_multi_char_phoneme_inventories(str(inventories_path))
    run_self_check(inventories)
    print(f"Phoneme inventories: {inventories_path} (voices configured: {sorted(inventories) or 'none'})")

    manifest_path = output_dir / args.manifest_name
    phoneme_vocab_path = output_dir / args.phoneme_vocab_name
    grapheme_vocab_path = output_dir / args.grapheme_vocab_name
    merged_vocab_path = output_dir / args.merged_vocab_name

    cpu_count = os.cpu_count() or 1
    auto_workers, auto_batch_size = recommend_runtime(cpu_count)
    workers = args.workers if args.workers > 0 else auto_workers
    batch_size = args.batch_size if args.batch_size > 0 else auto_batch_size
    executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor

    if args.show_progress:
        print("Preprocess: counting newlines in CSV for tqdm total…", file=sys.stderr, flush=True)
        if args.limit > 0:
            est_rows = args.limit
        else:
            est_rows = approx_csv_body_line_count(input_csv)
        est_batches = max(1, (est_rows + batch_size - 1) // batch_size) if est_rows > 0 else None
    else:
        est_batches = None

    phoneme_counter: Counter = Counter()
    grapheme_counter: Counter = Counter()
    processed_rows = 0
    manifest_buffer: List[str] = []

    def flush_manifest_buffer(manifest_f) -> None:
        nonlocal manifest_buffer
        if not manifest_buffer:
            return
        manifest_f.write("\n".join(manifest_buffer))
        manifest_f.write("\n")
        manifest_buffer = []

    # Splitting always yields punctuation-free segments, so the phoneme-side defensive
    # filter must run whenever either stripping or splitting is on.
    effective_strip = args.strip_punctuation or args.split_on_punctuation

    batch_iter = iter_batches(
        iter_csv_rows(
            input_csv,
            args.text_field,
            args.lang_field,
            args.limit,
            args.strip_punctuation,
            args.split_on_punctuation,
            args.normalize_numbers,
        ),
        batch_size,
    )

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        with executor_cls(max_workers=workers) as executor:
            ordered = ordered_process_batches(
                executor,
                max(1, workers * 2),
                batch_iter,
                args.default_voice,
                args.stress,
                effective_strip,
                inventories,
            )
            if args.show_progress:
                ordered = tqdm(
                    ordered,
                    desc="Preprocess",
                    unit="batch",
                    total=est_batches,
                    dynamic_ncols=True,
                    mininterval=0.25,
                    disable=False,
                    file=sys.stderr,
                )

            for processed in ordered:
                for text, phoneme_text, atoms in processed:
                    if not text or not phoneme_text:
                        continue
                    manifest_buffer.append(json.dumps({"text_graphemes": text, "text": phoneme_text}, ensure_ascii=False))
                    grapheme_counter.update(text)
                    # atoms 是最长匹配切出的原子单元（词内不含空格）；空格词边界单独统计一次，
                    # 保证它作为一个 token 进入音素词表（tokenizer 的 SPACE_TOKEN）。
                    phoneme_counter.update(atoms)
                    if " " in phoneme_text:
                        phoneme_counter.update([SPACE_TOKEN])
                    processed_rows += 1
                    if len(manifest_buffer) >= MANIFEST_WRITE_BUFFER_LINES:
                        flush_manifest_buffer(manifest_f)

            flush_manifest_buffer(manifest_f)

    if processed_rows == 0:
        raise RuntimeError(f"No valid rows processed from {input_csv}")

    if args.write_vocab:
        phoneme_tokens = [tok for tok, _ in sorted(phoneme_counter.items(), key=lambda kv: (-kv[1], kv[0]))]
        grapheme_tokens = sorted(grapheme_counter.keys())
        merged_tokens: List[str] = []
        seen = set(SPECIAL_TOKENS)
        for tok in phoneme_tokens:
            if tok not in seen:
                seen.add(tok)
                merged_tokens.append(tok)
        for tok in grapheme_tokens:
            if tok not in seen:
                seen.add(tok)
                merged_tokens.append(tok)

        write_vocab(phoneme_vocab_path, phoneme_tokens)
        write_vocab(grapheme_vocab_path, grapheme_tokens)
        write_vocab(merged_vocab_path, merged_tokens)

    print(f"Input CSV: {input_csv}")
    print(f"Processed rows: {processed_rows}")
    print(
        f"CPU: {cpu_count}, workers: {workers}, executor: {args.executor}, "
        f"batch size: {batch_size}"
    )
    print(f"Strip punctuation: {args.strip_punctuation} (no punctuation tokens in text_graphemes/text)")
    print(f"Split on punctuation: {args.split_on_punctuation} (one manifest line per segment; train==serve)")
    print(f"Normalize numbers: {args.normalize_numbers} (digits/codes -> words; MUST match frontend TN)")
    print(f"Manifest: {manifest_path}")
    if args.write_vocab:
        print(f"Phoneme vocab: {phoneme_vocab_path}")
        print(f"Grapheme vocab: {grapheme_vocab_path}")
        print(f"Merged vocab (for training): {merged_vocab_path}")
    else:
        print("Skipped vocab writing (--no-write-vocab).")


if __name__ == "__main__":
    main()
