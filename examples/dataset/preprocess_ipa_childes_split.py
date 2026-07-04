#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Executor, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

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

def is_attaching(ch: str) -> bool:
    if ch == LENGTH:
        return True
    return unicodedata.combining(ch) != 0

def segment_word(ipa: str, stress_mode: str = "attach") -> List[str]:
    units: List[str] = []
    cur = ""
    pending_stress = ""
    join = False

    def flush() -> None:
        nonlocal cur
        if cur:
            units.append(cur)
            cur = ""

    for ch in ipa:
        if ch in TIES:
            join = True
            continue

        if ch in STRESS:
            flush()
            if stress_mode == "separate":
                units.append(ch)
            elif stress_mode == "attach":
                pending_stress += ch
            join = False
            continue

        if ch.isspace():
            flush()
            pending_stress = ""
            join = False
            continue

        if is_attaching(ch):
            if not cur:
                cur = pending_stress
                pending_stress = ""
            cur += ch
            continue

        if join and cur:
            cur += ch
            join = False
        else:
            flush()
            cur = pending_stress + ch
            pending_stress = ""
            join = False

    flush()
    return units



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
) -> Iterator[List[Tuple[str, str]]]:
    """Like executor.map(process_batch, ...) but submit only max_in_flight tasks ahead; preserve order."""
    it = enumerate(batches_iter)
    pending: dict = {}
    saved: dict[int, List[Tuple[str, str]]] = {}
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
            fut = executor.submit(process_batch, batch, default_voice, stress_mode, strip_punct)
            pending[fut] = idx

    def emit_ready() -> Iterator[List[Tuple[str, str]]]:
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
) -> List[Tuple[str, str]]:
    """Phonemize a batch of (lang, grapheme_text, phonemize_text) rows.

    Returns (grapheme_text, phoneme_ipa) pairs.
    """
    results: List[Tuple[str, str]] = [("", "")] * len(batch_rows)
    by_voice: dict[str, List[Tuple[int, str, str]]] = {}
    for i, (lang, grapheme, ph_text) in enumerate(batch_rows):
        voice = normalize_voice(lang) if lang else default_voice
        if not voice:
            voice = default_voice
        by_voice.setdefault(voice, []).append((i, grapheme, ph_text))

    for voice, indexed_items in by_voice.items():
        phon_texts = [pt for _, _, pt in indexed_items]
        ipa_lines = run_phonemize_batch(phon_texts, voice=voice)
        for (src_idx, src_grapheme, _), ipa_line in zip(indexed_items, ipa_lines):
            phoneme_str = ipa_line.strip()
            if stress_mode == "drop":
                phoneme_str = phoneme_str.replace(PRIMARY_STRESS, "").replace(SECONDARY_STRESS, "")
            
            # 使用用户原来的字素级原子切分逻辑，恢复到7月2日之前的实现：
            # 这能完美将重音依附在下一个音素上，同时正确保留发音特征不超长。
            units = segment_word(phoneme_str, stress_mode=stress_mode)
            results[src_idx] = (src_grapheme, " ".join(units))

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
                for text, phoneme_text in processed:
                    if not text or not phoneme_text:
                        continue
                    manifest_buffer.append(json.dumps({"text_graphemes": text, "text": phoneme_text}, ensure_ascii=False))
                    grapheme_counter.update(text)
                    # 因为我们在 process_batch 已经用最长匹配原子正确切分并用空格拼接了，直接 split 即可
                    phoneme_counter.update(phoneme_text.split())
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
