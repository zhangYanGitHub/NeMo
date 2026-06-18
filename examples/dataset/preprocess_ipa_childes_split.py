#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

from tqdm import tqdm

LANG_SWITCH_RE = re.compile(r"\([^()]*\)")
ZWJ = "\u200d"
TIES = {ZWJ, "\u0361", "\u035c"}
PRIMARY_STRESS = "\u02c8"  # ˈ
SECONDARY_STRESS = "\u02cc"  # ˌ
STRESS = {PRIMARY_STRESS, SECONDARY_STRESS}
LENGTH = "\u02d0"  # ː
SPECIAL_TOKENS = ("<pad>", "<unk>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess ipa-childes-split CSV into NeMo G2P manifest and vocab files. "
            "Only reads language code + raw text columns, then regenerates phonemes with espeak-ng."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="ipa-childes-split style CSV path")
    parser.add_argument("--output-dir", type=Path, required=True, help="output directory")
    parser.add_argument("--text-field", type=str, default="sentence", help="raw text column name")
    parser.add_argument("--lang-field", type=str, default="espeak_lang_code", help="espeak language code column name")
    parser.add_argument("--default-voice", type=str, default="en-us", help="fallback voice when lang field is empty")
    parser.add_argument("--workers", type=int, default=0, help="thread workers; 0 means os.cpu_count()")
    parser.add_argument("--batch-size", type=int, default=1024, help="rows per worker batch")
    parser.add_argument("--limit", type=int, default=0, help="optional row limit")
    parser.add_argument("--stress", choices=["attach", "separate", "drop"], default="attach")
    parser.add_argument("--manifest-name", type=str, default="train.json")
    parser.add_argument("--phoneme-vocab-name", type=str, default="phoneme_vocab.txt")
    parser.add_argument("--grapheme-vocab-name", type=str, default="grapheme_vocab.txt")
    parser.add_argument("--merged-vocab-name", type=str, default="vocab.txt")
    parser.add_argument("--write-vocab", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def normalize_voice(voice: str) -> str:
    return voice.strip().lower().replace("_", "-")


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


def run_espeak(text: str, voice: str) -> str:
    proc = subprocess.run(
        ["espeak-ng", "-v", voice, "--ipa=3", "-q"],
        input=text,
        capture_output=True,
        text=True,
        check=True,
    )
    return LANG_SWITCH_RE.sub("", proc.stdout)


def run_espeak_batch(texts: Sequence[str], voice: str) -> List[str]:
    if not texts:
        return []
    out = run_espeak("\n".join(texts), voice=voice)
    out_lines = [line.strip() for line in out.split("\n")]
    while out_lines and out_lines[-1] == "":
        out_lines.pop()

    if len(out_lines) == len(texts):
        return out_lines

    # Rare alignment mismatch fallback: degrade to per-line for correctness.
    fixed: List[str] = []
    for text in texts:
        fixed.append(run_espeak(text, voice=voice).strip().replace("\n", " "))
    return fixed


def iter_csv_rows(csv_path: Path, text_field: str, lang_field: str, limit: int) -> Iterator[Tuple[str, str]]:
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
            text = normalize_text(row[text_idx])
            if not text:
                continue
            lang = row[lang_idx].strip() if lang_idx < len(row) else ""
            yield (lang, text)
            seen += 1
            if limit > 0 and seen >= limit:
                break


def iter_batches(rows: Iterator[Tuple[str, str]], batch_size: int) -> Iterator[List[Tuple[str, str]]]:
    batch: List[Tuple[str, str]] = []
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


def process_batch(
    batch_rows: Sequence[Tuple[str, str]],
    default_voice: str,
    stress_mode: str,
) -> List[Tuple[str, str]]:
    results: List[Tuple[str, str]] = [("", "")] * len(batch_rows)
    by_voice: dict[str, List[Tuple[int, str]]] = {}
    for i, (lang, text) in enumerate(batch_rows):
        voice = normalize_voice(lang) if lang else default_voice
        if not voice:
            voice = default_voice
        by_voice.setdefault(voice, []).append((i, text))

    for voice, indexed_texts in by_voice.items():
        texts = [t for _, t in indexed_texts]
        ipa_lines = run_espeak_batch(texts, voice=voice)
        for (src_idx, src_text), ipa_line in zip(indexed_texts, ipa_lines):
            units = segment_word(ipa_line, stress_mode=stress_mode)
            results[src_idx] = (src_text, " ".join(units))

    return results


def write_vocab(path: Path, tokens: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for tok in SPECIAL_TOKENS:
            f.write(tok + "\n")
        for tok in tokens:
            if tok and tok not in SPECIAL_TOKENS:
                f.write(tok + "\n")


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / args.manifest_name
    phoneme_vocab_path = output_dir / args.phoneme_vocab_name
    grapheme_vocab_path = output_dir / args.grapheme_vocab_name
    merged_vocab_path = output_dir / args.merged_vocab_name

    workers = args.workers or os.cpu_count() or 1
    batches = iter_batches(iter_csv_rows(input_csv, args.text_field, args.lang_field, args.limit), args.batch_size)

    if args.show_progress:
        if args.limit > 0:
            est_rows = args.limit
        else:
            est_rows = approx_csv_body_line_count(input_csv)
        est_batches = max(1, (est_rows + args.batch_size - 1) // args.batch_size) if est_rows > 0 else None
    else:
        est_batches = None

    phoneme_counter: Counter = Counter()
    grapheme_counter: Counter = Counter()
    processed_rows = 0

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            mapped = executor.map(
                process_batch,
                batches,
                itertools.repeat(args.default_voice),
                itertools.repeat(args.stress),
            )
            if args.show_progress:
                mapped = tqdm(
                    mapped,
                    desc="Preprocess",
                    unit="batch",
                    total=est_batches,
                    dynamic_ncols=True,
                    mininterval=0.5,
                    disable=False,
                )

            for processed in mapped:
                for text, phoneme_text in processed:
                    if not text or not phoneme_text:
                        continue
                    manifest_f.write(json.dumps({"text_graphemes": text, "text": phoneme_text}, ensure_ascii=False) + "\n")
                    grapheme_counter.update(text)
                    phoneme_counter.update(phoneme_text.split())
                    processed_rows += 1

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
    print(f"Workers: {workers}, batch size: {args.batch_size}")
    print(f"Manifest: {manifest_path}")
    if args.write_vocab:
        print(f"Phoneme vocab: {phoneme_vocab_path}")
        print(f"Grapheme vocab: {grapheme_vocab_path}")
        print(f"Merged vocab (for training): {merged_vocab_path}")
    else:
        print("Skipped vocab writing (--no-write-vocab).")


if __name__ == "__main__":
    main()
