# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate NeMo G2P JSONL manifest from IPA-CHILDES CSV using piper-phonemize.

Uses piper-phonemize (in-process espeak-ng C++ bindings) for fast batch G2P
without spawning subprocesses or playing audio.

Supports resume: if output JSON already exists, skips processed rows and appends.

Usage:
    pip install -r examples/dataset/requirements.txt
    python generate_g2p_manifest_espeak.py \\
        --input-csv dataset/ipa-childes-split/train/en-US/data.csv \\
        --output-vocab dataset/normal/en-US/vocab.txt \\
        --voice en-us

    python generate_g2p_manifest_espeak.py \\
        --input-csv dataset/ipa-childes-split/train/en-US/data.csv \\
        --output-vocab dataset/normal/en-US/vocab.txt \\
        --voice en-us \\
        --no-resume
"""

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator, List, Optional, Set, Tuple

from tqdm import tqdm

_VOICE = "en-us"
_PHONEMIZE_ESPEAK = None
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
WRITE_BUFFER_SIZE = 256
VOCAB_CACHE_SUFFIX = ".vocab_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate G2P manifest with piper-phonemize from IPA-CHILDES CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Input CSV file.",
    )
    parser.add_argument(
        "--output-json",
        "--output-manifest",
        type=Path,
        default=None,
        dest="output_json",
        help="Output JSONL manifest path (default: <output-vocab_dir>/train.json).",
    )
    parser.add_argument(
        "--output-vocab",
        type=Path,
        required=True,
        help="Output vocab.txt：从 JSON 的 text 字段（IPA 音素）收集字符集。",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="sentence",
        help="CSV column for input grapheme text (default: sentence).",
    )
    parser.add_argument(
        "--voice",
        type=str,
        required=True,
        help="espeak-ng voice passed to piper-phonemize, e.g. en-us.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=-1,
        help="Number of rows to process (default: -1 for all rows).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: CPU core count).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Number of sentences per worker task (default: 2048).",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing output JSON if present (default: True).",
    )
    return parser.parse_args()


def phoneme_lists_to_ipa(phoneme_lists: List[List[str]]) -> str:
    return "".join("".join(sentence) for sentence in phoneme_lists)


def _init_worker(voice: str) -> None:
    global _VOICE, _PHONEMIZE_ESPEAK
    from piper_phonemize import phonemize_espeak

    _VOICE = voice
    _PHONEMIZE_ESPEAK = phonemize_espeak


def _process_batch(texts: List[str]) -> List[Optional[dict]]:
    entries = []
    for text in texts:
        try:
            ipa = phoneme_lists_to_ipa(_PHONEMIZE_ESPEAK(text, _VOICE))
            if not ipa:
                entries.append(None)
                continue
            entries.append({"text_graphemes": text, "text": ipa})
        except Exception:
            entries.append(None)
    return entries


def get_text_column_index(csv_path: Path, text_field: str) -> int:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
    try:
        return header.index(text_field)
    except ValueError as exc:
        raise ValueError(f"Column '{text_field}' not found in {csv_path}. Available: {header}") from exc


def count_existing_lines(json_path: Path) -> int:
    if not json_path.exists():
        return 0
    count = 0
    with json_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_vocab_cache(cache_path: Path) -> Optional[Set[str]]:
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as f:
        return set(json.load(f))


def save_vocab_cache(cache_path: Path, vocab: Set[str]) -> None:
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(sorted(vocab), f, ensure_ascii=False)


def load_vocab_for_resume(json_path: Path, cache_path: Path) -> Set[str]:
    cached = load_vocab_cache(cache_path)
    if cached is not None:
        return cached
    return load_vocab_from_json(json_path)


def load_vocab_from_json(json_path: Path) -> Set[str]:
    vocab: Set[str] = set()
    if not json_path.exists():
        return vocab

    with json_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            vocab.update(entry["text"])
    return vocab


def count_valid_rows(csv_path: Path, text_col_idx: int, limit: int = -1) -> int:
    count = 0
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if text_col_idx >= len(row):
                continue
            if not row[text_col_idx].strip():
                continue
            count += 1
            if limit >= 0 and count >= limit:
                break
    return count


def iter_text_batches(
    csv_path: Path,
    text_col_idx: int,
    batch_size: int,
    skip_rows: int,
    num_samples: int,
) -> Iterator[List[str]]:
    batch: List[str] = []
    skipped = 0
    seen = 0

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            if text_col_idx >= len(row):
                continue
            text = row[text_col_idx].strip()
            if not text:
                continue

            if skipped < skip_rows:
                skipped += 1
                continue

            batch.append(text)
            seen += 1

            if len(batch) >= batch_size:
                yield batch
                batch = []

            if num_samples >= 0 and seen >= num_samples:
                break

    if batch:
        yield batch


def collect_chars(text: str, vocab: Set[str]) -> None:
    vocab.update(text)


def sort_vocab_chars(chars: Set[str]) -> List[str]:
    lowercase = [c for c in chars if "a" <= c <= "z"]
    uppercase = [c for c in chars if "A" <= c <= "Z"]
    ascii_other = [c for c in chars if c.isascii() and c not in lowercase and c not in uppercase]
    unicode_other = [c for c in chars if not c.isascii()]
    return sorted(lowercase) + sorted(uppercase) + sorted(ascii_other) + sorted(unicode_other)


def write_vocab(vocab_path: Path, chars: Set[str]) -> int:
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    reserved = set(SPECIAL_TOKENS) | {" "}
    body_chars = {c for c in chars if c not in reserved}

    with vocab_path.open("w", encoding="utf-8") as f:
        for token in SPECIAL_TOKENS:
            f.write(f"{token}\n")
        f.write(" \n")
        for char in sort_vocab_chars(body_chars):
            f.write(f"{char}\n")

    return len(body_chars)


def resolve_resume_state(
    output_json: Path,
    resume: bool,
    num_samples: int,
) -> Tuple[int, str]:
    existing = count_existing_lines(output_json)
    if existing == 0:
        return 0, "w"

    if not resume:
        print(f"Overwriting existing output ({existing} lines): {output_json}")
        return 0, "w"

    if num_samples >= 0:
        remaining = max(0, num_samples - existing)
        if remaining == 0:
            print(f"Already have {existing} lines (>= --num-samples {num_samples}). Nothing to do.")
            return existing, "done"
        print(f"Resuming: skip {existing} rows, process {remaining} more (target {num_samples}).")
    else:
        print(f"Resuming: skip {existing} already processed rows.")

    return existing, "a"


def main() -> None:
    args = parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    output_json = args.output_json or (args.output_vocab.parent / "train.json")
    num_workers = args.num_workers or os.cpu_count() or 1
    text_col_idx = get_text_column_index(args.input_csv, args.text_field)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_vocab.parent.mkdir(parents=True, exist_ok=True)

    skip_rows, file_mode = resolve_resume_state(output_json, args.resume, args.num_samples)
    if file_mode == "done":
        vocab = load_vocab_for_resume(output_json, args.output_vocab.with_suffix(VOCAB_CACHE_SUFFIX))
        char_count = write_vocab(args.output_vocab, vocab)
        print(f"Wrote vocab ({char_count} chars) to {args.output_vocab}")
        return

    remaining = args.num_samples - skip_rows if args.num_samples >= 0 else None
    map_chunksize = max(1, num_workers * 2)

    if args.num_samples >= 0:
        total_target = args.num_samples
    else:
        print("Counting CSV rows...")
        total_target = count_valid_rows(args.input_csv, text_col_idx)

    this_run_target = remaining if remaining is not None else max(0, total_target - skip_rows)

    print(f"Input CSV: {args.input_csv}")
    print(f"Output JSON: {output_json}")
    print(f"Output vocab: {args.output_vocab}")
    print(f"Voice: {args.voice}")
    print(f"Workers: {num_workers}, batch size: {args.batch_size}, chunksize: {map_chunksize}")
    print(f"Total target: {total_target}, already done: {skip_rows}, this run: {this_run_target}")

    success_count = 0
    error_count = 0
    vocab = (
        load_vocab_for_resume(output_json, args.output_vocab.with_suffix(VOCAB_CACHE_SUFFIX))
        if skip_rows
        else set()
    )
    write_buffer: List[str] = []

    def flush_buffer(out_f) -> None:
        nonlocal write_buffer
        if write_buffer:
            out_f.write("\n".join(write_buffer))
            out_f.write("\n")
            write_buffer = []
            save_vocab_cache(args.output_vocab.with_suffix(VOCAB_CACHE_SUFFIX), vocab)

    with output_json.open(file_mode, encoding="utf-8") as out_f:
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(args.voice,),
        ) as executor:
            batches = iter_text_batches(
                csv_path=args.input_csv,
                text_col_idx=text_col_idx,
                batch_size=args.batch_size,
                skip_rows=skip_rows,
                num_samples=remaining if remaining is not None else -1,
            )
            pbar = tqdm(
                total=total_target,
                initial=skip_rows,
                desc="G2P",
                unit="条",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            )
            for entries in executor.map(_process_batch, batches, chunksize=map_chunksize):
                for entry in entries:
                    if entry is None:
                        error_count += 1
                        pbar.update(1)
                        continue
                    collect_chars(entry["text"], vocab)
                    write_buffer.append(json.dumps(entry, ensure_ascii=False))
                    success_count += 1
                    pbar.update(1)
                    if len(write_buffer) >= WRITE_BUFFER_SIZE:
                        flush_buffer(out_f)
            pbar.close()
            flush_buffer(out_f)

    char_count = write_vocab(args.output_vocab, vocab)
    cache_path = args.output_vocab.with_suffix(VOCAB_CACHE_SUFFIX)
    if cache_path.exists():
        cache_path.unlink()
    total_lines = skip_rows + success_count

    print(f"Done. Wrote {success_count} new entries ({total_lines} total) to {output_json}")
    print(f"Wrote vocab ({char_count} chars from text) to {args.output_vocab}")
    if error_count:
        print(f"Skipped {error_count} failed samples in this run.")


if __name__ == "__main__":
    main()
