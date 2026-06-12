from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from tqdm import tqdm


# Curated en-US espeak-ng multi-character IPA units.
# These are always written into vocab.txt so the tokenizer can do longest-match
# tokenization even if some units are rare in the sampled manifest.
DEFAULT_EN_US_ESPEAK_IPA_TOKENS: Tuple[str, ...] = (
    # Affricates: tied and untied forms
    "t͡ʃ",
    "d͡ʒ",
    "tʃ",
    "dʒ",

    # Core en-US diphthongs
    "eɪ",
    "aɪ",
    "ɔɪ",
    "aʊ",
    "oʊ",

    # R-colored / rhotic vowel outputs
    "ɚ",
    "ɝ",

    # Triphthong / centering-style outputs
    "aɪə",
    "aɪɚ",

    # Syllabic consonants
    "l̩",
    "m̩",
    "n̩",
    "r̩",

    # Length-marked vowels
    "iː",
    "uː",
    "ɑː",
    "ɔː",
    "ɜː",

    # Rhotic sequences more compatible with IPA-style en-US output
    "ɑːɹ",
    "ɔːɹ",
    "ɛɹ",
    "ɪɹ",
    "ʊɹ",
)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
WRITE_BUFFER_SIZE = 256
VOCAB_CACHE_SUFFIX = ".vocab_cache"
RESUME_META_SUFFIX = ".resume_meta.json"
FAILED_SUFFIX = ".failed.jsonl"

_VOICE = "en-us"
_PHONEMIZE_ESPEAK = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate NeMo G2P manifest/vocab aligned to espeak-ng en-US IPA output."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-json", "--output-manifest", type=Path, default=None, dest="output_json")
    parser.add_argument("--output-vocab", type=Path, required=True)
    parser.add_argument("--text-field", type=str, default="sentence")
    parser.add_argument("--voice", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--failed-log", type=Path, default=None)
    return parser.parse_args()


def phoneme_lists_to_ipa(phoneme_lists: List[List[str]]) -> str:
    return "".join("".join(sentence) for sentence in phoneme_lists)


def _init_worker(voice: str) -> None:
    global _VOICE, _PHONEMIZE_ESPEAK
    from piper_phonemize import phonemize_espeak

    _VOICE = voice
    _PHONEMIZE_ESPEAK = phonemize_espeak


def _process_batch(texts: List[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for text in texts:
        try:
            ipa = phoneme_lists_to_ipa(_PHONEMIZE_ESPEAK(text, _VOICE))
            if not ipa:
                out.append({"ok": False, "text_graphemes": text, "error": "empty_ipa"})
                continue
            out.append({"ok": True, "text_graphemes": text, "text": ipa})
        except Exception as exc:
            out.append(
                {
                    "ok": False,
                    "text_graphemes": text,
                    "error": exc.__class__.__name__,
                    "error_message": str(exc),
                }
            )
    return out


def get_text_column_index(csv_path: Path, text_field: str) -> int:
    with csv_path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    try:
        return header.index(text_field)
    except ValueError as exc:
        raise ValueError(f"Column {text_field!r} not found in {csv_path}. Available: {header}") from exc


def count_existing_lines(json_path: Path) -> int:
    if not json_path.exists():
        return 0
    with json_path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def load_vocab_cache(cache_path: Path) -> Optional[Set[str]]:
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as f:
        return set(json.load(f))


def save_vocab_cache(cache_path: Path, vocab: Set[str]) -> None:
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(sorted(vocab), f, ensure_ascii=False)


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
            vocab.update(entry["text_graphemes"])
            vocab.update(entry["text"])
    return vocab


def load_vocab_for_resume(json_path: Path, cache_path: Path) -> Set[str]:
    cached = load_vocab_cache(cache_path)
    if cached is not None:
        return cached
    return load_vocab_from_json(json_path)


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


def fingerprint_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def iter_valid_texts(csv_path: Path, text_col_idx: int) -> Iterator[str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if text_col_idx >= len(row):
                continue
            text = row[text_col_idx].strip()
            if text:
                yield text


def iter_text_batches(
    csv_path: Path,
    text_col_idx: int,
    batch_size: int,
    skip_rows: int,
    num_samples: int,
) -> Iterator[List[str]]:
    batch: List[str] = []
    seen = 0
    for idx, text in enumerate(iter_valid_texts(csv_path, text_col_idx)):
        if idx < skip_rows:
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


def longest_match_tokenize(text: str, multi_tokens: List[str]) -> List[str]:
    tokens: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        matched = None
        for tok in multi_tokens:
            if text.startswith(tok, i):
                matched = tok
                break
        if matched is not None:
            tokens.append(matched)
            i += len(matched)
        else:
            tokens.append(text[i])
            i += 1
    return tokens


def collect_observed_multitokens(text: str, multi_tokens: List[str], observed: Set[str]) -> None:
    for tok in longest_match_tokenize(text, multi_tokens):
        if len(tok) > 1:
            observed.add(tok)


def sort_vocab_chars(chars: Set[str]) -> List[str]:
    lowercase = [c for c in chars if "a" <= c <= "z"]
    uppercase = [c for c in chars if "A" <= c <= "Z"]
    ascii_other = [c for c in chars if c.isascii() and c not in lowercase and c not in uppercase]
    unicode_other = [c for c in chars if not c.isascii()]
    return sorted(lowercase) + sorted(uppercase) + sorted(ascii_other) + sorted(unicode_other)


def write_vocab(vocab_path: Path, chars: Set[str], multitokens: Optional[Set[str]] = None) -> Tuple[int, int]:
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    reserved = set(SPECIAL_TOKENS) | {" "}
    body_chars = {c for c in chars if c not in reserved}
    multi_sorted: List[str] = []
    if multitokens:
        multi_sorted = sorted(
            (t for t in multitokens if len(t) >= 2 and t not in reserved),
            key=lambda x: (-len(x), x),
        )

    with vocab_path.open("w", encoding="utf-8") as f:
        for token in SPECIAL_TOKENS:
            f.write(f"{token}\n")
        f.write(" \n")
        for t in multi_sorted:
            f.write(f"{t}\n")
        for char in sort_vocab_chars(body_chars):
            f.write(f"{char}\n")
    return len(body_chars), len(multi_sorted)


def load_resume_meta(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_resume_meta(path: Path, data: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def resolve_resume_state(output_json: Path, resume: bool, num_samples: int) -> Tuple[int, str]:
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


def verify_resume_alignment(csv_path: Path, text_col_idx: int, skip_rows: int, meta_path: Path) -> None:
    if skip_rows == 0:
        return
    meta = load_resume_meta(meta_path)
    if not meta:
        print("Warning: resume metadata missing; cannot verify alignment robustly.")
        return
    expected = meta.get("last_text_sha1")
    if not expected:
        print("Warning: resume metadata has no last_text_sha1; cannot verify alignment robustly.")
        return

    last_text = None
    for idx, text in enumerate(iter_valid_texts(csv_path, text_col_idx), start=1):
        if idx == skip_rows:
            last_text = text
            break

    if last_text is None:
        raise RuntimeError("Resume verification failed: CSV has fewer valid rows than existing manifest lines.")

    actual = fingerprint_text(last_text)
    if actual != expected:
        raise RuntimeError(
            "Resume verification failed: input CSV ordering/content differs from previous run. "
            "Use --no-resume or restore the original CSV."
        )


def collect_observed_from_manifest(output_json: Path, multi_tokens: List[str]) -> Set[str]:
    observed: Set[str] = set()
    if not output_json.exists():
        return observed
    with output_json.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            collect_observed_multitokens(entry["text"], multi_tokens, observed)
    return observed


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    output_json = args.output_json or (args.output_vocab.parent / "train.json")
    failed_log = args.failed_log or output_json.with_suffix(output_json.suffix + FAILED_SUFFIX)
    cache_path = args.output_vocab.with_suffix(VOCAB_CACHE_SUFFIX)
    meta_path = output_json.with_suffix(RESUME_META_SUFFIX)
    num_workers = args.num_workers or os.cpu_count() or 1
    text_col_idx = get_text_column_index(args.input_csv, args.text_field)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_vocab.parent.mkdir(parents=True, exist_ok=True)
    failed_log.parent.mkdir(parents=True, exist_ok=True)

    skip_rows, file_mode = resolve_resume_state(output_json, args.resume, args.num_samples)
    verify_resume_alignment(args.input_csv, text_col_idx, skip_rows, meta_path)

    candidate_multi_sorted = sorted(
        (t for t in DEFAULT_EN_US_ESPEAK_IPA_TOKENS if len(t) >= 2),
        key=lambda x: (-len(x), x),
    )
    forced_multitokens = set(candidate_multi_sorted)

    if file_mode == "done":
        vocab = load_vocab_for_resume(output_json, cache_path)
        char_count, multi_count = write_vocab(args.output_vocab, vocab, forced_multitokens)
        print(f"Wrote vocab ({char_count} single-char tokens, {multi_count} forced multi-char IPA) to {args.output_vocab}")
        return

    remaining = args.num_samples - skip_rows if args.num_samples >= 0 else None
    map_chunksize = max(1, num_workers * 2)
    total_target = args.num_samples if args.num_samples >= 0 else count_valid_rows(args.input_csv, text_col_idx)
    this_run_target = remaining if remaining is not None else max(0, total_target - skip_rows)

    print(f"Input CSV: {args.input_csv}")
    print(f"Output JSON: {output_json}")
    print(f"Output vocab: {args.output_vocab}")
    print(f"Failed log: {failed_log}")
    print(f"Voice: {args.voice}")
    print(f"Workers: {num_workers}, batch size: {args.batch_size}, chunksize: {map_chunksize}")
    print(f"Total target: {total_target}, already done: {skip_rows}, this run: {this_run_target}")

    success_count = 0
    error_count = 0
    vocab = load_vocab_for_resume(output_json, cache_path) if skip_rows else set()
    observed_multitokens = collect_observed_from_manifest(output_json, candidate_multi_sorted) if skip_rows else set()
    write_buffer: List[str] = []
    last_success_text: Optional[str] = None

    def flush_buffer(out_f) -> None:
        nonlocal write_buffer
        if write_buffer:
            out_f.write("\n".join(write_buffer))
            out_f.write("\n")
            write_buffer = []
        save_vocab_cache(cache_path, vocab)
        save_resume_meta(
            meta_path,
            {
                "input_csv": str(args.input_csv),
                "text_field": args.text_field,
                "voice": args.voice,
                "processed_lines": skip_rows + success_count,
                "last_text_sha1": fingerprint_text(last_success_text) if last_success_text else None,
            },
        )

    failed_mode = "a" if (args.resume and failed_log.exists()) else "w"
    with output_json.open(file_mode, encoding="utf-8") as out_f, failed_log.open(failed_mode, encoding="utf-8") as fail_f:
        with ProcessPoolExecutor(max_workers=num_workers, initializer=_init_worker, initargs=(args.voice,)) as executor:
            batches = iter_text_batches(
                args.input_csv,
                text_col_idx,
                args.batch_size,
                skip_rows,
                remaining if remaining is not None else -1,
            )

            pbar = tqdm(total=total_target, initial=skip_rows, desc="G2P", unit="rows", dynamic_ncols=True)
            for entries in executor.map(_process_batch, batches, chunksize=map_chunksize):
                for entry in entries:
                    if not entry.get("ok", False):
                        error_count += 1
                        fail_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        pbar.update(1)
                        continue

                    text_g = str(entry["text_graphemes"])
                    text_ipa = str(entry["text"])
                    collect_chars(text_g, vocab)
                    collect_chars(text_ipa, vocab)

                    if candidate_multi_sorted:
                        collect_observed_multitokens(text_ipa, candidate_multi_sorted, observed_multitokens)

                    write_buffer.append(json.dumps({"text_graphemes": text_g, "text": text_ipa}, ensure_ascii=False))
                    success_count += 1
                    last_success_text = text_g
                    pbar.update(1)

                    if len(write_buffer) >= WRITE_BUFFER_SIZE:
                        flush_buffer(out_f)

            pbar.close()
            flush_buffer(out_f)

    char_count, multi_count = write_vocab(args.output_vocab, vocab, forced_multitokens)

    if cache_path.exists():
        cache_path.unlink()

    total_lines = skip_rows + success_count
    save_resume_meta(
        meta_path,
        {
            "input_csv": str(args.input_csv),
            "text_field": args.text_field,
            "voice": args.voice,
            "processed_lines": total_lines,
            "last_text_sha1": fingerprint_text(last_success_text) if last_success_text else None,
        },
    )

    print(f"Done. Wrote {success_count} new entries ({total_lines} total) to {output_json}")
    print(
        f"Wrote vocab ({char_count} single-char from text_graphemes + text, "
        f"{multi_count} forced multi-char IPA) to {args.output_vocab}"
    )
    if error_count:
        print(f"Logged {error_count} failed samples to {failed_log}")


if __name__ == "__main__":
    main()