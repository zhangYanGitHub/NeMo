from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from tqdm import tqdm

DEFAULT_EN_US_ESPEAK_IPA_TOKENS: Tuple[str, ...] = (
    "t͡ʃ", "d͡ʒ", "tʃ", "dʒ",
    "eɪ", "aɪ", "ɔɪ", "aʊ", "oʊ",
    "ɚ", "ɝ",
    "aɪə", "aɪɚ",
    "l̩", "m̩", "n̩", "r̩",
    "iː", "uː", "ɑː", "ɔː", "ɜː",
    "ɑːɹ", "ɔːɹ", "ɛɹ", "ɪɹ", "ʊɹ",
)
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
WRITE_BUFFER_SIZE = 256
VOCAB_CACHE_SUFFIX = ".vocab_cache"
RESUME_META_SUFFIX = ".resume_meta.json"
FAILED_SUFFIX = ".failed.jsonl"

_VOICE = "en-us"
_PHONEMIZE_ESPEAK = None
_CANDIDATE_MULTI_SORTED: Tuple[str, ...] = ()


def normalize_ipa_text(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_grapheme_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate NeMo G2P manifest/vocab aligned to espeak-ng en-US IPA output.")
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


def _init_worker(voice: str, candidate_multi_sorted: Sequence[str]) -> None:
    global _VOICE, _PHONEMIZE_ESPEAK, _CANDIDATE_MULTI_SORTED
    from piper_phonemize import phonemize_espeak
    _VOICE = voice
    _PHONEMIZE_ESPEAK = phonemize_espeak
    _CANDIDATE_MULTI_SORTED = tuple(candidate_multi_sorted)


def longest_match_tokenize(text: str, multi_tokens: Sequence[str]) -> List[str]:
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


def tokenize_ipa_units(text: str, multi_tokens: Sequence[str]) -> List[str]:
    return longest_match_tokenize(normalize_ipa_text(text), multi_tokens)


def collect_grapheme_chars(text: str, grapheme_chars: Set[str]) -> None:
    text = normalize_grapheme_text(text)
    grapheme_chars.update(text)


def collect_phoneme_units(
    text: str,
    multi_tokens: Sequence[str],
    single_chars: Set[str],
    observed_multi: Set[str],
) -> None:
    for tok in tokenize_ipa_units(text, multi_tokens):
        if tok in {"<pad>", "<unk>", "<bos>", "<eos>", " "}:
            continue
        if len(tok) > 1:
            observed_multi.add(tok)
        else:
            single_chars.add(tok)


def _process_batch(texts: List[str]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for text in texts:
        try:
            norm_text = normalize_grapheme_text(text)
            ipa = normalize_ipa_text(phoneme_lists_to_ipa(_PHONEMIZE_ESPEAK(norm_text, _VOICE)))
            if not ipa:
                out.append({"ok": False, "text_graphemes": norm_text, "error": "empty_ipa"})
                continue
            out.append({"ok": True, "text_graphemes": norm_text, "text": ipa})
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


def load_vocab_cache(cache_path: Path) -> Optional[Dict[str, List[str]]]:
    if not cache_path.exists():
        return None
    with cache_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return None
    return data


def save_vocab_cache(
    cache_path: Path,
    grapheme_chars: Set[str],
    phoneme_single_chars: Set[str],
    observed_multitokens: Set[str],
) -> None:
    payload = {
        "grapheme_chars": sorted(grapheme_chars),
        "phoneme_single_chars": sorted(phoneme_single_chars),
        "observed_multitokens": sorted(observed_multitokens),
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def load_vocab_from_json(json_path: Path, multi_tokens: Sequence[str]) -> tuple[Set[str], Set[str], Set[str]]:
    grapheme_chars: Set[str] = set()
    phoneme_single_chars: Set[str] = set()
    observed_multitokens: Set[str] = set()
    if not json_path.exists():
        return grapheme_chars, phoneme_single_chars, observed_multitokens

    with json_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            collect_grapheme_chars(entry["text_graphemes"], grapheme_chars)
            collect_phoneme_units(entry["text"], multi_tokens, phoneme_single_chars, observed_multitokens)

    return grapheme_chars, phoneme_single_chars, observed_multitokens


def load_vocab_for_resume(json_path: Path, cache_path: Path, multi_tokens: Sequence[str]) -> tuple[Set[str], Set[str], Set[str]]:
    cached = load_vocab_cache(cache_path)
    if cached is not None:
        return (
            set(cached.get("grapheme_chars", [])),
            set(cached.get("phoneme_single_chars", [])),
            set(cached.get("observed_multitokens", [])),
        )
    return load_vocab_from_json(json_path, multi_tokens)


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


def iter_text_batches(csv_path: Path, text_col_idx: int, batch_size: int, skip_rows: int, num_samples: int) -> Iterator[List[str]]:
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


def sort_vocab_chars(chars: Set[str]) -> List[str]:
    lowercase = [c for c in chars if "a" <= c <= "z"]
    uppercase = [c for c in chars if "A" <= c <= "Z"]
    ascii_other = [c for c in chars if c.isascii() and c not in lowercase and c not in uppercase]
    unicode_other = [c for c in chars if not c.isascii()]
    return sorted(lowercase) + sorted(uppercase) + sorted(ascii_other) + sorted(unicode_other)


def write_vocab(
    vocab_path: Path,
    phoneme_single_chars: Set[str],
    observed_multitokens: Set[str],
    forced_multitokens: Optional[Set[str]] = None,
) -> Tuple[int, int]:
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    reserved = set(SPECIAL_TOKENS) | {" "}
    phoneme_body = {c for c in phoneme_single_chars if c not in reserved}

    multi_union = set(observed_multitokens)
    if forced_multitokens:
        multi_union.update(t for t in forced_multitokens if len(t) >= 2 and t not in reserved)
    multi_sorted = sorted(multi_union, key=lambda x: (-len(x), x))

    with vocab_path.open("w", encoding="utf-8") as f:
        for token in SPECIAL_TOKENS:
            f.write(f"{token}\n")
        f.write(" \n")
        for t in multi_sorted:
            f.write(f"{t}\n")
        for char in sort_vocab_chars(phoneme_body):
            f.write(f"{char}\n")

    return len(phoneme_body), len(multi_sorted)


def save_grapheme_vocab_json(path: Path, grapheme_chars: Set[str]) -> None:
    payload = {
        "grapheme_chars": sort_vocab_chars(set(grapheme_chars)),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


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
            last_text = normalize_grapheme_text(text)
            break

    if last_text is None:
        raise RuntimeError("Resume verification failed: CSV has fewer valid rows than existing manifest lines.")

    actual = fingerprint_text(last_text)
    if actual != expected:
        raise RuntimeError(
            "Resume verification failed: input CSV ordering/content differs from previous run. "
            "Use --no-resume or restore the original CSV."
        )


def main() -> None:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    output_json = args.output_json or (args.output_vocab.parent / "train.json")
    failed_log = args.failed_log or output_json.with_suffix(output_json.suffix + FAILED_SUFFIX)
    cache_path = args.output_vocab.with_suffix(VOCAB_CACHE_SUFFIX)
    meta_path = output_json.with_suffix(RESUME_META_SUFFIX)
    grapheme_vocab_json = args.output_vocab.with_name(args.output_vocab.stem + ".graphemes.json")
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
        grapheme_chars, phoneme_single_chars, observed_multitokens = load_vocab_for_resume(
            output_json, cache_path, candidate_multi_sorted
        )
        char_count, multi_count = write_vocab(
            args.output_vocab,
            phoneme_single_chars,
            observed_multitokens,
            forced_multitokens,
        )
        save_grapheme_vocab_json(grapheme_vocab_json, grapheme_chars)
        print(f"Wrote IPA vocab ({char_count} single-char tokens, {multi_count} multi-char IPA tokens) to {args.output_vocab}")
        print(f"Wrote grapheme vocab JSON to {grapheme_vocab_json}")
        return

    remaining = args.num_samples - skip_rows if args.num_samples >= 0 else None
    map_chunksize = max(1, num_workers * 2)
    total_target = args.num_samples if args.num_samples >= 0 else count_valid_rows(args.input_csv, text_col_idx)
    this_run_target = remaining if remaining is not None else max(0, total_target - skip_rows)

    print(f"Input CSV: {args.input_csv}")
    print(f"Output JSON: {output_json}")
    print(f"Output vocab: {args.output_vocab}")
    print(f"Grapheme vocab JSON: {grapheme_vocab_json}")
    print(f"Failed log: {failed_log}")
    print(f"Voice: {args.voice}")
    print(f"Workers: {num_workers}, batch size: {args.batch_size}, chunksize: {map_chunksize}")
    print(f"Total target: {total_target}, already done: {skip_rows}, this run: {this_run_target}")

    success_count = 0
    error_count = 0
    if skip_rows:
        grapheme_chars, phoneme_single_chars, observed_multitokens = load_vocab_for_resume(
            output_json, cache_path, candidate_multi_sorted
        )
    else:
        grapheme_chars, phoneme_single_chars, observed_multitokens = set(), set(), set()

    write_buffer: List[str] = []
    last_success_text: Optional[str] = None

    def flush_buffer(out_f) -> None:
        nonlocal write_buffer
        if write_buffer:
            out_f.write("\n".join(write_buffer))
            out_f.write("\n")
            write_buffer = []
        save_vocab_cache(cache_path, grapheme_chars, phoneme_single_chars, observed_multitokens)
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
        with ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(args.voice, candidate_multi_sorted),
        ) as executor:
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

                    collect_grapheme_chars(text_g, grapheme_chars)
                    collect_phoneme_units(
                        text_ipa,
                        candidate_multi_sorted,
                        phoneme_single_chars,
                        observed_multitokens,
                    )

                    write_buffer.append(json.dumps({"text_graphemes": text_g, "text": text_ipa}, ensure_ascii=False))
                    success_count += 1
                    last_success_text = text_g
                    pbar.update(1)

                    if len(write_buffer) >= WRITE_BUFFER_SIZE:
                        flush_buffer(out_f)

            pbar.close()
            flush_buffer(out_f)

    char_count, multi_count = write_vocab(
        args.output_vocab,
        phoneme_single_chars,
        observed_multitokens,
        forced_multitokens,
    )
    save_grapheme_vocab_json(grapheme_vocab_json, grapheme_chars)

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
    print(f"Wrote IPA vocab ({char_count} single-char IPA tokens, {multi_count} multi-char IPA tokens) to {args.output_vocab}")
    print(f"Wrote grapheme vocab JSON to {grapheme_vocab_json}")
    if error_count:
        print(f"Logged {error_count} failed samples to {failed_log}")


if __name__ == "__main__":
    main()