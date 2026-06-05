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
Extract grapheme text lines from a NeMo G2P JSONL manifest.

Reads each JSON object from the input file and writes the ``text_graphemes``
field (one line per entry) to the output text file. Empty or missing values
are skipped.

Optimized for large manifests (millions of lines):
- binary I/O with large buffers
- fast field extraction without full JSON parse (NeMo G2P manifest format)
- batched writes to reduce syscall overhead
- parallel processing via byte-range sharding (default: all CPU cores)
- optional ``orjson`` fallback parser for non-standard lines

Usage:
    python extract_graphemes_from_manifest.py \\
        --input dataset/release/en-US/train.json \\
        --output dataset/release/en-US/graphemes.txt

    python extract_graphemes_from_manifest.py \\
        --input dataset/release/en-US/train.json \\
        --output dataset/release/en-US/graphemes.txt \\
        --workers 16 \\
        --progress
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import BinaryIO, Iterator, List, Optional, Tuple

try:
    import orjson

    _HAS_ORJSON = True
except ImportError:
    _HAS_ORJSON = False

IO_BUFFER_SIZE = 8 * 1024 * 1024
WRITE_BUFFER_LINES = 8192
_PREFIX = b'{"text_graphemes": "'
_SUFFIX = b'", "text":'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text_graphemes from a NeMo G2P JSONL manifest."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL manifest path (e.g. train.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output text file path (one grapheme line per entry).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel workers (default: all CPU cores). Use 1 to disable parallelism.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar while merging worker outputs (requires tqdm).",
    )
    parser.add_argument(
        "--write-buffer-lines",
        type=int,
        default=WRITE_BUFFER_LINES,
        help=f"Flush output after this many lines per worker (default: {WRITE_BUFFER_LINES}).",
    )
    return parser.parse_args()


def _extract_fast(line: bytes) -> Optional[bytes]:
    if not line.startswith(_PREFIX):
        return None
    end = line.find(_SUFFIX)
    if end < 0:
        return None
    return line[len(_PREFIX) : end]


def _extract_fallback(line: bytes) -> Optional[bytes]:
    if _HAS_ORJSON:
        value = orjson.loads(line).get("text_graphemes")
    else:
        value = json.loads(line.decode("utf-8")).get("text_graphemes")
    if not value:
        return None
    if isinstance(value, bytes):
        return value
    return str(value).encode("utf-8")


def _extract_text_graphemes(line: bytes) -> Optional[bytes]:
    text = _extract_fast(line)
    if text is None:
        text = _extract_fallback(line)
    if not text:
        return None
    text = text.strip()
    return text or None


def _flush_buffer(f_out: BinaryIO, buffer: List[bytes]) -> None:
    if buffer:
        f_out.write(b"\n".join(buffer))
        f_out.write(b"\n")
        buffer.clear()


def _iter_lines(f_in: BinaryIO, show_progress: bool) -> Iterator[bytes]:
    if not show_progress:
        yield from f_in
        return

    from tqdm import tqdm

    yield from tqdm(f_in, desc="Extracting", unit=" lines", mininterval=1.0)


def _split_byte_ranges(file_size: int, num_workers: int) -> List[Tuple[int, int, bool]]:
    if num_workers <= 1 or file_size == 0:
        return [(0, file_size, True)]

    chunk_size = file_size // num_workers
    ranges: List[Tuple[int, int, bool]] = []
    for idx in range(num_workers):
        start = idx * chunk_size
        end = file_size if idx == num_workers - 1 else (idx + 1) * chunk_size
        ranges.append((start, end, idx == num_workers - 1))
    return ranges


def _process_byte_range(
    input_path: str,
    start: int,
    end: int,
    is_last: bool,
    temp_path: str,
    write_buffer_lines: int,
) -> int:
    count = 0
    buffer: List[bytes] = []

    with open(input_path, "rb", buffering=IO_BUFFER_SIZE) as f_in, open(
        temp_path, "wb", buffering=IO_BUFFER_SIZE
    ) as f_out:
        if start > 0:
            f_in.seek(start)
            f_in.readline()

        while True:
            if not is_last and f_in.tell() >= end:
                break

            line = f_in.readline()
            if not line:
                break

            line = line.rstrip(b"\r\n")
            if not line:
                continue

            text = _extract_text_graphemes(line)
            if text is None:
                continue

            buffer.append(text)
            count += 1
            if len(buffer) >= write_buffer_lines:
                _flush_buffer(f_out, buffer)

        _flush_buffer(f_out, buffer)

    return count


def _merge_part_files(part_paths: List[Path], output_path: Path, show_progress: bool) -> None:
    iterator: Iterator[Path] = part_paths
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(part_paths, desc="Merging", unit=" part")

    with output_path.open("wb", buffering=IO_BUFFER_SIZE) as f_out:
        for part_path in iterator:
            with part_path.open("rb", buffering=IO_BUFFER_SIZE) as f_in:
                shutil.copyfileobj(f_in, f_out, length=IO_BUFFER_SIZE)
            part_path.unlink(missing_ok=True)


def extract_graphemes_parallel(
    input_path: Path,
    output_path: Path,
    num_workers: int,
    show_progress: bool,
    write_buffer_lines: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_size = input_path.stat().st_size
    ranges = _split_byte_ranges(file_size, num_workers)

    if len(ranges) == 1:
        return _process_byte_range(
            str(input_path),
            ranges[0][0],
            ranges[0][1],
            ranges[0][2],
            str(output_path),
            write_buffer_lines,
        )

    part_paths: List[Path] = []
    tasks = []
    for idx, (start, end, is_last) in enumerate(ranges):
        part_path = output_path.with_suffix(output_path.suffix + f".part{idx:04d}")
        part_paths.append(part_path)
        tasks.append((start, end, is_last, part_path))

    total = 0
    try:
        with ProcessPoolExecutor(max_workers=len(tasks)) as pool:
            futures = {
                pool.submit(
                    _process_byte_range,
                    str(input_path),
                    start,
                    end,
                    is_last,
                    str(part_path),
                    write_buffer_lines,
                ): part_path
                for start, end, is_last, part_path in tasks
            }
            for future in as_completed(futures):
                total += future.result()

        _merge_part_files(part_paths, output_path, show_progress)
    except Exception:
        for part_path in part_paths:
            part_path.unlink(missing_ok=True)
        raise

    return total


def extract_graphemes_serial(
    input_path: Path,
    output_path: Path,
    show_progress: bool,
    write_buffer_lines: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    buffer: List[bytes] = []

    with input_path.open("rb", buffering=IO_BUFFER_SIZE) as f_in, output_path.open(
        "wb", buffering=IO_BUFFER_SIZE
    ) as f_out:
        for raw_line in _iter_lines(f_in, show_progress):
            line = raw_line.rstrip(b"\r\n")
            if not line:
                continue

            text = _extract_text_graphemes(line)
            if text is None:
                continue

            buffer.append(text)
            count += 1
            if len(buffer) >= write_buffer_lines:
                _flush_buffer(f_out, buffer)

        _flush_buffer(f_out, buffer)

    return count


def extract_graphemes(
    input_path: Path,
    output_path: Path,
    num_workers: int = 0,
    show_progress: bool = False,
    write_buffer_lines: int = WRITE_BUFFER_LINES,
) -> int:
    workers = num_workers if num_workers > 0 else (os.cpu_count() or 1)
    workers = max(1, workers)

    if workers == 1:
        return extract_graphemes_serial(
            input_path, output_path, show_progress, write_buffer_lines
        )
    return extract_graphemes_parallel(
        input_path, output_path, workers, show_progress, write_buffer_lines
    )


def main() -> None:
    args = parse_args()
    workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
    count = extract_graphemes(
        args.input,
        args.output,
        num_workers=workers,
        show_progress=args.progress,
        write_buffer_lines=args.write_buffer_lines,
    )
    print(f"Wrote {count} grapheme lines to {args.output} ({workers} worker(s))")


if __name__ == "__main__":
    main()
