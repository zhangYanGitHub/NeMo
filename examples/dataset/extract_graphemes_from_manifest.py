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

Usage:
    python extract_graphemes_from_manifest.py \\
        --input dataset/release/en-US/train.json \\
        --output dataset/release/en-US/graphemes.txt
"""

import argparse
import json
from pathlib import Path


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
    return parser.parse_args()


def extract_graphemes(input_path: Path, output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(input_path, "r", encoding="utf-8") as f_in, open(
        output_path, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            entry = json.loads(line)
            text = entry.get("text_graphemes", "").strip()
            if text:
                f_out.write(text + "\n")
                count += 1
    return count


def main() -> None:
    args = parse_args()
    count = extract_graphemes(args.input, args.output)
    print(f"Wrote {count} grapheme lines to {args.output}")


if __name__ == "__main__":
    main()
