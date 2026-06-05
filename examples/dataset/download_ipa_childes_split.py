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
Download the IPA-CHILDES split G2P dataset from HuggingFace.

This dataset was used to fine-tune
https://huggingface.co/fdemelo/g2p-mbyt5-12l-ipa-childes-espeak

Dataset: https://huggingface.co/datasets/fdemelo/ipa-childes-split
License: CC-BY-4.0

Usage:
    python download_ipa_childes_split.py
    python download_ipa_childes_split.py --output-dir /path/to/dataset/ipa-childes-split
    python download_ipa_childes_split.py --splits train
    python download_ipa_childes_split.py --splits train test
    python download_ipa_childes_split.py --languages en-US
    python download_ipa_childes_split.py --languages en-US zh-CN --splits train
"""

import argparse
from pathlib import Path
from typing import List, Optional

from huggingface_hub import snapshot_download

DATASET_REPO = "fdemelo/ipa-childes-split"
MODEL_REPO = "fdemelo/g2p-mbyt5-12l-ipa-childes-espeak"
SUPPORTED_SPLITS = ("train", "test")


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Download the IPA-CHILDES split G2P dataset used to train {MODEL_REPO} "
            f"from HuggingFace ({DATASET_REPO})."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save the dataset (default: <repo_root>/dataset/ipa-childes-split).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SUPPORTED_SPLITS,
        default=list(SUPPORTED_SPLITS),
        help="Dataset splits to download (default: train test).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help=(
            "IETF language tags to download, e.g. en-US zh-CN. "
            "If omitted, all 31 languages are downloaded."
        ),
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional HuggingFace token for authenticated downloads.",
    )
    return parser.parse_args()


def build_allow_patterns(splits: List[str], languages: Optional[List[str]]) -> List[str]:
    if languages is None:
        return [f"{split}/**" for split in splits]

    return [f"{split}/{lang}/data.csv" for split in splits for lang in languages]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (get_repo_root() / "dataset" / "ipa-childes-split")
    output_dir.mkdir(parents=True, exist_ok=True)

    allow_patterns = build_allow_patterns(args.splits, args.languages)
    allow_patterns.append("README.md")

    print(f"Downloading {DATASET_REPO} to {output_dir}")
    print(f"Splits: {', '.join(args.splits)}")
    if args.languages:
        print(f"Languages: {', '.join(args.languages)}")
    else:
        print("Languages: all")

    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        local_dir=str(output_dir),
        allow_patterns=allow_patterns,
        token=args.hf_token,
    )

    print(f"Done. Dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
