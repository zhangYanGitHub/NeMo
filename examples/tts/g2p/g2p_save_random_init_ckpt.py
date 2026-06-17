# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Build a randomly initialized CTC Conformer G2P checkpoint (.nemo) without training.

Use this to benchmark on-device latency / export before you have trained weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
from omegaconf import OmegaConf

from nemo.collections.tts.g2p.models.ctc import CTCG2PModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_conf = Path(__file__).resolve().parent / "conf" / "g2p_conformer_ctc.yaml"
    parser.add_argument(
        "--config",
        type=Path,
        default=default_conf,
        help="Full Hydra-style YAML (same keys as training config).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("g2p_conformer_ctc_edge_random.nemo"),
        help="Output .nemo path.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Directory that contains vocab.txt for ipa_symbol tokenizer. "
        "Default: use model.tokenizer.dir from the YAML (g2p_conformer_ctc.yaml points to dataset/normal/en-US).",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        choices=("cpu", "gpu", "cuda", "mps"),
        help="Lightning accelerator for ModelPT init only (not used for real training here).",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.tokenizer_dir is not None:
        tok_dir = args.tokenizer_dir
    else:
        dir_cfg = cfg.model.tokenizer.get("dir")
        if dir_cfg:
            tok_dir = Path(dir_cfg)
            if not tok_dir.is_absolute():
                tok_dir = (Path.cwd() / tok_dir).resolve()
        else:
            tok_dir = args.config.parent
    tok_dir = tok_dir.resolve()
    vocab_txt = tok_dir / "vocab.txt"
    if not vocab_txt.is_file():
        raise FileNotFoundError(
            f"Expected IPA vocab at {vocab_txt}. "
            "Set model.tokenizer.dir in the YAML, or pass --tokenizer-dir to the folder that contains vocab.txt."
        )

    cfg.model.tokenizer.dir = str(tok_dir)

    accel = args.accelerator
    if accel == "cuda":
        accel = "gpu"

    trainer = pl.Trainer(
        accelerator=accel,
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    model = CTCG2PModel(cfg=cfg.model, trainer=trainer)
    out_path = args.out.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(out_path))
    print(f"Saved randomly initialized model to {out_path}")


if __name__ == "__main__":
    main()
