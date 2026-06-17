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

"""随机初始化 CTC Conformer G2P，导出 **.ckpt**（不经训练）。

- ``lightning``（默认）：PyTorch Lightning 检查点，可用 ``CTCG2PModel.load_from_checkpoint(path)`` 加载。
- ``weights``：与 .nemo 包内 ``model_weights.ckpt`` 相同，仅为 ``state_dict``；需先有 ``model_config.yaml``
  再用 ``from_config_dict`` 建模型后 ``load_state_dict``。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning.pytorch as pl
import torch
from omegaconf import OmegaConf

from nemo.collections.tts.g2p.models.ctc import CTCG2PModel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_conf = Path(__file__).resolve().parent / "conf" / "g2p_conformer_ctc.yaml"
    parser.add_argument(
        "--config",
        type=Path,
        default=default_conf,
        help="完整 YAML（与训练配置同一套键）。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("g2p_conformer_ctc_edge_random.ckpt"),
        help="输出 .ckpt 路径。",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="含 vocab.txt 的目录。默认使用 YAML 中 model.tokenizer.dir。",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        choices=("cpu", "gpu", "cuda", "mps"),
        help="仅用于构造 ModelPT / Trainer，不参与训练。",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=("lightning", "weights"),
        default="lightning",
        help="lightning: 可 load_from_checkpoint；weights: 纯权重（同 .nemo 内 model_weights.ckpt）。",
    )
    parser.add_argument(
        "--also-model-config",
        action="store_true",
        help="额外写出与 .nemo 内相同的 model_config.yaml（与 weights 格式搭配最常用）。",
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
            f"未找到 IPA 词表 {vocab_txt}。请在 YAML 中设置 model.tokenizer.dir，或使用 --tokenizer-dir。"
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

    if args.format == "weights":
        torch.save(model.state_dict(), out_path)
        print(f"已保存纯权重检查点（state_dict）: {out_path}")
    else:
        ckpt = {
            "state_dict": model.state_dict(),
            "hyper_parameters": {"cfg": OmegaConf.to_container(model.cfg, resolve=True)},
            "pytorch-lightning_version": pl.__version__,
        }
        torch.save(ckpt, out_path)
        print(f"已保存 Lightning 检查点: {out_path}")
        print("加载示例: CTCG2PModel.load_from_checkpoint(r\"...\", map_location=\"cpu\")")

    if args.also_model_config:
        yaml_path = out_path.with_name("model_config.yaml")
        model.to_config_file(path2yaml_file=str(yaml_path))
        print(f"已写出 model_config.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
