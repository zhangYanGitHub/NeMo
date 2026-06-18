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

"""随机初始化 CTC Conformer G2P：只生成 **一个目录** 内的 ``model.ckpt`` 与配套 **JSON**（不经训练、不导出 ONNX）。

- ``lightning``（默认）：Lightning 检查点，可用 ``CTCG2PModel.load_from_checkpoint(目录/model.ckpt)``。
- ``weights``：纯 ``state_dict``（与 .nemo 内 ``model_weights.ckpt`` 同类）。

``--out-dir`` 下默认生成：

- ``model.ckpt``
- ``model_config.json``：解析后的 ``model.cfg``
- ``inference.json``：音素/字素表、blank 下标、``max_source_len``、IPA ``vocab.txt`` 路径等
- ``io_shapes.json``：``forward_for_export`` 的输入输出名字与 ``dynamic_axes``（给部署侧对齐 I/O 维度假设）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo.collections.tts.g2p.models.ctc import CTCG2PModel


def _to_jsonable(obj: Any) -> Any:
    """将 OmegaConf / 容器递归转为可 ``json.dump`` 的结构。"""
    if isinstance(obj, DictConfig):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, ListConfig):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _grapheme_labels_in_id_order(model: CTCG2PModel) -> list[str]:
    g = model.tokenizer_grapheme
    inv = getattr(g, "inv_vocab", None)
    if not inv:
        return []
    n = getattr(g, "vocab_size", len(inv))
    out: list[str] = []
    for i in range(n):
        tok = inv.get(i)
        if tok is None:
            out.append("")
        else:
            out.append(tok if isinstance(tok, str) else str(tok))
    return out


def _write_json_bundle(model: CTCG2PModel, ipa_vocab_path: Path, out_dir: Path, ckpt_path: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(out_dir / "model_config.json", _to_jsonable(model.cfg))

    nwb = model.decoder.num_classes_with_blank
    phoneme_labels = [str(t) for t in list(model.vocabulary)]
    blank_class_index = nwb - 1

    _write_json(
        out_dir / "inference.json",
        {
            "schema_version": 1,
            "checkpoint_file": ckpt_path.name,
            "model_target": getattr(model.cfg, "target", None),
            "max_source_len": int(model._cfg.get("max_source_len", model.max_source_len)),
            "tokenizer_grapheme_do_lower": bool(model._cfg.tokenizer_grapheme.get("do_lower", True)),
            "ipa_vocab_path": str(ipa_vocab_path.resolve()),
            "phoneme_labels": phoneme_labels,
            "decoder_num_classes_with_blank": int(nwb),
            "ctc_blank_class_index": int(blank_class_index),
            "note": "log_probs 最后一维下标 ctc_blank_class_index 为 CTC blank；0..len(phoneme_labels)-1 与 phoneme_labels 对齐。",
            "grapheme_labels_by_id": _grapheme_labels_in_id_order(model),
        },
    )

    model._prepare_for_export()
    try:
        in_names = list(model.input_names)
        out_names = list(model.output_names)
        dyn = model.dynamic_shapes_for_export(use_dynamo=False)
        dyn_plain: Dict[str, Any] = {
            k: (list(v) if hasattr(v, "__iter__") and not isinstance(v, (str, dict)) else v)
            for k, v in dict(dyn).items()
        }
    finally:
        model._export_teardown()

    _write_json(
        out_dir / "io_shapes.json",
        {
            "schema_version": 1,
            "forward_for_export": "forward_for_export(input_ids, input_len) -> (log_probs, encoded_len)",
            "input_names": in_names,
            "output_names": out_names,
            "dynamic_axes": dyn_plain,
        },
    )

    print(f"已写出目录 {out_dir}：{ckpt_path.name}, model_config.json, inference.json, io_shapes.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_conf = Path(__file__).resolve().parent / "conf" / "g2p_conformer_ctc.yaml"
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("g2p_random_init_bundle"),
        help="输出目录（将创建）；内含 model.ckpt 与若干 JSON。",
    )
    parser.add_argument(
        "--ckpt-name",
        type=str,
        default="model.ckpt",
        help="写在 out-dir 下的 ckpt 文件名。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_conf,
        help="完整 YAML（与训练配置同一套键）。",
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
        help="lightning: 可 load_from_checkpoint；weights: 纯 state_dict。",
    )
    parser.add_argument(
        "--also-model-config",
        action="store_true",
        help="在同一目录额外写出 model_config.yaml。",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="只写 ckpt，不写 JSON。",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / args.ckpt_name

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

    if args.format == "weights":
        torch.save(model.state_dict(), ckpt_path)
        print(f"已保存纯权重: {ckpt_path}")
    else:
        ckpt = {
            "state_dict": model.state_dict(),
            "hyper_parameters": {"cfg": OmegaConf.to_container(model.cfg, resolve=True)},
            "pytorch-lightning_version": pl.__version__,
        }
        torch.save(ckpt, ckpt_path)
        print(f"已保存 Lightning 检查点: {ckpt_path}")

    if not args.no_json:
        _write_json_bundle(model, vocab_txt, out_dir, ckpt_path)
    else:
        print(f"输出目录: {out_dir}")

    if args.also_model_config:
        yaml_path = out_dir / "model_config.yaml"
        model.to_config_file(path2yaml_file=str(yaml_path))
        print(f"已写出 model_config.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
