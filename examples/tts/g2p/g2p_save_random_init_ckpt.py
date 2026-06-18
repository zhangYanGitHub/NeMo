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

"""随机初始化 Conformer-CTC G2P：仅在 ``--out-dir`` 生成 ``model.ckpt`` + ``model.json``。

与训练仓库中 ``export_nemo_g2p_ctc_onnx.py`` / ``g2p_nemo_client.py`` 对齐的要点：

- **model.ckpt**（默认 ``lightning``）：``hyper_parameters.cfg`` 存 **DictConfig**（不用纯 dict），
  以便 ``CTCG2PModel.load_from_checkpoint(path)`` 在反序列化后仍得到 ``cfg.model_name`` 等属性访问。
- **model.json**：含 ``model_config``、``inference``、``io_shapes``，以及 **onnx_runtime_meta** ——
  字段与导出脚本写入的 ``*.g2p_export_meta.json`` 一致（供与 ``g2p_nemo_client.load_g2p_export_meta`` 对照；ONNX 仍须用 export 脚本生成 sidecar）。

``--format weights``：仅 ``state_dict``；此时须用 ``model.json`` 的 ``model_config`` 自行 ``OmegaConf.create`` + ``CTCG2PModel`` + ``load_state_dict``。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo.collections.tts.g2p.models.ctc import CTCG2PModel


def _to_jsonable(obj: Any) -> Any:
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


def _grapheme_vocab_ordered(model: CTCG2PModel) -> List[str]:
    """与 export_nemo_g2p_ctc_onnx._grapheme_vocab_ordered 一致。"""
    tg = model.tokenizer_grapheme
    n = int(getattr(tg, "vocab_size", 0))
    if n <= 0:
        return []
    out: List[str] = []
    for i in range(n):
        if hasattr(tg, "ids_to_tokens"):
            tks = tg.ids_to_tokens([i])
            out.append(tks[0] if tks else "")
        else:
            out.append("")
    return out


def _grapheme_unk_id_optional(model: CTCG2PModel) -> Optional[int]:
    tg = model.tokenizer_grapheme
    try:
        return int(tg.unk_id)
    except Exception:
        return None


def _ipa_tokenizer_special_strings(model: CTCG2PModel) -> List[str]:
    tok = getattr(model, "tokenizer", None)
    inner = getattr(tok, "tokenizer", None) if tok is not None else None
    if inner is None:
        return []
    sp = getattr(inner, "all_special_tokens", None)
    if isinstance(sp, (list, tuple)) and sp:
        return [str(x) for x in sp if x is not None]
    out: List[str] = []
    for name in ("pad_token", "unk_token", "blank_token"):
        t = getattr(inner, name, None)
        if t is not None:
            out.append(str(t))
    return out


def _ctc_supported_punctuation_sorted(model: CTCG2PModel) -> List[str]:
    dec = getattr(model, "decoding", None)
    if dec is None:
        return []
    sp = getattr(dec, "supported_punctuation", None)
    if not sp:
        return []
    return sorted(str(x) for x in sp)


def _resolve_blank_id(model: CTCG2PModel) -> tuple[int, str]:
    if hasattr(model, "decoding") and hasattr(model.decoding, "blank_id"):
        return int(model.decoding.blank_id), "model.decoding.blank_id"
    tok = getattr(model, "tokenizer", None)
    inner = getattr(tok, "tokenizer", None) if tok is not None else None
    vs = getattr(inner, "vocab_size", None) if inner is not None else None
    if vs is not None:
        return int(vs), "model.tokenizer.tokenizer.vocab_size"
    n = len(getattr(model, "vocabulary", []))
    return n, "fallback len(model.vocabulary)"


def _resolve_num_classes(model: CTCG2PModel) -> tuple[int, str]:
    dec = getattr(model, "decoder", None)
    if dec is not None and hasattr(dec, "num_classes_with_blank"):
        return int(dec.num_classes_with_blank), "model.decoder.num_classes_with_blank"
    return -1, "unknown"


def _log_probs_layout_note(model: CTCG2PModel, device: torch.device) -> str:
    model.eval()
    T = min(8, int(getattr(model, "max_source_len", model.cfg.get("max_source_len", 128))))
    vocab = int(model.tokenizer_grapheme.vocab_size)
    ids = torch.randint(1, max(2, vocab), (1, T), device=device, dtype=torch.long)
    lens = torch.tensor([T], device=device, dtype=torch.long)
    with torch.inference_mode():
        lp, el = model.forward_for_export(ids, lens)
    return (
        f"log_probs_shape_example={tuple(lp.shape)}; encoded_len_shape_example={tuple(el.shape)}; "
        "通常 log_probs 为 [B, T_enc, num_classes_with_blank]（以实测为准）"
    )


def _build_onnx_runtime_meta(model: CTCG2PModel, device: torch.device) -> Dict[str, Any]:
    """
    与 export_nemo_g2p_ctc_onnx.write_sidecar_metadata 写入的 dict 对齐（除 fixed_batch/opset 等导出后才有的项）。
    g2p_nemo_client.load_g2p_export_meta 要求 phoneme_labels / blank_index / max_source_len / grapheme_vocab。
    """
    blank_id, blank_src = _resolve_blank_id(model)
    ncls, ncls_src = _resolve_num_classes(model)
    layout = _log_probs_layout_note(model, device)
    do_lower = bool(model.cfg.tokenizer_grapheme.get("do_lower", True))
    add_punct = bool(model.cfg.tokenizer_grapheme.get("add_punctuation", False))
    try:
        import nemo

        nemo_ver = getattr(nemo, "__version__", "unknown")
    except Exception:
        nemo_ver = "unknown"

    return {
        "phoneme_labels": list(model.vocabulary),
        "phoneme_label_count": len(model.vocabulary),
        "blank_index": blank_id,
        "blank_index_how_determined": blank_src,
        "blank_handling_note": (
            "CTC blank 与真实 token 列对齐：logits 最后一维为 num_classes_with_blank；"
            "NeMo CTCBPEDecoding 使用 blank_id = tokenizer.tokenizer.vocab_size（通常等于 len(phoneme_labels)）。"
            "解码前对每帧 argmax，再按 CTC 规则去 blank、合并连续重复 token。"
        ),
        "max_source_len": int(model._cfg.get("max_source_len", model.max_source_len)),
        "tokenizer_grapheme_do_lower": do_lower,
        "tokenizer_grapheme_add_punctuation": add_punct,
        "grapheme_vocab": _grapheme_vocab_ordered(model),
        "grapheme_unk_id": _grapheme_unk_id_optional(model),
        "tokenizer_special_tokens": _ipa_tokenizer_special_strings(model),
        "ctc_supported_punctuation": _ctc_supported_punctuation_sorted(model),
        "model_mode": str(getattr(model, "mode", "")),
        "model_cfg_model_name": str(model.cfg.get("model_name", "")),
        "fixed_batch": None,
        "fixed_seq_len": None,
        "opset_version": None,
        "export_backend": "not_exported_yet_random_init_bundle",
        "num_classes_with_blank": ncls,
        "num_classes_with_blank_how_determined": ncls_src,
        "log_probs_layout_note": layout,
        "onnx_expected_input_names": ["input_ids", "input_len"],
        "onnx_expected_output_names": ["log_probs", "encoded_len"],
        "pytorch_versions": {"torch": torch.__version__, "nemo_toolkit": nemo_ver},
    }


def _build_model_json(
    model: CTCG2PModel,
    ipa_vocab_path: Path,
    ckpt_name: str,
    device: torch.device,
) -> Dict[str, Any]:
    nwb = model.decoder.num_classes_with_blank
    phoneme_labels = [str(t) for t in list(model.vocabulary)]
    blank_class_index = nwb - 1

    inference = {
        "checkpoint_file": ckpt_name,
        "model_target": getattr(model.cfg, "target", None),
        "max_source_len": int(model._cfg.get("max_source_len", model.max_source_len)),
        "tokenizer_grapheme_do_lower": bool(model._cfg.tokenizer_grapheme.get("do_lower", True)),
        "ipa_vocab_path": str(ipa_vocab_path.resolve()),
        "phoneme_labels": phoneme_labels,
        "decoder_num_classes_with_blank": int(nwb),
        "ctc_blank_class_index": int(blank_class_index),
        "note": "log_probs 最后一维 ctc_blank_class_index 为 CTC blank 列下标。",
        "grapheme_vocab_ordered": _grapheme_vocab_ordered(model),
    }

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

    io_shapes = {
        "forward_for_export": "forward_for_export(input_ids, input_len) -> (log_probs, encoded_len)",
        "input_names": in_names,
        "output_names": out_names,
        "dynamic_axes": dyn_plain,
    }

    return {
        "schema_version": 3,
        "checkpoint_file": ckpt_name,
        "model_config": _to_jsonable(model.cfg),
        "inference": inference,
        "io_shapes": io_shapes,
        "onnx_runtime_meta": _build_onnx_runtime_meta(model, device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_conf = Path(__file__).resolve().parent / "conf" / "g2p_conformer_ctc.yaml"
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("g2p_random_init_bundle"),
        help="输出目录：model.ckpt + model.json",
    )
    parser.add_argument(
        "--ckpt-name",
        type=str,
        default="model.ckpt",
        help="ckpt 文件名",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_conf,
        help="Hydra 风格完整 YAML",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="含 vocab.txt；默认 YAML model.tokenizer.dir",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cpu",
        choices=("cpu", "gpu", "cuda", "mps"),
        help="仅构建 Trainer 用",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=("lightning", "weights"),
        default="lightning",
        help="lightning: 供 load_from_checkpoint（cfg 存 DictConfig）；weights: 仅 state_dict",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="只写 ckpt",
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
            f"未找到 IPA 词表 {vocab_txt}。请设置 model.tokenizer.dir 或 --tokenizer-dir。"
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
    if accel == "mps":
        model.to("mps")
    elif accel == "gpu":
        model.to(torch.device("cuda", 0))
    else:
        model.to("cpu")
    dev = next(model.parameters()).device

    if args.format == "weights":
        torch.save(model.state_dict(), ckpt_path)
        print(f"已保存纯权重: {ckpt_path}")
    else:
        ckpt = {
            "state_dict": model.state_dict(),
            "hyper_parameters": {"cfg": model.cfg},
            "pytorch-lightning_version": pl.__version__,
        }
        torch.save(ckpt, ckpt_path)
        print(f"已保存 Lightning ckpt（cfg 为 DictConfig）: {ckpt_path}")

    if not args.no_json:
        payload = _build_model_json(model, vocab_txt, ckpt_path.name, dev)
        _write_json(out_dir / "model.json", payload)
        print(f"已写出: {out_dir / 'model.json'}（含 onnx_runtime_meta，便于对照 g2p_nemo_client / export sidecar）")
    else:
        print(f"输出目录: {out_dir}")


if __name__ == "__main__":
    main()
