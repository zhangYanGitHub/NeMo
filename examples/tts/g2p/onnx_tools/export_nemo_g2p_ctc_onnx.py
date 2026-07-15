#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeMo CTCG2PModel (conformer_bpe) -> ONNX，含官方 export + torch.onnx.export fallback、
metadata sidecar、onnx checker、ORT CPU 校验、PyTorch/ONNX 数值自检工具函数。

Python 3.10+。依赖: nemo_toolkit[core], torch, onnx, onnxruntime, numpy，
以及 NeMo TTS 包内 import 链常用包（librosa/soundfile/scipy/einops/matplotlib/numba 等；G2P 导出本身不用 TTS 声学管线）。

示例:
  # 上线推荐：batch=1，序列长度动态（G2P 变长输入）
  python export_nemo_g2p_ctc_onnx.py --nemo model.nemo --out g2p.onnx --profile mobile_dynamic_seq

  # 直接从 Lightning .ckpt 导出（与 .nemo 二选一）
  python export_nemo_g2p_ctc_onnx.py --ckpt last.ckpt --out g2p.onnx --profile mobile_dynamic_seq

  # NNAPI / 极端静态 shape 试探
  python export_nemo_g2p_ctc_onnx.py --nemo model.nemo --out g2p_static.onnx --profile mobile_fixed_all --fixed-seq-len 128
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import onnx
import torch

try:
    from nemo.collections.tts.g2p.models.ctc import CTCG2PModel
except ImportError as e:
    _hints: list[str] = []
    _msg = str(e).lower()
    if any(x in _msg for x in ("librosa", "matplotlib", "soundfile", "scipy", "einops")):
        _hints.append(
            "G2P 虽不跑 TTS，但 NeMo import 会加载 tts/helpers 等；可 "
            'pip install "nemo_toolkit[core]" librosa soundfile scipy einops matplotlib numba，'
            '或兜底 pip install "nemo_toolkit[tts]"。'
        )
    if "sequenceparallel" in _msg.replace(" ", ""):
        _hints.append(
            "PyTorch 过旧，缺少 SequenceParallel；请升级 torch 与 nemo 要求一致（如 pip install -U torch），"
            '自检: python -c "from torch.distributed.tensor.parallel import SequenceParallel"。'
        )
    _extra = (" 提示: " + " | ".join(_hints)) if _hints else ""
    raise ImportError(
        "无法导入 CTCG2PModel。请安装 nemo_toolkit（建议 nemo_toolkit[core] + 上述 import 链依赖），"
        "若未 pip 安装 nemo，可将 NeMo git 源码根目录加入 PYTHONPATH。"
        f"{_extra}\n原始错误: {e}"
    ) from e

try:
    from nemo.utils.export_utils import replace_for_export
except ImportError as e:
    replace_for_export = None  # type: ignore[misc, assignment]
    _replace_import_err = e
else:
    _replace_import_err = None

_LOG = logging.getLogger("export_nemo_g2p_ctc_onnx")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def _device(s: str) -> torch.device:
    s = s.lower().strip()
    if s == "cpu":
        return torch.device("cpu")
    if s.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("指定 cuda 但 torch.cuda.is_available() 为 False")
        return torch.device(s)
    raise ValueError(f"不支持的 device: {s}")


def _resolve_blank_id(model: CTCG2PModel) -> tuple[int, str]:
    """
    NeMo CTCBPEDecoding: blank_id = tokenizer.tokenizer.vocab_size（见 ctc_decoding.py）。
    不同 NeMo 版本可能挂在 decoding / decoding.decoding；此处做多路径 introspection。
    """
    if hasattr(model, "decoding") and hasattr(model.decoding, "blank_id"):
        return int(model.decoding.blank_id), "model.decoding.blank_id"
    tok = getattr(model, "tokenizer", None)
    inner = getattr(tok, "tokenizer", None) if tok is not None else None
    vs = getattr(inner, "vocab_size", None) if inner is not None else None
    if vs is not None:
        return int(vs), "model.tokenizer.tokenizer.vocab_size"
    n = len(getattr(model, "vocabulary", []))
    return n, "fallback len(model.vocabulary)（若与 HF vocab_size 不一致请人工核对）"


def _resolve_num_classes(model: CTCG2PModel) -> tuple[int, str]:
    dec = getattr(model, "decoder", None)
    if dec is not None and hasattr(dec, "num_classes_with_blank"):
        return int(dec.num_classes_with_blank), "model.decoder.num_classes_with_blank"
    return -1, "unknown"


def _grapheme_vocab_ordered(model: CTCG2PModel) -> list[str]:
    """按 id 顺序导出 grapheme 符号，供纯 ONNX 侧手写 text_to_ids（对齐 CharTokenizer）。"""
    tg = model.tokenizer_grapheme
    n = int(getattr(tg, "vocab_size", 0))
    if n <= 0:
        return []
    out: list[str] = []
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


def _ipa_tokenizer_special_strings(model: CTCG2PModel) -> list[str]:
    """
    与 IPASymbolTokenizer.tokens_to_text 中过滤的 all_special_tokens 一致
    （pad / unk / blank 等不参与最终 IPA 字符串拼接）。
    """
    tok = getattr(model, "tokenizer", None)
    inner = getattr(tok, "tokenizer", None) if tok is not None else None
    if inner is None:
        return []
    sp = getattr(inner, "all_special_tokens", None)
    if isinstance(sp, (list, tuple)) and sp:
        return [str(x) for x in sp if x is not None]
    out: list[str] = []
    for name in ("pad_token", "unk_token", "blank_token"):
        t = getattr(inner, name, None)
        if t is not None:
            out.append(str(t))
    return out


def _ctc_supported_punctuation_sorted(model: CTCG2PModel) -> list[str]:
    """与 AbstractCTCDecoding.decode_tokens_to_str_with_strip_punctuation 中 regex 用到的标点集合一致（排序仅便于 JSON 稳定）。"""
    dec = getattr(model, "decoding", None)
    if dec is None:
        return []
    sp = getattr(dec, "supported_punctuation", None)
    if not sp:
        return []
    return sorted(str(x) for x in sp)


def _log_probs_layout(model: CTCG2PModel, device: torch.device) -> str:
    """用一次 forward_for_export 推断 log_probs 张量布局说明。"""
    model.eval()
    T = min(8, int(getattr(model, "max_source_len", model.cfg.get("max_source_len", 128))))
    vocab = int(model.tokenizer_grapheme.vocab_size)
    ids = torch.randint(1, max(2, vocab), (1, T), device=device, dtype=torch.long)
    lens = torch.tensor([T], device=device, dtype=torch.long)
    with torch.inference_mode():
        lp, el = model.forward_for_export(ids, lens)
    return f"log_probs_shape_example={tuple(lp.shape)}; encoded_len_shape_example={tuple(el.shape)}; " "通常 log_probs 为 [B, T_enc, num_classes_with_blank]（以实测为准）"


@dataclass
class ExportProfile:
    fixed_batch: bool
    fixed_seq_len: Optional[int]  # None => 序列维动态
    description: str


def _profile_from_arg(name: str) -> ExportProfile:
    name = name.strip().lower()
    if name in ("default", "server"):
        return ExportProfile(
            fixed_batch=False,
            fixed_seq_len=None,
            description="默认：动态 batch + 动态序列",
        )
    if name in ("mobile_dynamic_seq", "android_dynamic_seq"):
        return ExportProfile(
            fixed_batch=True,
            fixed_seq_len=None,
            description="Android 推荐：batch=1 固定，序列长度动态",
        )
    if name in ("mobile_fixed_all", "android_fixed_all"):
        return ExportProfile(
            fixed_batch=True,
            fixed_seq_len=None,
            description="batch=1 + 固定序列长度（需配合 --fixed-seq-len）",
        )
    raise ValueError(
        f"未知 --profile {name!r}；可选: default | mobile_dynamic_seq | mobile_fixed_all"
    )


def _dynamic_axes(profile: ExportProfile, fixed_seq_len: Optional[int]) -> Optional[dict]:
    """
    None: 让 NeMo 根据 NeuralType 自动推断（可能含 batch 动态）。
    dict: 传给 torch.onnx.export / model.export。
    全静态: 使用空 dict（PyTorch: 不声明 dynamic_axes => 本例用 dict 覆盖 NeMo 默认）。
    """
    fs = fixed_seq_len if fixed_seq_len is not None else profile.fixed_seq_len
    if profile.fixed_batch and fs is not None:
        return {}  # 全静态
    if profile.fixed_batch and fs is None:
        return {
            "input_ids": {1: "source_length"},
            "log_probs": {1: "encoder_time"},
        }
    # 动态 batch
    return {
        "input_ids": {0: "batch", 1: "source_length"},
        "input_len": {0: "batch"},
        "log_probs": {0: "batch", 1: "encoder_time"},
        "encoded_len": {0: "batch"},
    }


def _build_dummy(
    model: CTCG2PModel,
    batch: int,
    seq_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    vocab = int(model.tokenizer_grapheme.vocab_size)
    if vocab <= 1:
        raise RuntimeError(f"tokenizer_grapheme.vocab_size 异常: {vocab}")
    input_ids = torch.randint(1, vocab, (batch, seq_len), device=device, dtype=torch.long)
    input_len = torch.full((batch,), seq_len, device=device, dtype=torch.long)
    return input_ids, input_len


def export_fallback_torch_onnx(
    model: CTCG2PModel,
    out_path: Path,
    input_example: Tuple[torch.Tensor, torch.Tensor],
    dynamic_axes: Optional[dict],
    opset: int,
) -> None:
    if replace_for_export is None:
        raise RuntimeError(f"无法导入 replace_for_export: {_replace_import_err}")
    out_path = Path(out_path)
    for p in model.parameters():
        p.requires_grad = False

    replace_for_export(model)
    saved_forward = model.forward
    model.forward = types.MethodType(CTCG2PModel.forward_for_export, model)
    try:
        model._prepare_for_export(output=str(out_path), input_example=input_example)
        torch.onnx.export(
            model,
            input_example,
            str(out_path),
            input_names=["input_ids", "input_len"],
            output_names=["log_probs", "encoded_len"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    finally:
        model.forward = saved_forward
        if hasattr(model, "_export_teardown"):
            model._export_teardown()


def export_onnx_with_fallback(
    model: CTCG2PModel,
    out_path: Path,
    input_example: Tuple[torch.Tensor, torch.Tensor],
    dynamic_axes: Optional[dict],
    opset: int,
    check_trace: bool,
) -> str:
    """返回 'nemo_export' 或 'torch_onnx_fallback'。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _LOG.info("尝试 NeMo model.export() …")
        model.export(
            str(out_path),
            input_example=input_example,
            verbose=False,
            onnx_opset_version=opset,
            dynamic_axes=dynamic_axes,
            check_trace=check_trace,
            check_tolerance=0.08,
            do_constant_folding=True,
        )
        _LOG.info("NeMo model.export() 成功")
        return "nemo_export"
    except Exception as e:
        _LOG.warning("model.export 失败，进入 torch.onnx.export fallback: %s", e)
        if out_path.is_file():
            try:
                out_path.unlink()
            except OSError:
                pass
        export_fallback_torch_onnx(model, out_path, input_example, dynamic_axes, opset)
        _LOG.info("torch.onnx.export fallback 成功")
        return "torch_onnx_fallback"


def onnx_checker_validate(path: Path) -> None:
    m = onnx.load(str(path))
    onnx.checker.check_model(m, full_check=True)
    _LOG.info("onnx.checker.check_model(full_check=True) 通过")


def onnxruntime_cpu_smoke(path: Path, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    import onnxruntime

    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = onnxruntime.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    _LOG.info("ORT 输入顺序: %s", names)
    ordered = {k: feeds[k] for k in names if k in feeds}
    outs = sess.run(None, ordered)
    _LOG.info("ORT 输出个数=%d shapes=%s", len(outs), [getattr(o, "shape", None) for o in outs])
    return outs


def write_sidecar_metadata(
    path: Path,
    model: CTCG2PModel,
    *,
    blank_id: int,
    blank_source: str,
    num_classes: int,
    num_classes_source: str,
    max_source_len: int,
    fixed_batch: bool,
    fixed_seq_len: Optional[int],
    opset: int,
    export_backend: str,
    log_probs_layout_note: str,
) -> Path:
    do_lower = bool(model.cfg.tokenizer_grapheme.get("do_lower", True))
    add_punct = bool(model.cfg.tokenizer_grapheme.get("add_punctuation", False))
    try:
        import nemo

        nemo_ver = getattr(nemo, "__version__", "unknown")
    except Exception:
        nemo_ver = "unknown"
    meta: dict[str, Any] = {
        "phoneme_labels": list(model.vocabulary),
        "phoneme_label_count": len(model.vocabulary),
        "blank_index": blank_id,
        "blank_index_how_determined": blank_source,
        "blank_handling_note": (
            "CTC blank 与真实 token 列对齐：logits 最后一维为 num_classes_with_blank；"
            "NeMo CTCBPEDecoding 使用 blank_id = tokenizer.tokenizer.vocab_size（通常等于 len(phoneme_labels)）。"
            "解码前对每帧 argmax，再按 CTC 规则去 blank、合并连续重复 token。"
        ),
        "max_source_len": max_source_len,
        "tokenizer_grapheme_do_lower": do_lower,
        "tokenizer_grapheme_add_punctuation": add_punct,
        "grapheme_vocab": _grapheme_vocab_ordered(model),
        "grapheme_unk_id": _grapheme_unk_id_optional(model),
        "tokenizer_special_tokens": _ipa_tokenizer_special_strings(model),
        "ctc_supported_punctuation": _ctc_supported_punctuation_sorted(model),
        "model_mode": str(getattr(model, "mode", "")),
        "model_cfg_model_name": str(model.cfg.get("model_name", "")),
        "fixed_batch": fixed_batch,
        "fixed_seq_len": fixed_seq_len,
        "opset_version": opset,
        "export_backend": export_backend,
        "num_classes_with_blank": num_classes,
        "num_classes_with_blank_how_determined": num_classes_source,
        "log_probs_layout_note": log_probs_layout_note,
        "onnx_expected_input_names": ["input_ids", "input_len"],
        "onnx_expected_output_names": ["log_probs", "encoded_len"],
        "pytorch_versions": {"torch": torch.__version__, "nemo_toolkit": nemo_ver},
    }
    side = path.with_suffix(".g2p_export_meta.json")
    side.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _LOG.info("metadata 已写入 %s", side)
    return side


# ----- 4. PyTorch vs ONNX 自检（可独立 import） -----


def compare_torch_onnx_runtime(
    model: CTCG2PModel,
    onnx_path: str | Path,
    input_ids: np.ndarray,
    input_len: np.ndarray,
    *,
    device: str = "cpu",
    rtol: float = 1e-3,
    atol: float = 1e-4,
    raise_on_encoded_len_mismatch: bool = True,
) -> None:
    """
    使用真实 input_ids / input_len 对比 forward_for_export 与 ORT。
    input_ids: int64 [B,T]；input_len: int64 [B]。
    """
    import onnxruntime

    onnx_path = Path(onnx_path)
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    ids = torch.as_tensor(input_ids, dtype=torch.long, device=dev)
    lens = torch.as_tensor(input_len, dtype=torch.long, device=dev)

    with torch.inference_mode():
        lp_t, el_t = model.forward_for_export(ids, lens)

    lp_t_np = lp_t.detach().cpu().numpy()
    el_t_np = el_t.detach().cpu().numpy()

    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = onnxruntime.InferenceSession(str(onnx_path), sess_options=so, providers=["CPUExecutionProvider"])
    inames = [i.name for i in sess.get_inputs()]
    onames = [o.name for o in sess.get_outputs()]
    feeds = {}
    if "input_ids" in inames:
        feeds["input_ids"] = np.asarray(input_ids, dtype=np.int64)
    if "input_len" in inames:
        feeds["input_len"] = np.asarray(input_len, dtype=np.int64)
    outs = sess.run(None, feeds)
    if len(outs) < 2:
        raise RuntimeError(f"ORT 输出个数异常: {len(outs)}")
    by_name = dict(zip(onames, outs))
    lp_o = by_name.get("log_probs", outs[0])
    el_o = by_name.get("encoded_len", outs[1])

    if lp_t_np.shape != lp_o.shape:
        _LOG.error("log_probs shape 不一致 torch=%s onnx=%s", lp_t_np.shape, lp_o.shape)
    if el_t_np.shape != el_o.shape:
        _LOG.error("encoded_len shape 不一致 torch=%s onnx=%s", el_t_np.shape, el_o.shape)

    diff_lp = np.max(np.abs(lp_t_np - lp_o))
    close_lp = np.allclose(lp_t_np, lp_o, rtol=rtol, atol=atol)
    _LOG.info("log_probs max_abs_diff=%.6g allclose(rtol=%g,atol=%g)=%s", diff_lp, rtol, atol, close_lp)

    el_bad = True
    if el_t_np.shape == el_o.shape:
        el_t_cmp = np.asarray(el_t_np, dtype=np.int64)
        el_o_cmp = np.asarray(el_o, dtype=np.int64)
        el_bad = not np.array_equal(el_t_cmp, el_o_cmp)
        diff_el = int(np.max(np.abs(el_t_cmp - el_o_cmp)))
    else:
        diff_el = None

    if el_bad:
        _LOG.error(
            "encoded_len 与 PyTorch 不一致（须优先排查）: torch=%s onnx=%s max_abs=%s",
            el_t_np,
            el_o,
            diff_el,
        )
        if raise_on_encoded_len_mismatch:
            raise RuntimeError("encoded_len mismatch between PyTorch and ONNX Runtime")
    else:
        _LOG.info("encoded_len 完全一致")


# ----- 3. 最小 CTC greedy（与 NeMo fold_consecutive=True 主路径一致） -----


def ctc_collapse_indices_nemo_style(prediction: Sequence[int], blank_id: int, length: Optional[int] = None) -> list[int]:
    """
    对齐 NeMo AbstractCTCDecoding.decode_hypothesis 在 fold_consecutive=True 时的核心折叠逻辑
    （见 nemo/.../ctc_decoding.py 中 for p in prediction 循环；每步末尾 previous = p）。
    prediction: 每帧 argmax 得到的类别 id。
    """
    if length is not None:
        prediction = list(prediction)[: int(length)]
    else:
        prediction = list(prediction)

    decoded_prediction: list[int] = []
    previous = blank_id
    for p in prediction:
        p = int(p)
        if (p != previous or previous == blank_id) and p != blank_id:
            decoded_prediction.append(p)
        previous = p
    return decoded_prediction


def ids_to_ipa_string(ids: list[int], labels: list[str]) -> str:
    """仅 id→字符映射并拼接；完整 NeMo 路径另需 tokens_to_text 过滤 special + strip 标点前空格（见 g2p_nemo_client）。"""
    chars = []
    for i in ids:
        if 0 <= i < len(labels):
            chars.append(labels[i])
        else:
            chars.append(f"<unk_id_{i}>")
    return "".join(chars)


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--nemo", default=None, help="NeMo .nemo 路径（与 --ckpt 二选一）")
    src.add_argument("--ckpt", default=None, help="Lightning .ckpt 路径（与 --nemo 二选一）")
    ap.add_argument("--out", required=True, help="输出 .onnx")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument(
        "--profile",
        default="mobile_dynamic_seq",
        help="default | mobile_dynamic_seq | mobile_fixed_all",
    )
    ap.add_argument(
        "--fixed-seq-len",
        type=int,
        default=None,
        help="仅 mobile_fixed_all：固定导出序列长度（应与 max_source_len 或端上 padding 一致）",
    )
    ap.add_argument("--dummy-seq-len", type=int, default=None, help="trace 用序列长；默认 min(32, max_source_len)")
    ap.add_argument("--check-trace", action="store_true", help="NeMo export 数值 trace 检查（更慢）")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    _setup_logging(args.verbose)

    profile = _profile_from_arg(args.profile)
    if profile.fixed_seq_len is None and args.profile.strip().lower() in ("mobile_fixed_all", "android_fixed_all"):
        if args.fixed_seq_len is None:
            _LOG.error("profile=mobile_fixed_all 时必须提供 --fixed-seq-len")
            return 2
        profile = ExportProfile(
            fixed_batch=True,
            fixed_seq_len=int(args.fixed_seq_len),
            description="fixed batch=1 + fixed seq",
        )

    out = Path(args.out).expanduser().resolve()
    device = _device(args.device)

    if args.ckpt:
        ckpt = Path(args.ckpt).expanduser().resolve()
        if not ckpt.is_file():
            _LOG.error("找不到 .ckpt: %s", ckpt)
            return 2
        _LOG.info("load_from_checkpoint: %s -> %s", ckpt, device)
        model = CTCG2PModel.load_from_checkpoint(str(ckpt), map_location=device)
    else:
        nemo = Path(args.nemo).expanduser().resolve()
        if not nemo.is_file():
            _LOG.error("找不到 .nemo: %s", nemo)
            return 2
        _LOG.info("restore_from: %s -> %s", nemo, device)
        model = CTCG2PModel.restore_from(str(nemo), map_location=device)
    model.eval()
    model.to(device)

    mode = str(getattr(model, "mode", "")).lower()
    if mode != "conformer_bpe":
        _LOG.error("仅支持 conformer_bpe，当前 mode=%r", mode)
        return 3

    max_src = int(getattr(model, "max_source_len", model.cfg.get("max_source_len", 128)))
    fixed_seq = profile.fixed_seq_len if profile.fixed_seq_len is not None else args.fixed_seq_len
    dummy_t = args.dummy_seq_len
    if dummy_t is None:
        dummy_t = fixed_seq if fixed_seq is not None else min(32, max_src)

    batch = 1 if profile.fixed_batch else 1  # 上线默认 1；若需多 batch 用 profile=default 并改下行
    if not profile.fixed_batch:
        batch = 1  # 仍用 1 作为 dummy；dynamic_axes 会放开 batch 维

    input_ids, input_len = _build_dummy(model, batch=batch, seq_len=int(dummy_t), device=device)
    input_example = (input_ids, input_len)

    dyn = _dynamic_axes(profile, fixed_seq)
    _LOG.info(
        "导出配置: profile=%s fixed_batch=%s fixed_seq_len=%s dynamic_axes=%s dummy_shape=%s",
        args.profile,
        profile.fixed_batch,
        fixed_seq,
        dyn,
        tuple(input_ids.shape),
    )

    blank_id, blank_src = _resolve_blank_id(model)
    ncls, ncls_src = _resolve_num_classes(model)
    layout_note = _log_probs_layout(model, device)
    _LOG.info("blank_id=%d (%s) num_classes_with_blank=%s (%s)", blank_id, blank_src, ncls, ncls_src)
    _LOG.info("%s", layout_note)

    backend = export_onnx_with_fallback(
        model,
        out,
        input_example,
        dyn,
        opset=int(args.opset),
        check_trace=bool(args.check_trace),
    )

    onnx_checker_validate(out)

    feeds = {
        "input_ids": input_ids.detach().cpu().numpy().astype(np.int64),
        "input_len": input_len.detach().cpu().numpy().astype(np.int64),
    }
    onnxruntime_cpu_smoke(out, feeds)

    write_sidecar_metadata(
        out,
        model,
        blank_id=blank_id,
        blank_source=blank_src,
        num_classes=ncls,
        num_classes_source=ncls_src,
        max_source_len=max_src,
        fixed_batch=profile.fixed_batch,
        fixed_seq_len=fixed_seq,
        opset=int(args.opset),
        export_backend=backend,
        log_probs_layout_note=layout_note,
    )

    _LOG.info("完成: %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
