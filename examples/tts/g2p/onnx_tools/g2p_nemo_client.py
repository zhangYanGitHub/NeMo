# -*- coding: utf-8 -*-
"""
NeMo 训练得到的 Conformer-CTC G2P：纯 ONNX Runtime 推理（不依赖 NeMo / .nemo / torch 前向）。

一、设计说明（与 NeMo CTCG2PModel._infer 对齐要点）
----------------------------------------------------------------
导出阶段须固化到 ``*.g2p_export_meta.json``（由 export_nemo_g2p_ctc_onnx.py 生成）：

- ``phoneme_labels``：CTC 非 blank 类顺序，与 decoder 列对齐；``blank_index`` 为 CTC blank。
- ``grapheme_vocab``：按 id 0..V-1 排列的 grapheme 符号串（与 NeMo CharTokenizer 一致），供推理端手写 ``text_to_ids``。
- ``grapheme_unk_id``：未知 grapheme 映射（可为 null）；与 CharTokenizer 遇 unk 行为一致。
- ``max_source_len`` / ``tokenizer_grapheme_do_lower`` / ``tokenizer_grapheme_add_punctuation``：与 CTCG2PBPEDataset 推断前处理一致。
- ``tokenizer_special_tokens``：IPA 侧 ``IPASymbolTokenizer.all_special_tokens``（``ipa_symbol_tokenizer`` 模块），供推理端在拼接前剔除（等价 ``tokens_to_text``）。
- ``ctc_supported_punctuation``：``model.decoding.supported_punctuation`` 的稳定排序列表，供推理端做「标点前空格」剔除（等价
  ``decode_tokens_to_str_with_strip_punctuation``）。

推理端自实现：

- Grapheme：``do_lower`` 后整串 ``text_to_ids``；**仅当** ``len(ids) > max_source_len`` 时按 NeMo 推理 manifest 规则将 **字符串** 截断到 ``text[:max_source_len]`` 再编码；未知字符用 ``grapheme_unk_id`` 或丢弃（与无 unk 时 NeMo 一致）。
- ONNX：``input_ids`` int64 ``[B,T]``、``input_len`` int64 ``[B]`` → ``log_probs``、``encoded_len``。
- CTC：对 ``log_probs`` 最后一维 ``argmax``；按 ``encoded_len[b]`` 截断时间维；``fold_consecutive=True`` 折叠规则与 NeMo
  ``AbstractCTCDecoding.decode_hypothesis`` 主循环一致（见本模块 ``ctc_collapse_indices_nemo_style``）。
- IPA 串：``ids_to_tokens``（查 ``phoneme_labels``）→ 去 special → 拼接 → 标点前去空格；**不对输出再做 NFC**（与 NeMo ``hyp.text`` 一致）。

二、依赖与默认路径
----------------------------------------------------------------
依赖：``numpy``、``onnxruntime``（无 NeMo、无 torch）。

**纯 ONNX 推理**仅需同一次导出得到的 ``*.onnx`` + ``*.g2p_export_meta.json``（grapheme / blank / 解码字段均在 meta 内）。

默认模型路径（本仓库导出物放在 ``tools/g2p/model/``；可用 ``get_nemo_g2p_client(onnx=..., meta=...)`` 覆盖）::

    model/conformer_ctc_en_us_260w.onnx
    model/conformer_ctc_en_us_260w.g2p_export_meta.json

另：本模块中的 ``DEFAULT_VOCAB`` / ``nemo_g2p_config.json`` 仅给 ``reproduce_nemo_val_per.py`` 等 **NeMo 路径复现** 或评测脚本复用，**不参与** ``G2pNemoCtcClient`` 的加载与前向。

三、与 NeMo 对齐验证建议
----------------------------------------------------------------
同一批 ``text_graphemes``：

1. 用 ``CTCG2PModel.restore_from`` + ``forward`` + ``decoding.ctc_decoder_predictions_tensor`` 得到参考 IPA。
2. 用本客户端 ``phonemize_ipa_batch`` 得到 ONNX IPA；应逐句一致。
3. 若不一致：先 ``compare_torch_onnx_runtime``（export 脚本）核对 ``log_probs``/``encoded_len``；再核对 ``grapheme_vocab`` 是否最新导出、``do_lower`` 是否一致。
4. 若仍不一致：核对 sidecar 是否含 ``tokenizer_special_tokens`` / ``ctc_supported_punctuation``（需用最新 ``export_nemo_g2p_ctc_onnx.py`` 重新导出）。

用法::

    from g2p_nemo_client import get_nemo_g2p_client
    client = get_nemo_g2p_client()
    print(client.phonemize_ipa_one("hello world"))
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import AbstractSet, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnxruntime

# 单一真源：训练/推理共用的文本前端（与 NeMo examples/dataset/text_normalize.py 逐字对齐）。
# 客户端不再自实现 normalize_graphemes（旧的导航英文 TN，与训练不一致），改用此模块，
# 从而保证 train==serve（数字/符号 TN 按 locale、标点剥离、按停顿切段完全一致）。
from g2p_text_frontend import (
    is_arabic_g2p_lang,
    normalize_for_g2p,
    number_locale_for,
    phonemize_arabic_with_letter_lexicon,
    prepare_arabic_grapheme_text,
    split_arabic_latin_segments,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):  # type: ignore[misc]
        return iterable if iterable is not None else range(kwargs.get("total", 0))

# =============================================================================
# ⚠️ 遗留（LEGACY，客户端已不再调用）：下面的 normalize_graphemes 及其数字/缩写/路名
# 展开是早期 navigation_g2p 英文导航 TN，与**当前**训练前端（NeMo
# examples/dataset/text_normalize.py + preprocess_ipa_childes_split.py）**不一致**：
#   * 它做 St.->Street / I-5->Interstate five / 3rd->third / 英文数字词、保留标点、不切段；
#   * 训练前端按 locale 展开数字（德语 fünfhundert、英语 five hundred）+ 符号（%->Prozent、
#     +->plus）+ 剥标点 + 按停顿切段。二者是两套范式，混用会导致 train != serve。
# 现客户端统一改用 g2p_text_frontend.normalize_for_g2p（见 _phonemize_chunk）。
# 此处仅为兼容个别仍 import normalize_graphemes 的旧脚本而保留，**请勿在服务路径使用**。
# =============================================================================

# espeak/文本中偶发的零宽与不可见字符
_INVISIBLE_CHARS_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u2060\u00ad]")

# Unicode 标点替换（与训练 _PUNCT_REPLACEMENTS 一致）
_PUNCT_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
}

_HYPHEN_SPACE_RE = re.compile(r"\s*-\s*")
_MULTI_SPACE_RE = re.compile(r"\s+")

# --- 英文基数 / 序数（口语形式），与训练 normalizer.py 完全一致 ---
_UNDER_TWENTY = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
    "seventeen", "eighteen", "nineteen",
)
_TENS_WORDS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_ORDINAL_UNDER_TWENTY = (
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth",
    "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth", "fifteenth", "sixteenth",
    "seventeenth", "eighteenth", "nineteenth",
)
_TENS_ORDINAL = {
    20: "twentieth", 30: "thirtieth", 40: "fortieth", 50: "fiftieth",
    60: "sixtieth", 70: "seventieth", 80: "eightieth", 90: "ninetieth",
}


def _under_100_cardinal(n: int) -> str:
    if n < 20:
        return _UNDER_TWENTY[n]
    if n < 100:
        ten, one = divmod(n, 10)
        if one == 0:
            return _TENS_WORDS[ten]
        return f"{_TENS_WORDS[ten]} {_UNDER_TWENTY[one]}"
    raise ValueError(n)


def _under_1000_cardinal(n: int) -> str:
    if n < 100:
        return _under_100_cardinal(n)
    hundreds, rem = divmod(n, 100)
    head = f"{_UNDER_TWENTY[hundreds]} hundred"
    if rem == 0:
        return head
    return f"{head} {_under_100_cardinal(rem)}"


def int_to_cardinal_words(n: int) -> str:
    """0 <= n < 1_000_000 -> 口语英文基数。"""
    if n < 0:
        return "minus " + int_to_cardinal_words(-n)
    if n < 1000:
        return _under_1000_cardinal(n)
    if n < 1_000_000:
        thousands, rem = divmod(n, 1000)
        left = _under_1000_cardinal(thousands) + " thousand"
        if rem == 0:
            return left
        return f"{left} {_under_1000_cardinal(rem)}"
    return str(n)


def _ordinal_under_100(n: int) -> str:
    if n < 1:
        return int_to_cardinal_words(n)
    if n < 20:
        return _ORDINAL_UNDER_TWENTY[n - 1]
    if n < 100:
        if n % 10 == 0:
            return _TENS_ORDINAL[n]
        return f"{_TENS_WORDS[n // 10]} {_ordinal_under_100(n % 10)}"
    raise ValueError(n)


def int_to_ordinal_words(n: int) -> str:
    """1 <= n < 1000 -> 口语英文序数（例如 12 -> twelfth）。"""
    if n < 1:
        return int_to_cardinal_words(n)
    if n < 100:
        return _ordinal_under_100(n)
    hundreds, rem = divmod(n, 100)
    head_word = _UNDER_TWENTY[hundreds]
    if rem == 0:
        return f"{head_word} hundredth"
    if rem < 20:
        return f"{head_word} hundred {_ORDINAL_UNDER_TWENTY[rem - 1]}"
    return f"{head_word} hundred {_ordinal_under_100(rem)}"


def _decimal_to_words(int_part: str, frac_part: str) -> str:
    left = int_to_cardinal_words(int(int_part))
    digits = " ".join(_UNDER_TWENTY[int(c)] for c in frac_part if c.isdigit())
    return f"{left} point {digits}" if digits else left


def spoken_numbers_in_text(text: str) -> str:
    """阿拉伯数字 -> 英文口语，与训练 normalizer.spoken_numbers_in_text 一致。"""
    if not text:
        return text

    def repl_ordinal(m: "re.Match[str]") -> str:
        return int_to_ordinal_words(int(m.group(1)))

    result = re.sub(r"\b(\d{1,3})(st|nd|rd|th)\b", repl_ordinal, text, flags=re.I)

    def repl_exit(m: "re.Match[str]") -> str:
        n = int(m.group(1))
        suf = (m.group(2) or "").upper()
        num_w = int_to_cardinal_words(n)
        if not suf:
            return f"Exit {num_w}"
        return f"Exit {num_w} {suf}"

    result = re.sub(r"\bExit\s+(\d{1,3})([A-Za-z])?\b", repl_exit, result, flags=re.I)

    _space_routes = (
        (re.compile(r"\bI\s+(\d{1,4})\b", re.I), "Interstate {}"),
        (re.compile(r"\bUS\s+(\d{1,4})\b", re.I), "U S {}"),
        (re.compile(r"\bSR\s+(\d{1,4})\b", re.I), "State Route {}"),
        (re.compile(r"\bFM\s+(\d{1,4})\b", re.I), "Farm to Market Road {}"),
        (re.compile(r"\bTX\s+(\d{1,4})\b", re.I), "Texas Highway {}"),
    )
    for pat, fmt in _space_routes:
        def _mk_sp(fmt_inner: str):
            def _inner(m: "re.Match[str]") -> str:
                return fmt_inner.format(int_to_cardinal_words(int(m.group(1))))

            return _inner

        result = pat.sub(_mk_sp(fmt), result)

    _route_hyphen = (
        (re.compile(r"\bI-(\d{1,4})\b", re.I), "Interstate {}"),
        (re.compile(r"\bUS-(\d{1,4})\b", re.I), "U S {}"),
        (re.compile(r"\bSR-(\d{1,4})\b", re.I), "State Route {}"),
        (re.compile(r"\bFM-(\d{1,4})\b", re.I), "Farm to Market Road {}"),
        (re.compile(r"\bTX-(\d{1,4})\b", re.I), "Texas Highway {}"),
        (re.compile(r"\bHwy-(\d{1,4})\b", re.I), "Highway {}"),
    )
    for pat, fmt in _route_hyphen:
        def _mk(fmt_inner: str):
            def _inner(m: "re.Match[str]") -> str:
                return fmt_inner.format(int_to_cardinal_words(int(m.group(1))))

            return _inner

        result = pat.sub(_mk(fmt), result)

    result = re.sub(
        r"\bRoute\s+(\d{1,4})\b",
        lambda m: "Route " + int_to_cardinal_words(int(m.group(1))),
        result,
        flags=re.I,
    )
    result = re.sub(
        r"\bHighway\s+(\d{1,4})\b",
        lambda m: "Highway " + int_to_cardinal_words(int(m.group(1))),
        result,
        flags=re.I,
    )
    result = re.sub(
        r"\bHwy\s+(\d{1,4})\b",
        lambda m: "Highway " + int_to_cardinal_words(int(m.group(1))),
        result,
        flags=re.I,
    )

    def repl_decimal(m: "re.Match[str]") -> str:
        return _decimal_to_words(m.group(1), m.group(2))

    result = re.sub(r"\b(\d+)\.(\d+)\b", repl_decimal, result)

    def repl_int(m: "re.Match[str]") -> str:
        return int_to_cardinal_words(int(m.group(0)))

    result = re.sub(r"\b\d{1,6}\b", repl_int, result)
    return _MULTI_SPACE_RE.sub(" ", result).strip()


# 静态缩写表（带点），与训练 slot_values.ABBREVIATIONS 一致
_ABBREVIATIONS = {
    "St.": "Street",
    "Ave.": "Avenue",
    "Blvd.": "Boulevard",
    "Rd.": "Road",
    "Dr.": "Drive",
    "Mt.": "Mount",
    "Ft.": "Fort",
    "Ln.": "Lane",
    "Ct.": "Court",
    "Pl.": "Place",
    "Pkwy.": "Parkway",
    "Hwy.": "Highway",
    "Cir.": "Circle",
    "Ter.": "Terrace",
    "Expy.": "Expressway",
}

# 默认角色的多义缩写展开表（取自训练 MultiSenseAbbreviationStrategy.ROLE_EXPANSIONS["default"]）。
# 推理端无 slot 角色、且需确定性，固定取每个候选列表的首项（prefer_index=0）。
_DEFAULT_ABBREV_EXPANSIONS = {
    "St.": "Street",
    "Dr.": "Drive",
    "Mt.": "Mount",
    "Ft.": "Fort",
    "Ave.": "Avenue",
    "Blvd.": "Boulevard",
    "Rd.": "Road",
    "Ln.": "Lane",
    "Ct.": "Court",
    "Pl.": "Place",
    "Pkwy.": "Parkway",
    "Hwy.": "Highway",
    "Cir.": "Circle",
    "Ter.": "Terrace",
    "Expy.": "Expressway",
}


def _expand_default_abbrev(text: str) -> str:
    """等价 MultiSenseAbbreviationStrategy.expand(role='default') 的确定性版本。"""
    result = text
    for abbrev in sorted(_DEFAULT_ABBREV_EXPANSIONS.keys(), key=len, reverse=True):
        if abbrev in result:
            result = result.replace(abbrev, _DEFAULT_ABBREV_EXPANSIONS[abbrev])
    return result


def _apply_static_abbrev_map(text: str) -> str:
    result = text
    for abbrev in sorted(_ABBREVIATIONS.keys(), key=len, reverse=True):
        if abbrev in result:
            result = result.replace(abbrev, _ABBREVIATIONS[abbrev])
    return result


# 无点缩写（route/road builder 常见），与训练 _BARE_ABBREV_REPLACEMENTS 一致
_BARE_ABBREV_REPLACEMENTS: Tuple[Tuple["re.Pattern[str]", str], ...] = (
    (re.compile(r"\bHwy\b(?!\.)", re.I), "Highway"),
    (re.compile(r"\bAve\b(?!\.)", re.I), "Avenue"),
    (re.compile(r"\bBlvd\b(?!\.)", re.I), "Boulevard"),
    (re.compile(r"\bRd\b(?!\.)", re.I), "Road"),
    (re.compile(r"\bLn\b(?!\.)", re.I), "Lane"),
    (re.compile(r"\bCt\b(?!\.)", re.I), "Court"),
    (re.compile(r"\bPl\b(?!\.)", re.I), "Place"),
    (re.compile(r"\bPkwy\b(?!\.)", re.I), "Parkway"),
    (re.compile(r"\bCir\b(?!\.)", re.I), "Circle"),
    (re.compile(r"\bTer\b(?!\.)", re.I), "Terrace"),
    (re.compile(r"\bExpy\b(?!\.)", re.I), "Expressway"),
)


def _expand_bare_abbreviations(text: str) -> str:
    result = text
    for pat, repl in _BARE_ABBREV_REPLACEMENTS:
        result = pat.sub(repl, result)
    return result


def normalize_graphemes(
    text: str,
    *,
    expand_abbrev: bool = True,
    replace_punctuation: bool = True,
    normalize_hyphen_spacing: bool = True,
    spoken_numbers: bool = True,
) -> str:
    """与训练 navigation_g2p/normalizer.normalize_graphemes 等价（role 固定 default，无 RNG）。"""
    if text is None:
        return ""
    result = unicodedata.normalize("NFC", text)
    result = result.strip()
    if replace_punctuation:
        for src, dst in _PUNCT_REPLACEMENTS.items():
            result = result.replace(src, dst)
    result = _INVISIBLE_CHARS_RE.sub("", result)
    if normalize_hyphen_spacing:
        result = _HYPHEN_SPACE_RE.sub("-", result)
    result = _MULTI_SPACE_RE.sub(" ", result)
    if spoken_numbers:
        result = spoken_numbers_in_text(result)
        result = _MULTI_SPACE_RE.sub(" ", result).strip()
    if expand_abbrev:
        result = _expand_default_abbrev(result)
        result = _apply_static_abbrev_map(result)
        result = _expand_bare_abbreviations(result)
        result = _MULTI_SPACE_RE.sub(" ", result).strip()
    return result


_PKG_DIR = Path(__file__).resolve().parent
_MODEL_DIR = _PKG_DIR / "model"
_MODELS_DIR = _PKG_DIR / "models"

DEFAULT_ONNX = _MODEL_DIR / "conformer_ctc_en_us_260w.onnx"
DEFAULT_META = _MODEL_DIR / "conformer_ctc_en_us_260w.g2p_export_meta.json"
# 以下两项不参与 ONNX 客户端；供 reproduce_nemo_val_per 等脚本可选使用
DEFAULT_VOCAB = _MODELS_DIR / "vocab.txt"
DEFAULT_CONFIG = _MODELS_DIR / "nemo_g2p_config.json"

DEFAULT_G2P_BATCH_SIZE = 1
# 服务路径支持的 locale。前端 TN 按 number_locale 分派（en_US->en 数字/符号，de_DE->de）。
# 注意：de_DE 需配套导出的德语 ONNX+meta（当前 model/ 下仍只有英语 checkpoint）。
_SUPPORTED_LOCALES = frozenset({"en_US", "de_DE", "fr_FR"})

_CLIENT: Optional["G2pNemoCtcClient"] = None
_CLIENT_CACHE_KEY: Optional[Tuple[str, str]] = None


def resolve_nemo_vocab(path: Optional[Path] = None) -> Path:
    p = (path or DEFAULT_VOCAB).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"IPA vocab 不存在: {p}")
    return p


def resolve_g2p_onnx(path: Optional[Path] = None) -> Path:
    p = (path or DEFAULT_ONNX).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"ONNX 不存在: {p}。请先运行 export_nemo_g2p_ctc_onnx.py 导出，并用 --out 指向该路径。"
        )
    return p


def resolve_g2p_meta(path: Optional[Path] = None) -> Path:
    p = (path or DEFAULT_META).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(
            f"metadata 不存在: {p}。请使用与 ONNX 同一次导出生成的 *.g2p_export_meta.json（含 grapheme_vocab）。"
        )
    return p


def _load_config(path: Path) -> dict[str, Any]:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "max_source_len": 128,
        "do_lower": True,
        "add_punctuation": False,
    }


def locale_nemo_g2p_supported(locale: str) -> bool:
    return locale in _SUPPORTED_LOCALES


def load_g2p_export_meta(path: Path) -> dict[str, Any]:
    data = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    for k in ("phoneme_labels", "blank_index", "max_source_len", "grapheme_vocab"):
        if k not in data:
            raise KeyError(f"metadata 缺少 {k!r}，请用最新 export_nemo_g2p_ctc_onnx.py 重新导出: {path}")
    return data


def ctc_collapse_indices_nemo_style(
    prediction: Sequence[int], blank_id: int, length: Optional[int] = None
) -> list[int]:
    """
    与 NeMo AbstractCTCDecoding.decode_hypothesis 在 fold_consecutive=True 时的折叠一致
    （每步末尾 previous = p）。
    """
    if length is not None:
        prediction = list(prediction)[: int(length)]
    else:
        prediction = list(prediction)
    decoded_prediction: list[int] = []
    previous = blank_id
    for t, p in enumerate(prediction):
        p = int(p)
        if (p != previous or previous == blank_id) and p != blank_id:
            decoded_prediction.append(p)
        previous = p
    return decoded_prediction

def ids_to_ipa_string(ids: list[int], labels: list[str]) -> str:
    """仅作调试：不做 special 过滤与标点前空格处理（与 NeMo 最终 ``hyp.text`` 可能不同）。"""
    chars: list[str] = []
    for i in ids:
        if 0 <= i < len(labels):
            chars.append(labels[i])
        else:
            chars.append(f"<unk_id_{i}>")
    return "".join(chars)


def _build_grapheme_char_to_id(grapheme_vocab: list[str]) -> dict[str, int]:
    m: dict[str, int] = {}
    for i, ch in enumerate(grapheme_vocab):
        if ch not in m:
            m[ch] = i
    return m


def grapheme_text_to_ids(
    text: str,
    char_to_id: dict[str, int],
    *,
    do_lower: bool,
    max_source_len: int,
    unk_id: Optional[int],
    truncate_to_max_source_len: bool = True,
) -> list[int]:
    """对齐 NeMo CharTokenizer.text_to_ids + CTCG2PBPEDataset / ONNX 前向。

    ``truncate_to_max_source_len``：
    - ``True``（默认）：与 ``CTCG2PBPEDataset`` 在 ``with_labels=False``（推理 manifest）时一致：
      先对 **整串**（lower 后）做 ``text_to_ids``；**仅当** ``len(ids) > max_source_len`` 时再把 **原始 grapheme
      字符串** 截断到 ``text[:max_source_len]`` 后重新编码（见 NeMo ``data/ctc.py`` 推理分支）。
      这与「只要 ``len(text) > max`` 就先截断再编码」不同：后者在大量字符被丢弃（无 unk）时会多截断，
      与 NeMo ``_infer`` 不一致。
    - ``False``：与 ``CTCG2PBPEDataset`` 统计 ``grapheme_tokens_len`` 时一致，**不截断**整串再
      ``text_to_ids``，用于 ``ipa_len > grapheme_len`` / ``grapheme_len > max_source_len`` 过滤。
    """
    if do_lower:
        text = text.lower()

    def _encode(s: str) -> list[int]:
        out: list[int] = []
        for ch in s:
            if ch in char_to_id:
                out.append(char_to_id[ch])
            elif unk_id is not None:
                out.append(int(unk_id))
        return out

    ids = _encode(text)
    if truncate_to_max_source_len and len(ids) > max_source_len:
        text = text[:max_source_len]
        ids = _encode(text)
    return ids


def _normalize_log_probs_layout(
    log_probs: np.ndarray,
    *,
    num_classes_with_blank: int,
) -> np.ndarray:
    """
    规范化 ONNX 输出布局到 [B, T, C]。
    训练/导出链路在不同版本里可能给出 [B,T,C] 或 [B,C,T]。
    """
    lp = np.asarray(log_probs)
    if lp.ndim != 3:
        raise ValueError(f"log_probs 维度异常，期望 3D [B,T,C]/[B,C,T]，实际 shape={tuple(lp.shape)}")

    if lp.shape[2] == num_classes_with_blank:
        return lp  # [B,T,C]
    if lp.shape[1] == num_classes_with_blank:
        return np.transpose(lp, (0, 2, 1))  # [B,C,T] -> [B,T,C]

    raise ValueError(
        "无法根据 num_classes_with_blank 判定 log_probs 类别轴，"
        f"shape={tuple(lp.shape)} num_classes_with_blank={num_classes_with_blank}"
    )


def decode_ctc_batch(
    log_probs: np.ndarray,
    encoded_len: np.ndarray,
    *,
    blank_id: int,
    phoneme_labels: list[str],
    strip_token_strings: AbstractSet[str] = frozenset(),
    supported_punctuation: Sequence[str] = (),
) -> list[str]:
    """log_probs 规范为 [B,T,C]；encoded_len [B]。

    新版模型直接把词边界（空格 token）与重音标记预测进 CTC 序列，
    因此这里只需 CTC 折叠 → 去 blank/special → 按 token 顺序拼接，
    不再做任何词边界后处理（viterbi 对齐 / DP 切词等）。
    """
    pred = np.argmax(log_probs, axis=-1).astype(np.int64)
    b = pred.shape[0]
    out: list[str] = []
    id_to_phoneme = {i: lbl for i, lbl in enumerate(phoneme_labels)}

    for bi in range(b):
        tlen = int(encoded_len[bi])
        collapsed = ctc_collapse_indices_nemo_style(pred[bi].tolist(), blank_id, tlen)

        toks: list[str] = []
        for token_id in collapsed:
            if token_id == blank_id:
                continue
            tok_str = id_to_phoneme.get(token_id, f"<unk_{token_id}>")
            if tok_str in strip_token_strings:
                continue
            toks.append(tok_str)

        text = "".join(toks)
        # 等价 NeMo decode_tokens_to_str_with_strip_punctuation：标点前不留空格。
        for p in supported_punctuation:
            text = text.replace(f" {p}", p)
        # 合并多余空格并去首尾空格（模型偶发连续空格 token）。
        text = " ".join(text.split())
        out.append(text)
    return out


def _configure_cpu_threads() -> None:
    n = os.cpu_count() or 4
    s = str(n)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(key, s)


def _onnx_inputs_max_batch_cap(sess: onnxruntime.InferenceSession) -> Optional[int]:
    """
    从 ONNX 输入 meta 推断单次推理允许的最大 batch。

    导出 profile ``mobile_dynamic_seq`` 等会把 batch 固定为 1，此时 ``input_len`` 为 ``[1]``；
    若仍用 ``batch_size=64`` 喂 ``(64,)`` 会触发 ORT ``Expected: 1``。凡首维为正整数 ``B`` 的输入，
    取各 ``B`` 的最小值作为上界；若没有任何输入带固定首维（全为 None/符号维），返回 ``None`` 表示不钳制。
    """
    caps: list[int] = []
    for inp in sess.get_inputs():
        shp = inp.shape
        if not shp:
            continue
        d0 = shp[0]
        if isinstance(d0, int) and d0 > 0:
            caps.append(d0)
    if not caps:
        return None
    return min(caps)


class G2pNemoCtcClient:
    """Conformer-CTC G2P：纯 ONNX Runtime + metadata 解码。"""

    def __init__(
        self,
        onnx_path: Path,
        meta_path: Path,
        *,
        batch_size: int = DEFAULT_G2P_BATCH_SIZE,
        ort_providers: Optional[list[str]] = None,
        show_startup_progress: bool = True,
        normalize_input: bool = True,
        expand_abbrev: bool = True,
        spoken_numbers: bool = True,
    ) -> None:
        _configure_cpu_threads()
        self._onnx_path = Path(onnx_path).expanduser().resolve()
        self._meta_path = Path(meta_path).expanduser().resolve()
        self._meta = load_g2p_export_meta(self._meta_path)
        self.batch_size = max(1, batch_size)
        # 阶段A 训练文本归一化：现由 g2p_text_frontend.normalize_for_g2p 统一负责（按 locale 展开
        # 数字/符号），与训练逐字一致。expand_abbrev/spoken_numbers 为**遗留 no-op**（旧
        # normalize_graphemes 的开关），保留仅为兼容旧调用签名，不再影响行为。
        self._normalize_input = bool(normalize_input)
        self._expand_abbrev = bool(expand_abbrev)  # deprecated no-op
        self._spoken_numbers = bool(spoken_numbers)  # deprecated no-op

        self._phoneme_labels: list[str] = list(self._meta["phoneme_labels"])
        self._blank_id = int(self._meta["blank_index"])
        self._max_source_len = int(self._meta["max_source_len"])
        self._do_lower = bool(self._meta.get("tokenizer_grapheme_do_lower", True))
        self._unk_id: Optional[int] = None
        if "grapheme_unk_id" in self._meta and self._meta["grapheme_unk_id"] is not None:
            self._unk_id = int(self._meta["grapheme_unk_id"])

        gv = list(self._meta["grapheme_vocab"])
        self._char_to_id = _build_grapheme_char_to_id(gv)
        self._decode_strip_specials: frozenset[str] = frozenset(
            str(x) for x in self._meta.get("tokenizer_special_tokens", ()) if x is not None
        )
        self._decode_punct: tuple[str, ...] = tuple(
            str(x) for x in self._meta.get("ctc_supported_punctuation", ()) if x is not None
        )

        providers = ort_providers or ["CPUExecutionProvider"]
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = onnxruntime.InferenceSession(
            str(self._onnx_path), sess_options=so, providers=providers
        )
        self._inames = [i.name for i in self._sess.get_inputs()]
        self._onames = [o.name for o in self._sess.get_outputs()]

        onnx_batch_cap = _onnx_inputs_max_batch_cap(self._sess)
        requested_batch = self.batch_size
        if onnx_batch_cap is not None:
            self.batch_size = min(self.batch_size, onnx_batch_cap)

        if show_startup_progress:
            print(f"加载 G2P ONNX: {self._onnx_path}", flush=True)
            print(f"  metadata: {self._meta_path}", flush=True)
            print(
                f"  blank_index={self._blank_id} max_source_len={self._max_source_len} "
                f"do_lower={self._do_lower} grapheme_vocab_size={len(gv)} "
                f"decode_strip_specials={len(self._decode_strip_specials)} punct={len(self._decode_punct)}",
                flush=True,
            )
            print(
                f"  normalize_input={self._normalize_input}（阶段A: g2p_text_frontend.normalize_for_g2p，"
                f"按 locale 展开数字/符号，与训练逐字一致；expand_abbrev/spoken_numbers 已弃用为 no-op）",
                flush=True,
            )
            if onnx_batch_cap is not None and requested_batch > onnx_batch_cap:
                print(
                    f"  注意: ONNX 图首维 batch 上限为 {onnx_batch_cap}（请求 batch_size={requested_batch}），"
                    f"已钳制为 {self.batch_size}。若需更大 batch，请用 export 的 ``--profile default`` 重新导出。",
                    flush=True,
                )
            print(f"  ORT providers={providers} batch={self.batch_size}", flush=True)

    def phonemize_ipa_one(self, text: str, g2p_lang_tag: str = "en-US") -> str:
        number_locale = number_locale_for(g2p_lang_tag)
        if is_arabic_g2p_lang(g2p_lang_tag):
            return phonemize_arabic_with_letter_lexicon(
                text,
                lambda seg: self._phonemize_chunk([seg], number_locale, diacritize_arabic=False)[0],
                locale=number_locale,
                diacritize=True,
            )
        return self.phonemize_ipa_batch([text], g2p_lang_tag, show_progress=False)[0]

    def phonemize_ipa_batch(
        self,
        texts: Sequence[str],
        g2p_lang_tag: str = "en-US",
        *,
        jobs: int = 0,
        locale: str = "",
        show_progress: bool = True,
    ) -> List[str]:
        del jobs
        n = len(texts)
        if n == 0:
            return []
        # 数字/符号 TN 的语言按 locale（如 de_DE）或 g2p_lang_tag（如 de-DE）分派到 'en'/'de'。
        # locale 优先（调用方明确传入），否则回退 lang tag，未知一律 'en'，绝不崩。
        number_locale = number_locale_for(locale or g2p_lang_tag)
        results: List[str] = []
        batches = range(0, n, self.batch_size)
        it = batches
        if show_progress:
            it = tqdm(
                list(it),
                desc=f"{locale or g2p_lang_tag or 'en_US'} · G2P ONNX",
                unit="batch",
                leave=False,
            )
        for start in it:
            chunk = list(texts[start : start + self.batch_size])
            results.extend(self._phonemize_chunk(chunk, number_locale))
        return results

    def _phonemize_chunk(
        self,
        texts: list[str],
        number_locale: str = "en",
        *,
        diacritize_arabic: Optional[bool] = None,
    ) -> list[str]:
        rows: list[list[int]] = []
        lens: list[int] = []
        if diacritize_arabic is None:
            diacritize_arabic = number_locale == "ar"
        for t in texts:
            # 阶段A：与训练前端逐字一致的 TN。阿语先走 local_nav_diacritizer（与 preprocess
            # --diacritize-arabic 默认开启一致），再 normalize_for_g2p。
            if self._normalize_input:
                if diacritize_arabic:
                    t = prepare_arabic_grapheme_text(t, diacritize=True)
                else:
                    t = normalize_for_g2p(t, number_locale)
            # 阶段B：do_lower + 字符级 text_to_ids + max_source_len 截断（原有逻辑）。
            ids = grapheme_text_to_ids(
                t,
                self._char_to_id,
                do_lower=self._do_lower,
                max_source_len=self._max_source_len,
                unk_id=self._unk_id,
            )
            rows.append(ids)
            lens.append(len(ids))
        if not rows:
            return ["" for _ in texts]
        t_max = max(lens)
        b = len(rows)
        input_ids = np.zeros((b, t_max), dtype=np.int64)
        for i, row in enumerate(rows):
            input_ids[i, : len(row)] = np.asarray(row, dtype=np.int64)
        input_len = np.asarray(lens, dtype=np.int64)

        feeds: dict[str, np.ndarray] = {}
        if "input_ids" in self._inames:
            feeds["input_ids"] = input_ids
        if "input_len" in self._inames:
            feeds["input_len"] = input_len

        outs = self._sess.run(None, feeds)
        by_name = dict(zip(self._onames, outs))
        log_probs = by_name.get("log_probs", outs[0])
        enc = by_name.get("encoded_len", outs[1])

        num_classes_with_blank = int(self._meta.get("num_classes_with_blank", len(self._phoneme_labels) + 1))
        log_probs_btc = _normalize_log_probs_layout(
            np.asarray(log_probs),
            num_classes_with_blank=num_classes_with_blank,
        )
        enc_1d = np.asarray(enc).reshape(-1).astype(np.int64)

        decoded = decode_ctc_batch(
            log_probs_btc,
            enc_1d,
            blank_id=self._blank_id,
            phoneme_labels=self._phoneme_labels,
            strip_token_strings=self._decode_strip_specials,
            supported_punctuation=self._decode_punct,
        )
        return decoded


def get_nemo_g2p_client(
    *,
    onnx: Optional[Path] = None,
    meta: Optional[Path] = None,
    batch_size: int = DEFAULT_G2P_BATCH_SIZE,
    ort_providers: Optional[list[str]] = None,
    show_startup_progress: bool = True,
    normalize_input: bool = True,
    expand_abbrev: bool = True,
    spoken_numbers: bool = True,
) -> G2pNemoCtcClient:
    global _CLIENT, _CLIENT_CACHE_KEY
    op = str(resolve_g2p_onnx(onnx))
    mp = str(resolve_g2p_meta(meta))
    key = (op, mp)
    if _CLIENT is not None and _CLIENT_CACHE_KEY == key:
        return _CLIENT
    _CLIENT = G2pNemoCtcClient(
        onnx_path=Path(op),
        meta_path=Path(mp),
        batch_size=batch_size,
        ort_providers=ort_providers,
        show_startup_progress=show_startup_progress,
        normalize_input=normalize_input,
        expand_abbrev=expand_abbrev,
        spoken_numbers=spoken_numbers,
    )
    _CLIENT_CACHE_KEY = key
    return _CLIENT


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="G2P Conformer-CTC 纯 ONNX 推理")
    parser.add_argument("texts", nargs="*", help="输入文本；省略则从 stdin 读行")
    parser.add_argument("--onnx", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_G2P_BATCH_SIZE)
    parser.add_argument(
        "--no-normalize-input", dest="normalize_input", action="store_false", default=True,
        help="关闭阶段A 训练文本归一化（默认开启，与训练预处理对齐）",
    )
    parser.add_argument(
        "--no-expand-abbrev", dest="expand_abbrev", action="store_false", default=True,
        help="归一化时不展开缩写（默认展开，与训练一致）",
    )
    parser.add_argument(
        "--no-spoken-numbers", dest="spoken_numbers", action="store_false", default=True,
        help="归一化时不做数字口语化（默认开启，与训练一致）",
    )
    args = parser.parse_args()

    texts = list(args.texts)
    if not texts:
        import sys

        texts = [line.rstrip("\n") for line in sys.stdin if line.strip()]

    client = get_nemo_g2p_client(
        onnx=args.onnx,
        meta=args.meta,
        batch_size=args.batch_size,
        normalize_input=args.normalize_input,
        expand_abbrev=args.expand_abbrev,
        spoken_numbers=args.spoken_numbers,
    )
    for text, ipa in zip(texts, client.phonemize_ipa_batch(texts)):
        print(f"{text}\t{ipa}")


if __name__ == "__main__":
    _main()
