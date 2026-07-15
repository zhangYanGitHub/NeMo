#!/usr/bin/env python3
"""
g2p_evaluate.py

从包含 columns: text, target, predict 的 CSV/JSON 读取（或由 ``prepare_results`` 生成的明细含 ``phoneme_error_detail``），
计算 phone-level PER（基于标准编辑距离），并生成 report.html（Jinja2 模板）。

评估目标：
  - 以 espeak-ng / piper 产出的 IPA 作为 reference
  - 以待评估 G2P 模型产出的 IPA 作为 hypothesis
  - 衡量模型输出与 espeak-ng IPA 输出的一致性

PER 计算规则（target / predict 完全相同的流水线）：
  1. normalize_ipa_raw：仅合并空白，不做任何 Unicode 归一（两侧均为 piper 原生形式，直接比对）
  2. clean_ipa：去标点、去不可见字符、去常见 HTML 残留
  3. 将 clean 后的 IPA 去空格，按 phone 字符序列比较
  4. 使用标准编辑距离统计 substitution / deletion / insertion
  5. PER = 编辑距离 / reference 长度 * 100

用法:
  python tools/espeak_goruut_ipa_compare/g2p_evaluate.py --input path/to/file.csv --output report.html
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PUNCT_RE = re.compile(r"[.,!?;:]", flags=re.U)
_SPACE_RE = re.compile(r"\s+", flags=re.U)
_INVISIBLE_RE = re.compile(r"[\u200B\u200C\u200D\uFEFF\r\n\t]|[\x00-\x1f\x7f]", flags=re.U)
_TIES = {"\u200d", "\u0361", "\u035c"}
_STRESS = {"\u02c8", "\u02cc"}
_LENGTH = "\u02d0"


def normalize_ipa_raw(ipa_str: str) -> str:
    """
    对 espeak-ng 与 G2P 输出施加完全相同的第一阶段处理：仅合并空白，
    **不做任何 Unicode 归一（NFC/NFD）**。

    target 与 predict 都是 piper/espeak 原生形式（模型训练目标即 piper 原生输出，
    未经归一），两侧本就同形，直接逐码点比对即可；任何归一（尤其 NFC 会把 piper 原生的
    分解 ç=c+组合 cedilla 改成预组合 0xE7）都会让展示串偏离 piper 真实消费形态。
    """
    if ipa_str is None:
        return ""
    return _SPACE_RE.sub(" ", str(ipa_str)).strip()


def clean_ipa(ipa_str: str) -> str:
    """清洗 IPA：去标点、去不可见字符、合并空白，并去掉常见 HTML 残留标记。"""
    if ipa_str is None:
        return ""
    s = str(ipa_str)
    s = _INVISIBLE_RE.sub(" ", s)
    s = _PUNCT_RE.sub("", s)
    s = _SPACE_RE.sub(" ", s).strip()
    s = s.replace("tdtd", "").replace("span", "")
    return s.strip()


def ipa_eval_stages(ipa_str: str) -> Tuple[str, str]:
    """统一流水线：normalize_ipa_raw → clean_ipa。target/predict 均走此路径。"""
    raw = normalize_ipa_raw(ipa_str)
    return raw, clean_ipa(raw)


def ipa_units_from_clean(cleaned: str) -> List[str]:
    """将 clean 后的 IPA 转成音素 token 序列；真实空格表示词边界，参与 PER。"""
    if not cleaned:
        return []

    units: List[str] = []
    cur = ""
    pending_stress = ""
    join_next = False

    def flush() -> None:
        nonlocal cur
        if cur:
            units.append(cur)
            cur = ""

    for ch in cleaned:
        if ch in _TIES:
            join_next = True
            continue
        if ch in _STRESS:
            flush()
            pending_stress += ch
            join_next = False
            continue
        if ch.isspace():
            flush()
            if units and units[-1] != " ":
                units.append(" ")
            pending_stress = ""
            join_next = False
            continue

        if ch == _LENGTH or unicodedata.combining(ch) != 0:
            if not cur:
                cur = pending_stress
                pending_stress = ""
            cur += ch
            continue

        if join_next and cur:
            cur += ch
            join_next = False
        else:
            flush()
            cur = pending_stress + ch
            pending_stress = ""
            join_next = False

    flush()
    return units


def ipa_units_with_spans(cleaned: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """返回音素 token 及其在原字符串中的字符范围，供报告标红。"""
    units = ipa_units_from_clean(cleaned)
    spans: List[Tuple[int, int]] = []
    pos = 0
    for unit in units:
        if unit == " ":
            while pos < len(cleaned) and cleaned[pos].isspace():
                pos += 1
            spans.append((-1, -1))
            continue

        while pos < len(cleaned) and cleaned[pos].isspace():
            pos += 1
        if cleaned.startswith(unit, pos):
            start = pos
            pos += len(unit)
            spans.append((start, pos))
            continue

        found = cleaned.find(unit, pos)
        if found >= 0:
            start = found
            pos = found + len(unit)
            spans.append((start, pos))
        else:
            spans.append((-1, -1))
    return units, spans


def edit_distance(ref_units: List[str], hyp_units: List[str]) -> int:
    """标准 Levenshtein 编辑距离：替换 / 删除 / 插入 代价均为 1。"""
    m, n = len(ref_units), len(hyp_units)
    if m == 0:
        return n
    if n == 0:
        return m

    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        r = ref_units[i - 1]
        for j in range(1, n + 1):
            cost = 0 if r == hyp_units[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev
    return prev[n]


def _levenshtein_alignment_ops(ref_units: List[str], hyp_units: List[str]) -> List[Tuple[str, str, str]]:
    """
    对两个 phone 字符序列做最优 Levenshtein 对齐并回溯得到操作序列。
    每个元素为 (op, ref_sym, hyp_sym)，op ∈ match|sub|del|ins。
    """
    m, n = len(ref_units), len(hyp_units)
    inf = m + n + 10
    dp: List[List[int]] = [[inf] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + 1
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c = 0 if ref_units[i - 1] == hyp_units[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + c,
            )

    ops_rev: List[Tuple[str, str, str]] = []
    i, j = m, n
    while i > 0 or j > 0:
        took = False
        if i > 0 and j > 0:
            c = 0 if ref_units[i - 1] == hyp_units[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + c:
                if c == 0:
                    ops_rev.append(("match", ref_units[i - 1], hyp_units[j - 1]))
                else:
                    ops_rev.append(("sub", ref_units[i - 1], hyp_units[j - 1]))
                i, j = i - 1, j - 1
                took = True
        if took:
            continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops_rev.append(("del", ref_units[i - 1], ""))
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops_rev.append(("ins", "", hyp_units[j - 1]))
            j -= 1
            continue
        break
    ops_rev.reverse()
    return ops_rev


def phoneme_error_detail_zh(
    ref_units: List[str],
    hyp_units: List[str],
    *,
    max_items_per_group: Optional[int] = None,
) -> str:
    """
    按与 PER 相同的字符级单位序列，列举相对参考而言：缺失 / 错误 / 多了。

    - 缺失：对齐中为删除（reference 有、hypothesis 无）
    - 错误：替换（reference 与 hypothesis 各一音素）
    - 多了：插入（hypothesis 多出的音素）

    ``max_items_per_group``：每类最多列出多少项；``None`` 表示不截断（报告/HTML 全量展示）。
    """
    if ref_units == hyp_units:
        return ""

    ops = _levenshtein_alignment_ops(ref_units, hyp_units)
    missing: List[str] = []
    wrong: List[Tuple[str, str]] = []
    extra: List[str] = []
    missing_spaces: List[str] = []
    extra_spaces: List[str] = []
    wrong_spaces: List[str] = []

    def _space_context(units: List[str], pos: int) -> str:
        """返回空格左右相邻片段，帮助定位词边界错误。"""
        left_i = pos - 1
        while left_i >= 0 and units[left_i] == " ":
            left_i -= 1
        left_j = left_i
        while left_i >= 0 and units[left_i] != " ":
            left_i -= 1

        right_i = pos + 1
        while right_i < len(units) and units[right_i] == " ":
            right_i += 1
        right_j = right_i
        while right_j < len(units) and units[right_j] != " ":
            right_j += 1

        left = "".join(units[left_i + 1 : left_j + 1]) or "句首"
        right = "".join(units[right_i:right_j]) or "句尾"
        return f"{left} | {right}"

    ref_pos = 0
    hyp_pos = 0
    for op, rs, hs in ops:
        rs_disp = "␣" if rs == " " else rs
        hs_disp = "␣" if hs == " " else hs

        if op == "match":
            ref_pos += 1
            hyp_pos += 1
            continue
        if op == "del":
            if rs == " ":
                missing_spaces.append(_space_context(ref_units, ref_pos))
            else:
                missing.append(rs_disp)
            ref_pos += 1
        elif op == "sub":
            if rs == " ":
                wrong_spaces.append(f"target 空格 { _space_context(ref_units, ref_pos) } 被替换为 {hs_disp}")
            elif hs == " ":
                wrong_spaces.append(f"G2P 多出空格 { _space_context(hyp_units, hyp_pos) }，替代了 {rs_disp}")
            else:
                wrong.append((rs_disp, hs_disp))
            ref_pos += 1
            hyp_pos += 1
        elif op == "ins":
            if hs == " ":
                extra_spaces.append(_space_context(hyp_units, hyp_pos))
            else:
                extra.append(hs_disp)
            hyp_pos += 1

    def _clip(items: List[str], join_inner: str) -> str:
        if max_items_per_group is None or len(items) <= max_items_per_group:
            return join_inner.join(items)
        head = join_inner.join(items[:max_items_per_group])
        return f"{head} …(等共{len(items)}项)"

    def _append_phone_diffs(parts_out: List[str], ref_phone_units: List[str], hyp_phone_units: List[str]) -> None:
        phone_missing: List[str] = []
        phone_wrong: List[Tuple[str, str]] = []
        phone_extra: List[str] = []
        for phone_op, phone_rs, phone_hs in _levenshtein_alignment_ops(ref_phone_units, hyp_phone_units):
            if phone_op == "del":
                phone_missing.append(phone_rs)
            elif phone_op == "sub":
                phone_wrong.append((phone_rs, phone_hs))
            elif phone_op == "ins":
                phone_extra.append(phone_hs)

        if phone_missing:
            parts_out.append("缺失:" + _clip(phone_missing, " "))
        if phone_wrong:
            wstrs = [f"{a}→{b}" for a, b in phone_wrong]
            if max_items_per_group is not None and len(wstrs) > max_items_per_group:
                head = " ".join(wstrs[:max_items_per_group])
                parts_out.append(f"错误:{head} …(等共{len(wstrs)}项)")
            else:
                parts_out.append("错误:" + " ".join(wstrs))
        if phone_extra:
            parts_out.append("多了:" + _clip(phone_extra, " "))

    has_space_error = bool(missing_spaces or extra_spaces or wrong_spaces)
    if has_space_error:
        compact_ref = [x for x in ref_units if x != " "]
        compact_hyp = [x for x in hyp_units if x != " "]
        parts = ["空格错误"]
        if compact_ref != compact_hyp:
            _append_phone_diffs(parts, compact_ref, compact_hyp)
        return "；".join(parts)

    if not missing and not wrong and not extra:
        return ""

    parts: List[str] = []
    if missing:
        parts.append("缺失:" + _clip(missing, " "))
    if wrong:
        wstrs = [f"{a}→{b}" for a, b in wrong]
        if max_items_per_group is not None and len(wstrs) > max_items_per_group:
            head = " ".join(wstrs[:max_items_per_group])
            parts.append(f"错误:{head} …(等共{len(wstrs)}项)")
        else:
            parts.append("错误:" + " ".join(wstrs))
    if extra:
        parts.append("多了:" + _clip(extra, " "))
    return "；".join(parts)


def phoneme_error_detail_from_strings(
    ref_text: str,
    hyp_text: str,
    *,
    max_items_per_group: Optional[int] = None,
) -> str:
    """先去掉所有空格做字符级比对；字符一致时再判定词间空格错误。"""
    ref_chars = list(ref_text.replace(" ", ""))
    hyp_chars = list(hyp_text.replace(" ", ""))
    if ref_chars == hyp_chars:
        return "" if ref_text == hyp_text else "空格错误"

    missing: List[str] = []
    wrong: List[Tuple[str, str]] = []
    extra: List[str] = []
    for op, rs, hs in _levenshtein_alignment_ops(ref_chars, hyp_chars):
        if op == "del":
            missing.append(rs)
        elif op == "sub":
            wrong.append((rs, hs))
        elif op == "ins":
            extra.append(hs)

    def _clip(items: List[str], join_inner: str) -> str:
        if max_items_per_group is None or len(items) <= max_items_per_group:
            return join_inner.join(items)
        head = join_inner.join(items[:max_items_per_group])
        return f"{head} …(等共{len(items)}项)"

    parts: List[str] = []
    if missing:
        parts.append("缺失:" + _clip(missing, " "))
    if wrong:
        wstrs = [f"{a}→{b}" for a, b in wrong]
        if max_items_per_group is not None and len(wstrs) > max_items_per_group:
            head = " ".join(wstrs[:max_items_per_group])
            parts.append(f"错误:{head} …(等共{len(wstrs)}项)")
        else:
            parts.append("错误:" + " ".join(wstrs))
    if extra:
        parts.append("多了:" + _clip(extra, " "))
    return "；".join(parts)


def g2p_space_error_indices(
    ref_text: str,
    hyp_text: str,
) -> List[int]:
    """返回 G2P 模型输出串里需要标红的位置：正确词边界两侧相邻字符。"""
    if ref_text == hyp_text:
        return []
    if ref_text.replace(" ", "") != hyp_text.replace(" ", ""):
        return []

    marks: set[int] = set()
    hyp_char_positions = [i for i, ch in enumerate(hyp_text) if ch != " "]
    boundary = 0
    for i, ch in enumerate(ref_text):
        if ch == " ":
            if boundary <= 0 or boundary >= len(hyp_char_positions):
                continue
            left_pos = hyp_char_positions[boundary - 1]
            right_pos = hyp_char_positions[boundary]
            if " " not in hyp_text[left_pos + 1 : right_pos]:
                marks.add(left_pos)
                marks.add(right_pos)
        else:
            boundary += 1

    return sorted(marks)


def calc_per_from_clean(target_clean: str, predict_clean: str) -> float:
    """基于已清洗字符串计算 phone-level PER；词边界空格不参与主 PER。"""
    ref_units = [u for u in ipa_units_from_clean(target_clean) if u != " "]
    hyp_units = [u for u in ipa_units_from_clean(predict_clean) if u != " "]

    if not ref_units:
        return 0.0 if not hyp_units else 100.0

    dist = edit_distance(ref_units, hyp_units)
    return max(0.0, dist / len(ref_units) * 100.0)


def space_boundary_positions_from_units(units: List[str]) -> List[int]:
    """返回词边界位于第几个非空格音素之后，用于单独统计空格边界差异。"""
    positions: List[int] = []
    phone_count = 0
    for unit in units:
        if unit == " ":
            if phone_count > 0 and (not positions or positions[-1] != phone_count):
                positions.append(phone_count)
        else:
            phone_count += 1
    return positions


def space_error_flags_from_clean(target_clean: str, predict_clean: str) -> Tuple[bool, bool]:
    """返回 (任意空格边界错误, 纯空格错误)。"""
    ref_units = ipa_units_from_clean(target_clean)
    hyp_units = ipa_units_from_clean(predict_clean)
    ref_spaces = space_boundary_positions_from_units(ref_units)
    hyp_spaces = space_boundary_positions_from_units(hyp_units)
    has_space_error = ref_spaces != hyp_spaces
    compact_ref = [u for u in ref_units if u != " "]
    compact_hyp = [u for u in hyp_units if u != " "]
    pure_space_error = has_space_error and compact_ref == compact_hyp
    return has_space_error, pure_space_error


def calc_per(target: str, predict: str) -> float:
    """
    计算 phone-level PER（%）。

    reference / hypothesis 均经相同的 ipa_eval_stages 后再比较。
    """
    _, tgt_clean = ipa_eval_stages(target)
    _, pred_clean = ipa_eval_stages(predict)
    return calc_per_from_clean(tgt_clean, pred_clean)


def _row_engine_ipa(low: Dict[str, str], *, role: str) -> str:
    """从 CSV/JSON 行读取引擎原始 IPA；target/predict 与 target_raw/predict_raw 等价。"""
    if role == "target":
        keys = ("target", "target_raw")
    else:
        keys = ("predict", "predict_raw")
    for k in keys:
        if k in low and low[k]:
            return low[k]
    return ""


def read_input(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    _, ext = os.path.splitext(path.lower())
    rows: List[Dict[str, str]] = []

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        data_list = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), None)
        if data_list is None:
            raise ValueError("JSON must be a list or contain a list")

        for item in data_list:
            if not isinstance(item, dict):
                continue
            low_l = {
                str(k).lower(): (v or "")
                for k, v in item.items()
                if isinstance(k, str)
            }
            rows.append(
                {
                    "text": low_l.get("text", ""),
                    "target": _row_engine_ipa(low_l, role="target"),
                    "predict": _row_engine_ipa(low_l, role="predict"),
                }
            )
    else:
        with open(path, "r", encoding="utf-8", newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except Exception:
                dialect = csv.excel

            reader = csv.DictReader(f, dialect=dialect)
            for r in reader:
                low = {(k or "").lower(): (v or "") for k, v in r.items()}
                rows.append(
                    {
                        "text": low.get("text", ""),
                        "target": _row_engine_ipa(low, role="target"),
                        "predict": _row_engine_ipa(low, role="predict"),
                    }
                )

    return rows


def prepare_results(raw_rows: List[Dict[str, str]]) -> List[Dict]:
    results: List[Dict] = []
    for idx, r in enumerate(raw_rows, start=1):
        tgt_engine = r.get("target", "") or ""
        # 新版模型直接输出带词边界/重音的 IPA，无后处理；predict 即模型输出。
        pred_model = r.get("predict", "") or ""
        tgt_raw = normalize_ipa_raw(tgt_engine)
        pred_raw = normalize_ipa_raw(pred_model)
        tgt_eval = clean_ipa(tgt_raw)
        pred_eval = clean_ipa(pred_raw)
        per = calc_per_from_clean(tgt_eval, pred_eval)
        ref_u = ipa_units_from_clean(tgt_eval)
        hyp_u = ipa_units_from_clean(pred_eval)
        space_error, pure_space_error = space_error_flags_from_clean(tgt_eval, pred_eval)
        phoneme_error_detail = phoneme_error_detail_zh(ref_u, hyp_u)
        predict_error_indices = g2p_space_error_indices(tgt_eval, pred_eval)
        results.append(
            {
                "index": idx,
                "text": r.get("text", "") or "",
                "target_raw": tgt_raw,
                "predict_raw": pred_raw,
                "target_clean": tgt_raw,
                "predict_clean": pred_raw,
                "per": round(per, 2),
                "space_error": space_error,
                "pure_space_error": pure_space_error,
                "phoneme_error_detail": phoneme_error_detail,
                "predict_error_indices": predict_error_indices,
                "used_espeak_fallback": bool(r.get("_used_espeak_fallback")),
            }
        )
    return results


def apply_espeak_fallback_on_high_per(
    raw_rows: List[Dict[str, str]],
    threshold_per: float,
) -> tuple[List[Dict[str, str]], int]:
    """
    当 NeMo 预测与 espeak 参考的 phone-level PER 超过 ``threshold_per`` 时，将 ``predict`` 替换为 ``target``（espeak IPA），
    并打上 ``_used_espeak_fallback`` 供 ``prepare_results`` 写入明细（可选展示）。

    用于抑制 CTC G2P 在功能词等上的灾难性错误，同时保留低 PER 句子的模型输出。
    """
    if threshold_per <= 0:
        return [dict(r) for r in raw_rows], 0
    out: List[Dict[str, str]] = []
    n_fb = 0
    for r in raw_rows:
        row = dict(r)
        tgt_engine = row.get("target", "") or ""
        pred_engine = row.get("predict", "") or ""
        _, tgt_clean = ipa_eval_stages(tgt_engine)
        _, pred_clean = ipa_eval_stages(pred_engine)
        per = calc_per_from_clean(tgt_clean, pred_clean)
        if per > threshold_per:
            row["predict"] = row.get("target", "") or ""
            row["_used_espeak_fallback"] = "1"
            n_fb += 1
        else:
            row.pop("_used_espeak_fallback", None)
        out.append(row)
    return out, n_fb


# 评测 CSV / HTML 明细统一只保留 4 类核心信息：
#   text            输入文本（行标识）
#   target_raw      espeak-ng 原始输出（词内拼接、词间单空格）
#   predict_raw     G2P 模型原始输出（同口径）
#   per             phone-level PER（忽略词边界空格）
#   phoneme_error_detail  差异原因（缺失/错误/多了/空格错误）
EVAL_CSV_FIELDS = (
    "text",
    "target_raw",
    "predict_raw",
    "per",
    "phoneme_error_detail",
)


def write_eval_csv(path: str | Path, results: List[Dict]) -> None:
    """写出与 HTML 报告一致的评测 CSV。"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_CSV_FIELDS)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in EVAL_CSV_FIELDS})


def is_letter_a_pronunciation_error(result: Dict) -> bool:
    """
  判定是否为单纯的「字母 A」读音差异：弱读 /ɐ/（espeak 不定冠词）vs 字母 A /ˈeɪ/。

  典型出现在导航出口编号，如 ``exit two A``；差异原因常为 ``错误:ɐ→ɪ；多了:ˈe``。
  """
    detail = result.get("phoneme_error_detail") or ""
    if "ɐ→ɪ" in detail and "ˈe" in detail:
        return True
    if "ɪ→ɐ" in detail and "e" in detail:
        return True

    tgt_raw = result.get("target_raw") or result.get("target_clean") or ""
    pred_raw = result.get("predict_raw") or result.get("predict_clean") or ""
    if not tgt_raw or not pred_raw or tgt_raw == pred_raw:
        return False

    ref_units = [u for u in ipa_units_from_clean(clean_ipa(tgt_raw)) if u != " "]
    hyp_units = [u for u in ipa_units_from_clean(clean_ipa(pred_raw)) if u != " "]
    if len(ref_units) != len(hyp_units):
        return False
    diffs = [(a, b) for a, b in zip(ref_units, hyp_units) if a != b]
    if len(diffs) != 1:
        return False
    a, b = diffs[0]
    letter_a = {"ˈeɪ", "eɪ", "ˈe", "e", "ɪ"}
    return (a == "ɐ" and b in letter_a) or (b == "ɐ" and a in letter_a)


def aggregate_metrics_excluding_letter_a(results: List[Dict]) -> Dict:
    """将字母 A 读音差异样本视为正确后重新汇总指标。"""
    adjusted: List[Dict] = []
    excluded = 0
    for r in results:
        if is_letter_a_pronunciation_error(r):
            excluded += 1
            adj = dict(r)
            adj["per"] = 0.0
            adj["space_error"] = False
            adj["pure_space_error"] = False
            adjusted.append(adj)
        else:
            adjusted.append(r)
    stats = aggregate_metrics(adjusted)
    stats["letter_a_excluded"] = excluded
    return stats


def aggregate_metrics(results: List[Dict]) -> Dict:
    n = len(results)
    if n == 0:
        return {
            "n": 0,
            "avg_per": 0.0,
            "exact_match_rate": 0.0,
            "warn_count": 0,
            "fail_count": 0,
            "space_error_count": 0,
            "space_error_rate": 0.0,
            "pure_space_error_count": 0,
            "pure_space_error_rate": 0.0,
        }

    total_per = sum(r["per"] for r in results)
    avg_per = total_per / n
    exact_matches = sum(1 for r in results if abs(r["per"]) < 1e-9)
    warn_count = sum(1 for r in results if 0.0 < r["per"] <= 5.0)
    fail_count = sum(1 for r in results if r["per"] > 5.0)
    space_error_count = sum(1 for r in results if r.get("space_error"))
    pure_space_error_count = sum(1 for r in results if r.get("pure_space_error"))
    exact_rate = exact_matches / n
    return {
        "n": n,
        "avg_per": avg_per,
        "exact_match_rate": exact_rate,
        "exact_count": exact_matches,
        "warn_count": warn_count,
        "fail_count": fail_count,
        "space_error_count": space_error_count,
        "space_error_rate": space_error_count / n,
        "pure_space_error_count": pure_space_error_count,
        "pure_space_error_rate": pure_space_error_count / n,
    }


def _global_summary(summaries: List[Dict]) -> Dict:
    active = [s for s in summaries if s.get("g2p_supported", True)]
    total_n = sum(int(s["n"]) for s in active)
    if total_n == 0:
        return {
            "total_n": 0,
            "avg_per": 0.0,
            "exact_pct": 0.0,
            "space_error_pct": 0.0,
            "pure_space_error_pct": 0.0,
            "locale_count": len(summaries),
        }
    weighted_per = sum(float(s["avg_per"]) * int(s["n"]) for s in active) / total_n
    exact_total = sum(int(s.get("exact_count", 0)) for s in active)
    space_total = sum(int(s.get("space_error_count", 0)) for s in active)
    pure_space_total = sum(int(s.get("pure_space_error_count", 0)) for s in active)
    return {
        "total_n": total_n,
        "avg_per": round(weighted_per, 2),
        "exact_pct": round(exact_total / total_n * 100.0, 2),
        "space_error_pct": round(space_total / total_n * 100.0, 2),
        "pure_space_error_pct": round(pure_space_total / total_n * 100.0, 2),
        "locale_count": len(summaries),
    }


def _write_locale_detail_js(data_dir: Path, locale: str, payload: Dict) -> None:
    """按语言写出 report_data/<locale>.js，供 report.html 点击后动态加载。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, ensure_ascii=False)
    js = (
        "window.__LOCALE_DATA__=window.__LOCALE_DATA__||{};\n"
        f"window.__LOCALE_DATA__[{json.dumps(locale)}]={body};\n"
    )
    (data_dir / f"{locale}.js").write_text(js, encoding="utf-8")


_MULTI_LOCALE_TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>G2P × espeak-ng 多语言一致性报告</title>
  <style>
    :root{
      --bg:#0f172a; --bg2:#1e293b; --card:#ffffff; --muted:#64748b;
      --ok:#10b981; --warn:#f59e0b; --fail:#ef4444; --accent:#3b82f6; --line:#e2e8f0;
    }
    *{box-sizing:border-box;}
    body{margin:0;font-family:system-ui,-apple-system,sans-serif;background:linear-gradient(160deg,#0f172a 0%,#1e3a5f 40%,#f1f5f9 40%);color:#0f172a;}
    .hero{max-width:1280px;margin:0 auto;padding:32px 24px 24px;color:#f8fafc;}
    .hero h1{margin:0 0 8px;font-size:28px;font-weight:700;letter-spacing:-0.02em;}
    .hero .sub{opacity:.85;font-size:14px;line-height:1.5;}
    .hero-meta{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px;font-size:12px;opacity:.9;}
    .hero-meta span{background:rgba(255,255,255,.12);padding:6px 12px;border-radius:8px;}
    .wrap{max-width:1280px;margin:0 auto;padding:0 24px 48px;}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:14px;margin-bottom:24px;}
    .kpi{background:var(--card);border-radius:14px;padding:18px;box-shadow:0 4px 24px rgba(15,23,42,.08);}
    .kpi .lbl{font-size:12px;color:var(--muted);font-weight:500;}
    .kpi .val{font-size:26px;font-weight:700;margin-top:6px;color:#0f172a;}
    .panel{background:var(--card);border-radius:16px;padding:20px;margin-bottom:24px;box-shadow:0 8px 32px rgba(15,23,42,.1);}
    .panel h2{margin:0 0 16px;font-size:18px;font-weight:700;}
    .nav{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:20px;}
    .nav button{font:inherit;cursor:pointer;font-size:13px;font-weight:600;padding:8px 14px;border-radius:999px;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;}
    .nav button:hover{background:#dbeafe;}
    .nav button.active{background:#1d4ed8;color:#fff;border-color:#1d4ed8;}
    .load-hint{font-size:12px;color:var(--muted);padding:8px 0;}
    .warn-box{background:#fff7ed;border:1px solid #fed7aa;color:#9a3412;padding:10px 14px;border-radius:10px;font-size:12px;margin-bottom:12px;}
    table{width:100%;border-collapse:collapse;font-size:12px;}
    .sum-table th,.sum-table td{padding:12px 10px;text-align:left;border-bottom:1px solid var(--line);}
    .sum-table th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);background:#f8fafc;}
    .sum-table tr:hover td{background:#f8fafc;}
    .locale-block{margin-bottom:32px;scroll-margin-top:16px;}
    .locale-head{display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:12px;margin-bottom:12px;padding-bottom:12px;border-bottom:2px solid #e2e8f0;}
    .locale-head h3{margin:0;font-size:17px;}
    .locale-meta{font-size:12px;color:var(--muted);}
    .detail-wrap{border:1px solid var(--line);border-radius:12px;}
    .detail-tools{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:8px 0 12px;}
    .detail-tools button{font:inherit;cursor:pointer;font-size:12px;font-weight:600;padding:6px 10px;border-radius:8px;background:#eff6ff;color:#1d4ed8;border:1px solid #bfdbfe;}
    .detail-tools button:hover{background:#dbeafe;}
    .detail-tools button[disabled]{cursor:not-allowed;opacity:.5;}
    .detail-error-cell{white-space:normal;word-break:break-word;min-width:12em;}
    .detail-table thead th{position:sticky;top:0;z-index:2;background:#f1f5f9;padding:10px 8px;font-size:11px;font-weight:600;color:#334155;border-bottom:1px solid var(--line);}
    .detail-table td{padding:10px 8px;border-bottom:1px solid #f1f5f9;vertical-align:top;}
    .detail-table tr:hover td{background:#f8fafc;}
    .mono{font-family:ui-monospace,Menlo,Monaco,monospace;font-size:11px;line-height:1.45;word-break:break-all;color:#0f172a;}
    .mono.clean{color:#475569;background:#f8fafc;padding:4px 6px;border-radius:4px;display:block;margin-top:4px;}
    .err-mark{background:#fee2e2;color:#991b1b;border-radius:3px;padding:0 1px;font-weight:800;}
    .badge{display:inline-block;padding:4px 10px;border-radius:999px;font-size:11px;font-weight:700;color:#fff;}
    .badge.ok{background:var(--ok);} .badge.warn{background:var(--warn);color:#111;} .badge.fail{background:var(--fail);}
    .per-cell{min-width:100px;}
    .per-num{font-weight:700;font-size:13px;}
    .bar{height:6px;background:#e2e8f0;border-radius:4px;margin-top:6px;overflow:hidden;}
    .bar>i{display:block;height:100%;background:linear-gradient(90deg,#60a5fa,#2563eb);border-radius:4px;}
    .footer{text-align:center;font-size:12px;color:var(--muted);padding:24px;}
    .detail-table thead th.sortable{cursor:pointer;user-select:none;}
    .detail-table thead th.sortable:hover{background:#e2e8f0;}
    .detail-table thead th.sorted-asc,.detail-table thead th.sorted-desc{background:#dbeafe;}
    .detail-table thead th .sort-ind{font-weight:700;color:#2563eb;font-size:10px;margin-left:3px;}
  </style>
</head>
<body>
  <div class="hero">
    <h1>G2P × espeak-ng 发音一致性评估</h1>
    <div class="sub">对比 espeak-ng 原始输出与 G2P 模型原始输出 · phone-level PER 忽略词边界空格 · 差异原因含空格错误（参考列通常来自 espeak-ng；若需规避 GPL 请改用 manifest 等自有标注）</div>
    <div class="hero-meta">
      <span>生成时间 {{ generated_at }}</span>
      <span>耗时 {{ elapsed_s }}s</span>
      <span>G2P 模型 {{ g2p_model }}</span>
      <span>{{ locale_count }} 种语言</span>
    </div>
  </div>

  <div class="wrap">
    <div class="grid">
      <div class="kpi"><div class="lbl">总样本数</div><div class="val">{{ global.total_n }}</div></div>
      <div class="kpi"><div class="lbl">加权音素 PER</div><div class="val">{{ global.avg_per }}%</div></div>
      <div class="kpi"><div class="lbl">音素完全匹配率</div><div class="val">{{ global.exact_pct }}%</div></div>
      <div class="kpi"><div class="lbl">评测语言数</div><div class="val">{{ global.locale_count }}</div></div>
    </div>

    <div class="panel">
      <h2>语言汇总</h2>
      <div class="nav" id="locale-nav">
        {% for s in summaries %}
        <button type="button" class="locale-tab" data-locale="{{ s.locale }}">{{ s.label }}</button>
        {% endfor %}
      </div>
      <table class="sum-table">
        <thead>
          <tr>
            <th>语言</th>
            <th>Locale</th>
            <th>样本数</th>
            <th>平均 PER</th>
            <th>完全匹配</th>
            <th>Warn (0–5%)</th>
            <th>Fail (&gt;5%)</th>
            <th>espeak voice</th>
            <th>G2P tag</th>
          </tr>
        </thead>
        <tbody>
        {% for s in summaries %}
          <tr>
            <td><strong>{{ s.label }}</strong></td>
            <td>{{ s.locale }}</td>
            <td>{{ s.n }}</td>
            <td><strong>{% if s.get('g2p_supported', true) %}{{ '%.2f'|format(s.avg_per) }}%{% else %}<span title="G2P 模型不支持该语言">—</span>{% endif %}</strong></td>
            <td>{{ '%.1f'|format(s.exact_match_rate * 100) }}%</td>
            <td>{{ s.warn_count }}</td>
            <td>{{ s.fail_count }}</td>
            <td class="mono">{{ s.espeak_voice }}</td>
            <td class="mono">{{ s.g2p_lang_tag }}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>

    <div class="panel locale-block" id="locale-detail">
      <div class="locale-head">
        <div>
          <h3 id="detail-title">明细</h3>
          <div class="locale-meta" id="detail-meta">点击上方语言标签加载该语言明细 · <strong>明细表头可点击排序</strong>（同列再点反向）</div>
        </div>
      </div>
      <div id="detail-warn" class="warn-box" style="display:none"></div>
      <div class="detail-tools">
        <div class="load-hint" id="load-status"></div>
        <button type="button" id="show-all-btn" style="display:none">显示全部</button>
      </div>
      <div class="detail-wrap">
        <table class="detail-table">
          <thead>
            <tr>
              <th class="sortable" data-sort-key="index" title="点击排序">#<span class="sort-ind"></span></th>
              <th class="sortable" data-sort-key="text" title="点击排序">Text<span class="sort-ind"></span></th>
              <th class="sortable" data-sort-key="target_raw" title="点击排序">espeak-ng 原始输出<span class="sort-ind"></span></th>
              <th class="sortable" data-sort-key="predict_raw" title="点击排序">G2P 模型原始输出<span class="sort-ind"></span></th>
              <th class="sortable" data-sort-key="per" title="点击排序">PER<span class="sort-ind"></span></th>
              <th class="sortable" data-sort-key="phoneme_error_detail" title="点击排序（按差异全文）">原因<span class="sort-ind"></span></th>
            </tr>
          </thead>
          <tbody id="detail-tbody"></tbody>
        </table>
      </div>
    </div>

    <div class="footer">PER = 忽略空格后的 phone-level 编辑距离 / reference 音素长度；差异原因（含空格错误）见「原因」列 · 明细表头点击排序；数据从 report_data/&lt;locale&gt;.js 加载</div>
  </div>
  <script type="application/json" id="summaries-json">{{ summaries_json }}</script>
  <script>
  (function(){
    const summaries = JSON.parse(document.getElementById('summaries-json').textContent);
    const byLocale = Object.fromEntries(summaries.map(s => [s.locale, s]));
    const tbody = document.getElementById('detail-tbody');
    const titleEl = document.getElementById('detail-title');
    const metaEl = document.getElementById('detail-meta');
    const statusEl = document.getElementById('load-status');
    const warnEl = document.getElementById('detail-warn');
    const dataBase = {{ data_base_json }};
    let activeLocale = null;
    let lastDetailLocale = null;
    let detailRowsRaw = [];
    let sortDir = 1;
    let sortKey = 'index';
    let showAllRows = false;
    const RENDER_LIMIT = 200;

    const detailThead = document.querySelector('.detail-table thead');
    const showAllBtn = document.getElementById('show-all-btn');

    function esc(s){
      return String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }
    function renderMarked(s, indices){
      const markSet = new Set(indices || []);
      return Array.from(String(s ?? '')).map((ch, i) => {
        const safe = esc(ch);
        return markSet.has(i) ? `<span class="err-mark">${safe}</span>` : safe;
      }).join('');
    }
    function perBadge(per){
      if (per === 0) return '<span class="badge ok">OK</span>';
      if (per <= 5) return '<span class="badge warn">Warn</span>';
      return '<span class="badge fail">Fail</span>';
    }
    function renderRows(rows){
      const visibleRows = showAllRows ? rows : rows.slice(0, RENDER_LIMIT);
      if(rows.length > RENDER_LIMIT && !showAllRows){
        statusEl.textContent = `已加载 ${rows.length} 条，当前仅渲染前 ${RENDER_LIMIT} 条以保证秒开；排序会基于全量数据`;
        showAllBtn.style.display = '';
        showAllBtn.disabled = false;
      } else if(rows.length){
        statusEl.textContent = `已显示 ${rows.length} 条`;
        showAllBtn.style.display = rows.length > RENDER_LIMIT ? '' : 'none';
        showAllBtn.disabled = showAllRows;
      }
      tbody.innerHTML = visibleRows.map(r => {
        const barW = Math.min(100, r.per).toFixed(2);
        return `<tr>
          <td>${r.index}</td>
          <td>${esc(r.text)}</td>
          <td class="mono">${esc(r.target_raw)}</td>
          <td class="mono">${esc(r.predict_raw)}</td>
          <td class="per-cell">
            <div class="per-num">${r.per.toFixed(2)}%</div>
            ${perBadge(r.per)}
            ${r.used_espeak_fallback ? '<div style="font-size:10px;color:#0369a1;margin-top:4px;font-weight:600">NeMo→espeak 回退</div>' : ''}
            <div class="bar"><i style="width:${barW}%"></i></div>
          </td>
          <td class="mono detail-error-cell">${esc(r.phoneme_error_detail || '')}</td>
        </tr>`;
      }).join('');
    }
    function compareRowsAsc(a, b, key){
      if(key === 'index') return (Number(a.index)||0) - (Number(b.index)||0);
      if(key === 'per') return (Number(a.per)||0) - (Number(b.per)||0);
      if(key === 'space_error') return Number(Boolean(a.space_error)) - Number(Boolean(b.space_error));
      const sa = String(a[key] ?? '');
      const sb = String(b[key] ?? '');
      return sa.localeCompare(sb, 'zh-CN', {numeric:true,sensitivity:'base'});
    }
    function sortDetailRows(rows){
      const key = sortKey;
      const out = rows.slice();
      out.sort((a,b) => {
        const c = compareRowsAsc(a, b, key);
        return sortDir === 1 ? c : -c;
      });
      return out;
    }
    function updateSortHeaders(){
      detailThead.querySelectorAll('th[data-sort-key]').forEach(function(th){
        th.classList.remove('sorted-asc','sorted-desc');
        const ind = th.querySelector('.sort-ind');
        if(ind) ind.textContent = '';
        const k = th.getAttribute('data-sort-key');
        if(k === sortKey){
          th.classList.add(sortDir === 1 ? 'sorted-asc' : 'sorted-desc');
          if(ind) ind.textContent = sortDir === 1 ? '▲' : '▼';
        }
      });
    }
    function refreshDetailTable(){
      const sorted = sortDetailRows(detailRowsRaw);
      renderRows(sorted);
      updateSortHeaders();
    }
    function applyLocaleDetailRows(loc, rows, resetSort){
      activeLocale = loc;
      detailRowsRaw = (rows || []).slice();
      if(resetSort){
        sortKey = 'index';
        sortDir = 1;
        showAllRows = false;
      }
      refreshDetailTable();
    }
    function showSummary(loc){
      const s = byLocale[loc];
      if (!s) return;
      titleEl.textContent = `${s.label} (${loc})`;
      const perTxt = s.g2p_supported === false ? '—' : `${Number(s.avg_per).toFixed(2)}%`;
      metaEl.textContent = `espeak voice: ${s.espeak_voice || ''} · G2P: ${s.g2p_lang_tag || ''} · n=${s.n} · 音素 PER ${perTxt}` +
        (s.espeak_fallback_count ? ` · NeMo→espeak 回退 ${s.espeak_fallback_count} 条 (阈值 ${s.espeak_fallback_threshold ?? ''}%)` : '');
      if (s.g2p_supported === false) {
        warnEl.style.display = 'block';
        warnEl.textContent = s.g2p_note || 'G2P 模型不支持该语言 tag，PER 无参考价值。';
      } else {
        warnEl.style.display = 'none';
        warnEl.textContent = '';
      }
    }
    function loadLocale(loc){
      const resetSort = (loc !== lastDetailLocale);
      showSummary(loc);
      document.querySelectorAll('.locale-tab').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.locale === loc);
      });
      if (window.__LOCALE_DATA__ && window.__LOCALE_DATA__[loc]) {
        lastDetailLocale = loc;
        applyLocaleDetailRows(loc, window.__LOCALE_DATA__[loc].rows || [], resetSort);
        return;
      }
      statusEl.textContent = '加载中…';
      tbody.innerHTML = '';
      detailRowsRaw = [];
      const script = document.createElement('script');
      script.src = `${dataBase}/${loc}.js`;
      script.onload = function(){
        statusEl.textContent = '';
        lastDetailLocale = loc;
        const data = window.__LOCALE_DATA__ && window.__LOCALE_DATA__[loc];
        applyLocaleDetailRows(loc, (data && data.rows) || [], true);
        script.remove();
      };
      script.onerror = function(){
        statusEl.textContent = `无法加载 ${dataBase}/${loc}.js，请用 HTTP 打开报告目录（见 README）。`;
        detailRowsRaw = [];
        renderRows([]);
        updateSortHeaders();
        script.remove();
      };
      document.head.appendChild(script);
    }
    detailThead.addEventListener('click', function(e){
      const th = e.target.closest('th[data-sort-key]');
      if(!th) return;
      const key = th.getAttribute('data-sort-key');
      if(!key) return;
      if(sortKey === key){
        sortDir = -sortDir;
      } else {
        sortKey = key;
        sortDir = 1;
      }
      refreshDetailTable();
    });
    showAllBtn.addEventListener('click', function(){
      showAllRows = true;
      refreshDetailTable();
    });
    document.getElementById('locale-nav').addEventListener('click', e => {
      const btn = e.target.closest('.locale-tab');
      if (!btn) return;
      loadLocale(btn.dataset.locale);
    });
    if (summaries.length) loadLocale(summaries[0].locale);
  })();
  </script>
</body>
</html>
""".strip()


def render_multi_locale_report(
    output_path: str | Path,
    *,
    summaries: List[Dict],
    rows_by_locale: Dict[str, List[Dict]],
    generated_at: str,
    g2p_model: str,
    elapsed_s: float,
) -> None:
    global_stats = _global_summary(summaries)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = path.parent / "report_data"
    for s in summaries:
        loc = s["locale"]
        _write_locale_detail_js(
            data_dir,
            loc,
            {
                "locale": loc,
                "label": s.get("label", loc),
                "espeak_voice": s.get("espeak_voice", ""),
                "g2p_lang_tag": s.get("g2p_lang_tag", ""),
                "n": s["n"],
                "avg_per": s.get("avg_per", 0),
                "g2p_supported": s.get("g2p_supported", True),
                "g2p_note": s.get("g2p_note", ""),
                "rows": rows_by_locale.get(loc, []),
            },
        )
    ctx = {
        "generated_at": generated_at,
        "g2p_model": g2p_model,
        "elapsed_s": elapsed_s,
        "locale_count": global_stats["locale_count"],
        "global": global_stats,
        "summaries": summaries,
        "summaries_json": json.dumps(summaries, ensure_ascii=False),
        "data_base_json": json.dumps("report_data"),
    }
    from jinja2 import Template

    html = Template(_MULTI_LOCALE_TEMPLATE).render(**ctx)
    path.write_text(html, encoding="utf-8")


def render_html(results: List[Dict], stats: Dict, output_path: str) -> None:
    """单语言报告（兼容 g2p_evaluate.py CLI）。"""
    render_multi_locale_report(
        output_path,
        summaries=[{**stats, "locale": "single", "label": "单语言", "espeak_voice": "", "g2p_lang_tag": ""}],
        rows_by_locale={"single": results},
        generated_at="",
        g2p_model="",
        elapsed_s=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="G2P 发音一致性评估并生成 HTML 报告")
    parser.add_argument("--input", "-i", required=True, help="输入文件，CSV 或 JSON（包含 text,target,predict 列）")
    parser.add_argument("--output", "-o", default="report.html", help="输出 HTML 报告文件路径")
    args = parser.parse_args()

    raw = read_input(args.input)
    results = prepare_results(raw)
    stats = aggregate_metrics(results)
    render_html(results, stats, args.output)
    print(f"已生成报告: {args.output}")


if __name__ == "__main__":
    main()