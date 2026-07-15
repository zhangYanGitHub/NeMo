#!/usr/bin/env python3
"""
espeak_ng_client.py

提供最小化的 espeak-ng / piper_phonemize 调用封装：
- read_texts_from_csv(csv_path, text_col="text")
- resolve_piper_espeak_voice(locale_or_voice)
- piper_ipa_batch(texts, voice, locale, show_progress, stress_mode)
- write_eval_csv(path, texts, targets, predicts)

IPA 处理链与 NeMo G2P 训练前处理（NeMo/examples/dataset/preprocess_ipa_childes_split.py）
逐字符对齐：normalize_text(NFC) → phonemize_espeak → 扁平拼接 → segment_word(stress=attach)
→ 空格连接 token。底层 phonemize_espeak 调用与 piper TTS（piper_train/preprocess.py）完全一致；
segment_word 是 NeMo G2P 在 piper 音素之上的统一切分（去连接符 ͡、重音附着元音），使参考 IPA
与模型 phoneme 词表同口径，PER 比较才公平。

Voice id 映射参考 espeak-ng 官方 BCP47 Identifier：
https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md
"""
from __future__ import annotations

import csv
import threading
import unicodedata
from pathlib import Path
from typing import List

_ESPEAK_LOCK = threading.Lock()

# locale（如 it_IT）→ espeak-ng voice id（BCP47 Identifier）
# 仅包含 quantization/config/all_datasets/ 下 CSV 对应语言；
# voice id 必须来自 espeak-ng 官方 languages.md，勿自行编造。
LOCALE_TO_ESPEAK_VOICE: dict[str, str] = {
    # English
    "en_us": "en-us",              # American
    "en_gb": "en-gb-x-rp",         # Received Pronunciation（项目 en_GB 沿用 RP；也可用 en=British）
    # Germanic
    "de_de": "de",
    "nl_nl": "nl",
    "da_dk": "da",
    "sv_se": "sv",
    "nb_no": "nb",                 # Norwegian Bokmål
    "no_no": "nb",                 # 同上，CSV 文件名 no_NO
    "fi_fi": "fi",
    # Romance
    "fr_fr": "fr",                 # France
    "fr_ca": "fr",                 # espeak-ng 无 fr-CA，回退 fr
    "it_it": "it",
    "es_es": "es",                 # Spain
    "pt_pt": "pt",                 # Portugal
    "pt_br": "pt-br",              # Brazil
    "ca_es": "ca",                 # Catalan
    "ro_ro": "ro",
    # Slavic
    "ru_ru": "ru",
    "pl_pl": "pl",
    "cs_cz": "cs",
    "sk_sk": "sk",
    "hr_hr": "hr",
    "sl_si": "sl",
    "bg_bg": "bg",
    "uk_ua": "uk",
    # Other European
    "el_gr": "el",                 # Modern Greek
    "hu_hu": "hu",
    "et_ee": "et",
    "lt_lt": "lt",
    "lv_lv": "lv",
    # Turkic / Central Asian
    "tr_tr": "tr",
    "kk_kz": "kk",
    # Semitic / Arabic
    "ar_jo": "ar",
    # East Asian
    "ja_jp": "ja",
    "ko_kr": "ko",
    "zh_cn": "cmn",                # Mandarin（非 zh）
    # Southeast Asian
    "th_th": "th",
    "vi_vn": "vi",
}

# 已是 espeak-ng voice id 的直通表（小写 key → 官方 id）
ESPEAK_VOICE_IDS: frozenset[str] = frozenset(
    {
        "en-us",
        "en",
        "en-gb-x-rp",
        "en-gb-scotland",
        "en-gb-x-gbclan",
        "en-gb-x-gbcwmd",
        "en-029",
        "de",
        "nl",
        "da",
        "sv",
        "nb",
        "fi",
        "fr",
        "fr-be",
        "fr-ch",
        "it",
        "es",
        "es-419",
        "pt",
        "pt-br",
        "ca",
        "ro",
        "ru",
        "pl",
        "cs",
        "sk",
        "hr",
        "sl",
        "bg",
        "uk",
        "el",
        "hu",
        "et",
        "lt",
        "lv",
        "tr",
        "kk",
        "ar",
        "ja",
        "ko",
        "cmn",
        "th",
        "vi",
    }
)


def _normalize_locale_key(value: str) -> str:
    return str(value).strip().replace("-", "_").lower()


def resolve_piper_espeak_voice(voice_or_locale: str) -> str:
    """
    将 locale（it_IT）或 espeak-ng voice id（it / en-us）解析为官方 voice id。

    解析顺序：
      1. LOCALE_TO_ESPEAK_VOICE 精确映射
      2. 输入本身已是官方 voice id（如 en-us、en-gb-x-rp）
      3. 抛出 ValueError，提示用 --espeak-voice 或 espeak-ng --voices 查看
    """
    if not voice_or_locale:
        raise ValueError("espeak voice / locale 不能为空")

    raw = str(voice_or_locale).strip()
    key = _normalize_locale_key(raw)

    if key in LOCALE_TO_ESPEAK_VOICE:
        return LOCALE_TO_ESPEAK_VOICE[key]

    voice_id = raw.lower()
    if voice_id in ESPEAK_VOICE_IDS:
        return voice_id

    supported = ", ".join(sorted({v for v in LOCALE_TO_ESPEAK_VOICE.values()}))
    raise ValueError(
        f"无法将 {voice_or_locale!r} 映射到 espeak-ng voice。"
        f"请使用 --espeak-voice 指定官方 voice id，或检查 locale 是否在支持列表中。\n"
        f"已知 locale: {', '.join(sorted(LOCALE_TO_ESPEAK_VOICE))}\n"
        f"对应 voice 示例: {supported}\n"
        f"完整列表请运行: espeak-ng --voices"
    )


def read_texts_from_csv(csv_path: Path, *, text_col: str = "text") -> List[str]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or text_col not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} 需含列 {text_col!r}，当前: {reader.fieldnames}"
            )
        seen: set[str] = set()
        out: List[str] = []
        for row in reader:
            t = (row.get(text_col) or "").strip()
            if not t or t in seen:
                continue
            seen.add(t)
            out.append(t)
    return out


# ── 与 NeMo G2P 训练前处理逐字符对齐 ──────────────────────────────────────────
# 来源：NeMo/examples/dataset/preprocess_ipa_childes_split.py
# 目的：espeak 参考 IPA 必须与模型训练标签同一切分口径（segment_word + stress=attach），
# 否则 PER 比较会因连接符 ͡ / 重音附着方式不同而产生大量假性错误。
# 切勿在此处擅自调整逻辑——任何改动都必须与上述训练脚本保持同步。
ZWJ = "\u200d"
TIES = {ZWJ, "\u0361", "\u035c"}
PRIMARY_STRESS = "\u02c8"  # ˈ
SECONDARY_STRESS = "\u02cc"  # ˌ
STRESS = {PRIMARY_STRESS, SECONDARY_STRESS}
LENGTH = "\u02d0"  # ː
DEFAULT_STRESS_MODE = "attach"  # 与导出模型 phoneme_labels（重音附着元音）一致


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def is_attaching(ch: str) -> bool:
    if ch == LENGTH:
        return True
    return unicodedata.combining(ch) != 0


def segment_word(ipa: str, stress_mode: str = DEFAULT_STRESS_MODE) -> List[str]:
    units: List[str] = []
    cur = ""
    pending_stress = ""
    join = False

    def flush() -> None:
        nonlocal cur
        if cur:
            units.append(cur)
            cur = ""

    for ch in ipa:
        if ch in TIES:
            join = True
            continue

        if ch in STRESS:
            flush()
            if stress_mode == "separate":
                units.append(ch)
            elif stress_mode == "attach":
                pending_stress += ch
            join = False
            continue

        if ch.isspace():
            flush()
            units.append("_")
            pending_stress = ""
            join = False
            continue

        if is_attaching(ch):
            if not cur:
                cur = pending_stress
                pending_stress = ""
            cur += ch
            continue

        if join and cur:
            cur += ch
            join = False
        else:
            flush()
            cur = pending_stress + ch
            pending_stress = ""
            join = False

    flush()
    return units


def _piper_ipa_one(text: str, voice: str, *, stress_mode: str = DEFAULT_STRESS_MODE) -> str:
    """生成单句 IPA，处理链与 preprocess_ipa_childes_split.py 完全一致：

    normalize_text(NFC + 空白合并) → phonemize_espeak（piper TTS 同一底层调用）
    → 扁平拼接 → strip/换行转空格 → segment_word(stress_mode) → 空格连接 token。
    """
    try:
        from piper_phonemize import phonemize_espeak
    except ImportError as e:
        raise RuntimeError(
            "需要安装 piper-phonemize: pip install piper-phonemize"
        ) from e

    s = normalize_text(text or "")
    if not s:
        return ""

    piper_voice = resolve_piper_espeak_voice(voice)
    try:
        with _ESPEAK_LOCK:
            sentences = phonemize_espeak(s, piper_voice)
    except RuntimeError as e:
        if "voice" in str(e).lower():
            raise RuntimeError(
                f"piper_phonemize 无法使用 voice {piper_voice!r}（来自输入 {voice!r}）。"
                f"请确认系统已安装 espeak-ng，并运行 espeak-ng --voices 核对 voice id。"
            ) from e
        raise
    # 扁平拼接：与 piper preprocess / preprocess_ipa_childes_split 的 flatten 等价
    raw = "".join("".join(sentence) for sentence in sentences)
    raw = raw.strip().replace("\n", " ")
    units = segment_word(raw, stress_mode=stress_mode)
    
    # 所有的基元用单个空格连接，传给 NeMo 训练。
    # 空白符已经被 segment_word 转换成了 '_'
    return " ".join(units)


def _piper_ipa_raw_one(text: str, voice: str) -> str:
    """生成 espeak-ng/piper 原始 IPA：只扁平化底层输出，不做 segment_word。"""
    try:
        from piper_phonemize import phonemize_espeak
    except ImportError as e:
        raise RuntimeError(
            "需要安装 piper-phonemize: pip install piper-phonemize"
        ) from e

    s = normalize_text(text or "")
    if not s:
        return ""

    piper_voice = resolve_piper_espeak_voice(voice)
    try:
        with _ESPEAK_LOCK:
            sentences = phonemize_espeak(s, piper_voice)
    except RuntimeError as e:
        if "voice" in str(e).lower():
            raise RuntimeError(
                f"piper_phonemize 无法使用 voice {piper_voice!r}（来自输入 {voice!r}）。"
                f"请确认系统已安装 espeak-ng，并运行 espeak-ng --voices 核对 voice id。"
            ) from e
        raise
    return "".join("".join(sentence) for sentence in sentences).strip().replace("\n", " ")


def piper_ipa_batch(
    texts: List[str],
    voice: str,
    *,
    locale: str = "",
    show_progress: bool = False,
    stress_mode: str = DEFAULT_STRESS_MODE,
) -> List[str]:
    del locale, show_progress  # 保留参数以兼容 run_pipeline 调用签名
    n = len(texts)
    if n == 0:
        return []
    return [_piper_ipa_one(t, voice, stress_mode=stress_mode) for t in texts]


def piper_ipa_raw_batch(
    texts: List[str],
    voice: str,
    *,
    locale: str = "",
    show_progress: bool = False,
) -> List[str]:
    del locale, show_progress
    if not texts:
        return []
    return [_piper_ipa_raw_one(t, voice) for t in texts]


def write_eval_csv(
    path: Path,
    texts: List[str],
    targets: List[str],
    predicts: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("text", "target", "predict"))
        writer.writeheader()
        for t, tgt, pred in zip(texts, targets, predicts):
            writer.writerow({"text": t, "target": tgt, "predict": pred})
