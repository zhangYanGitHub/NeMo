#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Executor, ProcessPoolExecutor, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from text_normalize import normalize_for_g2p
from tqdm import tqdm

from ar_XA.diacritizer_frontend import diacritize_arabic_line, is_arabic_voice

ZWJ = "\u200d"
TIES = {ZWJ, "\u0361", "\u035c"}
PRIMARY_STRESS = "\u02c8"  # ˈ
SECONDARY_STRESS = "\u02cc"  # ˌ
STRESS = {PRIMARY_STRESS, SECONDARY_STRESS}
LENGTH = "\u02d0"  # ː
SPECIAL_TOKENS = ("<pad>", "<unk>")
MANIFEST_WRITE_BUFFER_LINES = 1024

# EVERYTHING language-specific (espeak voice, number-normalization locale, the multi-character
# phoneme atom whitelist, and the self-check regression cases) lives in lang_config.json, NOT
# here, so adding / switching a language is a DATA change in that file -- this script stays
# language-agnostic. Multi-character atoms are diphthongs/affricates/... that must survive the
# longest-match tokenizer whole; long vowels (iː, uː, ...), nasalization, syllabicity etc. are
# NOT listed there: they're "base char + combining mark" and handled generically by
# is_attaching() below for ANY language.
DEFAULT_LANG_CONFIG_PATH = Path(__file__).with_name("lang_config.json")


class LanguageProfile:
    """One language's worth of config from lang_config.json (see that file's _comment)."""

    __slots__ = (
        "key", "voice", "number_locale", "multi_char_phonemes",
        "self_check", "phoneme_inventory", "grapheme_inventory",
    )

    def __init__(
        self,
        key: str,
        voice: str,
        number_locale: str,
        multi_char_phonemes: Tuple[str, ...],
        self_check: Dict[str, Tuple[str, List[str]]],
        phoneme_inventory: Tuple[str, ...] = (),
        grapheme_inventory: frozenset = frozenset(),
    ) -> None:
        self.key = key
        self.voice = voice
        self.number_locale = number_locale
        self.multi_char_phonemes = multi_char_phonemes
        self.self_check = self_check
        # 该语言允许出现的单音素基字符白名单（去除重音 ˈˌ / 长音 ː / 组合符后的基底）。
        # 空 = 不做音素白名单校验（向后兼容）。见 build_voice_phone_sets()/atoms_are_in_language().
        self.phoneme_inventory = phoneme_inventory
        # 该语言 text_graphemes 允许出现的字素字符白名单。空 = 不校验（向后兼容）。
        self.grapheme_inventory = grapheme_inventory


@functools.lru_cache(maxsize=None)
def load_lang_config(path: str) -> Tuple[str, Dict[str, LanguageProfile]]:
    """Load lang_config.json -> (default_language_key, {language_key: LanguageProfile}).
    Cached per path so every worker process/thread only pays the (tiny) parse cost once."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Language config not found: {p}. It holds the per-language voice / number locale / "
            f"phoneme-atom whitelist / self-check cases and is required. See lang_config.json "
            f"next to this script."
        )
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    languages = raw.get("languages", {})
    if not isinstance(languages, dict) or not languages:
        raise ValueError(f"lang_config.json {p} has no 'languages' map.")

    profiles: Dict[str, LanguageProfile] = {}
    for key, prof in languages.items():
        if key.startswith("_") or not isinstance(prof, dict):
            continue  # e.g. a "_comment" key
        atoms = tuple(sorted(set(prof.get("multi_char_phonemes", [])), key=len, reverse=True))
        self_check_raw = prof.get("self_check", {}) or {}
        self_check = {word: (ipa, list(expected)) for word, (ipa, expected) in self_check_raw.items()}
        profiles[normalize_voice(key)] = LanguageProfile(
            key=normalize_voice(key),
            voice=normalize_voice(prof.get("voice", key)),
            number_locale=str(prof.get("number_locale", "en")).strip().lower(),
            multi_char_phonemes=atoms,
            self_check=self_check,
            phoneme_inventory=tuple(prof.get("phoneme_inventory", []) or []),
            grapheme_inventory=frozenset("".join(prof.get("grapheme_inventory", "") or "")),
        )

    default_language = normalize_voice(raw.get("default_language") or next(iter(profiles)))
    if default_language not in profiles:
        raise ValueError(
            f"lang_config.json default_language {default_language!r} is not one of the "
            f"configured languages: {sorted(profiles)}"
        )
    return default_language, profiles


def build_voice_inventories(profiles: Dict[str, LanguageProfile]) -> Dict[str, Tuple[str, ...]]:
    """Collapse the language profiles into the {voice_code: (atoms longest-first)} map that
    the per-row tokenizer looks up by espeak voice. Every configured language contributes its
    voice, so a mixed-language CSV still tokenizes each row with the right atom whitelist."""
    return {prof.voice: prof.multi_char_phonemes for prof in profiles.values()}


def get_multi_char_phonemes_for_voice(voice: str, inventories: Dict[str, Tuple[str, ...]]) -> Tuple[str, ...]:
    """Exact voice match first (e.g. 'en-us'), else fall back to the primary-language subtag
    (e.g. an unlisted 'en-gb' reuses 'en'), else () -- generic combining-mark merging only."""
    voice = normalize_voice(voice)
    if voice in inventories:
        return inventories[voice]
    lang = voice.split("-", 1)[0]
    return inventories.get(lang, ())


def build_voice_phone_sets(profiles: Dict[str, LanguageProfile]) -> Dict[str, Set[str]]:
    """{voice_code: set(allowed base phones)} —— 每语言的音素白名单（单音素基字符 ∪ 多字符原子）。
    用于剔除混入的外语/未声明音素。空集合 = 不校验。"""
    out: Dict[str, Set[str]] = {}
    for prof in profiles.values():
        if prof.phoneme_inventory:
            out[prof.voice] = set(prof.phoneme_inventory) | set(prof.multi_char_phonemes)
    return out


def build_voice_grapheme_sets(profiles: Dict[str, LanguageProfile]) -> Dict[str, frozenset]:
    """{voice_code: grapheme_inventory} for per-row grapheme whitelist gating."""
    return {prof.voice: prof.grapheme_inventory for prof in profiles.values() if prof.grapheme_inventory}


def graphemes_in_inventory(text: str, allowed: frozenset) -> bool:
    """True when every codepoint in *text* is in the language grapheme_inventory."""
    if not allowed:
        return True
    return all(ch in allowed for ch in text)


def _atom_base(atom: str) -> str:
    """Return the inventory-check base of a phoneme atom.

    Piper/espeak may emit canonically equivalent IPA in either precomposed or
    decomposed Unicode forms. Normalize to NFC before stripping stress, length,
    and true combining marks so equivalent phones hit the same inventory entry.
    """
    atom = unicodedata.normalize("NFC", atom)
    return "".join(ch for ch in atom if ch not in STRESS and ch != LENGTH and unicodedata.combining(ch) == 0)


def atoms_are_in_language(atoms: Sequence[str], allowed: Set[str]) -> bool:
    """所有原子的基底都在该语言音素白名单内则 True。allowed 为空 → 不校验（True）。"""
    if not allowed:
        return True
    for atom in atoms:
        base = _atom_base(atom)
        if base and base != SPACE_TOKEN and base not in allowed:
            return False
    return True


def assert_vocab_within_inventory(
    profile: "LanguageProfile", phoneme_counter: Counter, grapheme_counter: Counter
) -> None:
    """生成时对照元数据校验最终 vocab：音素 token 基底必须 ⊆ phoneme_inventory ∪ multi_char_phonemes，
    字素必须 ⊆ grapheme_inventory。任何越界都是数据/逻辑 bug，直接报错并列出越界项+频次，
    而不是把脏 token 塞进 vocab。inventory 为空则跳过（向后兼容）。"""
    problems: List[str] = []
    if profile.phoneme_inventory:
        # '?' 已按用户要求放开，视为合法音素 token，不计入越界。
        allowed = set(profile.phoneme_inventory) | set(profile.multi_char_phonemes) | {UNMAPPED_MARK}
        bad = {}
        for tok, cnt in phoneme_counter.items():
            if tok == SPACE_TOKEN:
                continue
            base = _atom_base(tok)
            if base and base not in allowed:
                bad[tok] = cnt
        if bad:
            top = sorted(bad.items(), key=lambda kv: (-kv[1], kv[0]))[:40]
            problems.append(
                "音素 vocab 出现 phoneme_inventory 之外的 token（基底越界）：\n    "
                + ", ".join(f"{t!r}(U+{'/'.join(f'{ord(c):04X}' for c in _atom_base(t))}, x{n})" for t, n in top)
            )
    if profile.grapheme_inventory:
        bad_g = {ch: cnt for ch, cnt in grapheme_counter.items() if ch not in profile.grapheme_inventory}
        if bad_g:
            top = sorted(bad_g.items(), key=lambda kv: (-kv[1], kv[0]))[:60]
            problems.append(
                "字素 vocab 出现 grapheme_inventory 之外的字符：\n    "
                + ", ".join(f"{c!r}(U+{ord(c):04X}, x{n})" for c, n in top)
            )
    if problems:
        raise RuntimeError(
            "Vocab 元数据校验失败——生成逻辑或输入数据把不该出现的符号带进了 vocab。\n"
            "请修数据清洗（prepare 的 char_policy）/ 音素闸门 / 或在 lang_config.json 补声明，"
            "而不是接受脏 vocab（注：'?' 已放开，不在此列）：\n- " + "\n- ".join(problems)
        )


# Word-boundary token. The manifest ``text`` field stores phonemes concatenated within a
# word and a single space between words (e.g. "bˈɔɪ hˈaʊs"), which is exactly espeak-ng's
# native output shape and also the model's decoded output shape. A literal space is the
# boundary token, so it must be present as its own entry in vocab.txt. This mirrors
# SPACE_TOKEN in nemo/collections/common/tokenizers/ipa_symbol_tokenizer.py, whose
# longest-match text_to_tokens() re-derives the atomic tokens from this string at train time.
SPACE_TOKEN = " "

# piper_phonemize 对个别音素无法映射时输出的占位符。按用户要求单独放开：
# 当作合法原子保留进 vocab（不丢行、不越界报错），不学外语的其它杂音素仍照常丢弃/报错。
UNMAPPED_MARK = "?"


def is_attaching(ch: str) -> bool:
    """True for codepoints that always attach to the PRECEDING symbol: the length
    mark 'ː' and any true Unicode combining mark (nasalization ̃, syllabic ̩, ...)."""
    if ch == LENGTH:
        return True
    return unicodedata.combining(ch) != 0


def tokenize_phoneme_word(
    ipa_word: str, stress_mode: str = "attach", multi_char_phonemes: Sequence[str] = ()
) -> List[str]:
    """Longest-match (maximal munch) tokenizer for a SINGLE word's espeak-ng IPA string
    against *multi_char_phonemes* (a language-specific whitelist, longest-first -- see
    get_multi_char_phonemes_for_voice()/lang_config.json) plus generic
    combining-mark attachment (is_attaching(), language-agnostic). Returns one atomic
    phoneme unit per token, with stress marks folded into the unit they belong to
    (mode-dependent). No NFC here: the espeak/piper native Unicode form is preserved so
    tokens stay drop-in compatible with the original piper TTS phoneme_id_map."""
    units: List[str] = []
    pending_stress = ""
    join_next = False
    i, n = 0, len(ipa_word)

    while i < n:
        ch = ipa_word[i]

        if ch in TIES:
            # Explicit tie bar (rare for en-us, kept for other espeak locales): merge
            # the next matched symbol into the previous unit instead of starting a new one.
            join_next = True
            i += 1
            continue

        if ch in STRESS:
            if stress_mode == "separate":
                units.append(ch)
            elif stress_mode == "attach":
                pending_stress += ch
            # stress_mode == "drop": stress marks are already stripped by the caller.
            i += 1
            continue

        matched = ch
        for cand in multi_char_phonemes:
            if ipa_word.startswith(cand, i):
                matched = cand
                break
        i += len(matched)

        while i < n and is_attaching(ipa_word[i]):
            matched += ipa_word[i]
            i += 1

        if join_next and units:
            units[-1] += matched
            join_next = False
        else:
            units.append(pending_stress + matched)
            pending_stress = ""

    return units


def phonemes_to_text_and_atoms(
    phoneme_str: str, stress_mode: str = "attach", multi_char_phonemes: Sequence[str] = ()
) -> Tuple[str, List[str]]:
    """Turn a full (multi-word) espeak-ng phoneme string into:
      1. the manifest ``text`` value: phonemes concatenated WITHIN each word, single space
         BETWEEN words (e.g. "bˈɔɪ hˈaʊs") — the model's target/decoded shape; and
      2. the flat list of atomic phoneme units (no spaces) for building the phoneme vocab,
         so vocab.txt covers exactly the tokens the tokenizer's longest-match will produce.
    Word-internal concatenation is loss-free: "".join(units) reproduces the original word,
    so tokenize + concat round-trips (verified in run_self_check())."""
    word_strs: List[str] = []
    atoms: List[str] = []
    for word in phoneme_str.split():
        units = tokenize_phoneme_word(word, stress_mode, multi_char_phonemes)
        word_strs.append("".join(units))
        atoms.extend(units)
    return " ".join(word_strs), atoms


def run_self_check(profile: LanguageProfile) -> None:
    """Assert tokenize_phoneme_word() still treats every diphthong/affricate in the ACTIVE
    language's multi_char_phonemes whitelist (plus generic length-mark/combining-mark
    attachment) as a single atomic unit, and that word-internal concatenation round-trips the
    original word. Regression cases are pinned per language in lang_config.json's 'self_check'.

    This guards against the exact incident that once happened here: the multi-char atom
    whitelist got silently reduced to a flat single-character inventory, which decomposed "aɪ"
    -> "a" + "ɪ" (and similarly for other diphthongs/affricates), inflating phoneme token
    counts enough to silently drop huge amounts of valid rows via the CTC T>=U length filter.
    Any regression -- in the code OR in lang_config.json -- fails loudly here instead of
    surfacing as silent data loss only noticed after a full preprocessing + training run."""
    atoms = profile.multi_char_phonemes
    if not atoms:
        raise AssertionError(
            f"run_self_check: language {profile.key!r} (voice {profile.voice!r}) has an empty "
            f"multi_char_phonemes whitelist in lang_config.json -- without it, diphthongs/"
            f"affricates silently stop being merged into atomic tokens for this language."
        )
    if not profile.self_check:
        raise AssertionError(
            f"run_self_check: language {profile.key!r} has no 'self_check' cases in "
            f"lang_config.json; add a few word -> [ipa, expected-tokens] pins against real "
            f"espeak-ng output so atom-merging regressions fail loudly."
        )
    for word, (ipa, expected) in profile.self_check.items():
        got = tokenize_phoneme_word(ipa, multi_char_phonemes=atoms)
        if got != expected:
            raise AssertionError(
                f"tokenize_phoneme_word regression for {word!r} (lang={profile.key!r}, "
                f"ipa={ipa!r}): got {got}, expected {expected}. This language's "
                f"multi_char_phonemes entry or is_attaching() was likely modified in a way that "
                f"stops merging diphthongs/affricates/length marks into single atomic tokens -- "
                f"this silently inflates phoneme token counts and can drop huge amounts of "
                f"training data via the CTC T>=U filter."
            )
        # Word-internal concatenation must be loss-free (attach mode keeps stress in-string).
        if "".join(got) != ipa:
            raise AssertionError(
                f"tokenize_phoneme_word is not round-trippable for {word!r}: "
                f"''.join({got}) = {''.join(got)!r} != original {ipa!r}"
            )
    # Full-string shape: phonemes concatenated within a word, single space between words.
    pairs = list(profile.self_check.values())[:2]
    if len(pairs) >= 2:
        pair_str = f"{pairs[0][0]} {pairs[1][0]}"
        text, _ = phonemes_to_text_and_atoms(pair_str, multi_char_phonemes=atoms)
        if text != pair_str:
            raise AssertionError(
                f"phonemes_to_text_and_atoms text regression (lang={profile.key!r}): "
                f"got {text!r}, expected {pair_str!r}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess ipa-childes-split CSV into NeMo G2P manifest and vocab files. "
            "Only reads language code + raw text columns, then regenerates phonemes "
            "with piper-phonemize (espeak backend)."
        )
    )
    parser.add_argument("--input-csv", type=Path, required=True, help="ipa-childes-split style CSV path")
    parser.add_argument("--output-dir", type=Path, required=True, help="output directory")
    parser.add_argument("--text-field", type=str, default="sentence", help="raw text column name")
    parser.add_argument("--lang-field", type=str, default="espeak_lang_code", help="espeak language code column name")
    parser.add_argument(
        "--lang-config",
        type=Path,
        default=DEFAULT_LANG_CONFIG_PATH,
        help=(
            "JSON file holding EVERYTHING language-specific: per-language espeak voice, "
            "number-normalization locale, multi-character IPA phoneme-atom whitelist "
            "(diphthongs/affricates kept atomic by the longest-match tokenizer), and the "
            "self-check regression cases. Add / switch a language here, not by editing this "
            "script. Defaults to lang_config.json next to this script."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help=(
            "Active language key in --lang-config (e.g. 'de', 'en-us'). Selects the default "
            "voice, number-normalization locale, and the self-check cases. Defaults to the "
            "config's 'default_language'."
        ),
    )
    parser.add_argument(
        "--default-voice",
        type=str,
        default=None,
        help=(
            "Fallback espeak voice for rows whose lang field is empty. Defaults to the active "
            "language's 'voice' from --lang-config."
        ),
    )
    parser.add_argument("--workers", type=int, default=0, help="parallel workers; 0 means auto by CPU count")
    parser.add_argument(
        "--executor",
        choices=["thread", "process"],
        default="process",
        help="parallel backend; process is usually faster for very large datasets",
    )
    parser.add_argument("--batch-size", type=int, default=0, help="rows per worker batch; 0 means auto by CPU count")
    parser.add_argument("--limit", type=int, default=0, help="optional row limit")
    parser.add_argument("--stress", choices=["attach", "separate", "drop"], default="attach")
    parser.add_argument(
        "--strip-punctuation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Strip pause/sentence punctuation from text BEFORE phonemizing so neither "
            "text_graphemes nor the phoneme target contains punctuation tokens "
            "(pure G2P; pauses are inserted by the frontend). Keeps apostrophes/hyphens."
        ),
    )
    parser.add_argument(
        "--split-on-punctuation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Split each sentence at pause/sentence punctuation into segments and emit "
            "ONE manifest line per segment, matching the frontend that sends each "
            "segment to the G2P model separately (train==serve). Segments are always "
            "punctuation-free. Use the shared split_into_segments() in the frontend."
        ),
    )
    parser.add_argument(
        "--normalize-numbers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Expand digits/codes to spoken words (text_normalize.normalize_for_g2p) "
            "BEFORE phonemizing, because the grapheme vocab has no digits (raw digits "
            "would be dropped). MUST be applied identically in the inference frontend."
        ),
    )
    parser.add_argument(
        "--diacritize-arabic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply ar_XA local_nav_diacritizer (navigation rules + tashkeel fallback) "
            "before phonemizing. Default: on when --language is 'ar'. MUST match the "
            "inference frontend when training Arabic G2P."
        ),
    )
    parser.add_argument(
        "--diacritizer-rules",
        type=Path,
        default=None,
        help="Override nav_diacritizer rules.json (default: ar_XA/resources/nav_diacritizer_rules.json).",
    )
    parser.add_argument(
        "--diacritizer-letter-map",
        type=Path,
        default=None,
        help="Override latin_letter_readings JSON (default: ar_XA/resources/latin_letter_readings_ar_XA.json).",
    )
    parser.add_argument("--manifest-name", type=str, default="train.json")
    parser.add_argument("--phoneme-vocab-name", type=str, default="phoneme_vocab.txt")
    parser.add_argument("--grapheme-vocab-name", type=str, default="grapheme_vocab.txt")
    parser.add_argument("--merged-vocab-name", type=str, default="vocab.txt")
    parser.add_argument("--write-vocab", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


# Pause/sentence punctuation to strip for the "pure G2P + frontend-inserted pause"
# pipeline. We deliberately KEEP characters that change pronunciation rather than
# act as pauses: apostrophes (contractions: don't, it's) and hyphen (well-known).
_PUNCT_STRIP = frozenset('.,!?;:"()[]{}<>…—–«»‹›„“”/\\|*&^%$#@~`')


def strip_punctuation(text: str) -> str:
    """Remove pause/sentence punctuation so the espeak target carries no
    punctuation tokens. Punct is replaced by a space (not deleted) to avoid
    merging adjacent words, then whitespace is collapsed."""
    cleaned = "".join(" " if ch in _PUNCT_STRIP else ch for ch in text)
    return " ".join(cleaned.split())


def is_punct_unit(unit: str) -> bool:
    """A segmented phoneme unit that is purely strip-punctuation (defensive)."""
    return bool(unit) and all(ch in _PUNCT_STRIP for ch in unit)


# Pause/sentence boundaries the frontend splits on before sending each segment to
# the G2P model. THIS RULE MUST BE IDENTICAL in training data prep and inference
# frontend, otherwise the model sees different inputs at train vs serve time.
_SEGMENT_SPLIT_RE = re.compile(r"[.,!?;:…]+|[—–]+")


def split_into_segments(text: str) -> List[str]:
    """Split a (normalized) sentence at pause/sentence punctuation into the exact
    punctuation-free segments the frontend will send to the G2P model. Each segment
    is additionally cleaned of any residual (non-splitting) punctuation."""
    segments: List[str] = []
    for piece in _SEGMENT_SPLIT_RE.split(text):
        seg = strip_punctuation(piece)
        if seg:
            segments.append(seg)
    return segments


def normalize_voice(voice: str) -> str:
    return voice.strip().lower().replace("_", "-")


# 已探明可用的 espeak 语音缓存（每进程一份）。不同 espeak-ng 打包对语音代码支持不一：
# 有的接受 'fr' 却拒绝 'fr-fr'（同时又接受 'en-us'）。CSV lang 列或 lang_config 里两种
# 写法都可能出现，故解析时先按原样试、失败再退回主语言子标签（'fr-fr' → 'fr'）。
_VOICE_RESOLVE_CACHE: Dict[str, str] = {}


def resolve_espeak_voice(voice: str) -> str:
    """把请求的 espeak-ng 语音代码解析为 *当前这套 espeak-ng 真正接受* 的代码。

    先按原样尝试；失败则退回主语言子标签（如 'fr-fr' → 'fr'）。按进程缓存结果。
    整条链路都拿不到可用语音时报错（而不是让每一行 phonemize 崩掉）。"""
    v = normalize_voice(voice)
    cached = _VOICE_RESOLVE_CACHE.get(v)
    if cached is not None:
        return cached
    phonemize_espeak = _phonemize_espeak()
    candidates = [v]
    base = v.split("-", 1)[0]
    if base and base != v:
        candidates.append(base)
    for cand in candidates:
        try:
            phonemize_espeak("a", cand)
        except Exception:
            continue
        _VOICE_RESOLVE_CACHE[v] = cand
        return cand
    raise RuntimeError(
        f"espeak-ng rejected voice {voice!r} (tried {candidates!r}). Set a valid 'voice' in "
        f"lang_config.json or fix the CSV lang column; run `espeak-ng --voices` to list installed voices."
    )


# piper_phonemize 实现（espeak-ng 的 Python 封装）。phonemize_espeak(text, voice) 返回
# 句 → 词 → 音素符号的嵌套列表，扁平拼接即得词内相连、词间空格的 IPA（模型目标形态）。
# 注：piper 内置 IPA 映射表对个别未映射音素会输出 '?'，属预期，交下游按需处理。
@functools.lru_cache(maxsize=1)
def _phonemize_espeak():
    try:
        from piper_phonemize import phonemize_espeak
    except ImportError as e:
        raise RuntimeError(
            "piper_phonemize not installed. Install it (pip install piper-phonemize) — "
            "this preprocessor phonemizes via piper_phonemize.phonemize_espeak."
        ) from e
    return phonemize_espeak


def run_phonemize_batch(texts: Sequence[str], voice: str) -> List[str]:
    """用 piper_phonemize 逐条音素化整批文本，返回与输入 **1:1 对齐** 的 IPA 串（词内音素相连、
    词间单空格，即模型目标形态）。空文本回填空串。返回行可能含 '(en)' 语言切换标记或 '?'，交由
    调用方判定丢弃。"""
    if not texts:
        return []
    phonemize_espeak = _phonemize_espeak()
    # 解析出当前 espeak-ng 真正接受的语音代码（如 'fr-fr' → 'fr'），避免逐行崩溃。
    voice = resolve_espeak_voice(voice)
    out: List[str] = [""] * len(texts)
    for i, t in enumerate(texts):
        if not (t and t.strip()):
            continue
        clean = t.replace("\n", " ").replace("\r", " ")
        sentences = phonemize_espeak(clean, voice)  # List[sentence[List[phoneme symbol]]]
        flat = "".join("".join(part) for part in sentences)
        # 不做 NFC：保留 piper/espeak 原生 Unicode 形式（如 ç = c+组合 cedilla），
        # 使训练目标与原生 piper TTS 逐码点 drop-in 一致；piper 输出对同一音素是确定且
        # 自洽的（不会混用预组合/分解），故无需归一。
        out[i] = flat.strip().replace("\n", " ")
    return out


def _pool_worker_init_diacritizer(
    rules_path: Optional[str],
    letter_map_path: Optional[str],
) -> None:
    """ProcessPool worker hook: load diacritizer rules once per worker before any batch."""
    from ar_XA.diacritizer_frontend import get_local_nav_diacritizer

    get_local_nav_diacritizer(rules_path, letter_map_path)


def _prepare_row_segments(
    lang: str,
    raw: str,
    *,
    default_voice: str,
    diacritize_arabic: bool,
    diacritizer_rules: Optional[str],
    diacritizer_letter_map: Optional[str],
    normalize_nums: bool,
    number_locale: str,
    strip_punct: bool,
    split_punct: bool,
) -> List[Tuple[str, str, str]]:
    """Diacritize / TN / segment one CSV row -> (lang, grapheme, phonemize) triples."""
    raw_for_row = raw
    if diacritize_arabic and is_arabic_voice(lang or default_voice):
        raw_for_row = diacritize_arabic_line(
            raw,
            rules_path=diacritizer_rules,
            letter_map_path=diacritizer_letter_map,
        )

    # Grapheme input and phonemize input are the same normalized text
    # (no letter-"A" / "A-" special-casing): the model sees exactly the
    # text espeak phonemizes into the target.
    text = normalize_for_g2p(raw_for_row, locale=number_locale) if normalize_nums else raw_for_row
    text = normalize_text(text)

    if split_punct:
        return [(lang, seg, seg) for seg in split_into_segments(text) if seg]
    if strip_punct:
        seg = strip_punctuation(text)
        return [(lang, seg, seg)] if seg else []
    return [(lang, text, text)] if text else []


def iter_csv_raw_rows(
    csv_path: Path,
    text_field: str,
    lang_field: str,
    limit: int,
) -> Iterator[Tuple[str, str]]:
    """Fast CSV reader: yield ``(lang, raw_text)`` without diacritize/TN (done in workers)."""
    seen = 0
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        try:
            text_idx = header.index(text_field)
        except ValueError as exc:
            raise ValueError(f"text field {text_field!r} not found in header: {header}") from exc
        try:
            lang_idx = header.index(lang_field)
        except ValueError as exc:
            raise ValueError(f"lang field {lang_field!r} not found in header: {header}") from exc

        for row in reader:
            if text_idx >= len(row):
                continue
            raw = row[text_idx]
            if not raw or not raw.strip():
                continue
            lang = row[lang_idx].strip() if lang_idx < len(row) else ""
            yield (lang, raw)
            seen += 1
            if limit > 0 and seen >= limit:
                return


def iter_csv_rows(
    csv_path: Path,
    text_field: str,
    lang_field: str,
    limit: int,
    strip_punct: bool = False,
    split_punct: bool = False,
    normalize_nums: bool = False,
    number_locale: str = "en",
    diacritize_arabic: bool = False,
    default_voice: str = "",
    diacritizer_rules: Optional[Path] = None,
    diacritizer_letter_map: Optional[Path] = None,
) -> Iterator[Tuple[str, str, str]]:
    """Yield ``(lang, grapheme_text, phonemize_text)`` triples.

    *grapheme_text* is the model's input (clean TN, no digits) and
    *phonemize_text* (fed to piper/espeak) are identical: the same normalized
    text goes to both, so training input and phoneme target stay in lockstep.
    """
    rules_s = str(diacritizer_rules) if diacritizer_rules else None
    letter_s = str(diacritizer_letter_map) if diacritizer_letter_map else None
    seen = 0
    for lang, raw in iter_csv_raw_rows(csv_path, text_field, lang_field, limit=0):
        for triple in _prepare_row_segments(
            lang,
            raw,
            default_voice=default_voice,
            diacritize_arabic=diacritize_arabic,
            diacritizer_rules=rules_s,
            diacritizer_letter_map=letter_s,
            normalize_nums=normalize_nums,
            number_locale=number_locale,
            strip_punct=strip_punct,
            split_punct=split_punct,
        ):
            yield triple
            seen += 1
            if limit > 0 and seen >= limit:
                return


def iter_batches(rows: Iterator, batch_size: int) -> Iterator[List]:
    batch: List = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def approx_csv_body_line_count(csv_path: Path) -> int:
    """Fast (data lines ≈ newlines − header) for tqdm total; assumes no multiline CSV fields."""
    try:
        with csv_path.open("rb") as f:
            n_newlines = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1024 * 1024), b""))
    except OSError:
        return 0
    return max(0, n_newlines - 1)


def ordered_process_batches(
    executor: Executor,
    max_in_flight: int,
    batches_iter: Iterator[List],
    process_fn,
) -> Iterator[List[Tuple[str, str, List[str]]]]:
    """Like executor.map(process_fn, ...) but submit only max_in_flight tasks ahead; preserve order."""
    it = enumerate(batches_iter)
    pending: dict = {}
    saved: dict[int, List[Tuple[str, str, List[str]]]] = {}
    next_emit = 0
    exhausted = False

    def fill_pending() -> None:
        nonlocal exhausted
        while len(pending) < max_in_flight and not exhausted:
            try:
                idx, batch = next(it)
            except StopIteration:
                exhausted = True
                return
            fut = executor.submit(process_fn, batch)
            pending[fut] = idx

    def emit_ready() -> Iterator[List[Tuple[str, str, List[str]]]]:
        nonlocal next_emit
        while next_emit in saved:
            yield saved.pop(next_emit)
            next_emit += 1

    fill_pending()
    while pending or saved:
        if pending:
            done, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                idx = pending.pop(fut)
                saved[idx] = fut.result()
        yield from emit_ready()
        if exhausted and not pending and not saved:
            break
        fill_pending()


def process_batch(
    batch_rows: Sequence[Tuple[str, str, str]],
    default_voice: str,
    stress_mode: str,
    strip_punct: bool = False,
    inventories: Dict[str, Tuple[str, ...]] = None,
    phone_sets: Dict[str, Set[str]] = None,
    grapheme_sets: Dict[str, frozenset] = None,
) -> List[Tuple[str, str, List[str]]]:
    """Phonemize a batch of (lang, grapheme_text, phonemize_text) rows.

    Returns (grapheme_text, phoneme_text, atom_tokens) triples, where phoneme_text is the
    manifest ``text`` value (phonemes concatenated within a word, single space between words)
    and atom_tokens is the flat list of atomic phoneme units for building the phoneme vocab.

    音素级净化闸门（保证 text 侧只含目标语言音素）：整行丢弃（回填空三元组，main 会跳过）当
      * 字素含 grapheme_inventory 之外的字符（如方案二阿语训练集中的孤立拉丁字母）；
      * 输出含 '(' 语言切换标记（如 (en)…(de)）→ 该行含外语发音；
      * 任一原子基底不在该语言 phoneme_inventory 白名单内 → 混入的外语/杂音素。
    注：'?'（piper 无法映射的符号）已按需放开，视为合法原子保留，不丢行。
    """
    inventories = inventories or {}
    phone_sets = phone_sets or {}
    grapheme_sets = grapheme_sets or {}
    results: List[Tuple[str, str, List[str]]] = [("", "", [])] * len(batch_rows)
    by_voice: dict[str, List[Tuple[int, str, str]]] = {}
    for i, (lang, grapheme, ph_text) in enumerate(batch_rows):
        voice = normalize_voice(lang) if lang else default_voice
        if not voice:
            voice = default_voice
        allowed_g = grapheme_sets.get(voice) or grapheme_sets.get(voice.split("-", 1)[0]) or frozenset()
        if not graphemes_in_inventory(grapheme, allowed_g):
            continue
        by_voice.setdefault(voice, []).append((i, grapheme, ph_text))

    for voice, indexed_items in by_voice.items():
        # Language-specific multi-char atom whitelist (diphthongs/affricates/...), looked up
        # per voice from lang_config.json -- adding a new language never touches this
        # function. Unlisted voices still tokenize correctly via the generic combining-mark
        # rule alone (is_attaching()), just without any language-specific atom merging.
        multi_char_phonemes = get_multi_char_phonemes_for_voice(voice, inventories)
        allowed_phones = phone_sets.get(voice) or phone_sets.get(voice.split("-", 1)[0]) or set()
        # '?' 单独放开：piper 无法映射的符号当作允许原子，不因它丢行/越界。
        # 其余闸门保留：语言切换 (en) 行仍丢，真正的外语/杂音素仍按 inventory 丢。
        allowed_phones = (allowed_phones | {UNMAPPED_MARK}) if allowed_phones else allowed_phones
        phon_texts = [pt for _, _, pt in indexed_items]
        ipa_lines = run_phonemize_batch(phon_texts, voice=voice)
        for (src_idx, src_grapheme, _), ipa_line in zip(indexed_items, ipa_lines):
            phoneme_str = ipa_line.strip()
            if not phoneme_str:  # piper 无输出 → 回填空三元组（main 跳过）
                continue
            # 语言切换标记 (en)/(de) → 丢弃（不学外语发音）；'?' 放开保留。
            if "(" in phoneme_str:
                continue
            if stress_mode == "drop":
                phoneme_str = phoneme_str.replace(PRIMARY_STRESS, "").replace(SECONDARY_STRESS, "")

            # text = 词内音素拼接、词间单空格（piper/espeak 原生形态，也是模型输出形态）；
            # atoms = 用完整 IPA 音素表最长匹配切出的原子单元（双元音/塞擦音/长音符等保持不可再分）。
            phoneme_text, atoms = phonemes_to_text_and_atoms(
                phoneme_str, stress_mode=stress_mode, multi_char_phonemes=multi_char_phonemes
            )
            # 音素白名单校验：任一原子越界（外语/杂音素）→ 丢弃整行；'?' 已并入白名单，放行。
            if not atoms_are_in_language(atoms, allowed_phones):
                continue
            results[src_idx] = (src_grapheme, phoneme_text, atoms)

    return results


def process_raw_batch(
    batch_rows: Sequence[Tuple[str, str]],
    default_voice: str,
    stress_mode: str,
    strip_punct: bool,
    split_punct: bool,
    phoneme_strip_punct: bool,
    normalize_nums: bool,
    number_locale: str,
    diacritize_arabic: bool,
    diacritizer_rules: Optional[str],
    diacritizer_letter_map: Optional[str],
    inventories: Dict[str, Tuple[str, ...]],
    phone_sets: Dict[str, Set[str]],
    grapheme_sets: Dict[str, frozenset],
) -> List[Tuple[str, str, List[str]]]:
    """Expand raw CSV rows in-worker (diacritize/TN/segment), then phonemize."""
    segment_rows: List[Tuple[str, str, str]] = []
    for lang, raw in batch_rows:
        segment_rows.extend(
            _prepare_row_segments(
                lang,
                raw,
                default_voice=default_voice,
                diacritize_arabic=diacritize_arabic,
                diacritizer_rules=diacritizer_rules,
                diacritizer_letter_map=diacritizer_letter_map,
                normalize_nums=normalize_nums,
                number_locale=number_locale,
                strip_punct=strip_punct,
                split_punct=split_punct,
            )
        )
    return process_batch(
        segment_rows,
        default_voice,
        stress_mode,
        phoneme_strip_punct,
        inventories,
        phone_sets,
        grapheme_sets,
    )


def write_vocab(path: Path, tokens: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for tok in SPECIAL_TOKENS:
            f.write(tok + "\n")
        for tok in tokens:
            if tok and tok not in SPECIAL_TOKENS:
                f.write(tok + "\n")


def recommend_runtime(cpu_count: int) -> tuple[int, int]:
    """Return (workers, batch_size) tuned for large CSV preprocessing."""
    cpu_count = max(1, cpu_count)
    # Leave a small CPU headroom for OS / IO threads. espeak phonemize is CPU-bound,
    # so scale with the full core count (no artificial 24-worker cap) for large datasets.
    workers = max(1, cpu_count - 2)

    if cpu_count <= 8:
        batch_size = 2048
    elif cpu_count <= 16:
        batch_size = 4096
    else:
        batch_size = 8192
    return workers, batch_size


def main() -> None:
    args = parse_args()
    input_csv = args.input_csv.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lang_config_path = args.lang_config.expanduser().resolve()
    default_language, profiles = load_lang_config(str(lang_config_path))
    language = normalize_voice(args.language) if args.language else default_language
    if language not in profiles:
        raise ValueError(
            f"--language {language!r} is not configured in {lang_config_path}. "
            f"Available languages: {sorted(profiles)}"
        )
    profile = profiles[language]
    default_voice = normalize_voice(args.default_voice) if args.default_voice else profile.voice
    diacritize_arabic = args.diacritize_arabic
    if diacritize_arabic is None:
        diacritize_arabic = language == "ar"
    diacritizer_rules = args.diacritizer_rules.expanduser().resolve() if args.diacritizer_rules else None
    diacritizer_letter_map = (
        args.diacritizer_letter_map.expanduser().resolve() if args.diacritizer_letter_map else None
    )
    inventories = build_voice_inventories(profiles)
    phone_sets = build_voice_phone_sets(profiles)
    grapheme_sets = build_voice_grapheme_sets(profiles)
    run_self_check(profile)
    print(
        f"Language config: {lang_config_path} "
        f"(languages: {sorted(profiles)}; active: {language!r}, "
        f"voice: {profile.voice!r}, number locale: {profile.number_locale!r}, "
        f"diacritize_arabic: {diacritize_arabic})"
    )

    manifest_path = output_dir / args.manifest_name
    phoneme_vocab_path = output_dir / args.phoneme_vocab_name
    grapheme_vocab_path = output_dir / args.grapheme_vocab_name
    merged_vocab_path = output_dir / args.merged_vocab_name

    cpu_count = os.cpu_count() or 1
    auto_workers, auto_batch_size = recommend_runtime(cpu_count)
    workers = args.workers if args.workers > 0 else auto_workers
    batch_size = args.batch_size if args.batch_size > 0 else auto_batch_size
    if diacritize_arabic and args.batch_size <= 0:
        # Smaller raw-row batches -> more parallel diacritize jobs; phonemize still batched per worker.
        batch_size = min(batch_size, 256)
    executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor

    # Splitting always yields punctuation-free segments, so the phoneme-side defensive
    # filter must run whenever either stripping or splitting is on.
    effective_strip = args.strip_punctuation or args.split_on_punctuation

    rules_s = str(diacritizer_rules) if diacritizer_rules else None
    letter_s = str(diacritizer_letter_map) if diacritizer_letter_map else None
    process_fn = functools.partial(
        process_raw_batch,
        default_voice=default_voice,
        stress_mode=args.stress,
        strip_punct=args.strip_punctuation,
        split_punct=args.split_on_punctuation,
        phoneme_strip_punct=effective_strip,
        normalize_nums=args.normalize_numbers,
        number_locale=profile.number_locale,
        diacritize_arabic=diacritize_arabic,
        diacritizer_rules=rules_s,
        diacritizer_letter_map=letter_s,
        inventories=inventories,
        phone_sets=phone_sets,
        grapheme_sets=grapheme_sets,
    )
    pool_initializer = None
    pool_initargs: Tuple = ()
    if diacritize_arabic:
        pool_initializer = _pool_worker_init_diacritizer
        pool_initargs = (rules_s, letter_s)
        print(
            "Arabic diacritization: rules preloaded per worker; diacritize+phonemize run in parallel.",
            flush=True,
        )

    if args.show_progress:
        print("Preprocess: counting newlines in CSV for tqdm total…", file=sys.stderr, flush=True)
        if args.limit > 0:
            est_rows = args.limit
        else:
            est_rows = approx_csv_body_line_count(input_csv)
        est_batches = max(1, (est_rows + batch_size - 1) // batch_size) if est_rows > 0 else None
    else:
        est_batches = None

    phoneme_counter: Counter = Counter()
    grapheme_counter: Counter = Counter()
    processed_rows = 0
    seen_rows = 0  # 进入音素化的总行数（含被音素闸门丢弃的），用于报告丢弃率
    manifest_buffer: List[str] = []

    def flush_manifest_buffer(manifest_f) -> None:
        nonlocal manifest_buffer
        if not manifest_buffer:
            return
        manifest_f.write("\n".join(manifest_buffer))
        manifest_f.write("\n")
        manifest_buffer = []

    batch_iter = iter_batches(
        iter_csv_raw_rows(input_csv, args.text_field, args.lang_field, args.limit),
        batch_size,
    )

    with manifest_path.open("w", encoding="utf-8") as manifest_f:
        with executor_cls(max_workers=workers, initializer=pool_initializer, initargs=pool_initargs) as executor:
            ordered = ordered_process_batches(
                executor,
                max(1, workers * 2),
                batch_iter,
                process_fn,
            )
            if args.show_progress:
                ordered = tqdm(
                    ordered,
                    desc="Preprocess",
                    unit="batch",
                    total=est_batches,
                    dynamic_ncols=True,
                    mininterval=0.25,
                    disable=False,
                    file=sys.stderr,
                )

            for processed in ordered:
                for text, phoneme_text, atoms in processed:
                    seen_rows += 1
                    if not text or not phoneme_text:
                        continue
                    manifest_buffer.append(json.dumps({"text_graphemes": text, "text": phoneme_text}, ensure_ascii=False))
                    grapheme_counter.update(text)
                    # atoms 是最长匹配切出的原子单元（词内不含空格）；空格词边界单独统计一次，
                    # 保证它作为一个 token 进入音素词表（tokenizer 的 SPACE_TOKEN）。
                    phoneme_counter.update(atoms)
                    if " " in phoneme_text:
                        phoneme_counter.update([SPACE_TOKEN])
                    processed_rows += 1
                    if len(manifest_buffer) >= MANIFEST_WRITE_BUFFER_LINES:
                        flush_manifest_buffer(manifest_f)

            flush_manifest_buffer(manifest_f)

    if processed_rows == 0:
        raise RuntimeError(f"No valid rows processed from {input_csv}")

    if args.write_vocab:
        # 生成时按元数据校验：越界 token 直接报错（不产出脏 vocab）。见 lang_config.json 的
        # phoneme_inventory / grapheme_inventory。
        assert_vocab_within_inventory(profile, phoneme_counter, grapheme_counter)
        phoneme_tokens = [tok for tok, _ in sorted(phoneme_counter.items(), key=lambda kv: (-kv[1], kv[0]))]
        grapheme_tokens = sorted(grapheme_counter.keys())
        merged_tokens: List[str] = []
        seen = set(SPECIAL_TOKENS)
        for tok in phoneme_tokens:
            if tok not in seen:
                seen.add(tok)
                merged_tokens.append(tok)
        for tok in grapheme_tokens:
            if tok not in seen:
                seen.add(tok)
                merged_tokens.append(tok)

        write_vocab(phoneme_vocab_path, phoneme_tokens)
        write_vocab(grapheme_vocab_path, grapheme_tokens)
        write_vocab(merged_vocab_path, merged_tokens)

    dropped_rows = seen_rows - processed_rows
    drop_pct = (100.0 * dropped_rows / seen_rows) if seen_rows else 0.0
    print(f"Input CSV: {input_csv}")
    print(f"Processed rows: {processed_rows}")
    print(
        f"Phoneme-gate dropped rows: {dropped_rows} / {seen_rows} ({drop_pct:.2f}%) "
        f"[(en) 语言切换 / 越界外语音素；'?' 已放开保留]"
    )
    print(
        f"CPU: {cpu_count}, workers: {workers}, executor: {args.executor}, "
        f"batch size: {batch_size}"
    )
    print(f"Strip punctuation: {args.strip_punctuation} (no punctuation tokens in text_graphemes/text)")
    print(f"Split on punctuation: {args.split_on_punctuation} (one manifest line per segment; train==serve)")
    print(f"Normalize numbers: {args.normalize_numbers} (digits/codes -> words; MUST match frontend TN)")
    print(f"Manifest: {manifest_path}")
    if args.write_vocab:
        print(f"Phoneme vocab: {phoneme_vocab_path}")
        print(f"Grapheme vocab: {grapheme_vocab_path}")
        print(f"Merged vocab (for training): {merged_vocab_path}")
    else:
        print("Skipped vocab writing (--no-write-vocab).")


if __name__ == "__main__":
    main()
