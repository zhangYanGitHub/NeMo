#!/usr/bin/env python3
"""Shared Arabic diacritization frontend for G2P preprocess + ONNX inference.

Uses the same ``local_nav_diacritizer.LocalNavDiacritizer`` as
``vivid-ai-app-tntts-training/run_ar_XA_diacritized.sh``:

    python tools/ar_XA/local_nav_diacritizer.py diacritize \\
      --rules .../exp1_all_train_test_all_min_conf_0.6_ngram_with_punct/rules.json \\
      --letter-map tools/ar_XA/latin_letter_readings_ar_XA.json

Default resource paths live under ``examples/dataset/ar_XA/resources/``.
"""
from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_AR_XA_DIR = Path(__file__).resolve().parent
_DEFAULT_RULES = _AR_XA_DIR / "resources" / "nav_diacritizer_rules.json"
_DEFAULT_LETTER_MAP = _AR_XA_DIR / "resources" / "latin_letter_readings_ar_XA.json"
_CONFIG_PATH = _AR_XA_DIR / "diacritizer_config.json"


def default_diacritizer_paths() -> Tuple[Path, Path]:
    """Return (rules_json, letter_map_json) — overridable via diacritizer_config.json."""
    rules = _DEFAULT_RULES
    letter_map = _DEFAULT_LETTER_MAP
    if _CONFIG_PATH.is_file():
        raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        if raw.get("rules"):
            rules = (_AR_XA_DIR / raw["rules"]).resolve()
        if raw.get("letter_map"):
            letter_map = (_AR_XA_DIR / raw["letter_map"]).resolve()
    return rules, letter_map


def is_arabic_voice(voice_or_lang: str) -> bool:
    if not voice_or_lang:
        return False
    key = voice_or_lang.strip().lower().replace("_", "-")
    if key in ("ar", "ar-xa"):
        return True
    return key.startswith("ar-")


def _engine_cache_path(rules_p: Path) -> Path:
    return rules_p.with_name(f"{rules_p.stem}.engine.pkl")


def _load_diacritizer_engine(rules_p: Path, letter_p: Path):
    """Load ``LocalNavDiacritizer``, using a pickle sidecar to skip JSON parse + ngram prep."""
    from ar_XA.local_nav_diacritizer import LocalNavDiacritizer, load_letter_map

    cache_p = _engine_cache_path(rules_p)
    rules_mtime = rules_p.stat().st_mtime
    letter_mtime = letter_p.stat().st_mtime
    letter_key = str(letter_p)
    if cache_p.is_file():
        try:
            with cache_p.open("rb") as f:
                cached_rules_mtime, cached_letter_mtime, cached_letter_key, engine = pickle.load(f)
            if (
                cached_rules_mtime == rules_mtime
                and cached_letter_mtime == letter_mtime
                and cached_letter_key == letter_key
            ):
                return engine
        except (pickle.PickleError, EOFError, ValueError, TypeError):
            pass

    rules: Dict[str, Any] = json.loads(rules_p.read_text(encoding="utf-8"))
    letter_map = load_letter_map(letter_p)
    engine = LocalNavDiacritizer(rules, letter_map)
    try:
        with cache_p.open("wb") as f:
            pickle.dump((rules_mtime, letter_mtime, letter_key, engine), f, protocol=pickle.HIGHEST_PROTOCOL)
    except OSError:
        pass
    return engine


@lru_cache(maxsize=4)
def get_local_nav_diacritizer(
    rules_path: Optional[str] = None,
    letter_map_path: Optional[str] = None,
):
    """Lazy singleton ``LocalNavDiacritizer`` (rules + letter map loaded once per process)."""
    default_rules, default_letter = default_diacritizer_paths()
    rules_p = Path(rules_path).expanduser().resolve() if rules_path else default_rules
    letter_p = Path(letter_map_path).expanduser().resolve() if letter_map_path else default_letter
    if not rules_p.is_file():
        raise FileNotFoundError(f"Arabic diacritizer rules not found: {rules_p}")
    if not letter_p.is_file():
        raise FileNotFoundError(f"Arabic letter map not found: {letter_p}")
    return _load_diacritizer_engine(rules_p, letter_p)


def reset_diacritizer_cache() -> None:
    get_local_nav_diacritizer.cache_clear()


def diacritize_arabic_line(
    text: str,
    *,
    rules_path: Optional[str] = None,
    letter_map_path: Optional[str] = None,
) -> str:
    """Apply navigation diacritizer to one line (plain or partially vocalized Arabic)."""
    if not text or not text.strip():
        return text
    engine = get_local_nav_diacritizer(rules_path, letter_map_path)
    return engine.diacritize_line(text)
