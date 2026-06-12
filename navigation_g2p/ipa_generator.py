"""IPA generation backends with a pluggable interface."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

from normalizer import normalize_ipa

logger = logging.getLogger(__name__)


class IpaGenerationError(Exception):
    """Raised when IPA generation fails after retries."""


def default_ipa_workers(requested: int = 0) -> int:
    """Parallel espeak workers. Default: CPU cores + 4 (good for subprocess I/O)."""
    if requested > 0:
        return requested
    return (os.cpu_count() or 4) + 4


def _default_workers(requested: int) -> int:
    return default_ipa_workers(requested)


def _run_espeak_subprocess(
    text: str,
    espeak_cmd: str,
    locale: str,
    ipa_mode: int,
    timeout_sec: float,
) -> str:
    cmd = [espeak_cmd, "-v", locale, f"--ipa={ipa_mode}", "-q", text]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_sec, check=False,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise IpaGenerationError(
            f"espeak-ng failed (code={proc.returncode}) for text={text!r}: {stderr}"
        )
    raw = (proc.stdout or "").strip()
    if not raw:
        stderr = (proc.stderr or "").strip()
        raise IpaGenerationError(f"espeak-ng empty IPA for text={text!r}: {stderr}")
    return normalize_ipa(raw)


class BaseIpaGenerator(ABC):
    """Abstract IPA generator interface."""

    @abstractmethod
    def generate(self, text: str, locale: Optional[str] = None) -> str:
        raise NotImplementedError

    def generate_batch(
        self,
        texts: Sequence[str],
        locale: Optional[str] = None,
        workers: int = 0,
    ) -> List[Optional[str]]:
        return [self._safe_generate(t, locale) for t in texts]

    def _safe_generate(self, text: str, locale: Optional[str] = None) -> Optional[str]:
        try:
            return self.generate(text, locale=locale)
        except IpaGenerationError:
            return None

    @abstractmethod
    def health_check(self) -> bool:
        raise NotImplementedError


class EspeakIpaGenerator(BaseIpaGenerator):
    """Generate IPA via espeak-ng subprocess calls (parallel batch supported)."""

    def __init__(
        self,
        espeak_cmd: str = "espeak-ng",
        locale: str = "en-us",
        timeout_sec: float = 10.0,
        retries: int = 1,
        ipa_mode: int = 3,
        workers: int = 0,
    ):
        self.espeak_cmd = espeak_cmd
        self.locale = locale
        self.timeout_sec = timeout_sec
        self.retries = retries
        self.ipa_mode = ipa_mode
        self.workers = workers

    def _run_espeak(self, text: str, locale: str) -> str:
        return _run_espeak_subprocess(
            text, self.espeak_cmd, locale, self.ipa_mode, self.timeout_sec,
        )

    def generate(self, text: str, locale: Optional[str] = None) -> str:
        loc = locale or self.locale
        last_err: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                return self._run_espeak(text, loc)
            except (IpaGenerationError, subprocess.TimeoutExpired, OSError) as exc:
                last_err = exc
                if attempt < self.retries:
                    time.sleep(0.02 * (attempt + 1))
        raise IpaGenerationError(f"IPA generation failed after retries: {last_err}")

    def generate_batch(
        self,
        texts: Sequence[str],
        locale: Optional[str] = None,
        workers: int = 0,
    ) -> List[Optional[str]]:
        if not texts:
            return []
        loc = locale or self.locale
        n_workers = _default_workers(workers or self.workers)
        if n_workers <= 1 or len(texts) == 1:
            return super().generate_batch(texts, locale=loc, workers=1)

        results: List[Optional[str]] = [None] * len(texts)

        def _task(idx_text: Tuple[int, str]) -> Tuple[int, Optional[str]]:
            idx, text = idx_text
            try:
                return idx, self.generate(text, locale=loc)
            except IpaGenerationError:
                return idx, None

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_task, (i, t)) for i, t in enumerate(texts)]
            for fut in as_completed(futures):
                idx, ipa = fut.result()
                results[idx] = ipa
        return results

    def health_check(self) -> bool:
        try:
            return bool(self.generate("test", locale=self.locale))
        except Exception:
            return False


class DictionaryBackedIpaGenerator(BaseIpaGenerator):
    """Dictionary lookup IPA generator (stub for future custom lexicon)."""

    def __init__(self, lexicon: Dict[str, str], fallback: Optional[BaseIpaGenerator] = None):
        self.lexicon = lexicon
        self.fallback = fallback

    def generate(self, text: str, locale: Optional[str] = None) -> str:
        if text in self.lexicon:
            return normalize_ipa(self.lexicon[text])
        if self.fallback:
            return self.fallback.generate(text, locale=locale)
        raise IpaGenerationError(f"No lexicon entry for: {text!r}")

    def health_check(self) -> bool:
        return bool(self.lexicon) or (self.fallback is not None and self.fallback.health_check())


class NeuralIpaGenerator(BaseIpaGenerator):
    """Neural G2P model backend (stub for future on-device model)."""

    def __init__(self, model_path: str, fallback: Optional[BaseIpaGenerator] = None):
        self.model_path = model_path
        self.fallback = fallback
        self._loaded = False

    def _load_model(self) -> None:
        self._loaded = True

    def generate(self, text: str, locale: Optional[str] = None) -> str:
        if not self._loaded:
            self._load_model()
        if self.fallback:
            return self.fallback.generate(text, locale=locale)
        raise IpaGenerationError("NeuralIpaGenerator is not configured with a fallback model")

    def health_check(self) -> bool:
        return self.fallback is not None and self.fallback.health_check()


def create_ipa_generator(
    backend: str = "espeak",
    espeak_cmd: str = "espeak-ng",
    locale: str = "en-us",
    timeout_sec: float = 10.0,
    retries: int = 1,
    workers: int = 0,
) -> BaseIpaGenerator:
    if backend == "espeak":
        return EspeakIpaGenerator(
            espeak_cmd=espeak_cmd,
            locale=locale,
            timeout_sec=timeout_sec,
            retries=retries,
            workers=workers,
        )
    raise ValueError(f"Unsupported IPA backend: {backend}")
