"""Default configuration for navigation G2P dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GeneratorConfig:
    """Runtime configuration for dataset generation."""

    seed: int = 42
    locale: str = "en-us"
    espeak_cmd: str = "espeak-ng"
    espeak_timeout_sec: float = 10.0
    espeak_retries: int = 1
    ipa_workers: int = 0  # 0 = auto (cpu_count * 2, cap 32)
    ipa_batch_size: int = 256
    fast_mode: bool = False
    expand_abbrev: bool = True
    dedupe: bool = True
    max_text_len: int = 300
    max_ipa_len: int = 500
    preview: int = 0
    save_metadata: bool = True
    output_dir: str = "output"
    dataset_profile: str = "nav_prod_v1"
    spoken_numbers: bool = True

    # Generation tuning
    max_generation_attempts_multiplier: float = 10.0
    skip_refinement: bool = False
    min_entity_coverage: int = 50

    # Length distribution targets (short / medium / long)
    length_distribution: tuple[float, float, float] = (0.25, 0.50, 0.25)
    short_text_max_words: int = 8
    long_text_min_words: int = 18

    # Abbreviation expansion strategy per slot role
    abbrev_strategy_overrides: dict[str, str] = field(default_factory=dict)


DEFAULT_CONFIG = GeneratorConfig()
