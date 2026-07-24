"""Arabic (ar_XA) navigation diacritizer — same implementation as vivid-ai-app-tntts-training."""

from ar_XA.diacritizer_frontend import (
    default_diacritizer_paths,
    diacritize_arabic_line,
    get_local_nav_diacritizer,
    is_arabic_voice,
    reset_diacritizer_cache,
)

__all__ = [
    "default_diacritizer_paths",
    "diacritize_arabic_line",
    "get_local_nav_diacritizer",
    "is_arabic_voice",
    "reset_diacritizer_cache",
]
