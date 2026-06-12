"""Shared utilities for navigation G2P dataset generation."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from normalizer import infer_abbrev_role, normalize_graphemes
from slot_values import fill_slot
from templates import Template


@dataclass
class SampleRecord:
    """A single generated dataset record."""

    text_graphemes: str
    text: str
    template_type: str
    template_id: str
    slots: Dict[str, str] = field(default_factory=dict)
    split: str = ""
    source_generator: str = "template_v1"
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_error: Optional[str] = None

    def to_json(self, with_meta: bool = False) -> Dict[str, Any]:
        if with_meta:
            return {
                "text_graphemes": self.text_graphemes,
                "text": self.text,
                "template_type": self.template_type,
                "template_id": self.template_id,
                "slots": self.slots,
                "split": self.split,
                "source_generator": self.source_generator,
                **({"metadata": self.metadata} if self.metadata else {}),
            }
        return {"text_graphemes": self.text_graphemes, "text": self.text}

    def dedupe_key(self) -> str:
        return normalize_key(self.text_graphemes)

    def entity_keys(self) -> Dict[str, str]:
        keys: Dict[str, str] = {}
        entity_slots = (
            "road_name", "poi_name", "city_name", "district_name",
            "route_name", "address_string", "street_name",
        )
        for slot in entity_slots:
            if slot in self.slots and self.slots[slot]:
                keys[slot] = normalize_key(self.slots[slot])
        return keys


def normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def word_count(text: str) -> int:
    return len(text.split())


def text_length_bucket(text: str, short_max: int = 8, long_min: int = 18) -> str:
    wc = word_count(text)
    if wc <= short_max:
        return "short"
    if wc >= long_min:
        return "long"
    return "medium"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stable_hash(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)


def choose_weighted(rng: random.Random, weights: Dict[str, float]) -> str:
    items = list(weights.items())
    total = sum(w for _, w in items)
    if total <= 0:
        raise ValueError("Weights must sum to a positive value")
    r = rng.random() * total
    upto = 0.0
    for key, weight in items:
        upto += weight
        if r <= upto:
            return key
    return items[-1][0]


def render_template(
    template: Template,
    rng: random.Random,
    slot_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, str]]:
    slots: Dict[str, str] = {}
    overrides = slot_overrides or {}
    for slot in template.required_slots:
        slots[slot] = fill_slot(slot, rng, overrides)
    text = template.pattern
    for slot_name, value in slots.items():
        text = text.replace("{" + slot_name + "}", value)
    return text, slots


def finalize_graphemes(
    text: str,
    slots: Dict[str, str],
    *,
    expand_abbrev: bool,
    abbrev_strategy=None,
    spoken_numbers: bool = True,
) -> str:
    role = infer_abbrev_role(slots)
    return normalize_graphemes(
        text,
        expand_abbrev=expand_abbrev,
        abbrev_role=role,
        abbrev_strategy=abbrev_strategy,
        spoken_numbers=spoken_numbers,
    )


def dedupe_records(records: List[SampleRecord]) -> Tuple[List[SampleRecord], int]:
    seen: Set[str] = set()
    unique: List[SampleRecord] = []
    dup_count = 0
    for rec in records:
        key = rec.dedupe_key()
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        unique.append(rec)
    return unique, dup_count


def preview_records(records: List[SampleRecord], n: int) -> None:
    for rec in records[:n]:
        print(f"G: {rec.text_graphemes}")
        print(f"I: {rec.text}")
        print(f"  type={rec.template_type} id={rec.template_id} split={rec.split}")
        print()
