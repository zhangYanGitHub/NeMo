"""Dedup keys, entity canonicalization, and generation-time near-duplicate tracking."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from utils import SampleRecord, normalize_key

try:
    from templates import TEMPLATE_ID_COOLDOWN_IDS, TEMPLATE_ID_SOFT_CAP_RATIO
except ImportError:
    TEMPLATE_ID_COOLDOWN_IDS = frozenset({"rn_002"})
    TEMPLATE_ID_SOFT_CAP_RATIO = 0.028

ENTITY_SLOTS = frozenset({
    "road_name", "poi_name", "city_name", "district_name",
    "route_name", "address_string", "street_name",
})
NUMERIC_SLOTS = frozenset({
    "distance_phrase", "distance_value", "distance_unit",
    "exit_no", "ordinal", "address_number",
})

_WORD_NUMBERS = re.compile(
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"first|second|third|fourth|fifth|sixth|seventh|eighth)\b",
    re.I,
)
_ORDINAL_RE = re.compile(r"\b\d+(st|nd|rd|th)\b", re.I)
_DIR_RE = re.compile(r"\b(north|south|east|west|n|s|e|w|ne|nw|se|sw)\b")
_RD_RE = re.compile(
    r"\b(st|street|ave|avenue|rd|road|blvd|boulevard|dr|drive|ln|lane|"
    r"ct|court|pl|place|pkwy|parkway|hwy|highway)\b",
)


def nfc_lower(text: str) -> str:
    return unicodedata.normalize("NFC", text.strip().lower())


def exact_key(text: str) -> str:
    """Byte-identical grapheme dedup after NFC + lower + space collapse."""
    return normalize_key(unicodedata.normalize("NFC", text))


def normalized_key(text: str) -> str:
    """Surface-form normalization for punctuation/spacing variants."""
    t = nfc_lower(text)
    t = re.sub(r"[.,;:'\"]", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def numeric_skeleton_key(text: str) -> str:
    """Collapse digits and spoken numbers — catches distance/exit swaps."""
    t = normalized_key(text)
    t = _WORD_NUMBERS.sub("#", t)
    t = _ORDINAL_RE.sub("#", t)
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _clean_canonical_tokens(t: str, *, keep_digits: bool) -> str:
    t = _DIR_RE.sub(" dir ", t)
    t = _RD_RE.sub(" rd ", t)
    if not keep_digits:
        t = re.sub(r"\d+", "#", t)
    t = re.sub(r"[^a-z0-9 ]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def canonical_route(name: str) -> str:
    """Route identity — preserve highway numbers (I-95 != I-101)."""
    return _clean_canonical_tokens(normalized_key(name), keep_digits=True)


def canonical_road(name: str) -> str:
    """Road identity — preserve street numbers (45th != 46th)."""
    return _clean_canonical_tokens(normalized_key(name), keep_digits=True)


def canonical_entity(name: str, slot: str = "") -> str:
    """Entity identity for split/leakage/coverage regardless of abbrev/compass."""
    if slot == "road_name":
        return canonical_road(name)
    if slot == "route_name" or re.match(r"^(i|us|sr|route|hwy|fm|tx)-?\d", normalized_key(name)):
        return canonical_route(name)
    return _clean_canonical_tokens(normalized_key(name), keep_digits=False)


def slot_skeleton_key(template_id: str, slots: Dict[str, str]) -> str:
    """Template + slot abstraction; numeric slots flattened to '#'."""
    parts = [template_id]
    for slot in sorted(slots):
        val = slots[slot]
        if slot in ENTITY_SLOTS:
            parts.append(f"{slot}={canonical_entity(val, slot)}")
        elif slot in NUMERIC_SLOTS:
            parts.append(f"{slot}=#")
        else:
            parts.append(f"{slot}={nfc_lower(val)}")
    return "|".join(parts)


@dataclass
class DedupCaps:
    """Per-key generation caps (nav_prod_v1 defaults)."""

    max_per_slot_skeleton: int = 1
    max_per_numeric_skeleton_per_entity: int = 2
    max_per_template_id_ratio: float = 0.03
    max_per_entity: int = 8
    max_per_template_entity_pair: int = 3


@dataclass
class DedupTracker:
    """Generation-time dedup state with capped near-duplicate rejection."""

    caps: DedupCaps = field(default_factory=DedupCaps)
    target_pool_size: int = 22000
    refinement_mode: Optional[str] = None

    _exact: Set[str] = field(default_factory=set)
    _normalized: Set[str] = field(default_factory=set)
    _slot_skeleton_counts: Counter = field(default_factory=Counter)
    _numeric_entity_counts: Counter = field(default_factory=Counter)
    _template_id_counts: Counter = field(default_factory=Counter)
    _entity_counts: Counter = field(default_factory=Counter)
    _template_entity_counts: Counter = field(default_factory=Counter)
    reject_stats: Counter = field(default_factory=Counter)
    _saved_entity_cap: Optional[int] = field(default=None, repr=False)

    def _template_id_cap(self) -> int:
        registered = sum(self._template_id_counts.values())
        denom = max(self.target_pool_size, registered, 1)
        return max(1, int(self.caps.max_per_template_id_ratio * denom))

    def _template_id_soft_cap(self) -> int:
        registered = sum(self._template_id_counts.values())
        denom = max(self.target_pool_size, registered, 1)
        return max(1, int(TEMPLATE_ID_SOFT_CAP_RATIO * denom))

    def begin_refinement(self, mode: str) -> None:
        self.refinement_mode = mode
        if mode in {"road_name_coverage", "poi_name_coverage", "route_name_coverage"}:
            self._saved_entity_cap = self.caps.max_per_entity
            self.caps.max_per_entity = 16
        elif mode in {"entity_heavy", "long_entity_heavy", "entity_heavy_final"}:
            self._saved_entity_cap = self.caps.max_per_entity
            self.caps.max_per_entity = 12

    def end_refinement(self) -> None:
        if self._saved_entity_cap is not None:
            self.caps.max_per_entity = self._saved_entity_cap
            self._saved_entity_cap = None
        self.refinement_mode = None

    def template_has_capacity(self, template_id: str) -> bool:
        count = self._template_id_counts[template_id]
        if template_id in TEMPLATE_ID_COOLDOWN_IDS and count >= self._template_id_soft_cap():
            return False
        return count < self._template_id_cap()

    def entity_has_capacity(self, etype: str, raw_value: str) -> bool:
        cent = f"{etype}::{canonical_entity(raw_value, etype)}"
        return self._entity_counts[cent] < self.caps.max_per_entity

    def template_entity_has_capacity(self, template_id: str, etype: str, raw_value: str) -> bool:
        cent = f"{etype}::{canonical_entity(raw_value, etype)}"
        te_key = f"{template_id}|{cent}"
        return self._template_entity_counts[te_key] < self.caps.max_per_template_entity_pair

    def slots_have_capacity(self, template_id: str, slots: Dict[str, str]) -> bool:
        """Preview cap headroom (ignores exact/normalized grapheme duplicates)."""
        if not self.template_has_capacity(template_id):
            return False
        if self._refinement_bypass_caps(slots):
            return True
        sk = slot_skeleton_key(template_id, slots)
        if self._slot_skeleton_counts[sk] >= self.caps.max_per_slot_skeleton:
            return False
        for etype, val in slots.items():
            if etype in ENTITY_SLOTS:
                if not self.entity_has_capacity(etype, val):
                    return False
                if not self.template_entity_has_capacity(template_id, etype, val):
                    return False
        return True

    def _refinement_bypass_caps(self, slots: Dict[str, str]) -> bool:
        """During entity/difficulty refinement, allow new entity+template combos."""
        if not self.refinement_mode:
            return False
        if self.refinement_mode in {
            "poi_name_coverage", "route_name_coverage", "road_name_coverage",
            "long_sentence", "long_entity_heavy", "entity_heavy", "entity_heavy_final",
            "long_tail", "near_duplicate_high",
        }:
            return True
        return False

    def should_reject(self, graphemes: str, template_id: str, slots: Dict[str, str]) -> Optional[str]:
        """Return rejection reason, or None if sample is acceptable."""
        reason: Optional[str] = None
        ek = exact_key(graphemes)
        if ek in self._exact:
            reason = "exact_duplicate"

        if reason is None:
            nk = normalized_key(graphemes)
            if nk in self._normalized:
                reason = "normalized_duplicate"

        # template_id cap always enforced (RULE_09)
        if reason is None and self._template_id_counts[template_id] >= self._template_id_cap():
            reason = "template_id_cap"

        if reason is None and not self._refinement_bypass_caps(slots):
            sk = slot_skeleton_key(template_id, slots)
            if self._slot_skeleton_counts[sk] >= self.caps.max_per_slot_skeleton:
                reason = "slot_skeleton_duplicate"

            # numeric skeleton scoped to primary entity (road/route/poi/address)
            if reason is None:
                nsk = numeric_skeleton_key(graphemes)
                primary_entity = self._primary_entity(slots)
                if primary_entity:
                    ne_key = f"{template_id}|{primary_entity}|{nsk}"
                    if self._numeric_entity_counts[ne_key] >= self.caps.max_per_numeric_skeleton_per_entity:
                        reason = "numeric_skeleton_duplicate"

            if reason is None:
                for etype, val in slots.items():
                    if etype in ENTITY_SLOTS:
                        cent = f"{etype}::{canonical_entity(val, etype)}"
                        if self._entity_counts[cent] >= self.caps.max_per_entity:
                            reason = "entity_cap"
                            break
                        te_key = f"{template_id}|{cent}"
                        if self._template_entity_counts[te_key] >= self.caps.max_per_template_entity_pair:
                            reason = "template_entity_cap"
                            break

        self.reject_stats[reason or "accepted"] += 1
        return reason

    def register(self, graphemes: str, template_id: str, slots: Dict[str, str]) -> None:
        self._exact.add(exact_key(graphemes))
        self._normalized.add(normalized_key(graphemes))
        sk = slot_skeleton_key(template_id, slots)
        self._slot_skeleton_counts[sk] += 1

        nsk = numeric_skeleton_key(graphemes)
        primary_entity = self._primary_entity(slots)
        if primary_entity:
            ne_key = f"{template_id}|{primary_entity}|{nsk}"
            self._numeric_entity_counts[ne_key] += 1

        self._template_id_counts[template_id] += 1
        for etype, val in slots.items():
            if etype in ENTITY_SLOTS:
                cent = f"{etype}::{canonical_entity(val, etype)}"
                self._entity_counts[cent] += 1
                self._template_entity_counts[f"{template_id}|{cent}"] += 1

    @staticmethod
    def _primary_entity(slots: Dict[str, str]) -> Optional[str]:
        for etype in ("road_name", "route_name", "poi_name", "address_string",
                      "city_name", "district_name", "street_name"):
            if etype in slots and slots[etype]:
                return f"{etype}::{canonical_entity(slots[etype], etype)}"
        return None


def compute_dedup_audit(records: List[SampleRecord]) -> Dict:
    """Post-hoc dedup audit for validator / summary.json."""
    n = len(records) or 1
    exact_c: Counter = Counter()
    norm_c: Counter = Counter()
    num_c: Counter = Counter()
    slot_c: Counter = Counter()

    for rec in records:
        exact_c[exact_key(rec.text_graphemes)] += 1
        norm_c[normalized_key(rec.text_graphemes)] += 1
        num_c[numeric_skeleton_key(rec.text_graphemes)] += 1
        slot_c[slot_skeleton_key(rec.template_id, rec.slots)] += 1

    def _dup_rate(counter: Counter) -> float:
        dup_samples = sum(c - 1 for c in counter.values() if c > 1)
        return dup_samples / n

    unique_slot_skeletons = len(slot_c)
    effective_unique_ratio = unique_slot_skeletons / n

    top_slot = [{"skeleton": k, "count": v} for k, v in slot_c.most_common(10)]
    top_numeric = [{"skeleton": k, "count": v} for k, v in num_c.most_common(10)]

    per_template_id = Counter(rec.template_id for rec in records)
    max_tid_share = max(per_template_id.values()) / n if per_template_id else 0.0

    entity_counts: Counter = Counter()
    for rec in records:
        for etype, val in rec.entity_keys().items():
            entity_counts[f"{etype}::{canonical_entity(val, etype)}"] += 1
    per_entity_max = dict(entity_counts.most_common(10))

    return {
        "exact_duplicate_rate": _dup_rate(exact_c),
        "normalized_duplicate_rate": _dup_rate(norm_c),
        "near_duplicate_rate_numeric_skeleton": _dup_rate(num_c),
        "near_duplicate_rate_slot_skeleton": _dup_rate(slot_c),
        "effective_unique_ratio": round(effective_unique_ratio, 4),
        "unique_slot_skeletons": unique_slot_skeletons,
        "top_slot_skeletons": top_slot,
        "top_numeric_skeletons": top_numeric,
        "per_template_id_max_share": round(max_tid_share, 4),
        "per_entity_max_count": per_entity_max,
    }
