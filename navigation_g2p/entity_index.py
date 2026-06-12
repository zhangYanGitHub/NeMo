"""O(1) entity picking by pre-indexing entities with remaining dedup capacity."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from dedup import DedupTracker, canonical_entity
from entity_sets import get_entity_pool

ENTITY_SLOTS = (
    "road_name", "poi_name", "city_name", "district_name", "route_name", "address_string",
)


@dataclass(frozen=True)
class _EntityEntry:
    raw: str
    canon: str
    entity_key: str


class EntityCapacityIndex:
    """Pre-built per-slot entity lists; pick is O(probe) not O(pool_size)."""

    def __init__(self, tracker: DedupTracker) -> None:
        self.tracker = tracker
        self._entries: Dict[str, List[_EntityEntry]] = {}
        self._available: Dict[str, List[_EntityEntry]] = {}
        self._rr: Dict[str, int] = defaultdict(int)
        self.rebuild_all()

    def rebuild_all(self) -> None:
        for slot in ENTITY_SLOTS:
            self.rebuild_slot(slot)

    def rebuild_slot(self, slot: str) -> None:
        pool = get_entity_pool(slot)
        cap = self.tracker.caps.max_per_entity
        entries: List[_EntityEntry] = []
        available: List[_EntityEntry] = []
        for raw in pool:
            canon = canonical_entity(raw, slot)
            key = f"{slot}::{canon}"
            entry = _EntityEntry(raw=raw, canon=canon, entity_key=key)
            entries.append(entry)
            if self.tracker._entity_counts[key] < cap:
                available.append(entry)
        self._entries[slot] = entries
        self._available[slot] = available

    def _entity_at_cap(self, entry: _EntityEntry) -> bool:
        return self.tracker._entity_counts[entry.entity_key] >= self.tracker.caps.max_per_entity

    def on_register(self, slots: Dict[str, str]) -> None:
        """Remove entities from available list when they hit entity_cap."""
        cap = self.tracker.caps.max_per_entity
        for slot, raw in slots.items():
            if slot not in self._available:
                continue
            canon = canonical_entity(raw, slot)
            key = f"{slot}::{canon}"
            if self.tracker._entity_counts[key] < cap:
                continue
            avail = self._available[slot]
            self._available[slot] = [e for e in avail if e.canon != canon]

    def pick(
        self,
        slot: str,
        template_id: str,
        uncovered_canons: Set[str],
        max_probes: int = 48,
    ) -> Optional[str]:
        avail = self._available.get(slot)
        if not avail:
            return None

        n = len(avail)
        probes = min(n, max_probes)
        uncovered_hits: List[_EntityEntry] = []
        covered_hits: List[_EntityEntry] = []

        for _ in range(probes):
            entry = avail[self._rr[slot] % n]
            self._rr[slot] += 1
            if self._entity_at_cap(entry):
                continue
            if not self.tracker.template_entity_has_capacity(template_id, slot, entry.raw):
                continue
            if entry.canon in uncovered_canons:
                uncovered_hits.append(entry)
            else:
                covered_hits.append(entry)

        if uncovered_hits:
            return uncovered_hits[0].raw
        if covered_hits:
            return covered_hits[0].raw
        return None
