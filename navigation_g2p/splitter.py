"""Entity-aware dataset splitting with leakage reporting."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple  # noqa: F401 used by _primary_entity_key

from dedup import canonical_entity, slot_skeleton_key
from entity_sets import FOREIGN_STYLE_NAMES
from utils import SampleRecord, stable_hash, word_count

LONG_TAIL_TOKENS = {t.lower() for t in FOREIGN_STYLE_NAMES}
PRIMARY_ENTITY_ORDER = (
    "road_name", "route_name", "poi_name", "address_string", "city_name", "district_name",
)

ENTITY_SLOTS_FOR_HEAVY = frozenset({
    "road_name", "poi_name", "city_name", "district_name", "route_name", "address_string",
})


ENTITY_SLOT_TYPES = (
    "road_name",
    "poi_name",
    "city_name",
    "district_name",
    "route_name",
    "address_string",
)


def _normalize_split_ratios(
    train_ratio: float, val_ratio: float, test_ratio: float,
) -> Tuple[float, float, float]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6 and total > 0:
        return train_ratio / total, val_ratio / total, test_ratio / total
    return train_ratio, val_ratio, test_ratio


def train_aligned_entity_slots(
    slots: Dict[str, str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> bool:
    """True if every filled entity hashes to the train bucket (reduces multi-entity conflicts)."""
    tr, vr, _ = _normalize_split_ratios(train_ratio, val_ratio, test_ratio)
    votes: List[str] = []
    for etype in ENTITY_SLOT_TYPES:
        val = slots.get(etype)
        if not val:
            continue
        gkey = f"{etype}::{canonical_entity(val, etype)}"
        bucket = (stable_hash(gkey) % 10000) / 10000.0
        if bucket < tr:
            votes.append("train")
        elif bucket < tr + vr:
            votes.append("val")
        else:
            votes.append("test")
    return bool(votes) and all(v == "train" for v in votes)


@dataclass
class LeakageReport:
    """Summary of entity leakage across splits."""

    leaked_entities: Dict[str, List[str]] = field(default_factory=dict)
    leakage_counts: Dict[str, int] = field(default_factory=dict)
    leakage_ratio: Dict[str, float] = field(default_factory=dict)
    total_entities_by_type: Dict[str, int] = field(default_factory=dict)
    weighted_leakage_ratio: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "leaked_entities": self.leaked_entities,
            "leakage_counts": self.leakage_counts,
            "leakage_ratio": self.leakage_ratio,
            "total_entities_by_type": self.total_entities_by_type,
            "weighted_leakage_ratio": self.weighted_leakage_ratio,
        }


@dataclass
class SplitResult:
    train: List[SampleRecord]
    val: List[SampleRecord]
    test: List[SampleRecord]
    leakage_report: LeakageReport
    template_distribution: Dict[str, Dict[str, int]]
    conflicts: List[SampleRecord] = field(default_factory=list)


def _canonical_entity_keys(rec: SampleRecord) -> Dict[str, str]:
    keys: Dict[str, str] = {}
    for etype in ENTITY_SLOT_TYPES:
        if etype in rec.slots and rec.slots[etype]:
            keys[etype] = canonical_entity(rec.slots[etype], etype)
    return keys


def _entity_groups(records: List[SampleRecord]) -> Dict[str, Set[int]]:
    groups: Dict[str, Set[int]] = defaultdict(set)
    for idx, rec in enumerate(records):
        for etype, value in _canonical_entity_keys(rec).items():
            key = f"{etype}::{value}"
            groups[key].add(idx)
    return groups


def _assign_group_to_split(
    group_key: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> str:
    h = stable_hash(group_key)
    bucket = (h % 10000) / 10000.0
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def _primary_entity_key(rec: SampleRecord) -> Optional[str]:
    """Single anchor entity for split — avoids multi-entity vote conflicts."""
    for etype in PRIMARY_ENTITY_ORDER:
        if etype in rec.slots and rec.slots[etype]:
            cent = canonical_entity(rec.slots[etype], etype)
            return f"{etype}::{cent}"
    return None


def _is_long_tail_sample(rec: SampleRecord) -> bool:
    text = rec.text_graphemes.lower()
    return any(tok in text for tok in LONG_TAIL_TOKENS)


def _assign_record_split(
    rec: SampleRecord,
    group_split: Dict[str, str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[str, bool]:
    """Return (split, is_conflict). Multi-entity vote mismatch -> conflict."""
    entity_keys = _canonical_entity_keys(rec)
    if len(entity_keys) >= 2:
        votes: Set[str] = set()
        for etype, value in entity_keys.items():
            gkey = f"{etype}::{value}"
            if gkey in group_split:
                votes.add(group_split[gkey])
            else:
                votes.add(_assign_group_to_split(gkey, train_ratio, val_ratio, test_ratio))
        if len(votes) > 1:
            return "conflict", True

    pkey = _primary_entity_key(rec)
    if pkey and pkey in group_split:
        return group_split[pkey], False
    if pkey:
        return _assign_group_to_split(pkey, train_ratio, val_ratio, test_ratio), False
    return _assign_group_to_split(rec.dedupe_key(), train_ratio, val_ratio, test_ratio), False


def _trim_split_stratified(
    records: List[SampleRecord],
    target: int,
    template_quotas: Optional[Dict[str, float]] = None,
    penalize_entity_heavy: bool = False,
) -> List[SampleRecord]:
    """Trim to target size, preferring rare slot_skeletons and template balance."""
    if len(records) <= target:
        return records

    sk_counts = Counter(slot_skeleton_key(r.template_id, r.slots) for r in records)
    tt_counts = Counter(r.template_type for r in records)

    def _score(rec: SampleRecord) -> Tuple[int, int, int, int]:
        sk = slot_skeleton_key(rec.template_id, rec.slots)
        rarity = sk_counts[sk]
        type_load = tt_counts[rec.template_type]
        is_long = word_count(rec.text_graphemes) >= 18
        n_ent = sum(1 for s in ENTITY_SLOTS_FOR_HEAVY if rec.slots.get(s))
        length_bonus = 0 if is_long else 1
        if penalize_entity_heavy and n_ent >= 2:
            entity_bonus = 2
        else:
            entity_bonus = 0 if n_ent >= 2 else 1
        return (rarity, type_load, length_bonus, entity_bonus)

    ranked = sorted(records, key=_score)
    kept = ranked[:target]
    return kept


def _trim_val_with_long_tail_floor(
    records: List[SampleRecord],
    target: int,
    min_long_tail_ratio: float = 0.10,
) -> List[SampleRecord]:
    """Trim val but reserve slots for long-tail samples (RULE_14)."""
    if len(records) <= target:
        return records
    tail = [r for r in records if _is_long_tail_sample(r)]
    non_tail = [r for r in records if not _is_long_tail_sample(r)]
    min_tail = max(1, int(target * min_long_tail_ratio))
    if len(tail) >= min_tail:
        kept_tail = tail[:min_tail]
        kept_other = _trim_split_stratified(non_tail, target - len(kept_tail))
        return kept_tail + kept_other
    return _trim_split_stratified(records, target)


def _ensure_val_long_tail_floor(
    buckets: Dict[str, List[SampleRecord]],
    min_ratio_of_train: float = 0.7,
    max_move_pct: float = 0.02,
) -> None:
    """Boost val long_tail ratio when pool is too small to trigger trim."""
    train = buckets.get("train", [])
    val = buckets.get("val", [])
    if not train or not val:
        return

    train_tail_ratio = sum(1 for r in train if _is_long_tail_sample(r)) / len(train)
    val_tail_count = sum(1 for r in val if _is_long_tail_sample(r))
    val_tail_ratio = val_tail_count / len(val)
    required = min_ratio_of_train * train_tail_ratio
    if val_tail_ratio >= required:
        return

    target_val_tail = max(1, int(len(val) * required) + 1)
    need = min(target_val_tail - val_tail_count, max(1, int(len(train) * max_move_pct)))
    if need <= 0:
        return

    moved: List[SampleRecord] = []
    for rec in train:
        if len(moved) >= need:
            break
        if _is_long_tail_sample(rec):
            moved.append(rec)

    for rec in moved:
        train.remove(rec)
        rec.split = "val"
        val.append(rec)

    buckets["train"] = train
    buckets["val"] = val


def split_records(
    records: List[SampleRecord],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    target_sizes: Optional[Dict[str, int]] = None,
    template_quotas: Optional[Dict[str, float]] = None,
) -> SplitResult:
    """Split records with entity-group assignment; never borrow across splits."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6 and total_ratio > 0:
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

    groups = _entity_groups(records)
    group_split: Dict[str, str] = {}
    for gkey in groups:
        group_split[gkey] = _assign_group_to_split(gkey, train_ratio, val_ratio, test_ratio)

    buckets: Dict[str, List[SampleRecord]] = {
        "train": [], "val": [], "test": [], "conflict": [],
    }
    for rec in records:
        chosen, is_conflict = _assign_record_split(
            rec, group_split, train_ratio, val_ratio, test_ratio,
        )
        if is_conflict:
            rec.split = "conflict"
            buckets["conflict"].append(rec)
        else:
            rec.split = chosen
            buckets[chosen].append(rec)

    conflicts = buckets.pop("conflict", [])

    train_heavy_ratio = 0.0
    if buckets.get("train"):
        n_train = len(buckets["train"])
        heavy = sum(
            1 for r in buckets["train"]
            if sum(1 for s in ENTITY_SLOTS_FOR_HEAVY if r.slots.get(s)) >= 2
        )
        train_heavy_ratio = heavy / n_train

    # Trim within each split only — never borrow from another split
    if target_sizes:
        for split_name in ("train", "val", "test"):
            target = target_sizes.get(split_name, 0)
            if target > 0:
                if split_name == "val":
                    buckets[split_name] = _trim_val_with_long_tail_floor(
                        buckets[split_name], target,
                    )
                else:
                    buckets[split_name] = _trim_split_stratified(
                        buckets[split_name], target, template_quotas,
                        penalize_entity_heavy=(
                            split_name == "train" and train_heavy_ratio > 0.40
                        ),
                    )
                for rec in buckets[split_name]:
                    rec.split = split_name

    _ensure_val_long_tail_floor(buckets)

    leakage = _compute_leakage(buckets)
    template_dist = _template_distribution(buckets)

    return SplitResult(
        train=buckets["train"],
        val=buckets["val"],
        test=buckets["test"],
        leakage_report=leakage,
        template_distribution=template_dist,
        conflicts=conflicts,
    )


def _compute_leakage(buckets: Dict[str, List[SampleRecord]]) -> LeakageReport:
    entity_to_splits: Dict[str, Set[str]] = defaultdict(set)
    entity_types: Dict[str, str] = {}

    for split_name, recs in buckets.items():
        for rec in recs:
            for etype, value in _canonical_entity_keys(rec).items():
                key = f"{etype}::{value}"
                entity_to_splits[key].add(split_name)
                entity_types[key] = etype

    leaked: Dict[str, List[str]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)
    totals: Dict[str, int] = defaultdict(int)

    for key, splits in entity_to_splits.items():
        etype = entity_types[key]
        totals[etype] += 1
        if len(splits) > 1:
            leaked[etype].append(key.split("::", 1)[1])
            counts[etype] += 1

    ratios = {
        etype: (counts[etype] / totals[etype] if totals[etype] else 0.0)
        for etype in totals
    }

    weights = {
        "address_string": 3.0, "poi_name": 2.0, "road_name": 2.0,
        "route_name": 1.5, "city_name": 1.0, "district_name": 1.0,
    }
    w_leaked = sum(counts.get(e, 0) * weights.get(e, 1.0) for e in totals)
    w_total = sum(totals[e] * weights.get(e, 1.0) for e in totals)
    weighted = w_leaked / w_total if w_total else 0.0

    return LeakageReport(
        leaked_entities=dict(leaked),
        leakage_counts=dict(counts),
        leakage_ratio=ratios,
        total_entities_by_type=dict(totals),
        weighted_leakage_ratio=round(weighted, 4),
    )


def _template_distribution(buckets: Dict[str, List[SampleRecord]]) -> Dict[str, Dict[str, int]]:
    dist: Dict[str, Dict[str, int]] = {}
    for split_name, recs in buckets.items():
        counts: Dict[str, int] = defaultdict(int)
        for rec in recs:
            counts[rec.template_type] += 1
        dist[split_name] = dict(counts)
    return dist


def assign_test_only(records: List[SampleRecord]) -> SplitResult:
    for rec in records:
        rec.split = "test"
    buckets = {"test": records}
    return SplitResult(
        train=[],
        val=[],
        test=records,
        leakage_report=_compute_leakage(buckets),
        template_distribution=_template_distribution(buckets),
        conflicts=[],
    )
