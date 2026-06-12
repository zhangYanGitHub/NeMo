#!/usr/bin/env python3
"""Generate navigation-domain G2P datasets for on-device model training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from config import GeneratorConfig
from dataset_profiles import DatasetProfile, get_profile, total_pool_size
from dedup import DedupCaps, DedupTracker, canonical_entity
from entity_index import EntityCapacityIndex
from entity_sets import ENTITY_POOLS, FOREIGN_STYLE_NAMES, MULTI_WORD_NAMES, get_entity_pool
from ipa_generator import BaseIpaGenerator, IpaGenerationError, create_ipa_generator, default_ipa_workers
from normalizer import MultiSenseAbbreviationStrategy
from replay_templates import REPLAY_TEMPLATES
from slot_values import fill_slot, iter_train_gap_entities, sample_entity_for_type
from splitter import assign_test_only, split_records, train_aligned_entity_slots
from templates import (
    ALL_TEMPLATES,
    LONG_ENTITY_HEAVY_PREFERRED_TYPES,
    LONG_ENTITY_HEAVY_TEMPLATES,
    LONG_SENTENCE_PREFERRED_TYPES,
    LONG_SENTENCE_TEMPLATES,
    ROAD_COVERAGE_TEMPLATES,
    TEMPLATE_CATEGORIES,
    TEMPLATE_TYPE_SOFT_CAP,
    Template,
    templates_with_length_hint,
    templates_with_min_entity_slots,
    templates_with_slot,
)
from utils import (
    SampleRecord,
    choose_weighted,
    dedupe_records,
    ensure_dir,
    finalize_graphemes,
    preview_records,
    render_template,
    text_length_bucket,
    word_count,
    write_jsonl,
)
from validator import Deficit, attach_debug_stats, validate_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ENTITY_SLOTS = ("road_name", "poi_name", "city_name", "district_name", "route_name", "address_string")
COVERAGE_DEFICIT_SLOTS = {
    "road_name_coverage": "road_name",
    "poi_name_coverage": "poi_name",
    "route_name_coverage": "route_name",
}
REFINEMENT_MODES = frozenset({
    "road_name_coverage", "poi_name_coverage", "route_name_coverage",
})
MIN_LONG_WORDS = 18
# Final pass (RULE_11): low-conflict types + train-hash alignment; avoid structured road / route-heavy ids
ENTITY_HEAVY_FINAL_EXCLUDED_TEMPLATE_IDS = frozenset({"rn_011"})
ENTITY_HEAVY_FINAL_ALLOWED_TYPES = frozenset({
    "poi_city_target", "address_based", "distance_prefixed", "road_navigation", "arrival",
})
LONG_DISTANCE_PHRASES = (
    "three quarters of a mile", "two thousand meters",
    "one point five kilometers", "two point five kilometers",
    "one thousand five hundred meters", "half a mile",
)


@dataclass
class _GraphemeCandidate:
    graphemes: str
    template: Template
    slots: Dict[str, str]


class DatasetGenerator:
    """Production-style navigation G2P dataset generator with quality-aware dedup."""

    def __init__(
        self,
        profile: DatasetProfile,
        config: GeneratorConfig,
        ipa_gen: BaseIpaGenerator,
    ):
        self.profile = profile
        self.config = config
        self.ipa_gen = ipa_gen
        self.rng = random.Random(config.seed)
        self.abbrev_strategy = MultiSenseAbbreviationStrategy(rng=self.rng)

        pool_target = total_pool_size(profile) or profile.test_size
        self.dedup_tracker = DedupTracker(
            caps=DedupCaps(),
            target_pool_size=pool_target,
        )
        self.entity_coverage: Dict[str, Set[str]] = defaultdict(set)
        self.template_type_counts: Counter = Counter()
        self.length_bucket_counts: Counter = Counter()

        # Targeted generation state (single primary deficit per round)
        self._primary_deficit: Optional[Deficit] = None
        self._forced_templates: Optional[List[Template]] = None
        self._forced_slot: Optional[str] = None
        self._forced_length_hint: Optional[str] = None
        self._min_entity_slots: int = 0
        self._force_long_tail: bool = False
        self._round_robin_entities: List[str] = []
        self._rr_entity_idx: int = 0
        self._rr_template_idx: int = 0
        self._capacity_template_idx: int = 0
        self._entity_rr_idx: Dict[str, int] = defaultdict(int)
        self.entity_index = EntityCapacityIndex(self.dedup_tracker)

    def _apply_deficit_plan(self, deficit: Deficit) -> None:
        """Hard constraints for one deficit — no merging / overwriting."""
        self._primary_deficit = deficit
        self._forced_templates = None
        self._forced_slot = None
        self._forced_length_hint = None
        self._min_entity_slots = 0
        self._force_long_tail = False
        self._round_robin_entities = []
        self._rr_entity_idx = 0
        self._rr_template_idx = 0
        self.dedup_tracker.begin_refinement(deficit.kind)
        if deficit.kind in REFINEMENT_MODES:
            for slot in COVERAGE_DEFICIT_SLOTS.values():
                self.entity_index.rebuild_slot(slot)

        kind = deficit.kind
        if kind == "poi_name_coverage":
            self._forced_slot = "poi_name"
            self._forced_templates = templates_with_slot("poi_name")
        elif kind == "route_name_coverage":
            self._forced_slot = "route_name"
            self._forced_templates = templates_with_slot("route_name") or TEMPLATE_CATEGORIES["numbered_routes"]
        elif kind == "road_name_coverage":
            self._forced_slot = "road_name"
            self._forced_templates = templates_with_slot("road_name")
        elif kind == "long_sentence":
            self._forced_length_hint = "long"
            self._forced_templates = [
                t for t in templates_with_length_hint("long")
                if sum(1 for s in t.required_slots if s in ENTITY_SLOTS) >= 2
            ] or templates_with_length_hint("long")
        elif kind == "entity_heavy":
            self._min_entity_slots = 2
            self._forced_templates = templates_with_min_entity_slots(2)
        elif kind == "long_tail":
            self._force_long_tail = True
            self._forced_templates = templates_with_min_entity_slots(1)
        elif kind == "near_duplicate_high":
            self._forced_templates = templates_with_min_entity_slots(2)

        if self._forced_slot:
            pool = get_entity_pool(self._forced_slot)
            self._round_robin_entities = [
                e for e in pool
                if canonical_entity(e, self._forced_slot) not in self.entity_coverage.get(self._forced_slot, set())
            ]
            if not self._round_robin_entities:
                self._round_robin_entities = list(pool)
        elif self._force_long_tail:
            self._round_robin_entities = list(FOREIGN_STYLE_NAMES)

    def _clear_deficit_plan(self) -> None:
        self._primary_deficit = None
        self._forced_slot = None
        self._forced_templates = None
        self._round_robin_entities = []
        self.dedup_tracker.end_refinement()

    def _template_type_share(self, template_type: str) -> float:
        total = sum(self.template_type_counts.values()) or 1
        return self.template_type_counts[template_type] / total

    def _type_share_budget_ok(self, template_type: str) -> bool:
        share = self._template_type_share(template_type)
        soft = TEMPLATE_TYPE_SOFT_CAP.get(template_type)
        if soft is not None:
            return share < soft
        return share < self.profile.quality_thresholds.max_template_type_share_max

    def _filter_templates_by_budget(self, pool: List[Template]) -> List[Template]:
        return [
            t for t in pool
            if self._type_share_budget_ok(t.template_type)
            and self.dedup_tracker.template_has_capacity(t.template_id)
        ]

    def _maybe_force_road_template(self, templates_by_type: Dict[str, List[Template]]) -> Optional[Template]:
        road_min = self.profile.entity_min_coverage.get("road_name", 0)
        if not road_min or len(self.entity_coverage["road_name"]) >= road_min * 0.7:
            return None
        if self.rng.random() > 0.40:
            return None
        pool = self._filter_templates_by_budget(templates_with_slot("road_name"))
        if not pool:
            return None
        uncovered = [
            e for e in get_entity_pool("road_name")
            if canonical_entity(e, "road_name") not in self.entity_coverage["road_name"]
        ]
        if uncovered:
            self._forced_slot = "road_name"
            self._round_robin_entities = uncovered
        return pool[self._rr_template_idx % len(pool)]

    def _select_templates(self) -> List[Template]:
        if self.profile.use_replay_templates:
            return REPLAY_TEMPLATES
        if self.profile.entity_focused:
            types = list(self.profile.template_quotas.keys())
            return [t for t in ALL_TEMPLATES if t.template_type in types]
        return ALL_TEMPLATES

    def _pick_template(self, templates_by_type: Dict[str, List[Template]]) -> Template:
        if self._forced_templates:
            pool = self._filter_templates_by_budget(list(self._forced_templates))
            if self._min_entity_slots:
                pool = [t for t in pool if sum(1 for s in t.required_slots if s in ENTITY_SLOTS) >= self._min_entity_slots]
            if self._forced_length_hint:
                hinted = [t for t in pool if t.length_hint == self._forced_length_hint]
                if hinted:
                    pool = hinted
            if pool:
                if self._round_robin_entities:
                    t = pool[self._rr_template_idx % len(pool)]
                    self._rr_template_idx += 1
                    return t
                return self.rng.choice(pool)

        forced_road = self._maybe_force_road_template(templates_by_type)
        if forced_road:
            self._rr_template_idx += 1
            return forced_road

        quotas = self.profile.template_quotas
        if not quotas:
            return self.rng.choice([t for ts in templates_by_type.values() for t in ts])
        adjusted: Dict[str, float] = {}
        total_generated = sum(self.template_type_counts.values()) or 1
        for ttype, target_frac in quotas.items():
            current_frac = self.template_type_counts.get(ttype, 0) / total_generated
            deficit = max(0.0, target_frac - current_frac)
            adjusted[ttype] = target_frac + deficit * 2.0
            if ttype == "basic_actions" and current_frac > 0.10:
                adjusted[ttype] *= 0.3
        chosen_type = choose_weighted(self.rng, adjusted)
        pool = templates_by_type.get(chosen_type) or TEMPLATE_CATEGORIES.get(chosen_type, ALL_TEMPLATES)
        pool = self._filter_templates_by_budget(pool)
        if not pool:
            cap_pool = [
                t for t in (templates_by_type.get(chosen_type) or ALL_TEMPLATES)
                if self.dedup_tracker.template_has_capacity(t.template_id)
                and self._type_share_budget_ok(t.template_type)
            ]
            if cap_pool:
                return self.rng.choice(cap_pool)
            all_cap = [
                t for t in ALL_TEMPLATES
                if self.dedup_tracker.template_has_capacity(t.template_id)
                and self._type_share_budget_ok(t.template_type)
            ]
            if all_cap:
                return self.rng.choice(all_cap)
            return self.rng.choice([t for ts in templates_by_type.values() for t in ts])
        return self.rng.choice(pool)

    def _needs_length_bucket(self, template: Template) -> bool:
        if self._forced_length_hint:
            return template.length_hint == self._forced_length_hint
        short_r, med_r, long_r = self.config.length_distribution
        total = sum(self.length_bucket_counts.values()) or 1
        targets = {"short": short_r, "medium": med_r, "long": long_r}
        current = {k: self.length_bucket_counts.get(k, 0) / total for k in targets}
        if current.get(template.length_hint, 0) < targets.get(template.length_hint, 0):
            return True
        return self.rng.random() < 0.7

    def _uncovered_entity(self, slot: str) -> Optional[str]:
        pool = get_entity_pool(slot)
        if self._force_long_tail and slot in ("poi_name", "city_name", "road_name"):
            pool = list(FOREIGN_STYLE_NAMES) + pool
        if not pool:
            return None
        uncovered = [e for e in pool if canonical_entity(e, slot) not in self.entity_coverage[slot]]
        if uncovered:
            return self.rng.choice(uncovered)
        return None

    def _entity_overrides(self, template: Template) -> Dict[str, str]:
        overrides: Dict[str, str] = {}
        long_tail = self.profile.long_tail_boost > 1.0 or self._force_long_tail
        candidate_slots = [s for s in template.required_slots if s in ENTITY_SLOTS]

        if self._round_robin_entities and self._forced_slot and self._forced_slot in candidate_slots:
            entity = self._round_robin_entities[self._rr_entity_idx % len(self._round_robin_entities)]
            self._rr_entity_idx += 1
            overrides[self._forced_slot] = entity

        if self._primary_deficit and self._primary_deficit.kind == "long_sentence":
            if "distance_phrase" in candidate_slots and "distance_phrase" not in overrides:
                overrides["distance_phrase"] = self.rng.choice([
                    "three quarters of a mile", "two thousand meters",
                    "one point five kilometers", "half a mile",
                ])
            for slot in ("poi_name", "city_name", "road_name", "district_name"):
                if slot in candidate_slots and slot not in overrides:
                    overrides[slot] = self.rng.choice(MULTI_WORD_NAMES)

        if template.template_type == "mixed_longform":
            for slot in ("poi_name", "city_name", "road_name", "district_name", "route_name"):
                if slot in candidate_slots and slot not in overrides:
                    pool = MULTI_WORD_NAMES if slot != "route_name" else get_entity_pool("route_name")
                    overrides[slot] = self.rng.choice(pool)

        for slot in candidate_slots:
            if slot in overrides:
                continue
            min_cov = self.profile.entity_min_coverage.get(slot, 0)
            covered = len(self.entity_coverage[slot])
            uncovered = self._uncovered_entity(slot)
            if uncovered:
                overrides[slot] = uncovered
            elif self.profile.entity_focused or covered < min_cov or self._primary_deficit is not None:
                pool = get_entity_pool(slot)
                if pool:
                    overrides[slot] = self.rng.choice(pool)
                else:
                    overrides[slot] = sample_entity_for_type(slot, self.rng, long_tail=long_tail)
        return overrides

    def _templates_with_capacity(self, templates: List[Template]) -> List[Template]:
        pool = self._filter_templates_by_budget(templates)
        return [t for t in pool if self.dedup_tracker.template_has_capacity(t.template_id)]

    def _pick_entity_with_capacity(self, slot: str, template_id: str) -> Optional[str]:
        return self.entity_index.pick(
            slot, template_id, self.entity_coverage.get(slot, set()),
        )

    def _capacity_overrides(self, template: Template) -> Optional[Dict[str, str]]:
        overrides: Dict[str, str] = {}
        for slot in template.required_slots:
            if slot in ENTITY_SLOTS:
                ent = self._pick_entity_with_capacity(slot, template.template_id)
                if ent is None:
                    return None
                overrides[slot] = ent
        return overrides

    def _generate_capacity_aware_candidate(
        self,
        templates: List[Template],
    ) -> Optional[_GraphemeCandidate]:
        """Round-robin template × entity with remaining dedup headroom."""
        viable = self._templates_with_capacity(templates)
        if not viable:
            return None

        n = len(viable)
        for _ in range(n):
            template = viable[self._capacity_template_idx % n]
            self._capacity_template_idx += 1

            overrides = self._capacity_overrides(template)
            if overrides is None:
                continue

            raw_text, slots = render_template(template, self.rng, overrides)
            graphemes = finalize_graphemes(
                raw_text, slots,
                expand_abbrev=self.config.expand_abbrev,
                abbrev_strategy=self.abbrev_strategy,
                spoken_numbers=self.config.spoken_numbers,
            )
            if not graphemes or len(graphemes) > self.config.max_text_len:
                continue
            if not self.dedup_tracker.slots_have_capacity(template.template_id, slots):
                continue
            if self.config.dedupe:
                reason = self.dedup_tracker.should_reject(graphemes, template.template_id, slots)
                if reason:
                    continue
            return _GraphemeCandidate(graphemes=graphemes, template=template, slots=slots)
        return None

    def _register_sample(self, rec: SampleRecord) -> None:
        self.dedup_tracker.register(rec.text_graphemes, rec.template_id, rec.slots)
        self.entity_index.on_register(rec.slots)
        self.template_type_counts[rec.template_type] += 1
        self.length_bucket_counts[text_length_bucket(rec.text_graphemes)] += 1
        for etype, val in rec.entity_keys().items():
            self.entity_coverage[etype].add(canonical_entity(val, etype))

    def _generate_grapheme_candidate(
        self, templates_by_type: Dict[str, List[Template]],
        max_tries: int = 10,
    ) -> Optional[_GraphemeCandidate]:
        for _ in range(max_tries):
            template = self._pick_template(templates_by_type)
            if self._min_entity_slots:
                n_ent = sum(1 for s in template.required_slots if s in ENTITY_SLOTS)
                if n_ent < self._min_entity_slots:
                    continue
            if not self._primary_deficit and not self._needs_length_bucket(template):
                if self.rng.random() < 0.3:
                    continue
            overrides = self._entity_overrides(template)
            raw_text, slots = render_template(template, self.rng, overrides)
            graphemes = finalize_graphemes(
                raw_text, slots,
                expand_abbrev=self.config.expand_abbrev,
                abbrev_strategy=self.abbrev_strategy,
                spoken_numbers=self.config.spoken_numbers,
            )
            if not graphemes or len(graphemes) > self.config.max_text_len:
                continue
            if self.config.dedupe:
                reason = self.dedup_tracker.should_reject(graphemes, template.template_id, slots)
                if reason:
                    continue
            return _GraphemeCandidate(graphemes=graphemes, template=template, slots=slots)
        return None

    def _candidates_to_records(self, candidates: List[_GraphemeCandidate]) -> List[SampleRecord]:
        if not candidates:
            return []
        texts = [c.graphemes for c in candidates]
        ipas = self.ipa_gen.generate_batch(
            texts, locale=self.config.locale, workers=self.config.ipa_workers,
        )
        records: List[SampleRecord] = []
        for cand, ipa in zip(candidates, ipas):
            if not ipa or len(ipa) > self.config.max_ipa_len:
                continue
            if not self.dedup_tracker.template_has_capacity(cand.template.template_id):
                continue
            if not self._type_share_budget_ok(cand.template.template_type):
                continue
            if self.config.dedupe:
                reason = self.dedup_tracker.should_reject(
                    cand.graphemes, cand.template.template_id, cand.slots,
                )
                if reason:
                    continue
            rec = SampleRecord(
                text_graphemes=cand.graphemes,
                text=ipa,
                template_type=cand.template.template_type,
                template_id=cand.template.template_id,
                slots=cand.slots,
                source_generator=self.profile.source_generator,
                metadata={
                    "length_bucket": text_length_bucket(cand.graphemes),
                    "word_count": len(cand.graphemes.split()),
                },
            )
            self._register_sample(rec)
            records.append(rec)
        return records

    def generate_entity_coverage(
        self,
        slot: str,
        target_size: int,
        train_existing_canon: Optional[Set[str]] = None,
    ) -> List[SampleRecord]:
        """Train-gap driven entity coverage — new canonical entities hashing to train."""
        mode = f"{slot}_coverage"
        if mode not in REFINEMENT_MODES:
            mode = "road_name_coverage"
        self.dedup_tracker.begin_refinement(mode)
        train_existing = train_existing_canon or set()

        if slot == "road_name":
            templates = list(ROAD_COVERAGE_TEMPLATES)
        else:
            templates = templates_with_slot(slot)
        templates = self._templates_with_capacity(templates) or templates

        entities = iter_train_gap_entities(slot, train_existing, self.profile.train_ratio)
        if not entities:
            logger.warning(
                "%s: no train-bucket entities left (train_existing=%d)",
                mode, len(train_existing),
            )
            self.dedup_tracker.end_refinement()
            return []

        records: List[SampleRecord] = []
        ti, ei = 0, 0
        attempts = 0
        max_attempts = max(target_size * 12, len(entities) * len(templates))

        try:
            while len(records) < target_size and attempts < max_attempts:
                entity = entities[ei % len(entities)]
                template = templates[ti % len(templates)]
                ti += 1
                if ti % len(templates) == 0:
                    ei += 1

                overrides: Dict[str, str] = {}
                for req in template.required_slots:
                    if req == slot:
                        overrides[req] = entity
                    elif req in ENTITY_SLOTS and req not in overrides:
                        overrides[req] = (
                            self._pick_entity_with_capacity(req, template.template_id)
                            or self._uncovered_entity(req)
                            or fill_slot(req, self.rng, overrides)
                        )

                raw_text, slots = render_template(template, self.rng, overrides)
                graphemes = finalize_graphemes(
                    raw_text, slots,
                    expand_abbrev=self.config.expand_abbrev,
                    abbrev_strategy=self.abbrev_strategy,
                    spoken_numbers=self.config.spoken_numbers,
                )
                attempts += 1
                if not graphemes or len(graphemes) > self.config.max_text_len:
                    continue
                if self.config.dedupe:
                    reason = self.dedup_tracker.should_reject(graphemes, template.template_id, slots)
                    if reason:
                        continue
                batch_recs = self._candidates_to_records([
                    _GraphemeCandidate(graphemes=graphemes, template=template, slots=slots),
                ])
                records.extend(batch_recs)
        finally:
            self._clear_deficit_plan()

        new_canon = {
            canonical_entity(r.slots[slot], slot)
            for r in records if r.slots.get(slot)
        } - train_existing
        logger.info(
            "%s: generated %d / %d (attempts=%d, new_train_canon=%d)",
            mode, len(records), target_size, attempts, len(new_canon),
        )
        return records[:target_size]

    def _entity_slot_count(self, slots: Dict[str, str]) -> int:
        return sum(1 for s in ENTITY_SLOTS if slots.get(s))

    def _heavy_entity_overrides(self, template: Template) -> Dict[str, str]:
        overrides: Dict[str, str] = {}
        for slot in template.required_slots:
            if slot == "distance_phrase":
                overrides[slot] = self.rng.choice(LONG_DISTANCE_PHRASES)
            elif slot in ("poi_name", "city_name", "road_name", "district_name"):
                overrides[slot] = self.rng.choice(MULTI_WORD_NAMES)
            elif slot == "route_name":
                pool = get_entity_pool("route_name")
                overrides[slot] = self.rng.choice(pool) if pool else sample_entity_for_type(slot, self.rng)
            elif slot in ENTITY_SLOTS:
                overrides[slot] = sample_entity_for_type(slot, self.rng)
            else:
                overrides[slot] = fill_slot(slot, self.rng, overrides)
        return overrides

    def _long_sentence_overrides(self, template: Template) -> Dict[str, str]:
        overrides: Dict[str, str] = {}
        for slot in template.required_slots:
            if slot == "distance_phrase":
                overrides[slot] = self.rng.choice(LONG_DISTANCE_PHRASES)
            elif slot in ("poi_name", "city_name", "road_name", "district_name"):
                overrides[slot] = self.rng.choice(MULTI_WORD_NAMES)
            elif slot == "route_name":
                pool = get_entity_pool("route_name")
                overrides[slot] = self.rng.choice(pool) if pool else sample_entity_for_type(slot, self.rng)
            elif slot in ENTITY_SLOTS:
                overrides[slot] = sample_entity_for_type(slot, self.rng)
            else:
                overrides[slot] = fill_slot(slot, self.rng, overrides)
        return overrides

    def _long_sentence_template_pool(self) -> List[Template]:
        preferred = [
            t for t in LONG_SENTENCE_TEMPLATES
            if t.template_type in LONG_SENTENCE_PREFERRED_TYPES
        ]
        fallback = [t for t in LONG_SENTENCE_TEMPLATES if t not in preferred]
        viable_pref = [
            t for t in preferred
            if self.dedup_tracker.template_has_capacity(t.template_id)
            and self._type_share_budget_ok(t.template_type)
        ]
        if viable_pref:
            return viable_pref
        viable_fb = [
            t for t in fallback
            if self.dedup_tracker.template_has_capacity(t.template_id)
            and self._type_share_budget_ok(t.template_type)
        ]
        return viable_fb or list(LONG_SENTENCE_TEMPLATES)

    def _pick_long_sentence_template(
        self, pool: List[Template], rr_idx: int,
    ) -> Optional[Template]:
        viable = [
            t for t in pool
            if self.dedup_tracker.template_has_capacity(t.template_id)
            and self._type_share_budget_ok(t.template_type)
        ]
        if not viable:
            return None
        viable.sort(key=lambda t: (
            self._template_type_share(t.template_type),
            self.dedup_tracker._template_id_counts[t.template_id],
        ))
        return viable[rr_idx % len(viable)]

    def generate_long_sentence(self, target_size: int) -> List[SampleRecord]:
        """Dedicated long-sentence path: non-mixed templates, word_count >= 18."""
        self.dedup_tracker.begin_refinement("long_sentence")
        records: List[SampleRecord] = []
        attempts = 0
        max_attempts = max(target_size * 12, 500)
        ti = 0
        batch_size = self.config.ipa_batch_size

        try:
            while len(records) < target_size and attempts < max_attempts:
                pool = self._long_sentence_template_pool()
                pending: List[_GraphemeCandidate] = []
                tries = 0
                while (
                    len(pending) < batch_size
                    and tries < batch_size * 5
                    and attempts < max_attempts
                    and len(records) + len(pending) < target_size
                ):
                    tries += 1
                    attempts += 1
                    template = self._pick_long_sentence_template(pool, ti)
                    ti += 1
                    if template is None:
                        continue
                    if not self.dedup_tracker.template_has_capacity(template.template_id):
                        continue
                    if not self._type_share_budget_ok(template.template_type):
                        continue
                    overrides = self._long_sentence_overrides(template)
                    raw_text, slots = render_template(template, self.rng, overrides)
                    graphemes = finalize_graphemes(
                        raw_text, slots,
                        expand_abbrev=self.config.expand_abbrev,
                        abbrev_strategy=self.abbrev_strategy,
                        spoken_numbers=self.config.spoken_numbers,
                    )
                    if not graphemes or len(graphemes) > self.config.max_text_len:
                        continue
                    if word_count(graphemes) < MIN_LONG_WORDS:
                        continue
                    if self.config.dedupe:
                        reason = self.dedup_tracker.should_reject(
                            graphemes, template.template_id, slots,
                        )
                        if reason:
                            continue
                    pending.append(_GraphemeCandidate(
                        graphemes=graphemes, template=template, slots=slots,
                    ))
                if not pending:
                    break
                for rec in self._candidates_to_records(pending):
                    records.append(rec)
                    if len(records) >= target_size:
                        break
        finally:
            self.dedup_tracker.end_refinement()

        logger.info(
            "long_sentence: generated %d / %d (attempts=%d)",
            len(records), target_size, attempts,
        )
        return records[:target_size]

    def _entity_heavy_template_pool(self, joint: bool = False) -> List[Template]:
        if joint:
            preferred = [
                t for t in LONG_ENTITY_HEAVY_TEMPLATES
                if t.template_type in LONG_ENTITY_HEAVY_PREFERRED_TYPES
            ]
            fallback = [t for t in LONG_ENTITY_HEAVY_TEMPLATES if t not in preferred]
        else:
            preferred = [
                t for t in templates_with_min_entity_slots(2)
                if t.template_type in LONG_ENTITY_HEAVY_PREFERRED_TYPES
            ]
            fallback = [
                t for t in templates_with_min_entity_slots(2) if t not in preferred
            ]
        viable_pref = [
            t for t in preferred
            if self.dedup_tracker.template_has_capacity(t.template_id)
            and self._type_share_budget_ok(t.template_type)
        ]
        if viable_pref:
            return viable_pref
        viable_fb = [
            t for t in fallback
            if self.dedup_tracker.template_has_capacity(t.template_id)
            and self._type_share_budget_ok(t.template_type)
        ]
        return viable_fb or (LONG_ENTITY_HEAVY_TEMPLATES if joint else templates_with_min_entity_slots(2))

    def _pick_entity_heavy_template(
        self, pool: List[Template], rr_idx: int,
    ) -> Optional[Template]:
        viable = [
            t for t in pool
            if self.dedup_tracker.template_has_capacity(t.template_id)
            and self._type_share_budget_ok(t.template_type)
        ]
        if not viable:
            return None
        viable.sort(key=lambda t: (
            self._template_type_share(t.template_type),
            self.dedup_tracker._template_id_counts[t.template_id],
        ))
        return viable[rr_idx % len(viable)]

    def _generate_heavy_refinement_batch(
        self,
        mode: str,
        target_size: int,
        *,
        require_long: bool,
    ) -> List[SampleRecord]:
        self.dedup_tracker.begin_refinement(mode)
        records: List[SampleRecord] = []
        attempts = 0
        max_attempts = max(target_size * 15, 500)
        ti = 0
        batch_size = self.config.ipa_batch_size
        min_entities = 2

        try:
            while len(records) < target_size and attempts < max_attempts:
                pool = self._entity_heavy_template_pool(joint=require_long)
                pending: List[_GraphemeCandidate] = []
                tries = 0
                while (
                    len(pending) < batch_size
                    and tries < batch_size * 6
                    and attempts < max_attempts
                    and len(records) + len(pending) < target_size
                ):
                    tries += 1
                    attempts += 1
                    template = self._pick_entity_heavy_template(pool, ti)
                    ti += 1
                    if template is None:
                        continue
                    overrides = self._heavy_entity_overrides(template)
                    raw_text, slots = render_template(template, self.rng, overrides)
                    graphemes = finalize_graphemes(
                        raw_text, slots,
                        expand_abbrev=self.config.expand_abbrev,
                        abbrev_strategy=self.abbrev_strategy,
                        spoken_numbers=self.config.spoken_numbers,
                    )
                    if not graphemes or len(graphemes) > self.config.max_text_len:
                        continue
                    if self._entity_slot_count(slots) < min_entities:
                        continue
                    if require_long and word_count(graphemes) < MIN_LONG_WORDS:
                        continue
                    if self.config.dedupe:
                        reason = self.dedup_tracker.should_reject(
                            graphemes, template.template_id, slots,
                        )
                        if reason:
                            continue
                    pending.append(_GraphemeCandidate(
                        graphemes=graphemes, template=template, slots=slots,
                    ))
                if not pending:
                    break
                for rec in self._candidates_to_records(pending):
                    records.append(rec)
                    if len(records) >= target_size:
                        break
        finally:
            self.dedup_tracker.end_refinement()

        logger.info(
            "%s: generated %d / %d (attempts=%d)",
            mode, len(records), target_size, attempts,
        )
        return records[:target_size]

    def _final_entity_heavy_template_pool(self) -> List[Template]:
        candidates = [
            t for t in templates_with_min_entity_slots(2)
            if t.template_type in ENTITY_HEAVY_FINAL_ALLOWED_TYPES
            and t.template_id not in ENTITY_HEAVY_FINAL_EXCLUDED_TEMPLATE_IDS
        ]
        viable = [
            t for t in candidates
            if self.dedup_tracker.template_has_capacity(t.template_id)
            and self._type_share_budget_ok(t.template_type)
        ]
        if viable:
            return viable
        return [
            t for t in candidates
            if self.dedup_tracker.template_has_capacity(t.template_id)
        ] or candidates

    def generate_entity_heavy_final_pass(self, target_size: int) -> List[SampleRecord]:
        """RULE_11 only: entity-heavy + train-bucket-aligned entities to land in train, not conflict."""
        self.dedup_tracker.begin_refinement("entity_heavy_final")
        records: List[SampleRecord] = []
        attempts = 0
        max_attempts = max(target_size * 25, 800)
        ti = 0
        batch_size = self.config.ipa_batch_size
        tr, vr, ter = (
            self.profile.train_ratio, self.profile.val_ratio, self.profile.test_ratio,
        )

        try:
            while len(records) < target_size and attempts < max_attempts:
                pool = self._final_entity_heavy_template_pool()
                pending: List[_GraphemeCandidate] = []
                tries = 0
                while (
                    len(pending) < batch_size
                    and tries < batch_size * 8
                    and attempts < max_attempts
                    and len(records) + len(pending) < target_size
                ):
                    tries += 1
                    attempts += 1
                    template = self._pick_entity_heavy_template(pool, ti)
                    ti += 1
                    if template is None:
                        continue
                    overrides = self._heavy_entity_overrides(template)
                    raw_text, slots = render_template(template, self.rng, overrides)
                    if self._entity_slot_count(slots) < 2:
                        continue
                    if not train_aligned_entity_slots(slots, tr, vr, ter):
                        continue
                    graphemes = finalize_graphemes(
                        raw_text, slots,
                        expand_abbrev=self.config.expand_abbrev,
                        abbrev_strategy=self.abbrev_strategy,
                        spoken_numbers=self.config.spoken_numbers,
                    )
                    if not graphemes or len(graphemes) > self.config.max_text_len:
                        continue
                    if self.config.dedupe:
                        reason = self.dedup_tracker.should_reject(
                            graphemes, template.template_id, slots,
                        )
                        if reason:
                            continue
                    pending.append(_GraphemeCandidate(
                        graphemes=graphemes, template=template, slots=slots,
                    ))
                if not pending:
                    break
                for rec in self._candidates_to_records(pending):
                    records.append(rec)
                    if len(records) >= target_size:
                        break
        finally:
            self.dedup_tracker.end_refinement()

        logger.info(
            "entity_heavy_final_pass: generated %d / %d (attempts=%d)",
            len(records), target_size, attempts,
        )
        return records[:target_size]

    def generate_long_entity_heavy(self, target_size: int) -> List[SampleRecord]:
        """Joint RULE_10 + RULE_11: word_count >= 18 and >= 2 entity slots."""
        return self._generate_heavy_refinement_batch(
            "long_entity_heavy", target_size, require_long=True,
        )

    def generate_entity_heavy(self, target_size: int) -> List[SampleRecord]:
        """Dedicated entity_heavy path with refinement bypass."""
        return self._generate_heavy_refinement_batch(
            "entity_heavy", target_size, require_long=False,
        )

    def generate_for_deficit(
        self,
        deficit: Deficit,
        target_size: int,
        train_gap_canon: Optional[Set[str]] = None,
    ) -> List[SampleRecord]:
        """Deterministic entity×template round-robin for one deficit."""
        if deficit.kind == "long_entity_heavy":
            return self.generate_long_entity_heavy(target_size)
        if deficit.kind == "long_sentence":
            return self.generate_long_sentence(target_size)
        if deficit.kind == "entity_heavy":
            return self.generate_entity_heavy(target_size)
        if deficit.kind in COVERAGE_DEFICIT_SLOTS:
            return self.generate_entity_coverage(
                COVERAGE_DEFICIT_SLOTS[deficit.kind], target_size, train_gap_canon,
            )
        self._apply_deficit_plan(deficit)
        templates = self._select_templates()
        templates_by_type: Dict[str, List[Template]] = defaultdict(list)
        for t in templates:
            templates_by_type[t.template_type].append(t)

        records: List[SampleRecord] = []
        attempts = 0
        max_attempts = max(target_size * 15, 500)
        batch_size = self.config.ipa_batch_size

        try:
            while len(records) < target_size and attempts < max_attempts:
                pending: List[_GraphemeCandidate] = []
                tries = 0
                while len(pending) < batch_size and tries < batch_size * 5 and attempts < max_attempts:
                    tries += 1
                    attempts += 1
                    cand = self._generate_grapheme_candidate(templates_by_type, max_tries=3)
                    if cand:
                        pending.append(cand)
                if not pending:
                    break
                for rec in self._candidates_to_records(pending):
                    records.append(rec)
                    if len(records) >= target_size:
                        break
        finally:
            self._clear_deficit_plan()

        logger.info(
            "Deficit %s: generated %d / %d (attempts=%d)",
            deficit.kind, len(records), target_size, attempts,
        )
        return records[:target_size]

    def generate_pool(self, target_size: int) -> List[SampleRecord]:
        all_templates = self._select_templates()
        self._capacity_template_idx = 0
        self._entity_rr_idx.clear()

        batch_size = self.config.ipa_batch_size
        pool_target = total_pool_size(self.profile) or self.profile.test_size
        min_viable = int(pool_target * 0.85)
        accept_window: deque = deque(maxlen=1500)
        accept_floor = 0.04
        records: List[SampleRecord] = []
        attempts = 0
        capacity_exhausted = False
        t0 = time.time()
        last_log = 0

        logger.info(
            "Main pool: capacity-aware sampling (target=%d, min_viable=%d)",
            target_size, min_viable,
        )

        while len(records) < target_size and not capacity_exhausted:
            viable = self._templates_with_capacity(all_templates)
            if not viable:
                capacity_exhausted = True
                logger.warning(
                    "No templates with dedup headroom, stopping pool at %d / %d",
                    len(records), target_size,
                )
                break

            need = min(batch_size, target_size - len(records))
            pending: List[_GraphemeCandidate] = []
            batch_attempts = 0
            max_batch_attempts = need * 4

            while len(pending) < need and batch_attempts < max_batch_attempts:
                batch_attempts += 1
                attempts += 1
                cand = self._generate_capacity_aware_candidate(viable)
                accepted = cand is not None
                accept_window.append(accepted)
                if cand:
                    pending.append(cand)

            if (
                len(accept_window) >= accept_window.maxlen
                and sum(accept_window) / len(accept_window) < accept_floor
            ):
                capacity_exhausted = True
                logger.warning(
                    "Accept rate below %.0f%% over last %d attempts, stopping at %d / %d",
                    accept_floor * 100, len(accept_window), len(records), target_size,
                )

            if not pending:
                if len(records) >= min_viable:
                    capacity_exhausted = True
                continue

            new_records = self._candidates_to_records(pending)
            for rec in new_records:
                records.append(rec)
                if len(records) >= target_size:
                    break

            if len(records) - last_log >= batch_size or capacity_exhausted:
                last_log = len(records)
                elapsed = time.time() - t0
                rate = len(records) / elapsed if elapsed > 0 else 0
                win_rate = sum(accept_window) / len(accept_window) if accept_window else 0
                logger.info(
                    "Generated %d / %d (attempts=%d, %.0f samples/s, window_accept=%.1f%%)",
                    len(records), target_size, attempts, rate, win_rate * 100,
                )

        logger.info(
            "Pool reject_stats final: %s", self.dedup_tracker.reject_stats.most_common(10),
        )
        if len(records) > target_size:
            records = records[:target_size]
        if len(records) < target_size:
            logger.warning(
                "Only generated %d / %d after %d attempts", len(records), target_size, attempts,
            )
        return records


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate navigation G2P JSONL datasets")
    parser.add_argument("--dataset-profile", default="nav_prod_v1",
                        choices=["mini_seed", "nav_baseline_v1", "nav_prod_v1",
                                 "nav_entities_v1", "nav_replay_v1"])
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--locale", default="en-us")
    parser.add_argument("--espeak-cmd", default="espeak-ng")
    parser.set_defaults(expand_abbrev=True, spoken_numbers=True)
    parser.add_argument(
        "--no-expand-abbrev", dest="expand_abbrev", action="store_false",
        help="Keep St./Ave./etc. abbreviations instead of full words",
    )
    parser.add_argument(
        "--no-spoken-numbers", dest="spoken_numbers", action="store_false",
        help="Keep Arabic digits instead of spoken English number words",
    )
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false", default=True)
    parser.add_argument("--max-text-len", type=int, default=300)
    parser.add_argument("--max-ipa-len", type=int, default=500)
    parser.add_argument("--preview", type=int, default=0)
    parser.add_argument("--save-metadata", action="store_true", default=True)
    parser.add_argument("--no-save-metadata", dest="save_metadata", action="store_false")
    parser.add_argument("--strict", action="store_true", default=None,
                        help="Exit 1 if quality gate fails (default: on for nav_prod_v1)")
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel espeak-ng workers (0=auto, cpu_count+4)",
    )
    parser.add_argument(
        "--ipa-batch-size", type=int, default=256,
        help="Grapheme candidates per parallel IPA batch",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Speed mode: skip refinement, lower oversample, fewer retries",
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> GeneratorConfig:
    return GeneratorConfig(
        seed=args.seed,
        locale=args.locale,
        espeak_cmd=args.espeak_cmd,
        expand_abbrev=args.expand_abbrev,
        spoken_numbers=args.spoken_numbers,
        dedupe=args.dedupe,
        max_text_len=args.max_text_len,
        max_ipa_len=args.max_ipa_len,
        preview=args.preview,
        save_metadata=args.save_metadata,
        output_dir=args.output_dir,
        dataset_profile=args.dataset_profile,
        ipa_workers=args.workers,
        ipa_batch_size=args.ipa_batch_size,
        fast_mode=args.fast,
        skip_refinement=args.fast,
        espeak_retries=0 if args.fast else 1,
    )


def write_outputs(
    profile: DatasetProfile,
    config: GeneratorConfig,
    split_result,
    summary: dict,
) -> Path:
    out_dir = ensure_dir(Path(config.output_dir) / profile.name)

    splits = {"train": split_result.train, "val": split_result.val, "test": split_result.test}
    for split_name, recs in splits.items():
        if not recs:
            continue
        write_jsonl(out_dir / f"{split_name}.jsonl", (r.to_json(with_meta=False) for r in recs))
        if config.save_metadata:
            write_jsonl(out_dir / f"{split_name}_with_meta.jsonl", (r.to_json(with_meta=True) for r in recs))

    if split_result.conflicts:
        write_jsonl(out_dir / "conflict.jsonl", (r.to_json(with_meta=True) for r in split_result.conflicts))

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Wrote outputs to %s", out_dir)
    return out_dir


def _estimate_deficit_gap(
    deficit: Deficit,
    train_cov: Dict[str, int],
    diff: Dict[str, float],
    th,
    pool_target: int,
) -> int:
    kind = deficit.kind
    if kind == "poi_name_coverage":
        return max(0, th.unique_poi_name_min - train_cov.get("poi_name", 0))
    if kind == "route_name_coverage":
        return max(0, th.unique_route_name_min - train_cov.get("route_name", 0))
    if kind == "road_name_coverage":
        gap = max(0, th.unique_road_name_min - train_cov.get("road_name", 0))
        return min(max(gap, 100), 800)
    if kind == "long_entity_heavy":
        lsr = diff.get("long_sentence_ratio", 0)
        ehr = diff.get("entity_heavy_ratio", 0)
        if lsr >= th.long_sentence_ratio_min and ehr >= th.entity_heavy_ratio_min:
            return 0
        long_need = max(0, int((th.long_sentence_ratio_min - lsr) * pool_target * 0.9))
        heavy_need = max(0, int((th.entity_heavy_ratio_min - ehr) * pool_target * 0.9))
        return min(max(min(long_need, heavy_need), 200), 800)
    if kind == "long_sentence":
        current = diff.get("long_sentence_ratio", 0)
        if current >= th.long_sentence_ratio_min:
            return 0
        need = int((th.long_sentence_ratio_min - current) * pool_target * 0.9)
        return min(max(need, 100), 700)
    if kind == "entity_heavy":
        current = diff.get("entity_heavy_ratio", 0)
        if current >= th.entity_heavy_ratio_min or current >= 0.38 or current > th.entity_heavy_ratio_max:
            return 0
        need = int((th.entity_heavy_ratio_min - current) * pool_target * 0.9) + 120
        return min(max(need, 200), 1000)
    if kind == "long_tail":
        current = diff.get("long_tail_ratio", 0)
        if current >= 0.12:
            return 0
        return max(int((0.12 - current) * pool_target), 300)
    if kind == "near_duplicate_high":
        return int(pool_target * 0.05)
    return 200


def _rebuild_entity_coverage(
    generator: DatasetGenerator,
    unique_pool: List[SampleRecord],
) -> None:
    """Align generator coverage with deduped unique pool (fixes false growth)."""
    generator.entity_coverage.clear()
    for rec in unique_pool:
        for etype, val in rec.entity_keys().items():
            generator.entity_coverage[etype].add(canonical_entity(val, etype))


def _log_diagnostic(summary: dict, generator: DatasetGenerator) -> None:
    dbg = summary.get("debug", {})
    if not dbg:
        return
    logger.info(
        "road diagnostic: pool_canon=%d train_canon=%d gen_cov=%d gap=%d",
        dbg.get("road_pool_canon_unique", 0),
        dbg.get("road_train_canon_unique", 0),
        dbg.get("generator_road_cov", 0),
        dbg.get("generator_train_road_gap", 0),
    )
    logger.info("top5 template_type: %s", dbg.get("top20_template_type", [])[:5])
    logger.info("top5 template_id: %s", dbg.get("top20_template_id", [])[:5])
    if dbg.get("reject_stats"):
        logger.info("reject_stats: %s", sorted(dbg["reject_stats"].items(), key=lambda x: -x[1])[:8])
    lt = dbg.get("long_tail_by_split", {})
    for split_name in ("train", "val", "test", "conflict"):
        if split_name in lt:
            logger.info(
                "long_tail %s: ratio=%.4f n=%d",
                split_name, lt[split_name].get("long_tail_ratio", 0), lt[split_name].get("n", 0),
            )


def _build_and_validate(
    profile: DatasetProfile,
    config: GeneratorConfig,
    raw_pool: List[SampleRecord],
    generator: DatasetGenerator,
) -> Tuple[object, dict]:
    unique_pool, dup_removed = dedupe_records(raw_pool)
    _rebuild_entity_coverage(generator, unique_pool)

    if profile.test_size > 0 and profile.train_size == 0:
        split_result = assign_test_only(unique_pool[: profile.test_size])
    else:
        split_result = split_records(
            unique_pool,
            train_ratio=profile.train_ratio,
            val_ratio=profile.val_ratio,
            test_ratio=profile.test_ratio,
            target_sizes={
                "train": profile.train_size,
                "val": profile.val_size,
                "test": profile.test_size,
            },
            template_quotas=profile.template_quotas,
        )

    summary_obj = validate_dataset(raw_pool, unique_pool, split_result, profile)
    attach_debug_stats(
        summary_obj,
        unique_pool,
        split_result,
        generator_road_cov=len(generator.entity_coverage.get("road_name", set())),
        reject_stats=dict(generator.dedup_tracker.reject_stats),
    )
    summary = summary_obj.to_dict()
    summary["profile"] = profile.name
    summary["config"] = {
        "seed": config.seed, "locale": config.locale,
        "expand_abbrev": config.expand_abbrev,
        "spoken_numbers": config.spoken_numbers,
        "dedupe": config.dedupe,
    }
    summary["duplicates_removed_in_pool"] = dup_removed
    summary["entity_coverage_achieved"] = {k: len(v) for k, v in generator.entity_coverage.items()}
    _log_diagnostic(summary, generator)
    return split_result, summary


def run(config: GeneratorConfig, strict: Optional[bool] = None) -> Path:
    profile = get_profile(config.dataset_profile)
    th = profile.quality_thresholds
    if strict is None:
        strict = th.enforce_quality_gate

    logger.info("Profile: %s — %s", profile.name, profile.description)

    n_workers = default_ipa_workers(config.ipa_workers)
    ipa_gen = create_ipa_generator(
        backend="espeak",
        espeak_cmd=config.espeak_cmd,
        locale=config.locale,
        retries=config.espeak_retries,
        workers=n_workers,
    )
    if not ipa_gen.health_check():
        logger.error("espeak-ng health check failed. Is '%s' installed?", config.espeak_cmd)
        sys.exit(1)

    pool_target = total_pool_size(profile) or profile.test_size
    if config.fast_mode:
        main_pool_target = int(pool_target * 1.08)
    else:
        # Capacity-aware main pool + refinement — no need for 1.25× oversample IPA pass
        main_pool_target = pool_target
    max_rounds = 0 if config.skip_refinement else th.max_refinement_rounds

    logger.info(
        "Speed: workers=%d, ipa_batch=%d, main_pool=%d, refinement_rounds=%d",
        n_workers, config.ipa_batch_size, main_pool_target, max_rounds,
    )

    generator = DatasetGenerator(profile, config, ipa_gen)
    t0 = time.time()
    raw_pool = generator.generate_pool(main_pool_target)
    logger.info("Pool generation took %.1fs (%d samples)", time.time() - t0, len(raw_pool))
    split_result, summary = _build_and_validate(profile, config, raw_pool, generator)

    # Refinement: one deficit at a time with hard constraints + relaxed dedup
    for round_idx in range(max_rounds):
        if summary["quality_gate"]["passed"]:
            break
        deficits = [Deficit(**d) for d in summary.get("deficits", [])]
        if not deficits:
            break
        deficits.sort(key=lambda d: d.priority)
        logger.info("Refinement round %d: %d deficits", round_idx + 1, len(deficits))
        for action in summary.get("recommended_actions", []):
            logger.info("  %s", action)

        train_cov = summary.get("entity_coverage", {}).get("train", {})
        diff = summary.get("difficulty_balance", {}).get("train", {})
        th = profile.quality_thresholds

        for deficit in deficits:
            if summary["quality_gate"]["passed"]:
                break
            ehr = diff.get("entity_heavy_ratio", 0)
            if deficit.kind == "entity_heavy" and (
                ehr >= 0.38 or ehr > th.entity_heavy_ratio_max
            ):
                continue
            gap = _estimate_deficit_gap(deficit, train_cov, diff, th, pool_target)
            if gap <= 0:
                continue
            batch = min(max(gap, 100), int(pool_target * 0.15))
            logger.info("  Targeting %s: gap=%d batch=%d", deficit.kind, gap, batch)

            train_gap_canon: Optional[Set[str]] = None
            coverage_slot = COVERAGE_DEFICIT_SLOTS.get(deficit.kind)
            before_cov = 0
            if coverage_slot:
                train_gap_canon = {
                    canonical_entity(r.slots[coverage_slot], coverage_slot)
                    for r in split_result.train
                    if r.slots.get(coverage_slot)
                }
                before_cov = len(train_gap_canon)

            extra = generator.generate_for_deficit(deficit, batch, train_gap_canon=train_gap_canon)
            raw_pool.extend(extra)
            split_result, summary = _build_and_validate(profile, config, raw_pool, generator)

            if coverage_slot:
                after_cov = summary.get("entity_coverage", {}).get("train", {}).get(coverage_slot, 0)
                logger.info(
                    "  %s delta: train_canon %d -> %d (+%d)",
                    coverage_slot, before_cov, after_cov, after_cov - before_cov,
                )

            train_cov = summary.get("entity_coverage", {}).get("train", {})
            diff = summary.get("difficulty_balance", {}).get("train", {})

    failures = summary["quality_gate"].get("failures", [])
    if not summary["quality_gate"]["passed"] and any("RULE_11" in f for f in failures):
        train_diff = summary.get("difficulty_balance", {}).get("train", {})
        ehr = train_diff.get("entity_heavy_ratio", 0)
        th = profile.quality_thresholds
        if ehr < th.entity_heavy_ratio_min:
            tn = max(len(split_result.train), profile.train_size)
            gap = int((th.entity_heavy_ratio_min - ehr) * tn) + 120
            batch = min(max(gap * 3, 400), 2000)
            logger.info(
                "RULE_11 entity_heavy_final_pass: train_heavy_ratio=%.4f target=%.2f batch=%d",
                ehr, th.entity_heavy_ratio_min, batch,
            )
            extra = generator.generate_entity_heavy_final_pass(batch)
            raw_pool.extend(extra)
            split_result, summary = _build_and_validate(profile, config, raw_pool, generator)

    if config.preview > 0:
        preview_records(split_result.train + split_result.val + split_result.test, config.preview)

    out = write_outputs(profile, config, split_result, summary)

    if summary["quality_gate"]["passed"]:
        logger.info("Quality gate PASSED")
    else:
        logger.warning("Quality gate FAILED: %s", summary["quality_gate"]["failures"])
        for w in summary["quality_gate"].get("warnings", []):
            logger.warning("  %s", w)
        if strict:
            sys.exit(1)

    return out


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    run(config, strict=args.strict)


if __name__ == "__main__":
    main()
