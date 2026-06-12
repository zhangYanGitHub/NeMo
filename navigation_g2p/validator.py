"""Dataset quality validation, quality gate, and deficit computation."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dataset_profiles import DatasetProfile, QualityThresholds
from dedup import canonical_entity, compute_dedup_audit
from entity_sets import FOREIGN_STYLE_NAMES
from splitter import LeakageReport, SplitResult
from utils import SampleRecord, text_length_bucket, word_count

ENTITY_SLOT_TYPES = (
    "road_name", "poi_name", "city_name", "district_name",
    "route_name", "address_string",
)
ENTITY_LEAKAGE_WEIGHTS = {
    "address_string": 3.0,
    "poi_name": 2.0,
    "road_name": 2.0,
    "route_name": 1.5,
    "city_name": 1.0,
    "district_name": 1.0,
    "street_name": 1.0,
}
LONG_TAIL_TOKENS = {t.lower() for t in FOREIGN_STYLE_NAMES}


@dataclass
class ValidationConfig:
    min_text_len: int = 3
    max_text_len: int = 300
    min_ipa_len: int = 1
    max_ipa_len: int = 500
    max_exact_duplicate_count: int = 3
    max_template_share: float = 0.25


@dataclass
class QualityGateResult:
    passed: bool
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {"passed": self.passed, "failures": self.failures, "warnings": self.warnings}


@dataclass
class Deficit:
    kind: str
    current: float
    target: float
    priority: int
    action: str

    def to_dict(self) -> Dict:
        return {
            "kind": self.kind,
            "current": self.current,
            "target": self.target,
            "priority": self.priority,
            "action": self.action,
        }


@dataclass
class ValidationSummary:
    total_generated: int = 0
    total_unique: int = 0
    split_counts: Dict[str, int] = field(default_factory=dict)
    template_type_counts: Dict[str, int] = field(default_factory=dict)
    entity_type_counts: Dict[str, int] = field(default_factory=dict)
    text_length_distribution: Dict[str, int] = field(default_factory=dict)
    ipa_length_distribution: Dict[str, int] = field(default_factory=dict)
    duplicate_stats: Dict[str, int] = field(default_factory=dict)
    leakage: Dict = field(default_factory=dict)
    anomaly_counts: Dict[str, int] = field(default_factory=dict)
    anomaly_samples: List[Dict] = field(default_factory=list)
    template_distribution_by_split: Dict[str, Dict[str, int]] = field(default_factory=dict)
    quality_gate: QualityGateResult = field(default_factory=lambda: QualityGateResult(passed=True))
    dedup_audit: Dict = field(default_factory=dict)
    entity_coverage: Dict[str, Dict[str, int]] = field(default_factory=dict)
    difficulty_balance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    deficits: List[Deficit] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    conflict_sample_count: int = 0
    debug: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "total_generated": self.total_generated,
            "total_unique": self.total_unique,
            "split_counts": self.split_counts,
            "template_type_counts": self.template_type_counts,
            "entity_type_counts": self.entity_type_counts,
            "text_length_distribution": self.text_length_distribution,
            "ipa_length_distribution": self.ipa_length_distribution,
            "duplicate_stats": self.duplicate_stats,
            "leakage": self.leakage,
            "anomaly_counts": self.anomaly_counts,
            "anomaly_samples": self.anomaly_samples[:50],
            "template_distribution_by_split": self.template_distribution_by_split,
            "quality_gate": self.quality_gate.to_dict(),
            "dedup_audit": self.dedup_audit,
            "entity_coverage": self.entity_coverage,
            "difficulty_balance": self.difficulty_balance,
            "deficits": [d.to_dict() for d in self.deficits],
            "recommended_actions": self.recommended_actions,
            "conflict_sample_count": self.conflict_sample_count,
            "debug": self.debug,
        }


def build_debug_stats(
    unique_records: List[SampleRecord],
    split_result: SplitResult,
    generator_road_cov: int = 0,
    reject_stats: Optional[Dict[str, int]] = None,
) -> Dict:
    """Diagnostic snapshot for summary.json and log output."""
    n = len(unique_records) or 1

    def _road_uniq(recs: List[SampleRecord], canon: bool = False) -> int:
        vals = [r.slots["road_name"] for r in recs if r.slots.get("road_name")]
        if canon:
            vals = [canonical_entity(v, "road_name") for v in vals]
        return len(set(vals))

    train_cov = compute_entity_coverage(split_result.train).get("road_name", 0)
    tt_top = Counter(r.template_type for r in unique_records).most_common(20)
    tid_top = [
        (k, v, round(v / n, 4))
        for k, v in Counter(r.template_id for r in unique_records).most_common(20)
    ]
    ent_top = Counter()
    for rec in unique_records:
        for et, val in rec.slots.items():
            if et in ENTITY_SLOT_TYPES:
                ent_top[f"{et}::{canonical_entity(val, et)}"] += 1

    long_tail = {}
    for split_name, recs in (
        ("train", split_result.train),
        ("val", split_result.val),
        ("test", split_result.test),
        ("conflict", split_result.conflicts),
    ):
        if recs:
            long_tail[split_name] = {
                "n": len(recs),
                **compute_difficulty_balance(recs),
            }

    audit = compute_dedup_audit(unique_records)
    return {
        "road_pool_raw_unique": _road_uniq(unique_records, False),
        "road_pool_canon_unique": _road_uniq(unique_records, True),
        "road_train_canon_unique": train_cov,
        "generator_road_cov": generator_road_cov,
        "generator_train_road_gap": generator_road_cov - train_cov,
        "top20_template_type": tt_top,
        "top20_template_id": tid_top,
        "top20_per_entity": ent_top.most_common(20),
        "top20_numeric_skeleton": audit.get("top_numeric_skeletons", [])[:20],
        "top20_slot_skeleton": audit.get("top_slot_skeletons", [])[:20],
        "reject_stats": reject_stats or {},
        "long_tail_by_split": long_tail,
        "conflict_count": len(split_result.conflicts),
    }


def _ipa_length_bucket(length: int) -> str:
    if length <= 20:
        return "short"
    if length <= 60:
        return "medium"
    return "long"


def _has_invalid_unicode(text: str) -> bool:
    for ch in text:
        if unicodedata.category(ch) == "Cn":
            return True
    return False


def is_long_sentence(text: str, min_words: int = 18) -> bool:
    return word_count(text) >= min_words


def is_entity_heavy(slots: Dict[str, str], min_entities: int = 2) -> bool:
    return sum(1 for s in ENTITY_SLOT_TYPES if slots.get(s)) >= min_entities


def is_long_tail_sample(rec: SampleRecord) -> bool:
    text = rec.text_graphemes.lower()
    return any(tok.lower() in text for tok in LONG_TAIL_TOKENS)


def compute_entity_coverage(records: List[SampleRecord]) -> Dict[str, int]:
    coverage: Dict[str, set] = defaultdict(set)
    for rec in records:
        for etype, val in rec.entity_keys().items():
            coverage[etype].add(canonical_entity(val, etype))
    return {k: len(v) for k, v in coverage.items()}


def compute_difficulty_balance(records: List[SampleRecord]) -> Dict[str, float]:
    n = len(records) or 1
    long_count = sum(1 for r in records if is_long_sentence(r.text_graphemes))
    heavy_count = sum(1 for r in records if is_entity_heavy(r.slots))
    tail_count = sum(1 for r in records if is_long_tail_sample(r))
    return {
        "long_sentence_ratio": round(long_count / n, 4),
        "entity_heavy_ratio": round(heavy_count / n, 4),
        "long_tail_ratio": round(tail_count / n, 4),
    }


def weighted_entity_leakage_ratio(leakage: Dict) -> float:
    counts = leakage.get("leakage_counts", {})
    totals = leakage.get("total_entities_by_type", {})
    if not totals:
        return 0.0
    weighted_leaked = 0.0
    weighted_total = 0.0
    for etype, total in totals.items():
        w = ENTITY_LEAKAGE_WEIGHTS.get(etype, 1.0)
        leaked = counts.get(etype, 0)
        weighted_leaked += leaked * w
        weighted_total += total * w
    return weighted_leaked / weighted_total if weighted_total else 0.0


def compute_quality_gate(
    summary_data: Dict,
    profile: DatasetProfile,
    train_records: List[SampleRecord],
    val_records: List[SampleRecord],
) -> QualityGateResult:
    """Evaluate HARD and SOFT rules against profile thresholds."""
    th = profile.quality_thresholds
    result = QualityGateResult(passed=True)
    dedup = summary_data.get("dedup_audit", {})
    n = summary_data.get("total_unique", 0) or (
        len(train_records) + len(val_records) + summary_data.get("split_counts", {}).get("test", 0)
    )
    n = n or 1

    # --- HARD RULES ---
    # RULE_01
    if dedup.get("exact_duplicate_rate", 0) > th.exact_duplicate_rate_max:
        result.failures.append(
            f"RULE_01 exact_duplicate_rate {dedup['exact_duplicate_rate']:.4f} > {th.exact_duplicate_rate_max}"
        )
    # RULE_02
    if dedup.get("near_duplicate_rate_slot_skeleton", 0) > th.near_duplicate_rate_slot_skeleton_max:
        result.failures.append(
            f"RULE_02 near_duplicate_rate_slot_skeleton "
            f"{dedup['near_duplicate_rate_slot_skeleton']:.4f} > {th.near_duplicate_rate_slot_skeleton_max}"
        )
    # RULE_03
    if dedup.get("effective_unique_ratio", 0) < th.effective_unique_ratio_min:
        result.failures.append(
            f"RULE_03 effective_unique_ratio {dedup.get('effective_unique_ratio')} < {th.effective_unique_ratio_min}"
        )
    # RULE_04
    wl = weighted_entity_leakage_ratio(summary_data.get("leakage", {}))
    if wl > th.weighted_entity_leakage_ratio_max:
        result.failures.append(
            f"RULE_04 weighted_entity_leakage_ratio {wl:.4f} > {th.weighted_entity_leakage_ratio_max}"
        )
    # RULE_05-07 entity coverage on train
    train_cov = compute_entity_coverage(train_records)
    if train_cov.get("road_name", 0) < th.unique_road_name_min:
        result.failures.append(
            f"RULE_05 unique_road_name {train_cov.get('road_name', 0)} < {th.unique_road_name_min}"
        )
    if train_cov.get("poi_name", 0) < th.unique_poi_name_min:
        result.failures.append(
            f"RULE_06 unique_poi_name {train_cov.get('poi_name', 0)} < {th.unique_poi_name_min}"
        )
    if train_cov.get("route_name", 0) < th.unique_route_name_min:
        result.failures.append(
            f"RULE_07 unique_route_name {train_cov.get('route_name', 0)} < {th.unique_route_name_min}"
        )
    # RULE_08-09 template shares (full pool)
    tmpl_type_c = summary_data.get("template_type_counts", {})
    if tmpl_type_c:
        max_type_share = max(tmpl_type_c.values()) / n
        if max_type_share > th.max_template_type_share_max:
            result.failures.append(
                f"RULE_08 max_template_type_share {max_type_share:.4f} > {th.max_template_type_share_max}"
            )
    if dedup.get("per_template_id_max_share", 0) > th.max_template_id_share_max:
        result.failures.append(
            f"RULE_09 max_template_id_share {dedup['per_template_id_max_share']:.4f} "
            f"> {th.max_template_id_share_max}"
        )
    # RULE_10-11 difficulty on train
    train_diff = compute_difficulty_balance(train_records)
    lsr = train_diff["long_sentence_ratio"]
    if lsr < th.long_sentence_ratio_min or lsr > th.long_sentence_ratio_max:
        result.failures.append(
            f"RULE_10 long_sentence_ratio {lsr:.4f} not in [{th.long_sentence_ratio_min}, {th.long_sentence_ratio_max}]"
        )
    ehr = train_diff["entity_heavy_ratio"]
    if ehr < th.entity_heavy_ratio_min or ehr > th.entity_heavy_ratio_max:
        result.failures.append(
            f"RULE_11 entity_heavy_ratio {ehr:.4f} not in [{th.entity_heavy_ratio_min}, {th.entity_heavy_ratio_max}]"
        )
    # RULE_12-13 anomalies
    anomalies = summary_data.get("anomaly_counts", {})
    if anomalies.get("generation_failed", 0) > 0:
        result.failures.append(f"RULE_12 ipa_generation_failure_count {anomalies['generation_failed']}")
    if anomalies.get("empty_field", 0) > 0:
        result.failures.append(f"RULE_13 empty_field_count {anomalies['empty_field']}")

    # RULE_14 val difficulty balance
    if val_records and train_records:
        val_diff = compute_difficulty_balance(val_records)
        train_tail = train_diff.get("long_tail_ratio", 0)
        val_tail = val_diff.get("long_tail_ratio", 0)
        if train_tail > 0 and val_tail < th.val_long_tail_ratio_of_train_min * train_tail:
            result.failures.append(
                f"RULE_14 val_long_tail_ratio {val_tail:.4f} < "
                f"{th.val_long_tail_ratio_of_train_min} * train {train_tail:.4f}"
            )

    # --- SOFT RULES (warnings) ---
    if dedup.get("near_duplicate_rate_numeric_skeleton", 0) > 0.06:
        result.warnings.append("WARN_01 near_duplicate_rate_numeric_skeleton > 0.06")
    per_entity = dedup.get("per_entity_max_count", {})
    if per_entity and max(per_entity.values()) > 8:
        result.warnings.append("WARN_02 per_entity_max_count > 8")
    basic_share = tmpl_type_c.get("basic_actions", 0) / n if tmpl_type_c else 0
    if basic_share > 0.15:
        result.warnings.append(f"WARN_03 basic_actions_share {basic_share:.4f} > 0.15")
    if summary_data.get("conflict_sample_count", 0) > 50:
        result.warnings.append("WARN_04 conflict_entity_sample_count > 50")

    result.passed = len(result.failures) == 0
    return result


def compute_deficits(
    summary_data: Dict,
    profile: DatasetProfile,
    train_records: List[SampleRecord],
) -> Tuple[List[Deficit], List[str]]:
    """Compute remediation deficits sorted by priority."""
    th = profile.quality_thresholds
    deficits: List[Deficit] = []
    actions: List[str] = []
    train_cov = compute_entity_coverage(train_records)
    train_diff = compute_difficulty_balance(train_records)
    dedup = summary_data.get("dedup_audit", {})
    wl = weighted_entity_leakage_ratio(summary_data.get("leakage", {}))

    if dedup.get("near_duplicate_rate_slot_skeleton", 0) > th.near_duplicate_rate_slot_skeleton_max:
        deficits.append(Deficit(
            "near_duplicate_high", dedup["near_duplicate_rate_slot_skeleton"],
            th.near_duplicate_rate_slot_skeleton_max, 1,
            "Stop random generation; emit new_entity x new_template_id pairs only",
        ))
    if wl > th.weighted_entity_leakage_ratio_max:
        deficits.append(Deficit(
            "leakage_high", wl, th.weighted_entity_leakage_ratio_max, 2,
            "Remove conflict samples; regenerate within same split pool only",
        ))
    if train_cov.get("poi_name", 0) < th.unique_poi_name_min:
        deficits.append(Deficit(
            "poi_name_coverage", train_cov.get("poi_name", 0), th.unique_poi_name_min, 3,
            "Target poi_city_target / mixed_longform with uncovered POI overrides",
        ))
    if train_cov.get("road_name", 0) < th.unique_road_name_min:
        deficits.append(Deficit(
            "road_name_coverage", train_cov.get("road_name", 0), th.unique_road_name_min, 3,
            "Target road_navigation / mixed_longform with uncovered road overrides",
        ))
    if train_cov.get("route_name", 0) < th.unique_route_name_min:
        deficits.append(Deficit(
            "route_name_coverage", train_cov.get("route_name", 0), th.unique_route_name_min, 4,
            "Target numbered_routes with HYPHENATED_ROUTES pool",
        ))
    lsr = train_diff["long_sentence_ratio"]
    ehr = train_diff["entity_heavy_ratio"]
    if lsr < th.long_sentence_ratio_min and ehr < th.entity_heavy_ratio_min:
        deficits.append(Deficit(
            "long_entity_heavy", min(lsr, ehr),
            th.long_sentence_ratio_min, 4,
            "Generate long (>=18 words) templates with >=2 entity slots",
        ))
    if lsr < th.long_sentence_ratio_min:
        deficits.append(Deficit(
            "long_sentence", lsr,
            th.long_sentence_ratio_min, 5,
            "Generate length_hint=long templates only (mixed_longform, replay)",
        ))
    if ehr < th.entity_heavy_ratio_min:
        deficits.append(Deficit(
            "entity_heavy", ehr,
            th.entity_heavy_ratio_min, 5,
            "Generate templates with >=2 entity slots",
        ))
    # Skip entity_heavy补样 when already near upper bound (RULE_11)
    if train_diff["long_tail_ratio"] < 0.12:
        deficits.append(Deficit(
            "long_tail", train_diff["long_tail_ratio"], 0.12, 6,
            "Inject FOREIGN_STYLE_NAMES via entity_overrides",
        ))

    deficits.sort(key=lambda d: d.priority)
    for d in deficits:
        actions.append(f"[P{d.priority}] {d.kind}: {d.action}")
    return deficits, actions


def validate_dataset(
    all_records: List[SampleRecord],
    unique_records: List[SampleRecord],
    split_result: SplitResult,
    profile: DatasetProfile,
    config: Optional[ValidationConfig] = None,
) -> ValidationSummary:
    cfg = config or ValidationConfig()
    summary = ValidationSummary(
        total_generated=len(all_records),
        total_unique=len(unique_records),
        leakage=split_result.leakage_report.to_dict(),
        template_distribution_by_split=split_result.template_distribution,
        conflict_sample_count=len(split_result.conflicts),
    )

    split_records_map = {
        "train": split_result.train,
        "val": split_result.val,
        "test": split_result.test,
    }
    summary.split_counts = {k: len(v) for k, v in split_records_map.items()}

    grapheme_counter: Counter = Counter()
    template_counter: Counter = Counter()
    entity_counter: Counter = Counter()

    for rec in unique_records:
        template_counter[rec.template_type] += 1
        grapheme_counter[rec.dedupe_key()] += 1
        bucket = text_length_bucket(rec.text_graphemes)
        summary.text_length_distribution[bucket] = summary.text_length_distribution.get(bucket, 0) + 1
        ipa_bucket = _ipa_length_bucket(len(rec.text))
        summary.ipa_length_distribution[ipa_bucket] = summary.ipa_length_distribution.get(ipa_bucket, 0) + 1
        for etype in rec.entity_keys():
            entity_counter[etype] += 1

    summary.template_type_counts = dict(template_counter)
    summary.entity_type_counts = dict(entity_counter)

    exact_dups = sum(1 for _, c in grapheme_counter.items() if c > 1)
    heavy_dups = sum(1 for _, c in grapheme_counter.items() if c > cfg.max_exact_duplicate_count)
    summary.duplicate_stats = {
        "duplicate_grapheme_keys": exact_dups,
        "heavy_duplicate_keys": heavy_dups,
        "max_duplicate_count": max(grapheme_counter.values()) if grapheme_counter else 0,
    }

    anomalies: Dict[str, int] = defaultdict(int)
    anomaly_samples: List[Dict] = []

    def _flag(kind: str, rec: SampleRecord) -> None:
        anomalies[kind] += 1
        if len(anomaly_samples) < 50:
            anomaly_samples.append({
                "type": kind,
                "text_graphemes": rec.text_graphemes,
                "text": rec.text,
                "template_id": rec.template_id,
            })

    for rec in unique_records:
        if not rec.text_graphemes or not rec.text:
            _flag("empty_field", rec)
        if len(rec.text_graphemes) < cfg.min_text_len:
            _flag("text_too_short", rec)
        if len(rec.text_graphemes) > cfg.max_text_len:
            _flag("text_too_long", rec)
        if len(rec.text) < cfg.min_ipa_len:
            _flag("ipa_too_short", rec)
        if len(rec.text) > cfg.max_ipa_len:
            _flag("ipa_too_long", rec)
        if rec.generation_error:
            _flag("generation_failed", rec)
        if _has_invalid_unicode(rec.text_graphemes) or _has_invalid_unicode(rec.text):
            _flag("invalid_unicode", rec)

    total = len(unique_records) or 1
    for ttype, count in template_counter.items():
        if count / total > cfg.max_template_share:
            anomalies["overrepresented_template"] += 1

    summary.anomaly_counts = dict(anomalies)
    summary.anomaly_samples = anomaly_samples

    # Extended audits
    summary.dedup_audit = compute_dedup_audit(unique_records)
    summary.entity_coverage = {
        split: compute_entity_coverage(recs)
        for split, recs in split_records_map.items() if recs
    }
    summary.difficulty_balance = {
        split: compute_difficulty_balance(recs)
        for split, recs in split_records_map.items() if recs
    }

    interim = summary.to_dict()
    summary.quality_gate = compute_quality_gate(
        interim, profile, split_result.train, split_result.val,
    )
    summary.deficits, summary.recommended_actions = compute_deficits(
        interim, profile, split_result.train,
    )
    return summary


def attach_debug_stats(
    summary: ValidationSummary,
    unique_records: List[SampleRecord],
    split_result: SplitResult,
    generator_road_cov: int = 0,
    reject_stats: Optional[Dict[str, int]] = None,
) -> None:
    summary.debug = build_debug_stats(
        unique_records, split_result, generator_road_cov, reject_stats,
    )
