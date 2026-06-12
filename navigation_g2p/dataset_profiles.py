"""Dataset profile definitions with target sizes and generation policies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class QualityThresholds:
    """Profile-level quality gate thresholds for training acceptance."""

    exact_duplicate_rate_max: float = 0.001
    near_duplicate_rate_slot_skeleton_max: float = 0.08
    effective_unique_ratio_min: float = 0.55
    weighted_entity_leakage_ratio_max: float = 0.02
    unique_road_name_min: int = 320
    unique_poi_name_min: int = 280
    unique_route_name_min: int = 240
    max_template_type_share_max: float = 0.20
    max_template_id_share_max: float = 0.03
    long_sentence_ratio_min: float = 0.15
    long_sentence_ratio_max: float = 0.25
    entity_heavy_ratio_min: float = 0.25
    entity_heavy_ratio_max: float = 0.40
    val_long_tail_ratio_of_train_min: float = 0.70
    oversample_factor: float = 1.25
    max_refinement_rounds: int = 3
    enforce_quality_gate: bool = True


# Relaxed thresholds for smoke / test-only profiles
_MINI_THRESHOLDS = QualityThresholds(
    exact_duplicate_rate_max=0.05,
    near_duplicate_rate_slot_skeleton_max=0.30,
    effective_unique_ratio_min=0.30,
    weighted_entity_leakage_ratio_max=0.20,
    unique_road_name_min=5,
    unique_poi_name_min=5,
    unique_route_name_min=5,
    max_template_type_share_max=0.50,
    max_template_id_share_max=0.20,
    long_sentence_ratio_min=0.0,
    long_sentence_ratio_max=1.0,
    entity_heavy_ratio_min=0.0,
    entity_heavy_ratio_max=1.0,
    val_long_tail_ratio_of_train_min=0.0,
    oversample_factor=1.15,
    max_refinement_rounds=1,
    enforce_quality_gate=False,
)

_ENTITIES_THRESHOLDS = QualityThresholds(
    exact_duplicate_rate_max=0.005,
    near_duplicate_rate_slot_skeleton_max=0.10,
    effective_unique_ratio_min=0.50,
    weighted_entity_leakage_ratio_max=1.0,
    unique_road_name_min=150,
    unique_poi_name_min=150,
    unique_route_name_min=100,
    max_template_type_share_max=0.30,
    max_template_id_share_max=0.05,
    long_sentence_ratio_min=0.10,
    long_sentence_ratio_max=0.40,
    entity_heavy_ratio_min=0.20,
    entity_heavy_ratio_max=0.60,
    val_long_tail_ratio_of_train_min=0.0,
    oversample_factor=1.05,
    max_refinement_rounds=2,
    enforce_quality_gate=False,
)


@dataclass
class DatasetProfile:
    """Defines scale, split ratios, and generation policy for a dataset variant."""

    name: str
    description: str
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0
    train_ratio: float = 0.0
    val_ratio: float = 0.0
    test_ratio: float = 0.0
    # template_type -> target fraction of generated pool (before split)
    template_quotas: Dict[str, float] = field(default_factory=dict)
    # entity slot -> minimum unique entities to cover
    entity_min_coverage: Dict[str, int] = field(default_factory=dict)
    # boost long-tail / difficult entities
    long_tail_boost: float = 1.0
    # prefer replay-style templates
    use_replay_templates: bool = False
    # entity-focused generation
    entity_focused: bool = False
    source_generator: str = "template_v1"
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)


PROFILES: Dict[str, DatasetProfile] = {
    "mini_seed": DatasetProfile(
        name="mini_seed",
        description="Tiny dataset for smoke testing the pipeline.",
        train_size=50,
        val_size=10,
        test_size=10,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        template_quotas={
            "basic_actions": 0.20,
            "distance_prefixed": 0.15,
            "road_navigation": 0.15,
            "numbered_routes": 0.10,
            "arrival": 0.10,
            "poi_city_target": 0.10,
            "address_based": 0.10,
            "mixed_longform": 0.10,
        },
        entity_min_coverage={
            "road_name": 10,
            "poi_name": 10,
            "city_name": 10,
            "route_name": 10,
            "address_string": 10,
        },
        quality_thresholds=_MINI_THRESHOLDS,
        source_generator="template_v1",
    ),
    "nav_baseline_v1": DatasetProfile(
        name="nav_baseline_v1",
        description="Medium-scale baseline for development iteration.",
        train_size=2000,
        val_size=200,
        test_size=200,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        template_quotas={
            "basic_actions": 0.15,
            "distance_prefixed": 0.15,
            "road_navigation": 0.15,
            "numbered_routes": 0.12,
            "arrival": 0.08,
            "poi_city_target": 0.12,
            "address_based": 0.10,
            "mixed_longform": 0.13,
        },
        entity_min_coverage={
            "road_name": 80,
            "poi_name": 80,
            "city_name": 60,
            "district_name": 40,
            "route_name": 60,
            "address_string": 60,
        },
        quality_thresholds=QualityThresholds(
            unique_road_name_min=80,
            unique_poi_name_min=70,
            unique_route_name_min=60,
            effective_unique_ratio_min=0.45,
            enforce_quality_gate=False,
        ),
        source_generator="template_v1",
    ),
    "nav_prod_v1": DatasetProfile(
        name="nav_prod_v1",
        description="Production training dataset for navigation G2P.",
        train_size=20000,
        val_size=1000,
        test_size=1000,
        train_ratio=0.90,
        val_ratio=0.05,
        test_ratio=0.05,
        template_quotas={
            "basic_actions": 0.06,
            "distance_prefixed": 0.12,
            "road_navigation": 0.10,
            "numbered_routes": 0.12,
            "arrival": 0.06,
            "poi_city_target": 0.15,
            "address_based": 0.10,
            "mixed_longform": 0.18,
        },
        long_tail_boost=1.5,
        entity_min_coverage={
            "road_name": 400,
            "poi_name": 350,
            "city_name": 250,
            "district_name": 150,
            "route_name": 300,
            "address_string": 250,
        },
        source_generator="template_v1",
    ),
    "nav_entities_v1": DatasetProfile(
        name="nav_entities_v1",
        description="Entity-focused test set for long-tail pronunciation coverage.",
        train_size=0,
        val_size=0,
        test_size=1000,
        train_ratio=0.0,
        val_ratio=0.0,
        test_ratio=1.0,
        template_quotas={
            "road_navigation": 0.18,
            "numbered_routes": 0.15,
            "poi_city_target": 0.20,
            "address_based": 0.22,
            "mixed_longform": 0.15,
            "distance_prefixed": 0.10,
        },
        entity_min_coverage={
            "road_name": 200,
            "poi_name": 200,
            "city_name": 150,
            "district_name": 120,
            "route_name": 150,
            "address_string": 180,
        },
        long_tail_boost=2.5,
        entity_focused=True,
        quality_thresholds=_ENTITIES_THRESHOLDS,
        source_generator="entity_v1",
    ),
    "nav_replay_v1": DatasetProfile(
        name="nav_replay_v1",
        description="Business-style replay test set mimicking real navigation prompts.",
        train_size=0,
        val_size=0,
        test_size=400,
        train_ratio=0.0,
        val_ratio=0.0,
        test_ratio=1.0,
        template_quotas={},
        entity_min_coverage={},
        use_replay_templates=True,
        quality_thresholds=_ENTITIES_THRESHOLDS,
        source_generator="replay_v1",
    ),
}


def get_profile(name: str) -> DatasetProfile:
    if name not in PROFILES:
        available = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown dataset profile '{name}'. Available: {available}")
    return PROFILES[name]


def total_pool_size(profile: DatasetProfile) -> int:
    return profile.train_size + profile.val_size + profile.test_size
