"""Navigation utterance templates organized by category."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass(frozen=True)
class Template:
    """A single navigation template with metadata."""

    template_id: str
    template_type: str
    pattern: str
    required_slots: tuple[str, ...] = ()
    optional_slots: tuple[str, ...] = ()
    tags: frozenset[str] = frozenset()
    length_hint: str = "medium"  # short | medium | long

    def slot_names(self) -> Set[str]:
        names: Set[str] = set()
        for group in (self.required_slots, self.optional_slots):
            for slot in group:
                if slot.startswith("?"):
                    names.add(slot[1:])
                else:
                    names.add(slot)
        return names


def _t(
    template_id: str,
    template_type: str,
    pattern: str,
    required: tuple[str, ...] = (),
    optional: tuple[str, ...] = (),
    tags: Optional[Set[str]] = None,
    length_hint: str = "medium",
) -> Template:
    return Template(
        template_id=template_id,
        template_type=template_type,
        pattern=pattern,
        required_slots=required,
        optional_slots=optional,
        tags=frozenset(tags or set()),
        length_hint=length_hint,
    )


# ---------------------------------------------------------------------------
# basic_actions
# ---------------------------------------------------------------------------
BASIC_ACTIONS: List[Template] = [
    _t("ba_001", "basic_actions", "Turn {direction}", ("direction",), length_hint="short"),
    _t("ba_002", "basic_actions", "Turn {direction} at the next intersection", ("direction",)),
    _t("ba_003", "basic_actions", "Turn {direction} at the traffic light", ("direction",)),
    _t("ba_004", "basic_actions", "Turn {direction} at the stop sign", ("direction",)),
    _t("ba_005", "basic_actions", "Keep {direction}", ("direction",), length_hint="short"),
    _t("ba_006", "basic_actions", "Keep {direction} at the fork", ("direction",)),
    _t("ba_007", "basic_actions", "Keep {direction} to stay on the current road", ("direction",)),
    _t("ba_008", "basic_actions", "Bear {direction}", ("direction",)),
    _t("ba_009", "basic_actions", "Continue straight", (), length_hint="short"),
    _t("ba_010", "basic_actions", "Continue straight ahead", (), length_hint="short"),
    _t("ba_011", "basic_actions", "Continue straight through the intersection", ()),
    _t("ba_012", "basic_actions", "Continue straight through the traffic light", ()),
    _t("ba_013", "basic_actions", "Make a U-turn", (), length_hint="short"),
    _t("ba_014", "basic_actions", "Make a U-turn when possible", ()),
    _t("ba_015", "basic_actions", "Make a U-turn at the next safe location", ()),
    _t("ba_016", "basic_actions", "Merge onto {route_name}", ("route_name",)),
    _t("ba_017", "basic_actions", "Merge left onto {route_name}", ("route_name",)),
    _t("ba_018", "basic_actions", "Merge right onto {route_name}", ("route_name",)),
    _t("ba_019", "basic_actions", "Take the exit", (), length_hint="short"),
    _t("ba_020", "basic_actions", "Take the next exit", (), length_hint="short"),
    _t("ba_021", "basic_actions", "Take Exit {exit_no}", ("exit_no",)),
    _t("ba_022", "basic_actions", "Take Exit {exit_no} toward {poi_name}", ("exit_no", "poi_name")),
    _t("ba_023", "basic_actions", "Enter the roundabout", (), length_hint="short"),
    _t("ba_024", "basic_actions", "Enter the roundabout and take the {ordinal} exit", ("ordinal",)),
    _t("ba_025", "basic_actions", "At the roundabout, take the {ordinal} exit", ("ordinal",)),
    _t("ba_026", "basic_actions", "At the roundabout, take the {ordinal} exit onto {road_name}",
         ("ordinal", "road_name")),
    _t("ba_027", "basic_actions", "Use {lane_count_phrase} to turn {direction}",
         ("lane_count_phrase", "direction")),
    _t("ba_028", "basic_actions", "Use {lane_count_phrase} to take Exit {exit_no}",
         ("lane_count_phrase", "exit_no")),
]

# ---------------------------------------------------------------------------
# distance_prefixed
# ---------------------------------------------------------------------------
DISTANCE_PREFIXED: List[Template] = [
    _t("dp_001", "distance_prefixed", "In {distance_phrase}, turn {direction}",
         ("distance_phrase", "direction")),
    _t("dp_002", "distance_prefixed", "In {distance_phrase}, turn {direction} onto {road_name}",
         ("distance_phrase", "direction", "road_name")),
    _t("dp_003", "distance_prefixed", "In {distance_phrase}, keep {direction}",
         ("distance_phrase", "direction")),
    _t("dp_004", "distance_prefixed", "In {distance_phrase}, continue straight",
         ("distance_phrase",), length_hint="short"),
    _t("dp_005", "distance_prefixed", "In {distance_phrase}, take Exit {exit_no}",
         ("distance_phrase", "exit_no")),
    _t("dp_006", "distance_prefixed", "In {distance_phrase}, take Exit {exit_no} toward {poi_name}",
         ("distance_phrase", "exit_no", "poi_name")),
    _t("dp_007", "distance_prefixed", "After {distance_phrase}, turn {direction}",
         ("distance_phrase", "direction")),
    _t("dp_008", "distance_prefixed", "After {distance_phrase}, turn {direction} onto {road_name}",
         ("distance_phrase", "direction", "road_name")),
    _t("dp_009", "distance_prefixed", "After {distance_phrase}, keep {direction}",
         ("distance_phrase", "direction")),
    _t("dp_010", "distance_prefixed", "Continue for {distance_phrase}, then turn {direction}",
         ("distance_phrase", "direction"), length_hint="long"),
    _t("dp_011", "distance_prefixed", "Continue for {distance_phrase}, then turn {direction} onto {road_name}",
         ("distance_phrase", "direction", "road_name"), length_hint="long"),
    _t("dp_012", "distance_prefixed", "Continue for {distance_phrase}, then take Exit {exit_no}",
         ("distance_phrase", "exit_no"), length_hint="long"),
    _t("dp_013", "distance_prefixed", "In half a mile, turn {direction} onto {road_name}",
         ("direction", "road_name")),
    _t("dp_014", "distance_prefixed", "In a quarter mile, turn {direction}",
         ("direction",)),
    _t("dp_015", "distance_prefixed", "In {distance_phrase}, merge onto {route_name}",
         ("distance_phrase", "route_name")),
    _t("dp_016", "distance_prefixed", "In {distance_phrase}, enter the roundabout",
         ("distance_phrase",)),
    _t("dp_017", "distance_prefixed", "In {distance_phrase}, make a U-turn",
         ("distance_phrase",)),
    _t("dp_018", "distance_prefixed",
         "In three quarters of a mile, turn {direction} onto {road_name} and continue straight toward {city_name}",
         ("direction", "road_name", "city_name"), length_hint="long"),
    _t("dp_019", "distance_prefixed",
         "Continue for two thousand meters, then turn {direction} onto {road_name} and head straight toward {poi_name}",
         ("direction", "road_name", "poi_name"), length_hint="long"),
    _t("dp_020", "distance_prefixed",
         "After one point five kilometers, keep {direction} to stay on {road_name} and continue toward {district_name}",
         ("direction", "road_name", "district_name"), length_hint="long"),
    _t("dp_021", "distance_prefixed",
         "After one point five kilometers, turn {direction} onto {road_name} and continue straight toward {poi_name} in {city_name}",
         ("direction", "road_name", "poi_name", "city_name"), length_hint="long"),
]

# ---------------------------------------------------------------------------
# road_navigation
# ---------------------------------------------------------------------------
ROAD_NAVIGATION: List[Template] = [
    _t("rn_001", "road_navigation", "Turn {direction} onto {road_name}", ("direction", "road_name")),
    _t("rn_002", "road_navigation", "Turn {direction} onto {road_name} toward {city_name}",
         ("direction", "road_name", "city_name"), length_hint="long"),
    _t("rn_003", "road_navigation", "Continue onto {road_name}", ("road_name",)),
    _t("rn_004", "road_navigation", "Continue straight onto {road_name}", ("road_name",)),
    _t("rn_005", "road_navigation", "Stay on {road_name}", ("road_name",), length_hint="short"),
    _t("rn_006", "road_navigation", "Stay on {road_name} toward {district_name}",
         ("road_name", "district_name")),
    _t("rn_007", "road_navigation", "Keep left to stay on {road_name}", ("road_name",)),
    _t("rn_008", "road_navigation", "Keep right to stay on {road_name}", ("road_name",)),
    _t("rn_009", "road_navigation", "Bear left onto {road_name}", ("road_name",)),
    _t("rn_010", "road_navigation", "Bear right onto {road_name}", ("road_name",)),
    _t("rn_011", "road_navigation", "Turn {direction} onto {road_direction_prefix} {road_core_name} {road_type}",
         ("direction", "road_direction_prefix", "road_core_name", "road_type"), tags={"structured_road"}),
    _t("rn_012", "road_navigation", "Continue on {road_name}", ("road_name",)),
    _t("rn_013", "road_navigation", "Continue on {road_name} for {distance_phrase}",
         ("road_name", "distance_phrase"), length_hint="long"),
    _t("rn_014", "road_navigation", "Merge onto {road_name}", ("road_name",)),
    _t("rn_015", "road_navigation", "Take the ramp onto {road_name}", ("road_name",)),
    _t("rn_016", "road_navigation", "Use the left lane to turn {direction} onto {road_name}",
         ("direction", "road_name")),
    _t("rn_017", "road_navigation", "Use the right lane to turn {direction} onto {road_name}",
         ("direction", "road_name")),
    _t("rn_018", "road_navigation",
         "Continue on {road_name} for three quarters of a mile, then head straight toward {district_name}",
         ("road_name", "district_name"), length_hint="long"),
    _t("rn_019", "road_navigation",
         "Use the left lane to turn {direction} onto {road_name} and continue straight toward {city_name}",
         ("direction", "road_name", "city_name"), length_hint="long"),
    _t("rn_021", "road_navigation",
         "In three quarters of a mile, turn {direction} onto {road_name} and head toward {poi_name} in {city_name}",
         ("direction", "road_name", "poi_name", "city_name"), length_hint="long"),
    _t("rn_022", "road_navigation",
         "Continue on {road_name} for one point five kilometers, then head toward {district_name} in {city_name}",
         ("road_name", "district_name", "city_name"), length_hint="long"),
]

# ---------------------------------------------------------------------------
# numbered_routes
# ---------------------------------------------------------------------------
NUMBERED_ROUTES: List[Template] = [
    _t("nr_001", "numbered_routes", "Turn {direction} onto {route_name}", ("direction", "route_name")),
    _t("nr_002", "numbered_routes", "Merge onto {route_name}", ("route_name",)),
    _t("nr_003", "numbered_routes", "Continue on {route_name}", ("route_name",)),
    _t("nr_004", "numbered_routes", "Stay on {route_name}", ("route_name",)),
    _t("nr_005", "numbered_routes", "Keep left to stay on {route_name}", ("route_name",)),
    _t("nr_006", "numbered_routes", "Keep right to stay on {route_name}", ("route_name",)),
    _t("nr_007", "numbered_routes", "Take Exit {exit_no}", ("exit_no",)),
    _t("nr_008", "numbered_routes", "Take Exit {exit_no} toward {city_name}", ("exit_no", "city_name")),
    _t("nr_009", "numbered_routes", "Take Exit {exit_no} toward {poi_name}", ("exit_no", "poi_name")),
    _t("nr_010", "numbered_routes", "Take Exit {exit_no} toward {district_name}", ("exit_no", "district_name")),
    _t("nr_011", "numbered_routes", "Take Exit {exit_no}A", ("exit_no",), tags={"exit_suffix"}),
    _t("nr_012", "numbered_routes", "Take Exit {exit_no}B toward Downtown", ("exit_no",), tags={"exit_suffix"}),
    _t("nr_013", "numbered_routes", "In {distance_phrase}, take Exit {exit_no} toward {poi_name}",
         ("distance_phrase", "exit_no", "poi_name"), length_hint="long"),
    _t("nr_014", "numbered_routes", "Turn {direction} onto {route_prefix}-{route_number} {route_direction}",
         ("direction", "route_prefix", "route_number", "route_direction"), tags={"structured_route"}),
    _t("nr_015", "numbered_routes", "Continue on {route_prefix}-{route_number} {route_direction}",
         ("route_prefix", "route_number", "route_direction"), tags={"structured_route"}),
    _t("nr_016", "numbered_routes", "Merge onto {route_prefix}-{route_number}",
         ("route_prefix", "route_number"), tags={"structured_route"}),
    _t("nr_017", "numbered_routes", "Take the exit for {route_name}", ("route_name",)),
    _t("nr_018", "numbered_routes", "Follow signs for {route_name}", ("route_name",)),
    _t("nr_019", "numbered_routes",
         "In one point five kilometers, continue on {route_name} and head straight toward {poi_name} on your route",
         ("route_name", "poi_name"), length_hint="long"),
    _t("nr_020", "numbered_routes",
         "In three quarters of a mile, take Exit {exit_no} and continue toward {poi_name} in {city_name}",
         ("exit_no", "poi_name", "city_name"), length_hint="long"),
]

# ---------------------------------------------------------------------------
# arrival
# ---------------------------------------------------------------------------
ARRIVAL: List[Template] = [
    _t("ar_001", "arrival", "Your destination is on the {side}", ("side",), length_hint="short"),
    _t("ar_002", "arrival", "The destination is on your {side}", ("side",), length_hint="short"),
    _t("ar_003", "arrival", "Your destination is on the {side} in {distance_phrase}",
         ("side", "distance_phrase")),
    _t("ar_004", "arrival", "The destination is on your {side} in {distance_phrase}",
         ("side", "distance_phrase")),
    _t("ar_005", "arrival", "You have arrived at your destination", (), length_hint="short"),
    _t("ar_006", "arrival", "You have arrived at {poi_name}", ("poi_name",)),
    _t("ar_007", "arrival", "You have arrived at {address_string}", ("address_string",)),
    _t("ar_008", "arrival", "The destination is ahead", (), length_hint="short"),
    _t("ar_009", "arrival", "The destination is ahead on the {side}", ("side",)),
    _t("ar_010", "arrival", "Arriving at {poi_name} on the {side}", ("poi_name", "side")),
    _t("ar_011", "arrival", "Arriving at {address_string}", ("address_string",)),
    _t("ar_012", "arrival", "Your destination, {poi_name}, is on the {side}", ("poi_name", "side")),
]

# ---------------------------------------------------------------------------
# poi_city_target
# ---------------------------------------------------------------------------
POI_CITY_TARGET: List[Template] = [
    _t("pct_001", "poi_city_target", "Head toward {poi_name}", ("poi_name",)),
    _t("pct_002", "poi_city_target", "Continue toward {poi_name}", ("poi_name",)),
    _t("pct_003", "poi_city_target", "Proceed toward {poi_name}", ("poi_name",)),
    _t("pct_004", "poi_city_target", "Head toward {city_name}", ("city_name",)),
    _t("pct_005", "poi_city_target", "Continue toward {city_name}", ("city_name",)),
    _t("pct_006", "poi_city_target", "Proceed toward {district_name}", ("district_name",)),
    _t("pct_007", "poi_city_target", "Head toward {poi_name} in {city_name}", ("poi_name", "city_name")),
    _t("pct_008", "poi_city_target", "Continue toward {poi_name} in {city_name}", ("poi_name", "city_name")),
    _t("pct_009", "poi_city_target", "Follow signs toward {poi_name}", ("poi_name",)),
    _t("pct_010", "poi_city_target", "Follow signs toward {city_name}", ("city_name",)),
    _t("pct_011", "poi_city_target", "Take Exit {exit_no} toward {poi_name}", ("exit_no", "poi_name")),
    _t("pct_012", "poi_city_target", "Take Exit {exit_no} toward {city_name}", ("exit_no", "city_name")),
    _t("pct_013", "poi_city_target", "Turn {direction} toward {poi_name}", ("direction", "poi_name")),
    _t("pct_014", "poi_city_target", "Turn {direction} toward {district_name}", ("direction", "district_name")),
    _t("pct_015", "poi_city_target", "In {distance_phrase}, head toward {poi_name}",
         ("distance_phrase", "poi_name")),
    _t("pct_016", "poi_city_target",
         "In two thousand meters, continue straight toward {poi_name} located in the city of {city_name}",
         ("poi_name", "city_name"), length_hint="long"),
    _t("pct_017", "poi_city_target",
         "In one point five kilometers, head straight toward {poi_name} in the {district_name} area",
         ("poi_name", "district_name"), length_hint="long"),
    _t("pct_018", "poi_city_target",
         "In two thousand meters, continue straight toward {poi_name} in {city_name} near the {district_name} area",
         ("poi_name", "city_name", "district_name"), length_hint="long"),
    _t("pct_019", "poi_city_target",
         "In one point five kilometers, follow signs toward {poi_name} located in the city of {city_name}",
         ("poi_name", "city_name"), length_hint="long"),
]

# ---------------------------------------------------------------------------
# address_based
# ---------------------------------------------------------------------------
ADDRESS_BASED: List[Template] = [
    _t("ab_001", "address_based", "Head toward {address_string}", ("address_string",)),
    _t("ab_002", "address_based", "Continue to {address_string}", ("address_string",)),
    _t("ab_003", "address_based", "Proceed to {address_string}", ("address_string",)),
    _t("ab_004", "address_based", "Your destination is {address_string}", ("address_string",)),
    _t("ab_005", "address_based", "The destination is {address_string}", ("address_string",)),
    _t("ab_006", "address_based", "Navigate to {address_string}", ("address_string",)),
    _t("ab_007", "address_based", "Turn {direction} toward {address_string}", ("direction", "address_string")),
    _t("ab_008", "address_based", "In {distance_phrase}, continue to {address_string}",
         ("distance_phrase", "address_string")),
    _t("ab_009", "address_based", "Arriving at {address_string}", ("address_string",)),
    _t("ab_010", "address_based", "You have arrived at {address_string}", ("address_string",)),
    _t("ab_011", "address_based", "Destination is {address_string}", ("address_string",), length_hint="short"),
    _t("ab_012", "address_based", "Continue on {road_name} to {address_string}",
         ("road_name", "address_string"), length_hint="long"),
    _t("ab_013", "address_based",
         "Continue for three quarters of a mile straight to {address_string} near {city_name} on your right",
         ("address_string", "city_name"), length_hint="long"),
    _t("ab_014", "address_based",
         "Continue for two thousand meters on {road_name} straight to {address_string} near {city_name}",
         ("road_name", "address_string", "city_name"), length_hint="long"),
]

# ---------------------------------------------------------------------------
# mixed_longform
# ---------------------------------------------------------------------------
MIXED_LONGFORM: List[Template] = [
    _t("ml_001", "mixed_longform",
         "In {distance_phrase}, turn {direction} onto {route_name} toward {city_name}",
         ("distance_phrase", "direction", "route_name", "city_name"), length_hint="long"),
    _t("ml_002", "mixed_longform",
         "Continue on {route_name} for {distance_phrase}, then take Exit {exit_no} toward {poi_name}",
         ("route_name", "distance_phrase", "exit_no", "poi_name"), length_hint="long"),
    _t("ml_003", "mixed_longform",
         "After {distance_phrase}, keep {direction} to stay on {road_name}",
         ("distance_phrase", "direction", "road_name"), length_hint="long"),
    _t("ml_004", "mixed_longform",
         "In a quarter mile, use {lane_count_phrase} to take Exit {exit_no} toward {poi_name}",
         ("lane_count_phrase", "exit_no", "poi_name"), length_hint="long"),
    _t("ml_005", "mixed_longform",
         "In {distance_phrase}, turn {direction} onto {road_name} toward {district_name}",
         ("distance_phrase", "direction", "road_name", "district_name"), length_hint="long"),
    _t("ml_006", "mixed_longform",
         "Continue for {distance_phrase}, then merge onto {route_name} toward {city_name}",
         ("distance_phrase", "route_name", "city_name"), length_hint="long"),
    _t("ml_007", "mixed_longform",
         "In {distance_phrase}, use {lane_count_phrase} to turn {direction} onto {road_name}",
         ("distance_phrase", "lane_count_phrase", "direction", "road_name"), length_hint="long"),
    _t("ml_008", "mixed_longform",
         "After {distance_phrase}, take Exit {exit_no} toward {poi_name}, then turn {direction} onto {road_name}",
         ("distance_phrase", "exit_no", "poi_name", "direction", "road_name"), length_hint="long"),
    _t("ml_009", "mixed_longform",
         "In {distance_phrase}, enter the roundabout and take the {ordinal} exit onto {road_name} toward {city_name}",
         ("distance_phrase", "ordinal", "road_name", "city_name"), length_hint="long"),
    _t("ml_010", "mixed_longform",
         "Continue on {route_name} for {distance_phrase}, then keep {direction} to stay on {route_name}",
         ("route_name", "distance_phrase", "direction"), length_hint="long"),
    _t("ml_011", "mixed_longform",
         "In {distance_phrase}, turn {direction} onto {route_name} toward Downtown {city_name}",
         ("distance_phrase", "direction", "route_name", "city_name"), length_hint="long"),
    _t("ml_012", "mixed_longform",
         "After {distance_phrase}, bear {direction} onto {road_name} toward {poi_name} in {city_name}",
         ("distance_phrase", "direction", "road_name", "poi_name", "city_name"), length_hint="long"),
    _t("ml_013", "mixed_longform",
         "In {distance_phrase}, continue on {route_name} toward {poi_name} in {city_name}, then take Exit {exit_no}",
         ("distance_phrase", "route_name", "poi_name", "city_name", "exit_no"), length_hint="long"),
    _t("ml_014", "mixed_longform",
         "After {distance_phrase}, merge onto {route_name} and follow signs toward {district_name} and {poi_name}",
         ("distance_phrase", "route_name", "district_name", "poi_name"), length_hint="long"),
    _t("ml_015", "mixed_longform",
         "Continue for {distance_phrase} on {road_name}, then turn {direction} toward {address_string} near {city_name}",
         ("distance_phrase", "road_name", "direction", "address_string", "city_name"), length_hint="long"),
    _t("ml_016", "mixed_longform",
         "In {distance_phrase}, use {lane_count_phrase} to merge onto {route_name} toward {poi_name} in {district_name}",
         ("distance_phrase", "lane_count_phrase", "route_name", "poi_name", "district_name"), length_hint="long"),
    _t("ml_017", "mixed_longform",
         "After {distance_phrase}, keep {direction} on {route_name} toward {city_name} and {poi_name}",
         ("distance_phrase", "direction", "route_name", "city_name", "poi_name"), length_hint="long"),
    _t("ml_018", "mixed_longform",
         "In {distance_phrase}, enter the roundabout and take the {ordinal} exit toward {poi_name} on {road_name}",
         ("distance_phrase", "ordinal", "poi_name", "road_name"), length_hint="long"),
]

ENTITY_SLOT_NAMES = frozenset({
    "road_name", "poi_name", "city_name", "district_name",
    "route_name", "address_string", "street_name",
})


def templates_with_slot(slot: str) -> List[Template]:
    return [t for t in ALL_TEMPLATES if slot in t.required_slots]


def templates_with_min_entity_slots(min_count: int) -> List[Template]:
    return [
        t for t in ALL_TEMPLATES
        if sum(1 for s in t.required_slots if s in ENTITY_SLOT_NAMES) >= min_count
    ]


def templates_with_length_hint(hint: str) -> List[Template]:
    return [t for t in ALL_TEMPLATES if t.length_hint == hint]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TEMPLATE_CATEGORIES: Dict[str, List[Template]] = {
    "basic_actions": BASIC_ACTIONS,
    "distance_prefixed": DISTANCE_PREFIXED,
    "road_navigation": ROAD_NAVIGATION,
    "numbered_routes": NUMBERED_ROUTES,
    "arrival": ARRIVAL,
    "poi_city_target": POI_CITY_TARGET,
    "address_based": ADDRESS_BASED,
    "mixed_longform": MIXED_LONGFORM,
}

ALL_TEMPLATES: List[Template] = [
    t for templates in TEMPLATE_CATEGORIES.values() for t in templates
]

# road_name slot templates excluding structured-only rn_011 (no road_name coverage credit)
ROAD_COVERAGE_TEMPLATES: List[Template] = [
    t for t in ALL_TEMPLATES if "road_name" in t.required_slots and t.template_id != "rn_011"
]

# Non-mixed templates for long_sentence refinement (RULE_10: word_count >= 18)
LONG_SENTENCE_TEMPLATES: List[Template] = [
    t for t in ALL_TEMPLATES
    if t.length_hint == "long" and t.template_type != "mixed_longform"
]

# template_id soft-cap cooldown (hard cap still enforced at 3%)
TEMPLATE_ID_COOLDOWN_IDS = frozenset({"rn_002", "nr_019", "dp_019"})
TEMPLATE_ID_SOFT_CAP_RATIO = 0.028

# type-level soft stop before RULE_08 hard limit (20%)
TEMPLATE_TYPE_SOFT_CAP: Dict[str, float] = {
    "numbered_routes": 0.195,
}

# long_sentence refinement prefers these types over numbered_routes
LONG_SENTENCE_PREFERRED_TYPES = frozenset({
    "road_navigation", "poi_city_target", "address_based", "distance_prefixed",
})

# long + entity_heavy joint refinement (RULE_10 + RULE_11)
LONG_ENTITY_HEAVY_TEMPLATES: List[Template] = [
    t for t in ALL_TEMPLATES
    if t.template_type != "mixed_longform"
    and t.length_hint == "long"
    and sum(1 for s in t.required_slots if s in ENTITY_SLOT_NAMES) >= 2
]

LONG_ENTITY_HEAVY_PREFERRED_TYPES = frozenset({
    "poi_city_target", "road_navigation", "address_based", "distance_prefixed",
})

TEMPLATES_BY_ID: Dict[str, Template] = {t.template_id: t for t in ALL_TEMPLATES}
TEMPLATES_BY_TYPE: Dict[str, List[Template]] = TEMPLATE_CATEGORIES


def get_templates_for_types(types: Optional[List[str]] = None) -> List[Template]:
    if not types:
        return ALL_TEMPLATES
    result: List[Template] = []
    for ttype in types:
        result.extend(TEMPLATE_CATEGORIES.get(ttype, []))
    return result
