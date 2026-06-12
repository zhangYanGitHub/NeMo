"""Business-style replay templates mimicking real navigation prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from templates import Template, _t


REPLAY_TEMPLATES: List[Template] = [
    _t(
        "rp_001", "replay",
        "In {distance_phrase}, turn {direction} onto {road_name} and continue toward {poi_name}",
        ("distance_phrase", "direction", "road_name", "poi_name"),
        length_hint="long",
    ),
    _t(
        "rp_002", "replay",
        "Continue on {route_name} for {distance_phrase}. Then take Exit {exit_no} toward {poi_name}",
        ("route_name", "distance_phrase", "exit_no", "poi_name"),
        length_hint="long",
    ),
    _t(
        "rp_003", "replay",
        "Your route continues on {route_name}. In {distance_phrase}, keep {direction} toward {city_name}",
        ("route_name", "distance_phrase", "direction", "city_name"),
        length_hint="long",
    ),
    _t(
        "rp_004", "replay",
        "In {distance_phrase}, use {lane_count_phrase} to take Exit {exit_no} toward {district_name}",
        ("distance_phrase", "lane_count_phrase", "exit_no", "district_name"),
        length_hint="long",
    ),
    _t(
        "rp_005", "replay",
        "After {distance_phrase}, merge onto {route_name} and follow signs for {poi_name}",
        ("distance_phrase", "route_name", "poi_name"),
        length_hint="long",
    ),
    _t(
        "rp_006", "replay",
        "GPS signal found. In {distance_phrase}, turn {direction} onto {road_name}",
        ("distance_phrase", "direction", "road_name"),
    ),
    _t(
        "rp_007", "replay",
        "Rerouting. Please make a U-turn when safe, then continue toward {poi_name}",
        ("poi_name",),
        length_hint="long",
    ),
    _t(
        "rp_008", "replay",
        "You are now on the fastest route to {poi_name} in {city_name}",
        ("poi_name", "city_name"),
        length_hint="long",
    ),
    _t(
        "rp_009", "replay",
        "Traffic ahead. In {distance_phrase}, keep {direction} to stay on {route_name}",
        ("distance_phrase", "direction", "route_name"),
        length_hint="long",
    ),
    _t(
        "rp_010", "replay",
        "In {distance_phrase}, enter the roundabout and take the {ordinal} exit toward {road_name}",
        ("distance_phrase", "ordinal", "road_name"),
        length_hint="long",
    ),
    _t(
        "rp_011", "replay",
        "Destination ahead in {distance_phrase}. {destination_phrase} will be on the {side}",
        ("distance_phrase", "destination_phrase", "side"),
        length_hint="long",
    ),
    _t(
        "rp_012", "replay",
        "You have arrived at {address_string}. {destination_phrase} is on the {side}",
        ("address_string", "destination_phrase", "side"),
        length_hint="long",
    ),
    _t(
        "rp_013", "replay",
        "Continue straight for {distance_phrase}, then turn {direction} onto {road_name} toward {city_name}",
        ("distance_phrase", "direction", "road_name", "city_name"),
        length_hint="long",
    ),
    _t(
        "rp_014", "replay",
        "In {distance_phrase}, bear {direction} onto {route_name} toward {poi_name}",
        ("distance_phrase", "direction", "route_name", "poi_name"),
        length_hint="long",
    ),
    _t(
        "rp_015", "replay",
        "Take the next left onto {road_name}, then in {distance_phrase} turn {direction} toward {district_name}",
        ("road_name", "distance_phrase", "direction", "district_name"),
        length_hint="long",
    ),
    _t(
        "rp_016", "replay",
        "Route guidance resumed. Head toward {poi_name} via {route_name}",
        ("poi_name", "route_name"),
        length_hint="long",
    ),
    _t(
        "rp_017", "replay",
        "In {distance_phrase}, keep {direction} at the fork to stay on {road_name} toward {poi_name}",
        ("distance_phrase", "direction", "road_name", "poi_name"),
        length_hint="long",
    ),
    _t(
        "rp_018", "replay",
        "Construction ahead. In {distance_phrase}, use {lane_count_phrase} to merge onto {route_name}",
        ("distance_phrase", "lane_count_phrase", "route_name"),
        length_hint="long",
    ),
    _t(
        "rp_019", "replay",
        "Toll road in {distance_phrase}. Continue on {route_name} toward {city_name}",
        ("distance_phrase", "route_name", "city_name"),
        length_hint="long",
    ),
    _t(
        "rp_020", "replay",
        "In {distance_phrase}, turn {direction} onto {road_name}. {poi_name} will be on the {side}",
        ("distance_phrase", "direction", "road_name", "poi_name", "side"),
        length_hint="long",
    ),
    _t(
        "rp_021", "replay",
        "Continue on {road_name} for {distance_phrase}, then take Exit {exit_no}B toward {poi_name}",
        ("road_name", "distance_phrase", "exit_no", "poi_name"),
        length_hint="long",
    ),
    _t(
        "rp_022", "replay",
        "Sharp turn ahead. In {distance_phrase}, turn {direction} onto {road_name} toward {address_string}",
        ("distance_phrase", "direction", "road_name", "address_string"),
        length_hint="long",
    ),
    _t(
        "rp_023", "replay",
        "Welcome to {city_name}. In {distance_phrase}, continue toward {district_name}",
        ("city_name", "distance_phrase", "district_name"),
        length_hint="long",
    ),
    _t(
        "rp_024", "replay",
        "Speed limit change ahead. Continue on {route_name} for {distance_phrase}, then exit toward {poi_name}",
        ("route_name", "distance_phrase", "poi_name"),
        length_hint="long",
    ),
    _t(
        "rp_025", "replay",
        "In {distance_phrase}, turn {direction} and stay on {route_name} toward Downtown",
        ("distance_phrase", "direction", "route_name"),
        length_hint="long",
    ),
]

REPLAY_BY_ID: Dict[str, Template] = {t.template_id: t for t in REPLAY_TEMPLATES}
