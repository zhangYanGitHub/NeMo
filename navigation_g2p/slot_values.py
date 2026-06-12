"""Structured slot vocabularies for navigation template filling."""

from __future__ import annotations

import itertools
import random
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Core slot lists
# ---------------------------------------------------------------------------

DIRECTION = ["left", "right", "straight"]

SLIGHT_DIRECTION = ["slightly left", "slightly right", "bear left", "bear right"]

SIDE = ["left", "right"]

DISTANCE_VALUE = [
    "50", "100", "150", "200", "250", "300", "400", "500", "600", "800",
    "1000", "1200", "1500", "2000", "half", "a quarter", "three quarters",
    "1", "2", "3", "0.5", "0.25",
]

DISTANCE_UNIT = ["meters", "meter", "feet", "foot", "kilometers", "kilometer", "miles", "mile", "yards", "yard"]

FRACTIONAL_DISTANCE = [
    "half a mile", "a quarter mile", "three quarters of a mile",
    "half a kilometer", "a quarter kilometer",
]

ORDINAL = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "1st", "2nd", "3rd", "4th", "5th",
]

EXIT_NO = ["1", "2", "3", "4", "5", "6", "7", "7A", "7B", "8", "9", "10", "11", "12", "12A", "12B", "15", "18", "22"]

ROUTE_PREFIX = ["I", "US", "SR", "Route", "Highway", "Hwy", "FM", "TX"]

ROUTE_NUMBER = ["1", "5", "10", "66", "85", "90", "95", "101", "280", "405", "520", "880", "1010"]

ROUTE_DIRECTION = ["North", "South", "East", "West", "N", "S", "E", "W"]

ROAD_DIRECTION_PREFIX = ["North", "South", "East", "West", "N", "S", "E", "W", "NE", "NW", "SE", "SW"]

ROAD_CORE_NAME = [
    "Main", "Oak", "Pine", "Maple", "Cedar", "Elm", "Washington", "Jefferson", "Lincoln",
    "Madison", "Franklin", "Market", "Broadway", "Central", "Park", "Lake", "River", "Hill",
    "Valley", "Sunset", "Sunrise", "Harbor", "Bay", "Coast", "Mission", "Union", "King",
    "Queen", "Victoria", "Cambridge", "Oxford", "Stanford", "Berkeley", "Howard", "Taylor",
    "Wilson", "Johnson", "Anderson", "Thompson", "Robinson", "Mitchell", "Campbell", "Parker",
    "Collins", "Stewart", "Morris", "Rogers", "Reed", "Cook", "Morgan", "Bell", "Murphy",
    "Bailey", "Rivera", "Gomez", "Gonzalez", "Lopez", "Martinez", "Hernandez", "Schuyler",
    "Joaquin", "Worcester", "Quetzal", "Des Plaines", "La Jolla",
    "Martin Luther King Jr", "George Washington", "Thomas Jefferson", "John F Kennedy",
    "Ronald Reagan", "Dwight D Eisenhower", "Woodrow Wilson",
]

ROAD_TYPE = [
    "Street", "St.", "Avenue", "Ave.", "Road", "Rd.", "Boulevard", "Blvd.",
    "Drive", "Dr.", "Lane", "Ln.", "Court", "Ct.", "Place", "Pl.",
    "Way", "Trail", "Parkway", "Pkwy.", "Highway", "Hwy.", "Circle", "Cir.",
    "Terrace", "Ter.", "Expressway", "Expy.",
]

CITY_NAME = [
    "San Francisco", "San Jose", "Los Angeles", "Seattle", "Portland", "Denver",
    "Chicago", "Boston", "New York", "Philadelphia", "Atlanta", "Miami",
    "Dallas", "Houston", "Austin", "Phoenix", "Las Vegas", "San Diego",
    "Sacramento", "Oakland", "Minneapolis", "Detroit", "Cleveland", "Pittsburgh",
    "Baltimore", "Charlotte", "Nashville", "Orlando", "Tampa", "Salt Lake City",
    "Albuquerque", "Boise", "Anchorage", "Honolulu", "Worcester", "Providence",
    "Richmond", "Raleigh", "Columbus", "Indianapolis", "Milwaukee", "Memphis",
    "Louisville", "Kansas City", "Omaha", "Tucson", "Fresno", "Bakersfield",
    "Santa Clara", "Palo Alto", "Mountain View", "Sunnyvale", "Cupertino",
    "Redmond", "Bellevue", "Cambridge", "Brookline", "Queens", "Brooklyn",
    "Jersey City", "Newark", "Arlington", "Alexandria", "Bethesda",
    "Silver Spring", "Rockville", "Gaithersburg", "Annapolis", "Frederick",
    "Xiang'an", "Shenzhen", "Guangzhou", "Shanghai", "Beijing",
]

DISTRICT_NAME = [
    "Downtown", "Midtown", "Uptown", "Financial District", "Theatre District",
    "Arts District", "Warehouse District", "Historic District", "Old Town",
    "North End", "South End", "West End", "East Village", "West Village",
    "SoMa", "Mission District", "Castro District", "Marina District",
    "Pacific Heights", "Nob Hill", "Chinatown", "Little Italy",
    "Koreatown", "Japantown", "French Quarter", "River North",
    "Capitol Hill", "Ballard", "Fremont", "Green Lake",
    "Xiang'an District", "Nanshan District", "Futian District",
    "Haidian District", "Chaoyang District", "Pudong New Area",
    "Central Business District", "Government Center", "University District",
    "Industrial Park", "Tech Park", "Innovation District",
]

POI_NAME = [
    "Central Station", "Union Square Station", "Grand Central Terminal",
    "Penn Station", "King Street Station", "South Station",
    "San Francisco International Airport", "San Jose International Airport",
    "Los Angeles International Airport", "Seattle-Tacoma International Airport",
    "O'Hare International Airport", "Logan International Airport",
    "Golden Gate Park", "Central Park", "Griffith Park", "Balboa Park",
    "City Hall", "County Courthouse", "State Capitol", "Public Library",
    "Convention Center", "Stadium", "Arena", "Museum of Modern Art",
    "Science Center", "Children's Hospital", "General Hospital",
    "University Campus", "Community College", "High School",
    "Shopping Mall", "Outlet Center", "Town Center", "Market Square",
    "Farmers Market", "Transit Center", "Bus Terminal", "Ferry Terminal",
    "Marina", "Harbor", "Pier 39", "Pike Place Market",
    "Disneyland", "Universal Studios", "SeaWorld", "Zoo",
    "National Monument", "Memorial Park", "Veterans Memorial",
    "Post Office", "Police Station", "Fire Station",
    "Costco", "Target", "Walmart", "Home Depot", "Whole Foods",
    "Starbucks Reserve", "Apple Park", "Googleplex", "Microsoft Campus",
    "Xiang'an District Government", "City Government Building",
    "Community Center", "Recreation Center", "Sports Complex",
    "Golf Course", "Country Club", "Beach Access", "Trailhead",
    "Rest Area", "Service Plaza", "Toll Plaza", "Weigh Station",
]

ADDRESS_NUMBER = [
    "1", "7", "12", "45", "88", "100", "128", "200", "350", "500",
    "888", "1000", "1280", "1500", "2000", "221", "221B", "3500", "4200",
    "10001", "12345",
]

STREET_NAME = [
    "Main Street", "Oak Avenue", "Pine Road", "Market Street", "Broadway",
    "NW 45th Street", "SE 12th Avenue", "SW 3rd Boulevard",
    "North Airport Road", "South Airport Road", "East Harbor Drive",
    "West Lake Boulevard", "Central Parkway", "Union Square",
    "Infinite Loop", "Baker Street", "Pennsylvania Avenue",
    "Constitution Avenue", "Independence Avenue", "Embarcadero",
    "Mission Street", "Howard Street", "Folsom Street", "Howard St.",
    "King St.", "Queen Ave.", "Park Blvd.", "River Rd.",
    "Martin Luther King Jr Boulevard", "George Bush Turnpike",
    "Des Plaines River Road", "La Jolla Shores Drive",
    "Schuyler Avenue", "Joaquin Miller Road", "Worcester Road",
    "Quetzal Lane", "Xiang'an Boulevard",
    "1280 NW 45th Street", "3500 South Airport Road",
    "1 Infinite Loop", "221B Baker Street",
]

COMPASS_PREFIX = ["N", "S", "E", "W", "NE", "NW", "SE", "SW", "North", "South", "East", "West"]

LANE_COUNT_PHRASE = [
    "the left lane", "the right lane", "the center lane",
    "the left two lanes", "the right two lanes", "the right three lanes",
    "both left lanes", "both right lanes",
    "the leftmost lane", "the rightmost lane",
]

DESTINATION_PHRASE = [
    "your destination", "the destination", "your final destination",
    "your next destination", "the next stop", "your stop",
]

# Abbreviation tokens used inside composed entities
ABBREVIATIONS = {
    "St.": "Street",
    "Ave.": "Avenue",
    "Blvd.": "Boulevard",
    "Rd.": "Road",
    "Dr.": "Drive",
    "Mt.": "Mount",
    "Ft.": "Fort",
    "Ln.": "Lane",
    "Ct.": "Court",
    "Pl.": "Place",
    "Pkwy.": "Parkway",
    "Hwy.": "Highway",
    "Cir.": "Circle",
    "Ter.": "Terrace",
    "Expy.": "Expressway",
}

# ---------------------------------------------------------------------------
# Composite builders
# ---------------------------------------------------------------------------


def build_road_name(
    rng: random.Random,
    *,
    use_abbrev: bool = True,
    use_compass: bool = True,
    use_hyphenated: bool = False,
) -> str:
    """Compose a road name from structured parts."""
    core = rng.choice(ROAD_CORE_NAME)
    road_type = rng.choice(ROAD_TYPE)
    parts: List[str] = []
    if use_compass and rng.random() < 0.35:
        parts.append(rng.choice(ROAD_DIRECTION_PREFIX))
    parts.append(core)
    parts.append(road_type)
    name = " ".join(parts)
    if use_hyphenated and rng.random() < 0.1:
        name = name.replace(" ", "-")
    return name


def build_route_name(rng: random.Random) -> str:
    prefix = rng.choice(ROUTE_PREFIX)
    number = rng.choice(ROUTE_NUMBER)
    if prefix in ("I", "US", "SR", "FM", "TX"):
        base = f"{prefix}-{number}"
    elif prefix in ("Route", "Highway", "Hwy"):
        base = f"{prefix} {number}"
    else:
        base = f"{prefix}-{number}"
    if rng.random() < 0.6:
        base = f"{base} {rng.choice(ROUTE_DIRECTION)}"
    return base


def build_address_string(rng: random.Random) -> str:
    if rng.random() < 0.3:
        return rng.choice(STREET_NAME)
    number = rng.choice(ADDRESS_NUMBER)
    if rng.random() < 0.4:
        compass = rng.choice(COMPASS_PREFIX)
        street_num = rng.choice(["1st", "2nd", "3rd", "4th", "5th", "7th", "12th", "45th", "99th"])
        road_type = rng.choice(["Street", "St.", "Avenue", "Ave.", "Road", "Rd.", "Boulevard", "Blvd."])
        return f"{number} {compass} {street_num} {road_type}"
    if rng.random() < 0.5:
        return f"{number} {build_road_name(rng)}"
    return f"{number} {rng.choice(STREET_NAME)}"


def build_distance_phrase(rng: random.Random) -> str:
    if rng.random() < 0.25:
        return rng.choice(FRACTIONAL_DISTANCE)
    value = rng.choice(DISTANCE_VALUE)
    unit = rng.choice(DISTANCE_UNIT)
    if value in ("half", "a quarter", "three quarters"):
        if unit in ("mile", "miles"):
            if value == "half":
                return "half a mile"
            if value == "a quarter":
                return "a quarter mile"
            return "three quarters of a mile"
        if value == "half":
            return "half a kilometer"
        if value == "a quarter":
            return "a quarter kilometer"
        return f"{value} {unit}"
    if value in ("0.5", "0.25"):
        return f"{value} {unit}"
    return f"{value} {unit}"


def build_exit_phrase(rng: random.Random) -> str:
    exit_no = rng.choice(EXIT_NO)
    if rng.random() < 0.5:
        target = rng.choice(POI_NAME + DISTRICT_NAME + CITY_NAME)
        return f"Exit {exit_no} toward {target}"
    return f"Exit {exit_no}"


# ---------------------------------------------------------------------------
# Slot registry
# ---------------------------------------------------------------------------

SLOT_BUILDERS: Dict[str, Callable[[random.Random], str]] = {
    "direction": lambda r: r.choice(DIRECTION),
    "slight_direction": lambda r: r.choice(SLIGHT_DIRECTION),
    "side": lambda r: r.choice(SIDE),
    "distance_value": lambda r: r.choice(DISTANCE_VALUE),
    "distance_unit": lambda r: r.choice(DISTANCE_UNIT),
    "fractional_distance": lambda r: r.choice(FRACTIONAL_DISTANCE),
    "distance_phrase": build_distance_phrase,
    "ordinal": lambda r: r.choice(ORDINAL),
    "exit_no": lambda r: r.choice(EXIT_NO),
    "route_prefix": lambda r: r.choice(ROUTE_PREFIX),
    "route_number": lambda r: r.choice(ROUTE_NUMBER),
    "route_direction": lambda r: r.choice(ROUTE_DIRECTION),
    "road_direction_prefix": lambda r: r.choice(ROAD_DIRECTION_PREFIX),
    "road_core_name": lambda r: r.choice(ROAD_CORE_NAME),
    "road_type": lambda r: r.choice(ROAD_TYPE),
    "road_name": lambda r: build_road_name(r),
    "route_name": build_route_name,
    "city_name": lambda r: r.choice(CITY_NAME),
    "district_name": lambda r: r.choice(DISTRICT_NAME),
    "poi_name": lambda r: r.choice(POI_NAME),
    "address_number": lambda r: r.choice(ADDRESS_NUMBER),
    "street_name": lambda r: r.choice(STREET_NAME),
    "address_string": build_address_string,
    "compass_prefix": lambda r: r.choice(COMPASS_PREFIX),
    "lane_count_phrase": lambda r: r.choice(LANE_COUNT_PHRASE),
    "destination_phrase": lambda r: r.choice(DESTINATION_PHRASE),
    "exit_phrase": build_exit_phrase,
}


def fill_slot(slot_name: str, rng: random.Random, overrides: Optional[Dict[str, str]] = None) -> str:
    if overrides and slot_name in overrides:
        return overrides[slot_name]
    builder = SLOT_BUILDERS.get(slot_name)
    if builder is None:
        raise KeyError(f"Unknown slot: {slot_name}")
    return builder(rng)


def get_all_slot_names() -> List[str]:
    return sorted(SLOT_BUILDERS.keys())


def _entity_hashes_to_train(group_key: str, train_ratio: float) -> bool:
    from utils import stable_hash

    bucket = (stable_hash(group_key) % 10000) / 10000.0
    return bucket < train_ratio


def iter_train_gap_entities(
    slot: str,
    train_existing_canon: set,
    train_ratio: float = 0.9,
) -> List[str]:
    """Raw entities: canonical not in train set and stable_hash maps to train bucket."""
    from dedup import canonical_entity
    from entity_sets import get_entity_pool

    entities: List[str] = []
    for raw in get_entity_pool(slot):
        canon = canonical_entity(raw, slot)
        if canon in train_existing_canon:
            continue
        if _entity_hashes_to_train(f"{slot}::{canon}", train_ratio):
            entities.append(raw)
    return entities


def iter_train_gap_roads(
    train_existing_canon: set,
    train_ratio: float = 0.9,
) -> List[str]:
    return iter_train_gap_entities("road_name", train_existing_canon, train_ratio)


def sample_entity_for_type(entity_type: str, rng: random.Random, long_tail: bool = False) -> str:
    """Sample entity with optional long-tail bias."""
    pools: Dict[str, List[str]] = {
        "road_name": [build_road_name(rng, use_abbrev=True) for _ in range(3)] + ROAD_CORE_NAME,
        "route_name": [build_route_name(rng) for _ in range(5)],
        "poi_name": POI_NAME,
        "city_name": CITY_NAME,
        "district_name": DISTRICT_NAME,
        "address_string": [build_address_string(rng) for _ in range(3)] + STREET_NAME,
    }
    long_tail_tokens = [
        "Schuyler", "Xiang'an", "Quetzal", "Joaquin", "Worcester", "La Jolla",
        "Des Plaines", "O'Hare", "San Jose International Airport",
        "Martin Luther King Jr Boulevard", "Xiang'an District Government",
    ]
    pool = pools.get(entity_type, [])
    if long_tail and rng.random() < 0.5:
        return rng.choice(long_tail_tokens + [p for p in pool if any(t in str(p) for t in long_tail_tokens)] or pool)
    if entity_type == "road_name":
        return build_road_name(rng, use_abbrev=rng.random() < 0.4)
    if entity_type == "route_name":
        return build_route_name(rng)
    if entity_type == "address_string":
        return build_address_string(rng)
    return rng.choice(pool) if pool else fill_slot(entity_type, rng)
