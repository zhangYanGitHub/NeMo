"""Curated entity sets for long-tail and entity-focused test generation."""

from __future__ import annotations

# Hyphenated route identifiers
HYPHENATED_ROUTES = [
    "I-5", "I-10", "I-15", "I-40", "I-70", "I-80", "I-90", "I-95",
    "I-280", "I-405", "I-880", "US-1", "US-101", "US-202", "US-395",
    "SR-1", "SR-17", "SR-85", "SR-92", "SR-520", "SR-237",
    "FM-1960", "TX-130",
]

# Multi-word proper names with pronunciation challenges
MULTI_WORD_NAMES = [
    "San Francisco International Airport",
    "San Jose International Airport",
    "Los Angeles International Airport",
    "Martin Luther King Jr Boulevard",
    "George Washington Memorial Parkway",
    "John F Kennedy International Airport",
    "Ronald Reagan Washington National Airport",
    "Dwight D Eisenhower National System",
    "Golden Gate Bridge Toll Plaza",
    "Union Square Station",
    "Xiang'an District Government",
    "Des Plaines River Road",
    "La Jolla Shores Drive",
    "New York State Thruway",
    "Pennsylvania Turnpike",
    "Garden State Parkway",
    "Santa Monica Boulevard",
    "Hollywood Boulevard",
    "Embarcadero Center",
    "Pike Place Market",
]

# Foreign-looking / uncommon spellings
FOREIGN_STYLE_NAMES = [
    "Schuyler", "Xiang'an", "Quetzal", "Joaquin", "Worcester",
    "La Jolla", "Des Plaines", "Nguyen", "Gonzalez", "Sioux Falls",
    "Boise", "Walla Walla", "Puyallup", "Skagit", "Tukwila",
    "Kissimmee", "Okeechobee", "Chattahoochee", "Natchitoches",
    "Schenectady", "Worcestershire", "Leicester", "Edinburgh",
    "Albuquerque", "Tucson", "Tehachapi", "Tejon", "Sepulveda",
]

# Abbreviation-heavy address forms
ABBREVIATION_ADDRESSES = [
    "1280 NW 45th St.",
    "3500 S Airport Rd.",
    "100 N Main St.",
    "2200 E Lake Ave.",
    "4500 W Broadway Blvd.",
    "1 Infinite Loop",
    "221B Baker St.",
    "500 NE 8th Ave.",
    "900 SW 1st Dr.",
    "1600 Pennsylvania Ave.",
    "700 Mt. View Rd.",
    "88 Ft. Worth Blvd.",
    "12 St. Charles Ave.",
    "300 Dr. Martin Luther King Jr Blvd.",
]

# POI entities with diverse naming patterns
ENTITY_POI_NAMES = [
    "Central Station", "Union Square Station", "Grand Central Terminal",
    "San Francisco International Airport", "O'Hare International Airport",
    "Golden Gate Park", "Xiang'an District Government",
    "Apple Park Visitor Center", "Googleplex Main Campus",
    "Costco Wholesale", "Whole Foods Market",
    "Children's Hospital of Philadelphia",
    "St. Mary's Medical Center", "Mt. Sinai Hospital",
    "Ft. Lauderdale Beach", "St. Louis Gateway Arch",
    "Dr. Phillips Performing Arts Center",
]

# Road names with mixed abbreviation / compass patterns
ENTITY_ROAD_NAMES = [
    "N Main St.", "S Oak Ave.", "E Pine Rd.", "W Cedar Blvd.",
    "NE 45th Street", "NW Market Street", "SE Airport Road",
    "SW Harbor Drive", "Martin Luther King Jr Boulevard",
    "George Bush Turnpike", "Des Plaines River Road",
    "La Jolla Shores Drive", "Schuyler Avenue",
    "Joaquin Miller Road", "Worcester Road", "Quetzal Lane",
    "Xiang'an Boulevard", "O'Farrell Street",
    "St. Charles Avenue", "Mt. Vernon Highway",
    "Ft. Worth Road", "Dr. Martin Luther King Jr Drive",
]

ENTITY_CITY_NAMES = [
    "San Francisco", "San Jose", "Los Angeles", "Worcester",
    "La Jolla", "Des Plaines", "Albuquerque", "Schenectady",
    "Xiang'an", "Shenzhen", "Boise", "Kissimmee",
    "Sioux Falls", "Oklahoma City", "Salt Lake City",
]

ENTITY_DISTRICT_NAMES = [
    "Financial District", "Mission District", "Castro District",
    "Xiang'an District", "Nanshan District", "SoMa",
    "North End", "French Quarter", "Capitol Hill",
    "Central Business District", "Historic District",
]

# Exit numbers with letter suffixes
ENTITY_EXIT_NUMBERS = [
    "1", "2A", "2B", "3", "4", "5", "6", "7", "7A", "7B",
    "8", "9", "10", "11", "12", "12A", "12B", "15", "18", "22",
]

def _build_expanded_route_pool() -> list[str]:
    """300+ distinct route identifiers for coverage quotas."""
    routes = set(HYPHENATED_ROUTES)
    nums = list(range(1, 201)) + [
        210, 220, 237, 280, 380, 405, 420, 440, 505, 520, 580, 680,
        780, 805, 880, 905, 1010,
    ]
    for n in nums:
        routes.add(f"I-{n}")
        routes.add(f"US-{n}")
        routes.add(f"SR-{n}")
        if n < 100:
            routes.add(f"Route {n}")
    for d in ("North", "South", "East", "West"):
        routes.add(f"I-95 {d}")
        routes.add(f"US-101 {d}")
    return sorted(routes)


def _build_expanded_poi_pool() -> list[str]:
    """350+ POI names for coverage quotas."""
    from slot_values import CITY_NAME, POI_NAME

    pool = set(POI_NAME) | set(ENTITY_POI_NAMES)
    suffixes = [
        "International Airport", "Regional Airport", "City Hall", "Convention Center",
        "Transit Center", "Medical Center", "Community Hospital", "University Campus",
        "Shopping Center", "Town Center", "Train Station", "Bus Station",
        "Visitor Center", "Sports Arena", "Memorial Park", "Public Library",
        "Tech Campus", "Industrial Park", "Waterfront Park", "Historic District",
    ]
    for city in CITY_NAME:
        for suf in suffixes[:12]:
            pool.add(f"{city} {suf}")
    for name in MULTI_WORD_NAMES:
        if "Airport" in name or "Station" in name or "Park" in name:
            pool.add(name)
    return sorted(pool)


def _build_expanded_road_pool() -> list[str]:
    """500+ road names — compass × core × type combinatorics."""
    from slot_values import ROAD_CORE_NAME, ROAD_DIRECTION_PREFIX, ROAD_TYPE

    pool: set[str] = set(ENTITY_ROAD_NAMES) | set(MULTI_WORD_NAMES)
    types = ["Street", "Avenue", "Road", "Boulevard", "Drive", "Lane", "Way"]
    for core in ROAD_CORE_NAME:
        for rtype in types:
            pool.add(f"{core} {rtype}")
    for prefix in ROAD_DIRECTION_PREFIX:
        for core in ROAD_CORE_NAME[:50]:
            pool.add(f"{prefix} {core} Street")
            pool.add(f"{prefix} {core} Avenue")
    for n in range(1, 150):
        pool.add(f"{n}th Street")
        pool.add(f"NE {n}th Avenue")
        pool.add(f"SW {n}th Road")
    return sorted(pool)


EXPANDED_ROUTE_POOL = _build_expanded_route_pool()
EXPANDED_POI_POOL = _build_expanded_poi_pool()
EXPANDED_ROAD_POOL = _build_expanded_road_pool()

# All entity pools keyed by slot / entity type
ENTITY_POOLS = {
    "road_name": EXPANDED_ROAD_POOL,
    "poi_name": EXPANDED_POI_POOL,
    "city_name": ENTITY_CITY_NAMES,
    "district_name": ENTITY_DISTRICT_NAMES,
    "route_name": EXPANDED_ROUTE_POOL,
    "address_string": ABBREVIATION_ADDRESSES,
    "exit_no": ENTITY_EXIT_NUMBERS,
    "foreign_name": FOREIGN_STYLE_NAMES,
    "multi_word_name": MULTI_WORD_NAMES,
    "hyphenated_route": EXPANDED_ROUTE_POOL,
}


def get_entity_pool(slot: str) -> list[str]:
    return list(ENTITY_POOLS.get(slot, []))
