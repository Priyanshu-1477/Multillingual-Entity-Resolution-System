"""Open-set country normalization module."""

from typing import Dict
from .unicode_utils import normalize_unicode

# Canonical mappings for high-frequency country designations
COUNTRY_CANONICAL_MAP: Dict[str, str] = {
    # India
    "in": "India",
    "ind": "India",
    "india": "India",
    "republic of india": "India",
    "bharat": "India",

    # France
    "fr": "France",
    "fra": "France",
    "france": "France",
    "french republic": "France",
    "republique francaise": "France",

    # United States
    "us": "United States",
    "usa": "United States",
    "u.s.": "United States",
    "u.s.a.": "United States",
    "united states": "United States",
    "united states of america": "United States",
}


def normalize_country(country: str) -> str:
    """Normalize country names consistently while supporting an open set of countries.

    Guarantees:
    - Known aliases (e.g. 'US', 'USA' -> 'United States'; 'IND', 'IN' -> 'India'; 'FRA', 'FR' -> 'France').
    - Open-set preservation: Any unrecognized country is preserved rather than discarded.
    - Missing/empty fields remain empty string "".
    """
    if not country:
        return ""

    cleaned = normalize_unicode(country).strip()
    if not cleaned:
        return ""

    lookup_key = cleaned.lower()
    if lookup_key in COUNTRY_CANONICAL_MAP:
        return COUNTRY_CANONICAL_MAP[lookup_key]

    # Open-set preservation: retain the normalized original name with clean casing
    return cleaned.title()
