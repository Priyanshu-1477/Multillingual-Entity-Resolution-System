"""Address normalization module preserving numbers, postal codes, and geographic components."""

import re
from .transliterator import transliterate_text
from .unicode_utils import normalize_unicode

# Rogue placeholder tokens inside address segments (handles <NULL>, <null>, null, none)
_RE_NULL_TOKEN = re.compile(r"(?i)(?:<null>|<none>|\bnull\b|\bnone\b)")
# Repeated commas or commas with whitespace (e.g. ', , ' or ' , ')
_RE_CLEAN_COMMAS = re.compile(r"\s*,\s*")
_RE_MULTI_COMMAS = re.compile(r",(?:\s*,)+")
_RE_MULTI_SPACE = re.compile(r"[ ]{2,}")


def normalize_address(address: str) -> str:
    """Conservatively normalize an address string.

    Guarantees:
    - Safe Unicode normalization (NFKC, control char removal).
    - Preserves house numbers, shop numbers, building numbers, and road numbers.
    - Preserves postal codes, PIN codes, and ZIP codes (including leading zeros: '00123').
    - Transliterates regional scripts in localities, states, and cities to Latin readable.
    - Removes rogue isolated placeholder tokens (like '<NULL>' or 'null' between commas).
    - Cleans comma formatting and spacing.
    - Does NOT geocode, infer coordinates, or add external information.
    - Handles empty/missing values safely, returning empty string "".
    """
    if not address:
        return ""

    # Step 1: Unicode normalization
    cleaned = normalize_unicode(address)
    if not cleaned:
        return ""

    # Check if the entire field is a null placeholder
    if cleaned.lower() in ("<null>", "null", "<none>", "none"):
        return ""

    # Step 2: Transliteration of non-Latin segments (states, cities, landmarks)
    cleaned = transliterate_text(cleaned)

    # Step 3: Remove isolated null placeholder tokens inside addresses
    cleaned = _RE_NULL_TOKEN.sub("", cleaned)

    # Step 4: Clean commas and punctuation spacing
    cleaned = _RE_CLEAN_COMMAS.sub(", ", cleaned)
    cleaned = _RE_MULTI_COMMAS.sub(", ", cleaned)

    # Strip leading/trailing commas, dashes, and whitespace
    cleaned = cleaned.strip(" ,-")

    # Step 5: Final whitespace collapse
    cleaned = _RE_MULTI_SPACE.sub(" ", cleaned).strip()

    return cleaned
