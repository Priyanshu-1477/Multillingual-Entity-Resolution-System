"""Conservative business name normalization module."""

import re
from .transliterator import transliterate_text
from .unicode_utils import normalize_unicode

# Leading and trailing noise punctuation (e.g., '-- Company', '<< Team Ecole >>', '* Acme *')
# Preserves parentheses '()' commonly used in legal suffixes like '(Limited)' or '(India)'
_RE_LEADING_NOISE = re.compile(r"^[\s\-_<>=+*•#~`\"'{}\[\]|\\/]+")
_RE_TRAILING_NOISE = re.compile(r"[\s\-_<>=+*•#~`\"'{}\[\]|\\/]+$")
_RE_MULTI_SPACE = re.compile(r"[ ]{2,}")


def normalize_business_name(name: str) -> str:
    """Conservatively normalize a business entity name.

    Guarantees:
    - Safe Unicode normalization (NFKC, control char removal).
    - Preserves original Latin language without translation.
    - Faithfully transliterates non-Latin scripts to Latin readable.
    - Strips unwanted leading and trailing noise artifacts ('--', '<<', '>>', etc.).
    - Preserves business words, trade names, acronyms, and legal suffixes (LLC, Inc, SAS, Pvt Ltd, etc.).
    - Collapses repeated spaces to a single space.
    - Handles empty/missing values safely, returning empty string "".
    """
    if not name:
        return ""

    # Step 1: Unicode normalization
    cleaned = normalize_unicode(name)
    if not cleaned:
        return ""

    # Check for literal representation of empty/null
    if cleaned.lower() in ("<null>", "<none>"):
        return ""

    # Step 2: Transliteration
    cleaned = transliterate_text(cleaned)

    # Step 3: Remove leading and trailing noise symbols
    cleaned = _RE_LEADING_NOISE.sub("", cleaned)
    cleaned = _RE_TRAILING_NOISE.sub("", cleaned)

    # Step 4: Final whitespace normalization
    cleaned = _RE_MULTI_SPACE.sub(" ", cleaned).strip()

    return cleaned
