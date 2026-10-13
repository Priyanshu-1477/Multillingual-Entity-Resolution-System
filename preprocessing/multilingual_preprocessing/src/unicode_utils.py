"""Unicode normalization and text sanitization utilities."""

import re
import unicodedata

# Compiled regex for Unicode whitespace characters to normalize to standard ASCII space
_RE_UNICODE_WHITESPACE = re.compile(r"[\u00A0\u1680\u2000-\u200A\u2028\u2029\u202F\u205F\u3000\t\r\n]+")

# Compiled regex for zero-width and invisible control characters
_RE_INVISIBLE_CHARS = re.compile(r"[\u200B\u200C\u200D\u200E\u200F\uFEFF\u2060\u00AD\u200E\u200F]")

# Compiled regex for non-printable control characters (ASCII 0-31 and 127-159)
_RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

# Repeated whitespace regex
_RE_MULTI_SPACE = re.compile(r"[ ]{2,}")


def normalize_unicode(text: str) -> str:
    """Safely normalize Unicode text.

    Operations:
    1. Removes invisible, zero-width, and BOM characters (\uFEFF, \u200B, etc.).
    2. Strips low-level non-printable control characters.
    3. Normalizes all Unicode whitespace variants (non-breaking, em, ideographic, stray tabs/newlines) to space.
    4. Applies NFKC decomposition/composition to standardize compatibility characters
       (such as fullwidth Latin 'ＡＢＣ' -> 'ABC', ligatures 'ﬁ' -> 'fi').
    5. Collapses repeated spaces into a single space and strips boundaries.
    """
    if not text:
        return ""

    # Fast path: pure clean ASCII without special control chars
    if text.isascii() and not ("\t" in text or "\r" in text or "\n" in text or "\x00" in text):
        cleaned = _RE_MULTI_SPACE.sub(" ", text).strip()
        return cleaned

    # Strip invisible formatting characters
    cleaned = _RE_INVISIBLE_CHARS.sub("", text)

    # Strip non-printable control characters
    cleaned = _RE_CONTROL_CHARS.sub("", cleaned)

    # Normalize compatibility characters
    cleaned = unicodedata.normalize("NFKC", cleaned)

    # Normalize Unicode whitespace and stray newlines/tabs to standard space
    cleaned = _RE_UNICODE_WHITESPACE.sub(" ", cleaned)

    # Collapse repeated spaces and trim
    cleaned = _RE_MULTI_SPACE.sub(" ", cleaned).strip()

    return cleaned


def unaccent_latin(text: str) -> str:
    """Normalize Latin diacritics to clean Latin-readable characters.

    Examples:
        'Farmacia San José' -> 'Farmacia San Jose'
        'Mercado São João' -> 'Mercado Sao Joao'
        'Boulangerie Dupont' -> 'Boulangerie Dupont'
        'Léarning Center' -> 'Learning Center'
    """
    if not text or text.isascii():
        return text

    # Decompose accented characters using NFKD, then strip combining diacritical marks (Mn)
    nfkd = unicodedata.normalize("NFKD", text)
    result = "".join(c for c in nfkd if unicodedata.category(c) != "Mn")
    return result
