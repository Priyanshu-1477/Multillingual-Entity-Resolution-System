"""Language analysis module combining script detection and geographic context."""

from typing import Optional, Set
from .script_detector import detect_scripts


def detect_probable_language(text: str, country: Optional[str] = None) -> str:
    """Infer the probable primary language of a business name or address.

    Given that short entity strings often lack sufficient n-grams for statistical
    language classification, this combines Unicode script identification with
    geographic and lexical indicators.
    """
    if not text:
        return "Unknown"

    scripts = detect_scripts(text)
    norm_country = (country or "").strip().lower()

    # Non-Latin script direct associations
    if "Devanagari" in scripts:
        # In Maharashtra, could be Marathi; default broadly to Hindi/Indic
        return "Hindi"
    if "Bengali" in scripts:
        return "Bengali"
    if "Gujarati" in scripts:
        return "Gujarati"
    if "Tamil" in scripts:
        return "Tamil"
    if "Telugu" in scripts:
        return "Telugu"
    if "Kannada" in scripts:
        return "Kannada"
    if "Malayalam" in scripts:
        return "Malayalam"
    if "Gurmukhi" in scripts:
        return "Punjabi"
    if "Odia" in scripts:
        return "Odia"
    if "Arabic" in scripts:
        return "Arabic"
    if "CJK" in scripts:
        return "Chinese"
    if "Cyrillic" in scripts:
        return "Russian"
    if "Greek" in scripts:
        return "Greek"
    if "Hebrew" in scripts:
        return "Hebrew"

    # Latin script contextual resolution
    if "Latin" in scripts or not scripts:
        if "france" in norm_country or norm_country == "fr":
            return "French"
        if "united states" in norm_country or norm_country in ("us", "usa"):
            # Check for prominent Spanish markers
            lower_text = text.lower()
            if any(w in lower_text.split() for w in ("san", "santa", "los", "las", "el", "la", "farmacia")):
                return "Spanish"
            return "English"
        if "india" in norm_country or norm_country == "in":
            return "English (India)"
        return "Latin"

    return "Unknown"
