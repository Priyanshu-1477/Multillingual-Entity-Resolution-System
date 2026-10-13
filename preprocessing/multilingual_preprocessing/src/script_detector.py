"""Unicode script detection and text segmentation."""

from typing import List, Set, Tuple


def get_char_script(char: str) -> str:
    """Return the Unicode script for a single character."""
    cp = ord(char)

    # Fast ASCII / Basic Latin & Latin extensions
    if (0x0000 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
        # Distinguish numbers / common punctuation from letters
        if (0x0030 <= cp <= 0x0039) or (0x0020 <= cp <= 0x002F) or (0x003A <= cp <= 0x0040) or \
           (0x005B <= cp <= 0x0060) or (0x007B <= cp <= 0x007E):
            return "Common"
        return "Latin"

    # Indic Scripts
    if 0x0900 <= cp <= 0x097F:
        return "Devanagari"
    if 0x0980 <= cp <= 0x09FF:
        return "Bengali"
    if 0x0A00 <= cp <= 0x0A7F:
        return "Gurmukhi"
    if 0x0A80 <= cp <= 0x0AFF:
        return "Gujarati"
    if 0x0B00 <= cp <= 0x0B7F:
        return "Odia"
    if 0x0B80 <= cp <= 0x0BFF:
        return "Tamil"
    if 0x0C00 <= cp <= 0x0C7F:
        return "Telugu"
    if 0x0C80 <= cp <= 0x0CFF:
        return "Kannada"
    if 0x0D00 <= cp <= 0x0D7F:
        return "Malayalam"

    # Middle Eastern
    if (0x0600 <= cp <= 0x06FF) or (0x0750 <= cp <= 0x077F) or (0x08A0 <= cp <= 0x08FF):
        return "Arabic"
    if 0x0590 <= cp <= 0x05FF:
        return "Hebrew"

    # East Asian (CJK)
    if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x20000 <= cp <= 0x2A6DF):
        return "CJK"
    if (0x3040 <= cp <= 0x309F) or (0x30A0 <= cp <= 0x30FF):
        return "Japanese"
    if 0xAC00 <= cp <= 0xD7AF:
        return "Hangul"

    # European / Eurasian non-Latin
    if (0x0400 <= cp <= 0x04FF) or (0x0500 <= cp <= 0x052F):
        return "Cyrillic"
    if (0x0370 <= cp <= 0x03FF) or (0x1F00 <= cp <= 0x1FFF):
        return "Greek"

    # Southeast Asian
    if 0x0E00 <= cp <= 0x0E7F:
        return "Thai"

    # Punctuation, symbols, digits in other Unicode blocks
    if (0x2000 <= cp <= 0x206F) or (0x2070 <= cp <= 0x209F) or (0x20A0 <= cp <= 0x20CF):
        return "Common"

    return "Other"


def detect_scripts(text: str) -> Set[str]:
    """Return all non-common scripts detected in text."""
    scripts = set()
    for ch in text:
        s = get_char_script(ch)
        if s != "Common":
            scripts.add(s)
    return scripts


def is_pure_latin(text: str) -> bool:
    """Return True if text contains only Latin and Common (digits, punctuation, spaces) characters."""
    for ch in text:
        s = get_char_script(ch)
        if s not in ("Latin", "Common"):
            return False
    return True


def has_non_latin(text: str) -> bool:
    """Fast check whether text contains any non-Latin alphabetic character."""
    if text.isascii():
        return False
    for ch in text:
        s = get_char_script(ch)
        if s not in ("Latin", "Common"):
            return True
    return False


def segment_by_script(text: str) -> List[Tuple[str, str]]:
    """Segment mixed-script text into contiguous script segments.

    Returns a list of (segment_text, script_name) tuples.
    Common characters (spaces, punctuation, digits) attach to adjacent segments or remain Common.
    """
    if not text:
        return []

    segments: List[Tuple[str, str]] = []
    current_chars = []
    current_script = None

    for ch in text:
        ch_script = get_char_script(ch)
        if current_script is None:
            current_script = ch_script
            current_chars.append(ch)
        elif ch_script == current_script or ch_script == "Common":
            current_chars.append(ch)
        elif current_script == "Common":
            current_script = ch_script
            current_chars.append(ch)
        else:
            segments.append(("".join(current_chars), current_script))
            current_chars = [ch]
            current_script = ch_script

    if current_chars:
        segments.append(("".join(current_chars), current_script or "Common"))

    return segments
