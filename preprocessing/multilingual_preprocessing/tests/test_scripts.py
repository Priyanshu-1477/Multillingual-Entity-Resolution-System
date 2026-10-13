"""Unit tests for Unicode script detection and segmentation."""

import pytest
from src.script_detector import detect_scripts, get_char_script, has_non_latin, is_pure_latin, segment_by_script


def test_char_scripts():
    assert get_char_script("A") == "Latin"
    assert get_char_script("é") == "Latin"
    assert get_char_script("1") == "Common"
    assert get_char_script(" ") == "Common"
    assert get_char_script("&") == "Common"

    assert get_char_script("क") == "Devanagari"
    assert get_char_script("গ") == "Bengali"
    assert get_char_script("ક") == "Gujarati"
    assert get_char_script("ਕ") == "Gurmukhi"
    assert get_char_script("க") == "Tamil"
    assert get_char_script("క") == "Telugu"
    assert get_char_script("ಕ") == "Kannada"
    assert get_char_script("ക") == "Malayalam"
    assert get_char_script("କ") == "Odia"

    assert get_char_script("م") == "Arabic"
    assert get_char_script("北") == "CJK"
    assert get_char_script("Д") == "Cyrillic"
    assert get_char_script("Ω") == "Greek"
    assert get_char_script("ש") == "Hebrew"


def test_detect_scripts():
    assert detect_scripts("ABC Trading Corp") == {"Latin"}
    assert detect_scripts("शिव शक्ति") == {"Devanagari"}
    assert detect_scripts("શ્રી ગણેશ") == {"Gujarati"}
    assert detect_scripts("ABC मोबाइल Store") == {"Latin", "Devanagari"}


def test_is_pure_latin():
    assert is_pure_latin("ABC Trading Corp 123 & Co.") is True
    assert is_pure_latin("Farmacia San José") is True
    assert is_pure_latin("शिव शक्ति") is False
    assert is_pure_latin("ABC मोबाइल Store") is False


def test_has_non_latin():
    assert has_non_latin("ABC Corp") is False
    assert has_non_latin("Farmacia San José") is False
    assert has_non_latin("शिव शक्ति") is True
    assert has_non_latin("ABC मोबाइल Store") is True


def test_segment_by_script():
    segments = segment_by_script("ABC मोबाइल Store")
    scripts = [s[1] for s in segments]
    assert "Latin" in scripts
    assert "Devanagari" in scripts
