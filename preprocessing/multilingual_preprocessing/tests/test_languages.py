"""Unit tests for language detection and contextual inference."""

import pytest
from src.language_detector import detect_probable_language


def test_detect_language_by_script():
    assert detect_probable_language("शिव शक्ति जनरल स्टोर", "India") == "Hindi"
    assert detect_probable_language("શ્રી ગણેશ મોબાઇલ", "India") == "Gujarati"
    assert detect_probable_language("শিব শক্তি", "India") == "Bengali"
    assert detect_probable_language("ஸ்ரீ கணேஷ்", "India") == "Tamil"
    assert detect_probable_language("శ్రీ గణేష్", "India") == "Telugu"
    assert detect_probable_language("ಶ್ರೀ ಗಣೇಶ್", "India") == "Kannada"
    assert detect_probable_language("ശ്രീ ഗണേഷ്", "India") == "Malayalam"
    assert detect_probable_language("متجر النور", "France") == "Arabic"
    assert detect_probable_language("北京贸易公司", "US") == "Chinese"


def test_detect_language_contextual():
    assert detect_probable_language("Boulangerie Dupont", "France") == "French"
    assert detect_probable_language("ABC Trading Corp", "United States") == "English"
    assert detect_probable_language("Farmacia San Jose", "US") == "Spanish"
    assert detect_probable_language("National Motors Pvt Ltd", "India") == "English (India)"
