"""Unit tests for Unicode normalization and sanitization."""

import pytest
from src.unicode_utils import normalize_unicode, unaccent_latin


def test_normalize_unicode_empty_and_ascii():
    assert normalize_unicode("") == ""
    assert normalize_unicode("   ") == ""
    assert normalize_unicode("ABC Trading Corp") == "ABC Trading Corp"
    assert normalize_unicode("  ABC   Trading   Corp  ") == "ABC Trading Corp"


def test_normalize_unicode_whitespace_variants():
    # Non-breaking space (\u00A0), em-space (\u2003), ideographic space (\u3000)
    text = "Acme\u00A0Corp\u2003LLC\u3000Services"
    assert normalize_unicode(text) == "Acme Corp LLC Services"


def test_normalize_unicode_invisible_and_control_chars():
    # Zero-width space (\u200B), BOM (\uFEFF), LTR mark (\u200E)
    text = "Super\u200B\uFEFFMarket\u200E Ltd"
    assert normalize_unicode(text) == "SuperMarket Ltd"

    # Stray tabs and newlines inside field
    text_stray = "Line1\nLine2\rLine3\tLine4"
    assert normalize_unicode(text_stray) == "Line1 Line2 Line3 Line4"


def test_normalize_unicode_nfkc_compatibility():
    # Fullwidth Latin
    fullwidth = "ＡＢＣ　Ｌｔｄ"
    assert normalize_unicode(fullwidth) == "ABC Ltd"

    # Ligatures
    ligature = "ﬁnance oﬃce"
    assert normalize_unicode(ligature) == "finance office"


def test_unaccent_latin():
    # French
    assert unaccent_latin("Boulangerie Dupont") == "Boulangerie Dupont"
    # Spanish
    assert unaccent_latin("Farmacia San José") == "Farmacia San Jose"
    # Portuguese
    assert unaccent_latin("Mercado São João") == "Mercado Sao Joao"
    # Accented English/French
    assert unaccent_latin("Léarning Center") == "Learning Center"
    assert unaccent_latin("Ptit Àmicale") == "Ptit Amicale"
