"""Unit tests for multilingual transliteration and identity preservation."""

import pytest
from src.transliterator import transliterate_text


def test_prompt_examples():
    # Example 1: Hindi
    assert transliterate_text("शिव शक्ति जनरल स्टोर") == "Shiv Shakti General Store"

    # Example 2: Gujarati
    assert transliterate_text("શ્રી ગણેશ મોબાઇલ") == "Shri Ganesh Mobile"

    # Example 3: French (Latin preserved)
    assert transliterate_text("Boulangerie Dupont") == "Boulangerie Dupont"

    # Example 4: Spanish (Latin unaccented)
    assert transliterate_text("Farmacia San José") == "Farmacia San Jose"

    # Example 5: Mixed script
    assert transliterate_text("ABC मोबाइल Store") == "ABC Mobile Store"


def test_indic_business_and_geographic_terms():
    # Legal terms
    assert transliterate_text("राम मार्केटिंग प्राइवेट लिमिटेड") == "Ram Marketing Private Limited"
    assert transliterate_text("आदित्य प्रॉपर्टीज एलएलपी") == "Aditya Properties LLP"
    assert transliterate_text("सन कंस्ट्रक्शंस प्राइवेट लिमिटेड") == "Sun Constructions Private Limited"

    # Geographic locations in addresses
    assert "Maharashtra" in transliterate_text("H.no 910 A 3503, Mumbai, महाराष्ट्र")
    assert "Gujarat" in transliterate_text("CRYSTAL PLAZA, VADODARA, ગુજરાત")
    assert "Karnataka" in transliterate_text("Door No 183, Bangalore, ಕರ್ನಾಟಕ")
    assert "Tamil Nadu" in transliterate_text("Coimbatore, தமிழ்நாடு")
    assert "Delhi" in transliterate_text("New Delhi, दिल्ली")


def test_mixed_scripts():
    assert transliterate_text("Shree राम Traders") == "Shree Ram Traders"
    assert transliterate_text("Paris दुकान") == "Paris Dukan"
    assert "ABC" in transliterate_text("شركة ABC")


def test_other_scripts():
    # Arabic
    ar = transliterate_text("متجر النور")
    assert ar.isascii() and len(ar) > 0

    # Chinese
    zh = transliterate_text("北京贸易公司")
    assert "BeiJing" in zh or "beijing" in zh.lower()

    # Cyrillic
    cyr = transliterate_text("ООО Вектор")
    assert cyr.isascii() and "Vektor" in cyr or "vektor" in cyr.lower()


def test_latin_preservation():
    # Ensure English and European names are preserved faithfully
    assert transliterate_text("ABC Trading Corporation") == "ABC Trading Corporation"
    assert transliterate_text("Mercado São João") == "Mercado Sao Joao"
    assert transliterate_text("LLC Moncada Léarning Center") == "LLC Moncada Learning Center"
    assert transliterate_text("Fractales Amis Groupe S.A.S") == "Fractales Amis Groupe S.A.S"
