"""Unit tests for business name conservative normalization."""

import pytest
from src.name_normalizer import normalize_business_name


def test_clean_standard_names():
    assert normalize_business_name("Acme Corporation LLC") == "Acme Corporation LLC"
    assert normalize_business_name("B+ Retail Inc") == "B+ Retail Inc"
    assert normalize_business_name("Fractales Amis Groupe S.A.S") == "Fractales Amis Groupe S.A.S"


def test_strip_leading_trailing_noise():
    # Noise symbols from dataset
    assert normalize_business_name("-- Holloway Peak Inc Seafood") == "Holloway Peak Inc Seafood"
    assert normalize_business_name("<< Team Ecole >>") == "Team Ecole"
    assert normalize_business_name("** Prime Solutions **") == "Prime Solutions"
    assert normalize_business_name("- Global Logistics -") == "Global Logistics"


def test_empty_and_null_names():
    assert normalize_business_name("") == ""
    assert normalize_business_name("   ") == ""
    assert normalize_business_name("<null>") == ""
    assert normalize_business_name("<none>") == ""


def test_multilingual_names():
    assert normalize_business_name("शिव शक्ति जनरल स्टोर") == "Shiv Shakti General Store"
    assert normalize_business_name("શ્રી ગણેશ મોબાઇલ") == "Shri Ganesh Mobile"
    assert normalize_business_name("Boulangerie Dupont") == "Boulangerie Dupont"
    assert normalize_business_name("Farmacia San José") == "Farmacia San Jose"
    assert normalize_business_name("ABC मोबाइल Store") == "ABC Mobile Store"
