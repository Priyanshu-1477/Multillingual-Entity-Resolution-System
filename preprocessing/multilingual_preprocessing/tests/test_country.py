"""Unit tests for open-set country normalization."""

import pytest
from src.country_normalizer import normalize_country


def test_core_country_normalization():
    # India
    assert normalize_country("IND") == "India"
    assert normalize_country("IN") == "India"
    assert normalize_country("India") == "India"
    assert normalize_country("Republic of India") == "India"

    # France
    assert normalize_country("FRA") == "France"
    assert normalize_country("FR") == "France"
    assert normalize_country("France") == "France"

    # United States
    assert normalize_country("US") == "United States"
    assert normalize_country("USA") == "United States"
    assert normalize_country("United States") == "United States"
    assert normalize_country("United States of America") == "United States"


def test_open_set_country_preservation():
    # The country field is open-set: other countries must be preserved, never dropped
    assert normalize_country("Canada") == "Canada"
    assert normalize_country("GERMANY") == "Germany"
    assert normalize_country("japan") == "Japan"
    assert normalize_country("Brazil") == "Brazil"


def test_empty_country():
    assert normalize_country("") == ""
    assert normalize_country("   ") == ""
