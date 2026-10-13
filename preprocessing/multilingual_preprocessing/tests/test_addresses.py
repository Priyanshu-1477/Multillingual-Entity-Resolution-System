"""Unit tests for address normalization."""

import pytest
from src.address_normalizer import normalize_address


def test_standard_address_preservation():
    addr = "1795 Westchester Drive, High Point, NC 27262"
    assert normalize_address(addr) == "1795 Westchester Drive, High Point, NC 27262"


def test_preserve_pin_and_leading_zeros():
    # Crucial rule: preserve leading zeros in postal codes, unit numbers
    addr = "Shop 004, Sector 012, PIN 00123"
    assert "004" in normalize_address(addr)
    assert "012" in normalize_address(addr)
    assert "00123" in normalize_address(addr)


def test_clean_rogue_null_tokens():
    # Dataset pattern: addresses with <NULL> or null inserted between commas
    addr1 = "33466 WARWICK HILLS ROAD, <NULL>, YUCAIPA, CA"
    assert normalize_address(addr1) == "33466 WARWICK HILLS ROAD, YUCAIPA, CA"

    addr2 = "G.t. Karnal Road, New Delhi, null, A-68, Delhi"
    assert normalize_address(addr2) == "G.t. Karnal Road, New Delhi, A-68, Delhi"


def test_multilingual_address_transliteration():
    addr = "H.no 910 A 3503, Mumbai, महाराष्ट्र"
    assert normalize_address(addr) == "H.no 910 A 3503, Mumbai, Maharashtra"

    addr_guj = "VADODARA, ગુજરાત"
    assert normalize_address(addr_guj) == "VADODARA, Gujarat"


def test_empty_and_placeholder_addresses():
    assert normalize_address("") == ""
    assert normalize_address("   ") == ""
    assert normalize_address("<NULL>") == ""
    assert normalize_address("null") == ""
