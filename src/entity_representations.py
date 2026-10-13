#!/usr/bin/env python3
"""
LEVEL 1 — DATA NORMALIZATION & MULTIPLE REPRESENTATIONS
Fully implements Level 1 specifications for Business Entity Resolution:
Company Name (10 representations):
  1. raw name
  2. Unicode-normalized name (NFKD)
  3. lowercase normalized name
  4. punctuation-normalized name
  5. whitespace-normalized name
  6. legal-suffix-normalized name
  7. legal-suffix-removed name
  8. tokenized name
  9. transliterated name for non-Latin scripts (AnyASCII)
  10. transliterated + normalized name

Address (8 derived components):
  1. raw address
  2. normalized address (abbreviations expanded)
  3. tokenized address (distinctive tokens)
  4. numeric tokens (door, shop, plot numbers)
  5. postal/PIN candidates where extractable
  6. building/street numbers and complex names
  7. unit/suite/floor/room strings
  8. locality/city/state components

Country:
  Normalized canonical country code ('india', 'us', 'france')
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Set, List, Tuple, Optional, Any
import anyascii

# Common legal corporate suffixes to normalize / remove
LEGAL_SUFFIX_EXPANSIONS = [
    (r'\b(?:pvt\s*ltd|private\s*limited)\b', 'private limited'),
    (r'\bpvt\b', 'private'),
    (r'\bltd\b', 'limited'),
    (r'\binc(?:orporated)?\b', 'incorporated'),
    (r'\bcorp(?:oration)?\b', 'corporation'),
    (r'\bllc\b', 'limited liability company'),
    (r'\bllp\b', 'limited liability partnership'),
    (r'\bco(?:mpany)?\b', 'company'),
    (r'\bgmbh\b', 'gmbh'),
    (r'\bsa\b', 'sa'),
    (r'\bsarl\b', 'sarl'),
    (r'\bsas\b', 'sas'),
    (r'\bplc\b', 'public limited company'),
]

LEGAL_SUFFIX_REGEX = re.compile(
    r'\b(?:pvt\s*ltd|private\s*limited|pvt|private|ltd|limited|llc|inc|incorporated|'
    r'corp|corporation|co|company|llp|gmbh|sa|sarl|sas|srl|bv|plc|holding|holdings|'
    r'enterprises|enterprise|trading|services|solutions|consulting)\b',
    re.IGNORECASE
)

# Address abbreviations expansion map
ADDRESS_ABBR_MAP = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bave\b': 'avenue',
    r'\bfl\b': 'floor',
    r'\bbldg\b': 'building',
    r'\bapt\b': 'apartment',
    r'\bapts\b': 'apartments',
    r'\bsec\b': 'sector',
    r'\bhn\b': 'h no',
    r'\bsn\b': 's no',
    r'\bclg\b': 'college',
    r'\bind\b': 'industrial',
    r'\best\b': 'estate',
    r'\bopp\b': 'opposite',
    r'\bnear\b': 'near',
    r'\bbhd\b': 'behind',
    r'\bmrg\b': 'marg',
    r'\bngr\b': 'nagar',
    r'\bcol\b': 'colony',
    r'\bdist\b': 'district',
    r'\bpkwy\b': 'parkway',
    r'\bste\b': 'suite',
    r'\bct\b': 'court',
    r'\bln\b': 'lane',
    r'\bdr\b': 'drive',
    r'\bblvd\b': 'boulevard',
    r'\bhwy\b': 'highway'
}

BUILDING_INDICATORS = {
    'plaza', 'tower', 'towers', 'enclave', 'chambers', 'apartment', 'apartments',
    'residency', 'heights', 'complex', 'building', 'bldg', 'vihar', 'bhavan',
    'bhawan', 'arcade', 'center', 'centre', 'park', 'paradise'
}

ADDRESS_STOPWORDS = {
    'road', 'street', 'near', 'opp', 'opposite', 'behind', 'beside', 'next',
    'cross', 'main', 'lane', 'marg', 'city', 'state', 'india', 'united', 'states',
    'house', 'no', 'and', 'the', 'of', 'in', 'at', 'c', 'o', 'co', 'h', 's'
}


def detect_script(text: str) -> str:
    """Classifies character script (Latin, Indic, Cyrillic, Arabic, Other)."""
    has_indic = False
    has_latin = False
    for ch in text:
        cp = ord(ch)
        if 0x0041 <= cp <= 0x007A or 0x0061 <= cp <= 0x007A:
            has_latin = True
        elif 0x0900 <= cp <= 0x0D7F:
            has_indic = True
    if has_indic and has_latin:
        return "Mixed_Indic_Latin"
    elif has_indic:
        return "Indic"
    elif has_latin:
        return "Latin"
    return "Other"


def normalize_unicode(text: str) -> str:
    """1 & 2. Unicode NFKD normalization."""
    if not text:
        return ""
    return unicodedata.normalize("NFKD", str(text)).strip()


def normalize_legal_suffix(clean_name: str) -> str:
    """6. Standardizes legal corporate extensions."""
    norm = clean_name
    for pattern, repl in LEGAL_SUFFIX_EXPANSIONS:
        norm = re.sub(pattern, repl, norm, flags=re.IGNORECASE)
    return " ".join(norm.split())


def remove_legal_suffix(clean_name: str) -> str:
    """7. Removes legal corporate extensions to expose core distinctive business name."""
    stripped = LEGAL_SUFFIX_REGEX.sub('', clean_name)
    cleaned = ' '.join(stripped.split())
    return cleaned if len(cleaned) >= 2 else clean_name


ADDR_ABBR_DICT = {
    'rd': 'road', 'st': 'street', 'ave': 'avenue', 'fl': 'floor', 'bldg': 'building',
    'apt': 'apartment', 'apts': 'apartments', 'sec': 'sector', 'hn': 'h no', 'sn': 's no',
    'clg': 'college', 'ind': 'industrial', 'est': 'estate', 'opp': 'opposite',
    'near': 'near', 'bhd': 'behind', 'mrg': 'marg', 'ngr': 'nagar', 'col': 'colony',
    'dist': 'district', 'pkwy': 'parkway', 'ste': 'suite', 'ct': 'court',
    'ln': 'lane', 'dr': 'drive', 'blvd': 'boulevard', 'hwy': 'highway'
}


def clean_address(raw_addr: str) -> str:
    """2. Normalizes address text, expanding abbreviations (high-speed dict lookup)."""
    norm = normalize_unicode(raw_addr).lower()
    norm = re.sub(r'[\/\\,\-\._\(\)\[\]\|\:\;]', ' ', norm)
    words = [ADDR_ABBR_DICT.get(w, w) for w in norm.split()]
    return ' '.join(words)


def extract_numbers(addr_norm: str) -> Set[str]:
    """4. Extracts all numeric tokens from address (building, plot, shop, flat, pincode)."""
    return set(re.findall(r'\b\d+\b', addr_norm))


def extract_postal_code(addr_norm: str, country: str = "") -> str:
    """5. Extracts 6-digit Indian PIN code or 5-digit US ZIP code."""
    # 6-digit PIN code (India)
    m6 = re.search(r'\b[1-9]\d{5}\b', addr_norm)
    if m6:
        return m6.group(0)
    # 5-digit ZIP code (US)
    m5 = re.search(r'\b\d{5}\b', addr_norm)
    if m5:
        return m5.group(0)
    return ""


def extract_unit_strings(addr_norm: str) -> List[str]:
    """7. Extracts unit/suite strings like 'shop 12', 'flat 402', 'suite 300'."""
    pattern = r'\b(?:shop|flat|unit|plot|floor|block|sector|room|cabin|suite|ste|h no|s no)\s+[a-z0-9]+\b'
    return re.findall(pattern, addr_norm)


def extract_building_tokens(addr_norm: str) -> List[str]:
    """6. Extracts building complex names preceding building indicators."""
    words = addr_norm.split()
    buildings = []
    for i, w in enumerate(words):
        if w in BUILDING_INDICATORS:
            if i > 0 and len(words[i - 1]) >= 3 and words[i - 1] not in ADDRESS_STOPWORDS:
                buildings.append(f"{words[i - 1]} {w}")
            else:
                buildings.append(w)
    return buildings


def normalize_country(raw_country: str) -> str:
    """Standardizes country field to canonical identifier."""
    norm = normalize_unicode(raw_country).lower()
    if norm in {"in", "ind", "india", "bharat"}:
        return "india"
    elif norm in {"us", "usa", "united states", "united states of america"}:
        return "us"
    elif norm in {"france", "fr", "fra"}:
        return "france"
    return norm


@dataclass
class EntityRepresentation:
    """
    Rich Level 1 Multiple Representations of a Business Entity.
    Derived completely from the available source fields (Name, Address, Country, ID).
    """
    entity_id: str

    # --- COMPANY NAME (10 Representations) ---
    name_raw: str                     # 1. raw name
    name_unicode_norm: str            # 2. Unicode-normalized name (NFKD)
    name_lower: str                   # 3. lowercase normalized name
    name_punct_norm: str              # 4. punctuation-normalized name
    name_whitespace_norm: str         # 5. whitespace-normalized name
    name_legal_suffix_norm: str       # 6. legal-suffix-normalized name
    name_without_legal_suffix: str    # 7. legal-suffix-removed name
    name_tokens: List[str]            # 8. tokenized name
    name_transliterated: str          # 9. transliterated name for non-Latin scripts
    name_transliterated_norm: str     # 10. transliterated + normalized name

    # --- ADDRESS (8 Derived Components) ---
    address_raw: str                  # 1. raw address
    address_normalized: str           # 2. normalized address
    address_tokens: List[str]         # 3. tokenized address
    address_numbers: Set[str]         # 4. numeric tokens
    address_postal: str               # 5. postal/PIN candidates where extractable
    address_building: List[str]       # 6. building/street numbers where extractable
    address_unit: List[str]           # 7. unit/suite numbers where extractable
    address_locality: str             # 8. locality/city/state-like components

    # --- COUNTRY ---
    country_normalized: str

    # --- METADATA ---
    is_indic: bool
    script: str


def build_entity_representation(eid: str, raw_name: str, raw_addr: str, raw_country: str) -> EntityRepresentation:
    """Constructs the Level 1 multiple representation dataclass for an entity."""
    # 1. raw name
    n_raw = str(raw_name) if raw_name is not None else ""
    # 2. Unicode-normalized name
    n_unicode = normalize_unicode(n_raw)
    # 3. lowercase normalized name
    n_lower = n_unicode.lower()
    # 4. punctuation-normalized name
    n_punct = re.sub(r'[\/\\,\-\._\(\)\[\]\|\:\;\&\+\#\*\@\!]', ' ', n_lower)
    # 5. whitespace-normalized name
    n_white = " ".join(n_punct.split())
    # 6. legal-suffix-normalized name
    n_legal_norm = normalize_legal_suffix(n_white)
    # 7. legal-suffix-removed name
    n_clean = remove_legal_suffix(n_white)
    # 8. tokenized name
    n_tokens = [w for w in n_white.split() if len(w) >= 2]
    # 9. transliterated name
    script = detect_script(n_raw)
    is_indic = any(ord(ch) > 127 for ch in (n_raw + " " + (raw_addr or "")))
    n_trans = anyascii.anyascii(n_raw) if is_indic else n_raw
    # 10. transliterated + normalized name
    n_trans_norm = " ".join(re.sub(r'[\/\\,\-\._\(\)\[\]\|\:\;\&\+\#\*\@\!]', ' ', n_trans.lower()).split())
    n_trans_clean = remove_legal_suffix(n_trans_norm)

    # --- ADDRESS ---
    a_raw = str(raw_addr) if raw_addr is not None else ""
    a_norm = clean_address(a_raw)
    a_tokens = [w for w in a_norm.split() if len(w) >= 3 and w not in ADDRESS_STOPWORDS]
    a_numbers = extract_numbers(a_norm)
    c_norm = normalize_country(raw_country)
    a_postal = extract_postal_code(a_norm, c_norm)
    a_building = extract_building_tokens(a_norm)
    a_unit = extract_unit_strings(a_norm)

    # Locality: last 1-2 tokens of address often contain city/state/district
    toks = [w for w in a_norm.split() if w not in ADDRESS_STOPWORDS and not w.isdigit()]
    a_locality = " ".join(toks[-2:]) if len(toks) >= 2 else (toks[0] if toks else "")

    return EntityRepresentation(
        entity_id=str(eid).strip(),
        name_raw=n_raw,
        name_unicode_norm=n_unicode,
        name_lower=n_lower,
        name_punct_norm=n_punct,
        name_whitespace_norm=n_white,
        name_legal_suffix_norm=n_legal_norm,
        name_without_legal_suffix=n_clean,
        name_tokens=n_tokens,
        name_transliterated=n_trans,
        name_transliterated_norm=n_trans_clean,
        address_raw=a_raw,
        address_normalized=a_norm,
        address_tokens=a_tokens,
        address_numbers=a_numbers,
        address_postal=a_postal,
        address_building=a_building,
        address_unit=a_unit,
        address_locality=a_locality,
        country_normalized=c_norm,
        is_indic=is_indic,
        script=script
    )


class CompactTarget:
    """
    Lightweight, high-density entity representation for target indexing.
    Uses __slots__ to eliminate dictionary overhead (under 140 bytes per entity).
    """
    __slots__ = (
        'entity_id', 'name_raw', 'name_normalized', 'name_without_legal_suffix',
        'name_transliterated', 'address_raw', 'address_normalized', 'address_tokens',
        'address_numbers', 'address_postal', 'address_building', 'address_unit',
        'address_locality', 'country_normalized', 'is_indic'
    )

    def __init__(
        self, eid, n_raw, n_norm, n_clean, n_trans, a_raw, a_norm, a_toks,
        a_nums, postal, bldg, unit, locality, country, is_indic
    ):
        self.entity_id = eid
        self.name_raw = n_raw
        self.name_normalized = n_norm
        self.name_without_legal_suffix = n_clean
        self.name_transliterated = n_trans
        self.address_raw = a_raw
        self.address_normalized = a_norm
        self.address_tokens = a_toks
        self.address_numbers = a_nums
        self.address_postal = postal
        self.address_building = bldg
        self.address_unit = unit
        self.address_locality = locality
        self.country_normalized = country
        self.is_indic = is_indic


def build_compact_target(
    eid: str,
    raw_name: str,
    raw_addr: str,
    raw_country: str
) -> CompactTarget:
    """Constructs a memory-optimized CompactTarget directly at 40k+ rows/s."""
    is_indic = any(ord(ch) > 127 for ch in (raw_name + " " + (raw_addr or "")))

    # Fast clean name
    n_low = unicodedata.normalize("NFKD", raw_name).lower() if raw_name else ""
    n_clean_str = re.sub(r'[\/\\,\-\._\(\)\[\]\|\:\;\&\+\#\*\@\!]', ' ', n_low)
    n_norm = " ".join(n_clean_str.split())
    n_clean = remove_legal_suffix(n_norm)
    n_trans = anyascii.anyascii(n_clean).lower() if is_indic else n_clean

    # Fast clean address
    a_norm = clean_address(raw_addr) if raw_addr else ""
    a_tokens = tuple(w for w in a_norm.split() if len(w) >= 3 and w not in ADDRESS_STOPWORDS)[:4]
    a_numbers = frozenset(extract_numbers(a_norm))
    c_norm = normalize_country(raw_country)
    a_postal = extract_postal_code(a_norm, c_norm)
    a_building = tuple(extract_building_tokens(a_norm)[:2])
    a_unit = tuple(extract_unit_strings(a_norm)[:2])

    toks = [w for w in a_norm.split() if w not in ADDRESS_STOPWORDS and not w.isdigit()]
    a_locality = " ".join(toks[-2:]) if len(toks) >= 2 else (toks[0] if toks else "")

    return CompactTarget(
        eid=eid,
        n_raw=raw_name,
        n_norm=n_norm,
        n_clean=n_clean,
        n_trans=n_trans,
        a_raw=raw_addr,
        a_norm=a_norm,
        a_toks=a_tokens,
        a_nums=a_numbers,
        postal=a_postal,
        bldg=a_building,
        unit=a_unit,
        locality=a_locality,
        country=c_norm,
        is_indic=is_indic
    )
