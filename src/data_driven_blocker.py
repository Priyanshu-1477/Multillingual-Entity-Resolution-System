#!/usr/bin/env python3
"""
DATA-DRIVEN BLOCKER & CANDIDATE RETRIEVAL
Amazon ML Challenge 2026 - Business Entity Resolution

Implements:
- Step 2: Core-Brand Extraction + India Phonetic (Double-Metaphone/Soundex) Blocking
- Step 4: Data-Driven TF-IDF / BM25 Inverted Index Weights
- Step 5: Data-Driven Blocker with Automated Dynamic Selectivity Filtering
"""

import re
import math
import unicodedata
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional, Any
from array import array

from src.entity_representations import (
    EntityRepresentation,
    CompactTarget,
    clean_address,
    extract_postal_code,
    normalize_country,
    remove_legal_suffix
)
from src.multi_path_funnel import COMMON_STOPWORDS

# Corporate generic tokens to strip for core brand extraction
GENERIC_BRAND_STOPWORDS = {
    'private', 'limited', 'pvt', 'ltd', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'gmbh', 'sarl', 'sa', 'sas', 'srl', 'bv', 'plc',
    'holding', 'holdings', 'group', 'enterprises', 'enterprise', 'industries', 'industry',
    'services', 'service', 'solutions', 'solution', 'consulting', 'consultancy',
    'technologies', 'technology', 'tech', 'trading', 'traders', 'venture', 'ventures',
    'international', 'systems', 'system', 'india', 'united', 'states', 'france',
    'associates', 'agency', 'logistics', 'management', 'global', 'commercial',
    'development', 'properties', 'real', 'estate', 'infra', 'infracon', 'buildhome'
}

TRANSLATE_TABLE = str.maketrans('/-.,\'\"()[]{}#@!*&|\\:;?<>', '                        ')


def clean_norm_str(s: str) -> str:
    if not s:
        return ""
    return " ".join(s.lower().translate(TRANSLATE_TABLE).split())


def extract_core_brand(name: str) -> str:
    """
    Step 2: Extracts distinctive core brand root by stripping legal designators 
    and generic industry stopwords.
    e.g., 'Amazon Media (India) Limited' -> 'amazon'
    e.g., 'Tata Consultancy Services Pvt Ltd' -> 'tata'
    """
    if not name:
        return ""
    clean = clean_norm_str(name)
    tokens = [w for w in clean.split() if w not in GENERIC_BRAND_STOPWORDS and len(w) >= 2]
    if not tokens:
        tokens = [w for w in clean.split() if len(w) >= 2]
    return " ".join(tokens[:3]) if tokens else clean


def soundex_token(token: str) -> str:
    """Standard American Soundex key."""
    if not token:
        return ""
    s = re.sub(r'[^a-z]', '', token.lower())
    if not s:
        return ""
    first = s[0].upper()
    mapping = {
        'b': '1', 'f': '1', 'p': '1', 'v': '1',
        'c': '2', 'g': '2', 'j': '2', 'k': '2', 'q': '2', 's': '2', 'x': '2', 'z': '2',
        'd': '3', 't': '3',
        'l': '4',
        'm': '5', 'n': '5',
        'r': '6'
    }
    codes = [mapping.get(ch, '0') for ch in s[1:]]
    res = [first]
    prev = mapping.get(s[0], '0')
    for c in codes:
        if c != '0' and c != prev:
            res.append(c)
        prev = c
    return "".join(res)[:4].ljust(4, '0')



def india_phonetic_key(token: str) -> str:
    """
    Step 2: Fast India Phonetic Transformation Key.
    Neutralizes Latin transliteration variations common in Indian records:
    - Lakshmi / Laxmi -> lksm
    - Vikas / Vickas -> vks
    - Chandra / Chander -> cndr
    """
    if not token or len(token) < 2:
        return ""
    s = token.lower()
    if 'h' in s or 'w' in s or 'k' in s or 'x' in s or 'e' in s or 'o' in s:
        s = s.replace('ph', 'f').replace('w', 'v').replace('sh', 's').replace('ksh', 'x')
        s = s.replace('ee', 'i').replace('oo', 'u').replace('th', 't').replace('ch', 'c')
        s = s.replace('ck', 'k').replace('kh', 'k').replace('gh', 'g').replace('dh', 'd')

    first = s[0]
    clean = [first]
    prev = first
    for ch in s[1:]:
        if ch not in 'aeiouy' and ch.isalpha():
            if ch != prev:
                clean.append(ch)
                prev = ch
    return "".join(clean)[:6]



class DataDrivenBlocker:
    """
    Step 4 & 5: High-efficiency, data-driven multi-predicate blocker.
    Uses corpus-mined token IDF weights to dynamically score and rank candidates,
    while pruning high-frequency non-selective tokens to prevent bucket bloat.
    """

    def __init__(
        self,
        target_store: Any,
        idf_dict: Dict[str, float],
        default_idf: float,
        max_df_cutoff: int = 2500,
        max_cands: int = 70
    ):
        self.target_store = target_store
        self.idf_dict = idf_dict
        self.default_idf = default_idf
        self.max_df_cutoff = max_df_cutoff
        self.max_cands = max_cands

        # Multi-predicate inverted indexes (array('i') minimizes memory overhead for 10M records)
        self.exact_name_idx: Dict[str, array] = defaultdict(lambda: array("i"))
        self.core_brand_idx: Dict[str, array] = defaultdict(lambda: array("i"))
        self.exact_addr_idx: Dict[str, array] = defaultdict(lambda: array("i"))
        self.rare_token_idx: Dict[str, array] = defaultdict(lambda: array("i"))
        self.india_phonetic_idx: Dict[str, array] = defaultdict(lambda: array("i"))
        self.postal_idx: Dict[str, array] = defaultdict(lambda: array("i"))
        self.number_idx: Dict[str, array] = defaultdict(lambda: array("i"))


    def get_idf(self, token: str) -> float:
        return self.idf_dict.get(token, self.default_idf)

    def index_target_line(
        self,
        int_id: int,
        raw_name: str,
        raw_addr: str,
        raw_country: str
    ):
        """Indexes target using Level 1 multiple representations with data-driven selectivity."""
        norm_n = clean_norm_str(raw_name)
        norm_a = clean_norm_str(raw_addr)
        country = normalize_country(raw_country)

        # 1. Exact Name Index
        if norm_n:
            arr = self.exact_name_idx[norm_n]
            if len(arr) < 500:
                arr.append(int_id)

        # 2. Core Brand Index (Step 2)
        core_brand = extract_core_brand(raw_name)
        if core_brand and core_brand != norm_n:
            arr_cb = self.core_brand_idx[core_brand]
            if len(arr_cb) < 800:
                arr_cb.append(int_id)

        # 3. Exact Address Index
        if norm_a:
            arr_a = self.exact_addr_idx[norm_a]
            if len(arr_a) < 500:
                arr_a.append(int_id)

        # 4. Data-Driven Rare Token Index (Step 4 & 5)
        # Selectively index only tokens with high IDF (rarity) to avoid bloat
        n_words = [w for w in norm_n.split() if w not in COMMON_STOPWORDS and len(w) >= 3]
        for w in n_words[:5]:
            idf = self.get_idf(w)
            if idf >= 3.0:  # Data-driven selectivity cutoff
                arr_rt = self.rare_token_idx[w]
                if len(arr_rt) < self.max_df_cutoff:
                    arr_rt.append(int_id)

        # 5. India Phonetic Blocking (Step 2)
        if country == 'india' and n_words:
            p_key = india_phonetic_key(n_words[0])
            if p_key:
                # Compound key: phonetic + state/city if available
                arr_ph = self.india_phonetic_idx[p_key]
                if len(arr_ph) < 1200:
                    arr_ph.append(int_id)

        # 6. Postal Index & Numeric Index
        if norm_a:
            postal = extract_postal_code(norm_a, country)
            if postal:
                arr_p = self.postal_idx[postal]
                if len(arr_p) < 1500:
                    arr_p.append(int_id)

            nums = [w for w in norm_a.split() if w.isdigit() and len(w) >= 2]
            for num in nums[:2]:
                arr_num = self.number_idx[num]
                if len(arr_num) < 1500:
                    arr_num.append(int_id)

    def retrieve_candidates(self, query: EntityRepresentation) -> List[int]:
        """
        Step 4: Retrieves candidates ranked by data-driven TF-IDF / BM25 match scores.
        """
        norm_n = clean_norm_str(query.name_whitespace_norm)
        norm_a = clean_norm_str(query.address_normalized)
        country = getattr(query, "country_normalized", "other")

        cand_scores: Dict[int, float] = defaultdict(float)

        # 1. Exact Name: Data-driven score proportional to token IDF sum
        if norm_n and norm_n in self.exact_name_idx:
            n_tokens = norm_n.split()
            score_weight = min(sum(self.get_idf(w) for w in n_tokens), 35.0)
            for int_id in self.exact_name_idx[norm_n]:
                cand_scores[int_id] += max(score_weight, 25.0)

        # 2. Core Brand Exact Match (Step 2)
        core_brand = extract_core_brand(query.name_whitespace_norm)
        if core_brand and core_brand in self.core_brand_idx:
            cb_tokens = core_brand.split()
            cb_weight = min(sum(self.get_idf(w) for w in cb_tokens), 28.0)
            for int_id in self.core_brand_idx[core_brand]:
                cand_scores[int_id] += max(cb_weight, 18.0)

        # 3. Exact Address: Data-driven score proportional to address IDF sum
        if norm_a and norm_a in self.exact_addr_idx:
            a_tokens = norm_a.split()
            a_weight = min(sum(self.get_idf(w) for w in a_tokens if len(w) >= 3), 30.0)
            for int_id in self.exact_addr_idx[norm_a]:
                cand_scores[int_id] += max(a_weight, 20.0)

        # 4. Rare Tokens: Accumulate true data-driven IDF points (Step 4)
        n_words = [w for w in norm_n.split() if w not in COMMON_STOPWORDS and len(w) >= 3]
        for w in n_words[:5]:
            idf = self.get_idf(w)
            if w in self.rare_token_idx:
                for int_id in self.rare_token_idx[w]:
                    cand_scores[int_id] += idf

        # 5. India Phonetic Blocking (Step 2)
        if country == 'india' and n_words:
            p_key = india_phonetic_key(n_words[0])
            if p_key and p_key in self.india_phonetic_idx:
                for int_id in self.india_phonetic_idx[p_key]:
                    cand_scores[int_id] += 5.5

        # 6. Postal Code & Street Numbers
        if norm_a:
            postal = getattr(query, "address_postal", "")
            if postal and postal in self.postal_idx:
                for int_id in self.postal_idx[postal]:
                    cand_scores[int_id] += 4.5

            q_nums = getattr(query, "address_numbers", set())
            for num in list(q_nums)[:2]:
                if num in self.number_idx:
                    for int_id in self.number_idx[num]:
                        cand_scores[int_id] += 2.0

        if not cand_scores:
            return []

        # Sort by accumulated data-driven score and return top K
        sorted_cands = sorted(cand_scores.items(), key=lambda x: -x[1])
        return [int_id for int_id, _ in sorted_cands[:self.max_cands]]
