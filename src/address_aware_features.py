"""
ADDRESS-AWARE FEATURE EXTRACTOR
Amazon ML Challenge 2026 - Business Entity Resolution

Implements Rules A to F, the Address Decision Matrix, and explicit cross-field interactions:
- Rule A: exact_name + exact_address + exact_name_and_address
- Rule B: exact_address + exact_address_country
- Rule C: same_name_different_address + address_conflict
- Rule D: same_name_missing_address (distinguished from conflict)
- Rule E: different_name_exact_address
- Rule F: different_name_different_address
- Numeric address features (overlap, conflict, postal code exact)
- Character n-gram similarities (3-gram, 4-gram)
- Consonant skeleton Levenshtein (transliteration robustness)
- Structural & country agreement features
"""

import re
from typing import Dict, List, Tuple, Set, Any
from rapidfuzz import fuzz

from src.entity_representations import (
    EntityRepresentation,
    CompactTarget,
    clean_address,
    extract_numbers,
    extract_postal_code,
    normalize_country,
    remove_legal_suffix
)
from src.multi_path_funnel import COMMON_STOPWORDS


def get_char_ngrams(s: str, n: int = 3) -> Set[str]:
    clean = re.sub(r'[^a-z0-9]', '', s.lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i+n] for i in range(len(clean) - n + 1)}


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    g1 = get_char_ngrams(s1, n)
    g2 = get_char_ngrams(s2, n)
    union = len(g1.union(g2))
    return len(g1.intersection(g2)) / float(union) if union > 0 else 0.0


def consonant_skeleton(s: str) -> str:
    """Retains first character and consonants only, neutralizing transliteration vowel shifts."""
    clean = re.sub(r'[^a-z]', '', s.lower())
    if not clean:
        return ""
    first = clean[0]
    rest = re.sub(r'[aeiouy]', '', clean[1:])
    return first + rest


class AddressAwareFeatureExtractor:
    """Extracts 32 dense pairwise features explicitly modeling address and name interactions."""

    FEATURE_NAMES = [
        # --- Name Similarity Features (0..6) ---
        "name_lev",
        "name_clean_lev",
        "name_token_sort",
        "name_token_set",
        "name_char_3gram_jaccard",
        "name_char_4gram_jaccard",
        "name_consonant_skeleton_lev",

        # --- Name Structural Features (7..9) ---
        "name_len_diff",
        "name_token_overlap_ratio",
        "exact_name",

        # --- Address Similarity Features (10..15) ---
        "addr_lev",
        "addr_token_jaccard",
        "addr_token_overlap",
        "addr_char_3gram_jaccard",
        "addr_len_diff",
        "exact_address",

        # --- Address Missingness & Conflict (16..19) ---
        "addr_nonempty_both",
        "addr_missing_s1",
        "addr_missing_cand",
        "address_conflict",

        # --- Address Numeric & Postal Components (20..23) ---
        "postal_code_exact",
        "numeric_token_overlap",
        "numeric_token_conflict",
        "numeric_count_diff",

        # --- Cross-Field & Interaction Features (24..30) ---
        "exact_name_and_exact_address",
        "exact_address_same_country",
        "same_name_missing_address",
        "different_name_exact_address",
        "similar_name_similar_address",
        "similar_name_address_conflict",
        "different_name_different_address",

        # --- Country Signal (31) ---
        "country_match"
    ]

    @staticmethod
    def extract_features(
        q_rep: EntityRepresentation,
        t_rep: CompactTarget
    ) -> List[float]:
        # 1. Names
        n1 = getattr(q_rep, "name_whitespace_norm", "")
        n2 = getattr(t_rep, "name_normalized", "")
        n_lev = fuzz.ratio(n1, n2) / 100.0

        n1_c = getattr(q_rep, "name_without_legal_suffix", "")
        n2_c = getattr(t_rep, "name_without_legal_suffix", "")
        n_clean_lev = fuzz.ratio(n1_c, n2_c) / 100.0
        n_token_sort = fuzz.token_sort_ratio(n1_c, n2_c) / 100.0
        n_token_set = fuzz.token_set_ratio(n1_c, n2_c) / 100.0

        n_3gram = char_ngram_jaccard(n1_c, n2_c, 3)
        n_4gram = char_ngram_jaccard(n1_c, n2_c, 4)

        skel1 = consonant_skeleton(n1_c)
        skel2 = consonant_skeleton(n2_c)
        skel_lev = fuzz.ratio(skel1, skel2) / 100.0 if (skel1 and skel2) else n_clean_lev

        max_n_len = max(len(n1), len(n2), 1)
        name_len_diff = abs(len(n1) - len(n2)) / float(max_n_len)

        n1_toks = set(n1_c.split())
        n2_toks = set(n2_c.split())
        tok_overlap = len(n1_toks.intersection(n2_toks)) / float(max(len(n1_toks), 1))

        exact_name = 1.0 if (n1 and n2 and n1.lower() == n2.lower()) else 0.0

        # 2. Addresses
        a1 = getattr(q_rep, "address_normalized", "")
        a2 = getattr(t_rep, "address_normalized", "")

        has_a1 = bool(a1)
        has_a2 = bool(a2)
        addr_both = 1.0 if (has_a1 and has_a2) else 0.0
        addr_miss_s1 = 1.0 if not has_a1 else 0.0
        addr_miss_cand = 1.0 if not has_a2 else 0.0

        if has_a1 and has_a2:
            a_lev = fuzz.ratio(a1, a2) / 100.0
            a_3gram = char_ngram_jaccard(a1, a2, 3)
            max_a_len = max(len(a1), len(a2), 1)
            addr_len_diff = abs(len(a1) - len(a2)) / float(max_a_len)
            exact_addr = 1.0 if a1.lower() == a2.lower() else 0.0
        else:
            a_lev = 0.0
            a_3gram = 0.0
            addr_len_diff = 1.0
            exact_addr = 0.0

        a1_toks = set(getattr(q_rep, "address_tokens", []))
        a2_toks = set(getattr(t_rep, "address_tokens", []))
        u_toks = len(a1_toks.union(a2_toks))
        i_toks = len(a1_toks.intersection(a2_toks))
        a_jaccard = (i_toks / float(u_toks)) if u_toks > 0 else 0.0
        a_overlap = float(min(i_toks, 8)) / 8.0

        # Numeric components & postal
        q_nums = getattr(q_rep, "address_numbers", set())
        t_nums = getattr(t_rep, "address_numbers", set())
        num_overlap = float(len(q_nums.intersection(t_nums)))
        has_numeric_conflict = 1.0 if (q_nums and t_nums and not q_nums.intersection(t_nums)) else 0.0
        num_count_diff = float(abs(len(q_nums) - len(t_nums)))

        q_post = getattr(q_rep, "address_postal", "")
        t_post = getattr(t_rep, "address_postal", "")
        postal_match = 1.0 if (q_post and t_post and q_post == t_post) else 0.0

        # Address conflict (Rule C: both present, but different addresses with low similarity or conflicting numbers)
        addr_conflict = 0.0
        if has_a1 and has_a2:
            if exact_addr == 0.0 and (a_lev < 0.45 or has_numeric_conflict == 1.0):
                addr_conflict = 1.0

        # 3. Country Signal
        c1 = getattr(q_rep, "country_normalized", "")
        c2 = getattr(t_rep, "country_normalized", "")
        country_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0

        # 4. Cross-Field & Interaction Features
        # Rule A: Exact Name + Exact Address
        exact_both = 1.0 if (exact_name == 1.0 and exact_addr == 1.0) else 0.0

        # Rule B: Exact Address + Same Country
        exact_addr_country = 1.0 if (exact_addr == 1.0 and country_match == 1.0) else 0.0

        # Rule C: Same Name + Address Conflict
        same_name_diff_addr = 1.0 if (exact_name == 1.0 and addr_conflict == 1.0) else 0.0

        # Rule D: Same Name + Missing Address
        same_name_miss_addr = 1.0 if (exact_name == 1.0 and (addr_miss_s1 == 1.0 or addr_miss_cand == 1.0)) else 0.0

        # Rule E: Different Name + Exact Address
        diff_name_exact_addr = 1.0 if (exact_name == 0.0 and exact_addr == 1.0) else 0.0

        # High/Low combinations
        sim_name_sim_addr = 1.0 if (n_clean_lev >= 0.75 and a_lev >= 0.70) else 0.0
        sim_name_addr_conflict = 1.0 if (n_clean_lev >= 0.75 and addr_conflict == 1.0) else 0.0
        diff_name_diff_addr = 1.0 if (n_clean_lev < 0.50 and a_lev < 0.50) else 0.0

        return [
            n_lev,
            n_clean_lev,
            n_token_sort,
            n_token_set,
            n_3gram,
            n_4gram,
            skel_lev,
            name_len_diff,
            tok_overlap,
            exact_name,
            a_lev,
            a_jaccard,
            a_overlap,
            a_3gram,
            addr_len_diff,
            exact_addr,
            addr_both,
            addr_miss_s1,
            addr_miss_cand,
            addr_conflict,
            postal_match,
            num_overlap,
            has_numeric_conflict,
            num_count_diff,
            exact_both,
            exact_addr_country,
            same_name_miss_addr,
            diff_name_exact_addr,
            sim_name_sim_addr,
            sim_name_addr_conflict,
            diff_name_diff_addr,
            country_match
        ]


class EnhancedAddressFeaturesV4:
    """
    Extends feature extraction to 40 features incorporating:
    - Step 2: Core-brand matching and Levenshtein similarity
    - Step 3: Strict postal-code conflict and pincode numeric conflict hard penalties
    - Backward compatible with all 36 features of EXP_003
    """

    FEATURE_NAMES = [
        "name_lev",
        "name_clean_lev",
        "name_token_sort",
        "name_token_set",
        "name_char_3gram_jaccard",
        "name_char_4gram_jaccard",
        "name_consonant_skeleton_lev",
        "name_len_diff",
        "name_token_overlap_ratio",
        "exact_name",
        "addr_lev",
        "addr_token_jaccard",
        "addr_token_overlap",
        "addr_char_3gram_jaccard",
        "addr_len_diff",
        "exact_address",
        "addr_nonempty_both",
        "addr_missing_s1",
        "addr_missing_cand",
        "address_conflict",
        "postal_code_exact",
        "numeric_token_overlap",
        "numeric_token_conflict",
        "numeric_count_diff",
        "exact_name_and_exact_address",
        "exact_address_same_country",
        "same_name_missing_address",
        "different_name_exact_address",
        "similar_name_similar_address",
        "similar_name_address_conflict",
        "different_name_different_address",
        "country_match",
        # EXP_003 additions:
        "name_soundex_match",
        "name_rare_token_overlap",
        "name_weighted_jaccard",
        "max_token_idf",
        # Step 2 & 3 enhancements:
        "core_brand_exact",
        "core_brand_lev",
        "postal_code_conflict",
        "pincode_numeric_conflict"
    ]

    @staticmethod
    def extract_features(
        q_rep: EntityRepresentation,
        t_rep: CompactTarget,
        idf_dict: Dict[str, float],
        default_idf: float,
        q_core_brand: str = "",
        t_core_brand: str = ""
    ) -> List[float]:
        # Base 32 features
        base_32 = AddressAwareFeatureExtractor.extract_features(q_rep, t_rep)

        # 1. EXP_003: Soundex & Rare-Token IDF
        n1 = getattr(q_rep, "name_without_legal_suffix", "")
        n2 = getattr(t_rep, "name_without_legal_suffix", "")
        toks1 = [w for w in n1.split() if w not in COMMON_STOPWORDS and len(w) >= 2]
        toks2 = [w for w in n2.split() if w not in COMMON_STOPWORDS and len(w) >= 2]

        # Soundex
        from src.data_driven_blocker import soundex_token, extract_core_brand
        sx1 = soundex_token(toks1[0]) if toks1 else ""
        sx2 = soundex_token(toks2[0]) if toks2 else ""
        soundex_match = 1.0 if (sx1 and sx2 and sx1 == sx2) else 0.0

        # Rare token overlap & weighted Jaccard
        set1 = set(toks1)
        set2 = set(toks2)
        inter = set1.intersection(set2)
        rare_overlap = sum(1.0 for w in inter if idf_dict.get(w, default_idf) >= 3.5)

        w_inter = sum(idf_dict.get(w, default_idf) for w in inter)
        w_union = sum(idf_dict.get(w, default_idf) for w in set1.union(set2))
        w_jaccard = (w_inter / float(w_union)) if w_union > 0 else 0.0
        max_idf = max([idf_dict.get(w, default_idf) for w in inter], default=0.0)

        # 2. Step 2: Core-Brand Matching
        if not q_core_brand:
            q_core_brand = extract_core_brand(getattr(q_rep, "name_whitespace_norm", ""))
        if not t_core_brand:
            t_core_brand = extract_core_brand(getattr(t_rep, "name_normalized", ""))

        cb_exact = 1.0 if (q_core_brand and t_core_brand and q_core_brand.lower() == t_core_brand.lower()) else 0.0
        cb_lev = fuzz.ratio(q_core_brand, t_core_brand) / 100.0 if (q_core_brand and t_core_brand) else base_32[1]

        # 3. Step 3: Postal-Conflict Hard Penalties
        q_post = getattr(q_rep, "address_postal", "")
        t_post = getattr(t_rep, "address_postal", "")
        # Conflict: both present, valid length (>= 3 chars), but different
        if q_post and t_post and len(q_post) >= 3 and len(t_post) >= 3:
            postal_conflict = 1.0 if q_post != t_post else 0.0
        else:
            postal_conflict = 0.0

        num_conflict = base_32[22]  # numeric_token_conflict
        pincode_numeric_conflict = 1.0 if (postal_conflict == 1.0 or num_conflict == 1.0) else 0.0

        return base_32 + [
            soundex_match,
            float(rare_overlap),
            float(w_jaccard),
            float(max_idf),
            cb_exact,
            cb_lev,
            postal_conflict,
            pincode_numeric_conflict
        ]

