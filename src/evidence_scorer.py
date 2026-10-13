#!/usr/bin/env python3
"""
LEVEL 3 — FOUR CORE EVIDENCE SCORES
LEVEL 4 — CANDIDATE SCREENING
Calculates the 4 major consolidated evidence scores:
  - Score 1: Normalized Company Name (Levenshtein, Jaro-Winkler, Token Sort, Token Set)
  - Score 2: Transliteration Name (Levenshtein, Jaro-Winkler, Token Sort, Token Set on transliterated text)
  - Score 3: Address Similarity (Full Address Levenshtein, Token Sort, Token Jaccard, Token Overlap)
  - Score 4: Address Structure (Numeric tokens, Postal/PIN, Building, Unit, Locality)

Implements permissive, high-recall Candidate Screening:
  Retains candidate if ANY major channel provides sufficient evidence.
"""

from typing import Tuple, Set, List, Any
from rapidfuzz import fuzz, distance


class EvidenceScorer:
    """
    Computes Level 3 Four Core Evidence Scores between query (S1) and target (S2/S3).
    All scores are normalized in [0.0, 1.0].
    """

    @staticmethod
    def compute_score_1_name(query: Any, target: Any) -> float:
        """
        SCORE 1 — Normalized Company Name
        Combines:
          - normalized Levenshtein similarity
          - Jaro-Winkler similarity
          - token sort similarity
          - token set similarity
        """
        q_name = getattr(query, "name_without_legal_suffix", "") or getattr(query, "name_whitespace_norm", "")
        t_name = getattr(target, "name_without_legal_suffix", "") or getattr(target, "name_normalized", "")

        if not q_name or not t_name:
            return 0.0

        # Exact match fast path
        if q_name == t_name:
            return 1.0

        lev = fuzz.ratio(q_name, t_name) / 100.0
        jw = distance.JaroWinkler.similarity(q_name, t_name)
        tok_sort = fuzz.token_sort_ratio(q_name, t_name) / 100.0
        tok_set = fuzz.token_set_ratio(q_name, t_name) / 100.0

        # Consolidated normalized-name score
        return (0.35 * lev) + (0.35 * jw) + (0.15 * tok_sort) + (0.15 * tok_set)

    @staticmethod
    def compute_score_2_transliterated_name(query: Any, target: Any) -> float:
        """
        SCORE 2 — Transliteration Name
        Specifically intended to recover cross-script matches (e.g. English <-> Devanagari).
        """
        q_trans = getattr(query, "name_transliterated_norm", "") or getattr(query, "name_transliterated", "")
        t_trans = getattr(target, "name_transliterated", "")

        if not q_trans or not t_trans:
            return 0.0

        if q_trans == t_trans:
            return 1.0

        lev = fuzz.ratio(q_trans, t_trans) / 100.0
        jw = distance.JaroWinkler.similarity(q_trans, t_trans)
        tok_sort = fuzz.token_sort_ratio(q_trans, t_trans) / 100.0
        tok_set = fuzz.token_set_ratio(q_trans, t_trans) / 100.0

        # Consolidated transliterated-name score
        return (0.35 * lev) + (0.35 * jw) + (0.15 * tok_sort) + (0.15 * tok_set)

    @staticmethod
    def compute_score_3_address(query: Any, target: Any) -> float:
        """
        SCORE 3 — Address Similarity
        Combines:
          - full normalized-address similarity
          - address token similarity
          - token Jaccard
          - token overlap
        """
        q_addr = getattr(query, "address_normalized", "")
        t_addr = getattr(target, "address_normalized", "")

        if not q_addr or not t_addr:
            return 0.0

        if q_addr == t_addr:
            return 1.0

        full_sim = fuzz.ratio(q_addr, t_addr) / 100.0
        tok_sim = fuzz.token_sort_ratio(q_addr, t_addr) / 100.0

        q_tokens = set(getattr(query, "address_tokens", []))
        t_tokens = set(getattr(target, "address_tokens", []))

        if q_tokens and t_tokens:
            inter_len = len(q_tokens.intersection(t_tokens))
            union_len = len(q_tokens.union(t_tokens))
            jaccard = inter_len / float(union_len) if union_len > 0 else 0.0
            overlap = inter_len / float(min(len(q_tokens), len(t_tokens))) if min(len(q_tokens), len(t_tokens)) > 0 else 0.0
        else:
            jaccard = 0.0
            overlap = 0.0

        # Consolidated address score
        return (0.35 * full_sim) + (0.25 * tok_sim) + (0.20 * jaccard) + (0.20 * overlap)

    @staticmethod
    def compute_score_4_address_structure(query: Any, target: Any) -> float:
        """
        SCORE 4 — Address Structure
        Compares extracted address components:
          - numeric-token agreement (door/plot/building numbers)
          - postal/PIN agreement
          - building-number agreement
          - unit-number agreement
          - locality/city agreement
        """
        # 1. Numeric token agreement
        q_nums = set(getattr(query, "address_numbers", []))
        t_nums = set(getattr(target, "address_numbers", []))
        if q_nums and t_nums:
            if q_nums == t_nums:
                num_agree = 1.0
            else:
                inter = q_nums.intersection(t_nums)
                num_agree = len(inter) / float(len(q_nums.union(t_nums))) if inter else 0.0
        elif not q_nums and not t_nums:
            num_agree = 0.50  # neutral when neither has numbers
        else:
            num_agree = 0.20

        # 2. Postal / PIN agreement
        q_postal = getattr(query, "address_postal", "")
        t_postal = getattr(target, "address_postal", "")
        if q_postal and t_postal:
            postal_agree = 1.0 if q_postal == t_postal else 0.0
        else:
            postal_agree = 0.50  # neutral

        # 3. Building agreement
        q_bldg = set(getattr(query, "address_building", []))
        t_bldg = set(getattr(target, "address_building", []))
        if q_bldg and t_bldg:
            bldg_agree = 1.0 if q_bldg.intersection(t_bldg) else 0.0
        else:
            bldg_agree = 0.50

        # 4. Unit / Suite agreement
        q_unit = set(getattr(query, "address_unit", []))
        t_unit = set(getattr(target, "address_unit", []))
        if q_unit and t_unit:
            unit_agree = 1.0 if q_unit.intersection(t_unit) else 0.0
        else:
            unit_agree = 0.50

        # 5. Locality agreement
        q_loc = getattr(query, "address_locality", "")
        t_loc = getattr(target, "address_locality", "")
        if q_loc and t_loc:
            loc_agree = fuzz.ratio(q_loc, t_loc) / 100.0
        else:
            loc_agree = 0.50

        # Consolidated address structure score
        return (
            (0.35 * num_agree) +
            (0.25 * postal_agree) +
            (0.15 * bldg_agree) +
            (0.10 * unit_agree) +
            (0.15 * loc_agree)
        )

    @classmethod
    def compute_all_scores(cls, query: Any, target: Any) -> Tuple[float, float, float, float]:
        """Returns (score_1_name, score_2_transliterated_name, score_3_address, score_4_address_structure)."""
        s1 = cls.compute_score_1_name(query, target)
        s2 = cls.compute_score_2_transliterated_name(query, target)
        s3 = cls.compute_score_3_address(query, target)
        s4 = cls.compute_score_4_address_structure(query, target)
        return s1, s2, s3, s4

    compute_evidence_scores = compute_all_scores


class CandidateScreening:
    """
    LEVEL 4: CANDIDATE SCREENING
    Do NOT use `name_score > X AND address_score > Y`.
    Instead retain a candidate if there is sufficient evidence through ANY major channel:
      - strong name evidence
      - OR strong transliterated-name evidence
      - OR strong address evidence
      - OR strong address-structure evidence
      - OR moderate evidence from multiple channels
    """

    @staticmethod
    def screen_candidate(scores: Tuple[float, float, float, float]) -> bool:
        s1, s2, s3, s4 = scores

        # 1. Strong name evidence
        if s1 >= 0.65:
            return True

        # 2. Strong transliterated-name evidence (cross-script rescue)
        if s2 >= 0.65:
            return True

        # 3. Strong address evidence (DBA / rebranding rescue)
        if s3 >= 0.60:
            return True

        # 4. Strong address-structure evidence (identical street/building numbers + postal)
        if s4 >= 0.75:
            return True

        # 5. Moderate evidence from multiple channels
        # (e.g. moderate name + moderate address)
        if max(s1, s2) >= 0.35 and (s3 >= 0.35 or s4 >= 0.50):
            return True

        # (e.g. solid address with non-zero name)
        if s3 >= 0.45 and max(s1, s2) >= 0.20:
            return True

        return False
