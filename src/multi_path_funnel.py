#!/usr/bin/env python3
"""
LEVEL 2 — HIGH-RECALL MULTI-PATH CANDIDATE GENERATION
LEVEL 6 — FULL PAIRWISE FEATURE ENGINEERING
Implements independent candidate-generation paths and their UNION:
  - Candidate Path A — normalized company name (exact name, prefixes, informative tokens)
  - Candidate Path B — transliterated company name (transliterated normalized name, transliterated tokens)
  - Candidate Path C — address (address tokens, rare tokens, numeric tokens, building numbers, locality, postal)
  - Candidate Path D — combined address blocks (postal + token, locality + token, building + locality, building + postal)
  - Candidate Path E — country-aware blocking (prefer candidates from same normalized country)
UNION:
  candidates = name_candidates UNION transliterated_name_candidates UNION address_candidates UNION combined_address_candidates
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
from rapidfuzz import fuzz, distance

from .entity_representations import EntityRepresentation, CompactTarget, build_entity_representation
from .evidence_scorer import EvidenceScorer, CandidateScreening


COMMON_STOPWORDS = {
    "inc", "corp", "corporation", "llc", "ltd", "limited", "co", "company",
    "the", "and", "services", "group", "enterprises", "solutions", "holdings",
    "pvt", "private", "sa", "sarl", "sas", "associates", "partners", "management"
}

# Bitmask flags for paths
PATH_FLAG_NAME = (1 << 0)         # 1: Path A
PATH_FLAG_TRANSLIT = (1 << 1)     # 2: Path B
PATH_FLAG_ADDR = (1 << 2)         # 4: Path C
PATH_FLAG_COMBINED = (1 << 3)     # 8: Path D
PATH_FLAG_POSTAL = (1 << 4)       # 16: Postal
PATH_FLAG_LOCALITY = (1 << 5)     # 32: Locality


class MultiPathIndex:
    """
    LEVEL 2: High-Recall Multi-Path Candidate Generation.
    Maintains compact inverted index maps with distinctive token indexing:
    - Path A: Normalized distinctive company-name words & prefixes
    - Path B: Transliterated distinctive words
    - Path C: Address distinctive tokens & street/building numbers
    - Path D: Combined address blocks (postal + token, locality + token, building + postal)
    - Path E: Country-aware partitioning
    """

    def __init__(self, max_bucket_size: int = 2500):
        self.max_bucket_size = max_bucket_size

        # Inverted index tables: key = (country, token) -> list of target int IDs
        self.path_a_name_idx: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.path_b_trans_idx: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.path_c_addr_idx: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.path_c_num_idx: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.path_c_postal_idx: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.path_c_locality_idx: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        self.path_c_bldg_idx: Dict[Tuple[str, str], List[int]] = defaultdict(list)

        # Path D: Combined address blocks
        self.path_d_postal_tok_idx: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        self.path_d_loc_tok_idx: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
        self.path_d_bldg_loc_idx: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)

        self.target_list: List[Any] = []
        self.tid_to_idx: Dict[str, int] = {}

    def get_target(self, tid: str, default=None):
        """Direct O(1) target lookup without proxy allocation."""
        idx = self.tid_to_idx.get(tid)
        return self.target_list[idx] if idx is not None else default

    @property
    def target_store(self):
        class TargetStoreProxy:
            def __init__(self, tlist, tmap):
                self._tlist = tlist
                self._tmap = tmap
            def __getitem__(self, tid):
                return self._tlist[self._tmap[tid]]
            def get(self, tid, default=None):
                idx = self._tmap.get(tid)
                return self._tlist[idx] if idx is not None else default
            def __contains__(self, tid):
                return tid in self._tmap
            def __len__(self):
                return len(self._tlist)
        return TargetStoreProxy(self.target_list, self.tid_to_idx)

    def index_entity(self, rep: Any):
        """Indexes a target entity across all candidate paths."""
        int_id = len(self.target_list)
        self.target_list.append(rep)
        self.tid_to_idx[rep.entity_id] = int_id
        c = rep.country_normalized

        # -------------------------------------------------------------
        # Path A: Normalized Company Name (Tokens & Prefixes)
        # -------------------------------------------------------------
        clean_name = getattr(rep, "name_without_legal_suffix", "") or getattr(rep, "name_normalized", "")
        clean_tokens = clean_name.split()
        name_words = [w for w in clean_tokens if w not in COMMON_STOPWORDS and len(w) >= 3]

        for w in name_words[:3]:
            k = (c, w)
            if len(self.path_a_name_idx[k]) < self.max_bucket_size:
                self.path_a_name_idx[k].append(int_id)

        if clean_tokens:
            first_w = clean_tokens[0]
            if len(first_w) >= 4:
                k4 = (c, first_w[:4])
                if len(self.path_a_name_idx[k4]) < self.max_bucket_size:
                    self.path_a_name_idx[k4].append(int_id)

        # -------------------------------------------------------------
        # Path B: Transliterated Company Name
        # -------------------------------------------------------------
        if getattr(rep, "is_indic", False) or any(ord(ch) > 127 for ch in clean_name):
            trans_str = getattr(rep, "name_transliterated", "")
            trans_toks = trans_str.split()
            trans_words = [w for w in trans_toks if w not in COMMON_STOPWORDS and len(w) >= 3]
            for tw in trans_words[:2]:
                tb = (c, tw)
                if len(self.path_b_trans_idx[tb]) < self.max_bucket_size:
                    self.path_b_trans_idx[tb].append(int_id)

        # -------------------------------------------------------------
        # Path C: Address Tokens, Numbers, Postal, Locality, Buildings
        # -------------------------------------------------------------
        addr_toks = getattr(rep, "address_tokens", [])
        for tok in addr_toks:
            if tok not in COMMON_STOPWORDS and len(tok) >= 3:
                kc = (c, tok)
                if len(self.path_c_addr_idx[kc]) < self.max_bucket_size:
                    self.path_c_addr_idx[kc].append(int_id)

        addr_nums = getattr(rep, "address_numbers", set())
        for num in addr_nums:
            if len(num) >= 2:
                k_num = (c, num)
                if len(self.path_c_num_idx[k_num]) < self.max_bucket_size:
                    self.path_c_num_idx[k_num].append(int_id)

        postal = getattr(rep, "address_postal", "")
        if postal:
            kd = (c, postal)
            if len(self.path_c_postal_idx[kd]) < self.max_bucket_size:
                self.path_c_postal_idx[kd].append(int_id)

        locality = getattr(rep, "address_locality", "")
        if locality:
            ke = (c, locality)
            if len(self.path_c_locality_idx[ke]) < self.max_bucket_size:
                self.path_c_locality_idx[ke].append(int_id)

        bldgs = getattr(rep, "address_building", [])
        for bldg in bldgs[:1]:
            kf = (c, bldg)
            if len(self.path_c_bldg_idx[kf]) < self.max_bucket_size:
                self.path_c_bldg_idx[kf].append(int_id)

        # -------------------------------------------------------------
        # Path D: Combined Address Blocks
        # -------------------------------------------------------------
        if postal and addr_toks:
            for tok in addr_toks[:2]:
                k_post_tok = (c, postal, tok)
                if len(self.path_d_postal_tok_idx[k_post_tok]) < self.max_bucket_size:
                    self.path_d_postal_tok_idx[k_post_tok].append(int_id)

        if locality and addr_toks:
            for tok in addr_toks[:1]:
                k_loc_tok = (c, locality, tok)
                if len(self.path_d_loc_tok_idx[k_loc_tok]) < self.max_bucket_size:
                    self.path_d_loc_tok_idx[k_loc_tok].append(int_id)

        if bldgs and locality:
            k_bldg_loc = (c, bldgs[0], locality)
            if len(self.path_d_bldg_loc_idx[k_bldg_loc]) < self.max_bucket_size:
                self.path_d_bldg_loc_idx[k_bldg_loc].append(int_id)

    def retrieve_candidates(
        self,
        query: Any,
        max_cands: int = 80
    ) -> List[Tuple[Any, int]]:
        """
        LEVEL 2: High-Recall Multi-Path Retrieval with High-Speed Bitmask Tracking.
        Returns list of (target_entity_rep, path_bitmask) sorted by match relevance.
        """
        c = getattr(query, "country_normalized", "")
        cand_scores: Dict[int, int] = defaultdict(int)
        cand_flags: Dict[int, int] = defaultdict(int)

        # -------------------------------------------------------------
        # Candidate Path A — Normalized Company Name
        # -------------------------------------------------------------
        clean_name = getattr(query, "name_without_legal_suffix", "") or getattr(query, "name_whitespace_norm", "")
        clean_tokens = clean_name.split()
        name_words = [w for w in clean_tokens if w not in COMMON_STOPWORDS and len(w) >= 3]

        for w in name_words[:3]:
            for int_id in self.path_a_name_idx.get((c, w), []):
                cand_scores[int_id] += 6
                cand_flags[int_id] |= PATH_FLAG_NAME

        if clean_tokens:
            first_w = clean_tokens[0]
            if len(first_w) >= 4:
                for int_id in self.path_a_name_idx.get((c, first_w[:4]), []):
                    cand_scores[int_id] += 4
                    cand_flags[int_id] |= PATH_FLAG_NAME

        # -------------------------------------------------------------
        # Candidate Path B — Transliterated Company Name
        # -------------------------------------------------------------
        trans_str = getattr(query, "name_transliterated_norm", "") or getattr(query, "name_transliterated", "")
        trans_toks = trans_str.split()
        trans_words = [w for w in trans_toks if w not in COMMON_STOPWORDS and len(w) >= 3]

        for tw in trans_words[:2]:
            for int_id in self.path_b_trans_idx.get((c, tw), []):
                cand_scores[int_id] += 5
                cand_flags[int_id] |= PATH_FLAG_TRANSLIT

        # -------------------------------------------------------------
        # Candidate Path C — Address (Tokens, Street Numbers, Postal, Locality)
        # -------------------------------------------------------------
        addr_toks = getattr(query, "address_tokens", [])
        for tok in addr_toks[:3]:
            if tok not in COMMON_STOPWORDS and len(tok) >= 3:
                for int_id in self.path_c_addr_idx.get((c, tok), []):
                    cand_scores[int_id] += 2
                    cand_flags[int_id] |= PATH_FLAG_ADDR

        addr_nums = getattr(query, "address_numbers", set())
        for num in addr_nums:
            if len(num) >= 2:
                for int_id in self.path_c_num_idx.get((c, num), []):
                    cand_scores[int_id] += 4
                    cand_flags[int_id] |= PATH_FLAG_ADDR

        postal = getattr(query, "address_postal", "")
        if postal:
            for int_id in self.path_c_postal_idx.get((c, postal), []):
                cand_scores[int_id] += 3
                cand_flags[int_id] |= PATH_FLAG_POSTAL

        locality = getattr(query, "address_locality", "")
        if locality:
            for int_id in self.path_c_locality_idx.get((c, locality), []):
                cand_scores[int_id] += 2
                cand_flags[int_id] |= PATH_FLAG_LOCALITY

        bldgs = getattr(query, "address_building", [])
        for bldg in bldgs[:1]:
            for int_id in self.path_c_bldg_idx.get((c, bldg), []):
                cand_scores[int_id] += 3
                cand_flags[int_id] |= PATH_FLAG_ADDR

        # -------------------------------------------------------------
        # Candidate Path D — Combined Address Blocks
        # -------------------------------------------------------------
        if postal and addr_toks:
            for tok in addr_toks[:2]:
                for int_id in self.path_d_postal_tok_idx.get((c, postal, tok), []):
                    cand_scores[int_id] += 5
                    cand_flags[int_id] |= PATH_FLAG_COMBINED

        if locality and addr_toks:
            for tok in addr_toks[:1]:
                for int_id in self.path_d_loc_tok_idx.get((c, locality, tok), []):
                    cand_scores[int_id] += 4
                    cand_flags[int_id] |= PATH_FLAG_COMBINED

        if bldgs and locality:
            for int_id in self.path_d_bldg_loc_idx.get((c, bldgs[0], locality), []):
                cand_scores[int_id] += 4
                cand_flags[int_id] |= PATH_FLAG_COMBINED

        if not cand_scores:
            return []

        # Sort candidate pool by multi-signal match score descending
        top_cands = sorted(cand_scores.items(), key=lambda x: -x[1])[:max_cands]
        return [
            (self.target_list[int_id], cand_flags[int_id])
            for int_id, _ in top_cands
        ]

    def retrieve_candidates_with_paths(
        self,
        query: Any,
        max_cands: int = 150
    ) -> List[Tuple[Any, Set[str], int]]:
        """Extended retrieval for dry-run reports that maps bitmask to path names."""
        cands = self.retrieve_candidates(query, max_cands=max_cands)
        results = []
        for t_rep, flags in cands:
            path_set = set()
            if flags & PATH_FLAG_NAME:
                path_set.add("Path A (Name)")
            if flags & PATH_FLAG_TRANSLIT:
                path_set.add("Path B (Transliteration)")
            if flags & PATH_FLAG_ADDR:
                path_set.add("Path C (Address)")
            if flags & PATH_FLAG_POSTAL:
                path_set.add("Path C (Postal)")
            if flags & PATH_FLAG_LOCALITY:
                path_set.add("Path C (Locality)")
            if flags & PATH_FLAG_COMBINED:
                path_set.add("Path D (Combined Address Blocks)")
            results.append((t_rep, path_set, flags))
        return results


class PairwiseFeatureExtractor:
    """
    LEVEL 6: Full Pairwise Feature Engineering.
    Extracts 20 pairwise features capturing name, transliteration, address,
    structure, script, and interaction signals for LightGBM.
    """

    FEATURE_NAMES = [
        "name_lev",
        "name_clean_lev",
        "name_token_sort",
        "name_token_set",
        "name_trans_lev",
        "name_len_diff",
        "addr_lev",
        "addr_jaccard",
        "addr_token_overlap",
        "addr_num_agreement",
        "addr_postal_agreement",
        "addr_unit_agreement",
        "addr_building_agreement",
        "addr_len_diff",
        "country_match",
        "is_cross_script",
        "path_flags",
        "name_addr_interaction",
        "max_evidence",
        "struct_agreement"
    ]

    @staticmethod
    def extract_features(
        e1: EntityRepresentation,
        e2: Any,
        path_flags: int = 1
    ) -> List[float]:
        # Name features
        n1 = getattr(e1, "name_whitespace_norm", "")
        n2 = getattr(e2, "name_normalized", "")
        n_lev = fuzz.ratio(n1, n2) / 100.0

        n1_c = getattr(e1, "name_without_legal_suffix", "")
        n2_c = getattr(e2, "name_without_legal_suffix", "")
        n_clean_lev = fuzz.ratio(n1_c, n2_c) / 100.0
        n_token_sort = fuzz.token_sort_ratio(n1_c, n2_c) / 100.0
        n_token_set = fuzz.token_set_ratio(n1_c, n2_c) / 100.0

        t1 = getattr(e1, "name_transliterated_norm", "") or getattr(e1, "name_transliterated", "")
        t2 = getattr(e2, "name_transliterated", "")
        n_trans_lev = fuzz.ratio(t1, t2) / 100.0

        max_n_len = max(len(n1), len(n2), 1)
        name_len_diff = abs(len(n1) - len(n2)) / float(max_n_len)

        # Address features
        a1 = getattr(e1, "address_normalized", "")
        a2 = getattr(e2, "address_normalized", "")
        a_lev = fuzz.ratio(a1, a2) / 100.0 if (a1 and a2) else 0.0

        s1_toks = set(getattr(e1, "address_tokens", []))
        s2_toks = set(getattr(e2, "address_tokens", []))
        union_len = len(s1_toks.union(s2_toks))
        inter_len = len(s1_toks.intersection(s2_toks))
        addr_jaccard = (inter_len / float(union_len)) if union_len > 0 else 0.0
        addr_token_overlap = float(min(inter_len, 10))

        # Structural address agreements
        nums1 = getattr(e1, "address_numbers", set())
        nums2 = getattr(e2, "address_numbers", set())
        if nums1 and nums2:
            num_agree = 1.00 if nums1 == nums2 else (0.80 if nums1.intersection(nums2) else 0.00)
        elif not nums1 and not nums2:
            num_agree = 0.50
        else:
            num_agree = 0.20

        p1 = getattr(e1, "address_postal", "")
        p2 = getattr(e2, "address_postal", "")
        postal_agree = 1.00 if (p1 and p2 and p1 == p2) else (0.50 if not (p1 and p2) else 0.00)

        u1 = set(getattr(e1, "address_unit", []))
        u2 = set(getattr(e2, "address_unit", []))
        unit_agree = 1.00 if (u1 and u2 and u1.intersection(u2)) else 0.00

        b1 = set(getattr(e1, "address_building", []))
        b2 = set(getattr(e2, "address_building", []))
        bldg_agree = 1.00 if (b1 and b2 and b1.intersection(b2)) else 0.00

        max_a_len = max(len(a1), len(a2), 1)
        addr_len_diff = abs(len(a1) - len(a2)) / float(max_a_len)

        # Metadata
        c1 = getattr(e1, "country_normalized", "")
        c2 = getattr(e2, "country_normalized", "")
        country_match = 1.0 if (c1 and c2 and c1 == c2) else 0.0

        is_indic1 = getattr(e1, "is_indic", False)
        is_indic2 = getattr(e2, "is_indic", False)
        is_cross_script = 1.0 if (is_indic1 != is_indic2) else 0.0
        path_bitmask = float(path_flags)

        # Interactions
        best_name_sim = max(n_lev, n_clean_lev, n_trans_lev, n_token_sort)
        name_addr_interaction = best_name_sim * a_lev
        max_evidence = max(best_name_sim, a_lev)
        struct_agreement = (num_agree * 0.5) + (postal_agree * 0.3) + (unit_agree * 0.2)

        return [
            n_lev,
            n_clean_lev,
            n_token_sort,
            n_token_set,
            n_trans_lev,
            name_len_diff,
            a_lev,
            addr_jaccard,
            addr_token_overlap,
            num_agree,
            postal_agree,
            unit_agree,
            bldg_agree,
            addr_len_diff,
            country_match,
            is_cross_script,
            path_bitmask,
            name_addr_interaction,
            max_evidence,
            struct_agreement
        ]
