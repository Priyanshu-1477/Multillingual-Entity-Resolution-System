#!/usr/bin/env python3
"""
Test SuperBlockerV4:
Adding:
1. Postal + Name Token Key (e.g. '10001_godwin', '560001_tata')
2. Postal + Street Number Key (e.g. '80127_4491', '110001_24')
"""
from typing import List, Dict, Set, Tuple, Optional
import os, sys, re, time
from collections import defaultdict, Counter
from array import array

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_DIR = os.path.join(REPO_ROOT, "data")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.data_driven_blocker import clean_norm_str, extract_core_brand, soundex_token, india_phonetic_key
def consonant_skeleton(s: str) -> str:
    if not isinstance(s, str): return ""
    return "".join(c for c in s.lower() if c in 'bcdfghjklmnpqrstvwxyz')
from src.entity_representations import (
    build_entity_representation,
    CompactTarget,
    clean_address,
    extract_postal_code,
    normalize_country,
    remove_legal_suffix
)
from src.multi_path_funnel import COMMON_STOPWORDS
from scripts.optimize_country_thresholds import UnifiedTargetStore, compute_token_idf

ADDRESS_STOPWORDS = {
    'road', 'street', 'avenue', 'lane', 'drive', 'boulevard', 'highway', 'way', 'circle',
    'court', 'place', 'square', 'terrace', 'parkway', 'dr', 'rd', 'st', 'ave', 'blvd',
    'hwy', 'ct', 'pl', 'sq', 'ln', 'cir', 'pkwy', 'near', 'opp', 'opposite', 'behind',
    'floor', 'fl', 'bldg', 'building', 'suite', 'ste', 'apt', 'apartment', 'unit', 'no',
    'number', 'plot', 'shop', 'sector', 'sec', 'block', 'phase', 'cross', 'main', 'nagar',
    'colony', 'marg', 'city', 'state', 'india', 'usa', 'united', 'states', 'pmb'
}


class SuperBlockerV4:
    def __init__(self, target_store, idf_dict, default_idf, max_cands=100):
        self.target_store = target_store
        self.idf_dict = idf_dict
        self.default_idf = default_idf
        self.max_cands = max_cands

        # Country-partitioned inverted indexes: 'india' and 'us'
        self.exact_name_idx = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}
        self.sorted_name_idx = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}
        self.core_brand_idx = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}
        self.brand_prefix_idx = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}
        self.token_pair_idx = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}
        self.rare_token_idx = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}
        self.compound_addr_idx = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}
        self.rare_addr_idx = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}
        self.postal_name_idx = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}
        self.postal_num_idx = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}
        self.postal_idx = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}
        self.india_phonetic_idx = defaultdict(lambda: array("i"))
        self.name_skel_idx = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}
        self.highest_idf_name = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}
        self.highest_idf_addr = {'india': defaultdict(lambda: array("i")), 'us': defaultdict(lambda: array("i"))}

    def get_idf(self, w: str) -> float:
        return self.idf_dict.get(w, self.default_idf)

    def extract_name_tokens(self, raw_name: str):
        clean = clean_norm_str(remove_legal_suffix(raw_name))
        return [w for w in clean.split() if w not in COMMON_STOPWORDS and len(w) >= 2]

    def extract_addr_features(self, raw_addr: str, country: str):
        norm_a = clean_address(raw_addr)
        nums = [w for w in norm_a.split() if w.isdigit() and len(w) >= 1]
        addr_words = [w for w in norm_a.split() if len(w) >= 3 and w not in ADDRESS_STOPWORDS and not w.isdigit()]
        postal = extract_postal_code(norm_a, country)
        return norm_a, nums, addr_words, postal

    def index_target(self, int_id: int, raw_name: str, raw_addr: str, raw_country: str):
        norm_n = clean_norm_str(raw_name)
        country = normalize_country(raw_country)
        if country not in ('india', 'us'):
            country = 'us'

        tokens = self.extract_name_tokens(raw_name)

        # 1. Exact Name
        if norm_n:
            arr = self.exact_name_idx[country][norm_n]
            if len(arr) < 4000:
                arr.append(int_id)

        # 2. Sorted Name Key
        if len(tokens) >= 2:
            s_key = " ".join(sorted(tokens[:4]))
            arr = self.sorted_name_idx[country][s_key]
            if len(arr) < 2000:
                arr.append(int_id)

        # 3. Core Brand & Prefix
        cb = extract_core_brand(raw_name)
        if cb and cb != norm_n:
            arr = self.core_brand_idx[country][cb]
            if len(arr) < 2000:
                arr.append(int_id)
            if len(cb) >= 4:
                prefix = cb[:4]
                arr_p = self.brand_prefix_idx[country][prefix]
                if len(arr_p) < 1000:
                    arr_p.append(int_id)

        # 4. Token Pairs
        if len(tokens) >= 2:
            for i in range(min(len(tokens), 3)):
                for j in range(i + 1, min(len(tokens), 4)):
                    pair = f"{tokens[i]}_{tokens[j]}"
                    arr = self.token_pair_idx[country][pair]
                    if len(arr) < 800:
                        arr.append(int_id)

        # 5. Rare Unigrams
        for w in tokens[:5]:
            if self.get_idf(w) >= 3.2:
                arr = self.rare_token_idx[country][w]
                if len(arr) < 1500:
                    arr.append(int_id)

        # 6. India Phonetics
        if country == 'india' and tokens:
            pk = india_phonetic_key(tokens[0])
            if pk:
                arr = self.india_phonetic_idx[pk]
                if len(arr) < 1500:
                    arr.append(int_id)
        
        # 6b. Consonant Skeleton & Highest IDF Name
        if tokens:
            skel = consonant_skeleton(tokens[0])
            if skel and len(skel) >= 3:
                arr = self.name_skel_idx[country][skel]
                if len(arr) < 1500:
                    arr.append(int_id)
            
            best_n = max(tokens[:4], key=self.get_idf)
            if self.get_idf(best_n) >= 2.0:
                arr = self.highest_idf_name[country][best_n]
                if len(arr) < 2000:
                    arr.append(int_id)

        # 7. Address: Compound Keys & Postal Combos
        if raw_addr:
            norm_a, nums, addr_words, postal = self.extract_addr_features(raw_addr, country)
            if postal:
                arr = self.postal_idx[country][postal]
                if len(arr) < 2500:
                    arr.append(int_id)

                # Postal + Name Token
                for tok in tokens[:2]:
                    pn_key = f"{postal}_{tok}"
                    arr_pn = self.postal_name_idx[country][pn_key]
                    if len(arr_pn) < 500:
                        arr_pn.append(int_id)

                # Postal + Street Number
                if nums:
                    pnum_key = f"{postal}_{nums[0]}"
                    arr_pnum = self.postal_num_idx[country][pnum_key]
                    if len(arr_pnum) < 500:
                        arr_pnum.append(int_id)

            if nums and addr_words:
                primary_num = nums[0]
                for aw in addr_words[:2]:
                    ckey = f"{primary_num}_{aw}"
                    arr = self.compound_addr_idx[country][ckey]
                    if len(arr) < 500:
                        arr.append(int_id)

            for aw in addr_words[:3]:
                if self.get_idf(aw) >= 5.0:
                    arr_ra = self.rare_addr_idx[country][aw]
                    if len(arr_ra) < 400:
                        arr_ra.append(int_id)
            
            if addr_words:
                best_a = max(addr_words[:4], key=self.get_idf)
                if self.get_idf(best_a) >= 2.0:
                    arr_ha = self.highest_idf_addr[country][best_a]
                    if len(arr_ha) < 1500:
                        arr_ha.append(int_id)

    def retrieve(self, q_name: str, q_addr: str, q_country: str) -> List[int]:
        cand_scores = defaultdict(float)
        norm_n = clean_norm_str(q_name)
        country = normalize_country(q_country)
        if country not in ('india', 'us'):
            country = 'us'

        tokens = self.extract_name_tokens(q_name)

        # 1. Exact Name: TOP PRIORITY (100 pts)
        if norm_n in self.exact_name_idx[country]:
            for cid in self.exact_name_idx[country][norm_n]:
                cand_scores[cid] += 100.0

        # 2. Sorted Name Key (60 pts)
        if len(tokens) >= 2:
            s_key = " ".join(sorted(tokens[:4]))
            if s_key in self.sorted_name_idx[country]:
                for cid in self.sorted_name_idx[country][s_key]:
                    cand_scores[cid] += 60.0

        # 3. Core Brand & Prefix (40 pts & 8 pts)
        cb = extract_core_brand(q_name)
        if cb and cb in self.core_brand_idx[country]:
            for cid in self.core_brand_idx[country][cb]:
                cand_scores[cid] += 40.0
            if len(cb) >= 4:
                prefix = cb[:4]
                if prefix in self.brand_prefix_idx[country]:
                    for cid in self.brand_prefix_idx[country][prefix]:
                        cand_scores[cid] += 8.0

        # 4. Token Pairs
        if len(tokens) >= 2:
            for i in range(min(len(tokens), 3)):
                for j in range(i + 1, min(len(tokens), 4)):
                    pair = f"{tokens[i]}_{tokens[j]}"
                    if pair in self.token_pair_idx[country]:
                        pair_idf = self.get_idf(tokens[i]) + self.get_idf(tokens[j])
                        score_boost = max(pair_idf * 2.0, 15.0)
                        for cid in self.token_pair_idx[country][pair]:
                            cand_scores[cid] += score_boost

        # 5. Rare Unigrams
        for w in tokens[:5]:
            idf = self.get_idf(w)
            if idf >= 3.2 and w in self.rare_token_idx[country]:
                for cid in self.rare_token_idx[country][w]:
                    cand_scores[cid] += idf

        # 6. India Phonetics
        if country == 'india' and tokens:
            pk = india_phonetic_key(tokens[0])
            if pk and pk in self.india_phonetic_idx:
                for cid in self.india_phonetic_idx[pk]:
                    cand_scores[cid] += 8.0

        if tokens:
            skel = consonant_skeleton(tokens[0])
            if skel and len(skel) >= 3 and skel in self.name_skel_idx[country]:
                for cid in self.name_skel_idx[country][skel]:
                    cand_scores[cid] += 8.0
                    
            best_n = max(tokens[:4], key=self.get_idf)
            if self.get_idf(best_n) >= 2.0 and best_n in self.highest_idf_name[country]:
                for cid in self.highest_idf_name[country][best_n]:
                    cand_scores[cid] += 10.0

        # 7. Address & Postal Combos
        if q_addr:
            norm_a, nums, addr_words, postal = self.extract_addr_features(q_addr, country)
            if postal:
                for tok in tokens[:2]:
                    pn_key = f"{postal}_{tok}"
                    if pn_key in self.postal_name_idx[country]:
                        for cid in self.postal_name_idx[country][pn_key]:
                            cand_scores[cid] += 35.0

                if nums:
                    pnum_key = f"{postal}_{nums[0]}"
                    if pnum_key in self.postal_num_idx[country]:
                        for cid in self.postal_num_idx[country][pnum_key]:
                            cand_scores[cid] += 25.0

                if postal in self.postal_idx[country]:
                    for cid in self.postal_idx[country][postal]:
                        cand_scores[cid] += 3.0

            if nums and addr_words:
                primary_num = nums[0]
                for aw in addr_words[:2]:
                    ckey = f"{primary_num}_{aw}"
                    if ckey in self.compound_addr_idx[country]:
                        for cid in self.compound_addr_idx[country][ckey]:
                            cand_scores[cid] += 30.0

            for aw in addr_words[:3]:
                if self.get_idf(aw) >= 5.0 and aw in self.rare_addr_idx[country]:
                    for cid in self.rare_addr_idx[country][aw]:
                        cand_scores[cid] += 12.0
                        
            if addr_words:
                best_a = max(addr_words[:4], key=self.get_idf)
                if self.get_idf(best_a) >= 2.0 and best_a in self.highest_idf_addr[country]:
                    for cid in self.highest_idf_addr[country][best_a]:
                        cand_scores[cid] += 6.0

        if not cand_scores:
            return []

        sorted_c = sorted(cand_scores.items(), key=lambda x: -x[1])
        return [cid for cid, _ in sorted_c[:self.max_cands]]


def main():
    print("Loading Validation Dataset...")
    val_s1_path = os.path.join(WORKSPACE_DIR, 'experiments/val_s1_entities.tsv')
    val_gt_path = os.path.join(WORKSPACE_DIR, 'experiments/val_ground_truth.tsv')

    val_queries = []
    with open(val_s1_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) >= 4:
                val_queries.append((parts[0].strip(), parts[1].strip(), parts[2].strip(), parts[3].strip()))

    val_gt = {}
    with open(val_gt_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if parts:
                eid = parts[0].strip()
                matches = parts[1].strip().split(',') if len(parts) > 1 and parts[1].strip() else []
                val_gt[eid] = matches

    idf_dict, default_idf = compute_token_idf()
    target_files = [
        os.path.join(WORKSPACE_DIR, 'train_source2.tsv'),
        os.path.join(WORKSPACE_DIR, 'train_source3.tsv')
    ]
    store = UnifiedTargetStore(target_files)
    sblocker = SuperBlockerV4(store, idf_dict, default_idf, max_cands=100)

    print("Indexing targets with SuperBlockerV4...")
    t0 = time.time()
    for f_idx, tf in enumerate(target_files):
        with open(tf, 'rb') as f:
            f.readline()
            byte_pos = f.tell()
            line = f.readline()
            while line:
                parts = line.decode('utf-8', errors='replace').rstrip('\r\n').split('\t')
                if len(parts) >= 4:
                    int_id = store.add_target(parts[0].strip(), f_idx, byte_pos)
                    sblocker.index_target(int_id, parts[1].strip(), parts[2].strip(), parts[3].strip())
                byte_pos = f.tell()
                line = f.readline()
    print(f"Indexed {len(store.int_to_id):,} targets in {time.time()-t0:.1f}s.")

    print("Evaluating Candidate Recall on 5,000 validation queries...")
    t_val = time.time()
    total_gt = 0
    recalled_gt = 0

    for eid, q_name, q_addr, q_country in val_queries[:5000]:
        gt_targets = val_gt.get(eid, [])
        if not gt_targets:
            continue
        cands = sblocker.retrieve(q_name, q_addr, q_country)
        cand_ids = set(store.int_to_id[cid] for cid in cands)

        for gt_id in gt_targets:
            total_gt += 1
            if gt_id in cand_ids:
                recalled_gt += 1

    rec_rate = recalled_gt / total_gt if total_gt > 0 else 0
    print(f"EVALUATION COMPLETE in {time.time()-t_val:.1f}s:")
    print(f"  Total Ground Truth Pairs in Sample: {total_gt:,}")
    print(f"  Recalled Pairs:                     {recalled_gt:,}")
    print(f"  SuperBlockerV4 Recall Rate:         {rec_rate*100:.2f}% (Baseline was 62.46%)")
    print(f"  Absolute Recall Improvement:        {(rec_rate - 0.6246)*100:+.2f}%")

if __name__ == "__main__":
    main()
