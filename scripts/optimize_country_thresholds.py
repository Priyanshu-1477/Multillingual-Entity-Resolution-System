#!/usr/bin/env python3
"""
STEP 1: COUNTRY-SPECIFIC THRESHOLD OPTIMIZATION (QUICK WIN)
Amazon ML Challenge 2026 - Business Entity Resolution

Evaluates 2D threshold grid sweep (tau_india vs tau_us) on the existing EXP_003 model
to immediately eliminate the precision/recall penalty caused by a single global threshold.
"""

import os
import sys
import json
import time
import math
from typing import Dict, List, Tuple, Set, Any
from array import array
from collections import defaultdict, Counter
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.entity_representations import (
    EntityRepresentation,
    CompactTarget,
    build_entity_representation,
    build_compact_target,
    clean_address,
    extract_postal_code,
    normalize_country
)
from src.multi_path_funnel import COMMON_STOPWORDS
from src.address_aware_features import AddressAwareFeatureExtractor
from src.evaluation import evaluate_predictions
from src.data_driven_blocker import soundex_token, extract_core_brand, clean_norm_str

DATASET_DIR = os.path.join(REPO_ROOT, "data")
EXPERIMENTS_DIR = os.path.join(REPO_ROOT, "experiments")
EXP_003_DIR = os.path.join(EXPERIMENTS_DIR, "EXP_003")


def log(msg: str):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{t_str}] {msg}", flush=True)


class UnifiedTargetStore:
    def __init__(self, target_files: List[str]):
        self.target_files = target_files
        self.file_handles = [open(f, "rb") for f in target_files]
        self.offsets = array("q")
        self.file_ids = array("B")
        self.id_to_int: Dict[str, int] = {}
        self.int_to_id: List[str] = []
        self._cache: Dict[int, CompactTarget] = {}

    def add_target(self, tid: str, f_idx: int, byte_pos: int) -> int:
        int_id = len(self.int_to_id)
        self.int_to_id.append(tid)
        self.id_to_int[tid] = int_id
        self.file_ids.append(f_idx)
        self.offsets.append(byte_pos)
        return int_id

    def get_target(self, int_id: int) -> CompactTarget:
        if int_id in self._cache:
            return self._cache[int_id]
        f_idx = self.file_ids[int_id]
        byte_pos = self.offsets[int_id]
        fh = self.file_handles[f_idx]
        fh.seek(byte_pos)
        line = fh.readline().decode("utf-8", errors="replace")
        parts = line.rstrip("\r\n").split("\t")
        ct = build_compact_target(parts[0].strip(), parts[1].strip(), parts[2].strip() if len(parts) > 2 else "", parts[3].strip() if len(parts) > 3 else "")
        self._cache[int_id] = ct
        return ct

    def close(self):
        for fh in self.file_handles:
            try:
                fh.close()
            except Exception:
                pass


def compute_token_idf() -> Tuple[Dict[str, float], float]:
    log("Computing token IDF dictionary from train_source1.tsv...")
    df = Counter()
    N = 0
    with open(os.path.join(DATASET_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            N += 1
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 2:
                norm_n = clean_norm_str(parts[1])
                tokens = set(w for w in norm_n.split() if len(w) >= 2)
                for t in tokens:
                    df[t] += 1
    idf_dict = {}
    default_idf = math.log((N + 1) / 1.0)
    for tok, count in df.items():
        idf_dict[tok] = math.log((N + 1) / (count + 1))
    return idf_dict, default_idf


class EnhancedFeatureExtractorV3:
    @staticmethod
    def extract_features(
        q_rep: EntityRepresentation,
        t_rep: CompactTarget,
        idf_dict: Dict[str, float],
        default_idf: float
    ) -> List[float]:
        base_32 = AddressAwareFeatureExtractor.extract_features(q_rep, t_rep)
        n1 = getattr(q_rep, "name_without_legal_suffix", "")
        n2 = getattr(t_rep, "name_without_legal_suffix", "")
        toks1 = [w for w in n1.split() if w not in COMMON_STOPWORDS and len(w) >= 2]
        toks2 = [w for w in n2.split() if w not in COMMON_STOPWORDS and len(w) >= 2]
        sx1 = soundex_token(toks1[0]) if toks1 else ""
        sx2 = soundex_token(toks2[0]) if toks2 else ""
        soundex_match = 1.0 if (sx1 and sx2 and sx1 == sx2) else 0.0
        set1 = set(toks1)
        set2 = set(toks2)
        inter = set1.intersection(set2)
        rare_overlap = sum(1.0 for w in inter if idf_dict.get(w, default_idf) >= 3.5)
        w_inter = sum(idf_dict.get(w, default_idf) for w in inter)
        w_union = sum(idf_dict.get(w, default_idf) for w in set1.union(set2))
        w_jaccard = (w_inter / float(w_union)) if w_union > 0 else 0.0
        max_idf = max([idf_dict.get(w, default_idf) for w in inter], default=0.0)
        return base_32 + [soundex_match, float(rare_overlap), float(w_jaccard), float(max_idf)]


def run_step_1():
    log("=" * 80)
    log("STEP 1: OPTIMIZING COUNTRY-SPECIFIC THRESHOLDS (tau_India vs tau_US) ON EXP_003 MODEL")
    log("=" * 80)

    # 1. Load validation benchmark
    val_meta_path = os.path.join(EXPERIMENTS_DIR, "val_metadata.json")
    with open(val_meta_path, "r", encoding="utf-8") as f:
        val_meta = json.load(f)

    val_s1_path = os.path.join(EXPERIMENTS_DIR, "val_s1_entities.tsv")
    val_gt_path = os.path.join(EXPERIMENTS_DIR, "val_ground_truth.tsv")

    val_queries = []
    val_country_map = {}
    with open(val_s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4:
                eid = parts[0].strip()
                val_queries.append((eid, parts[1].strip(), parts[2].strip(), parts[3].strip()))
                val_country_map[eid] = normalize_country(parts[3].strip())

    val_gt = defaultdict(list)
    with open(val_gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if parts:
                eid = parts[0].strip()
                matches = parts[1].strip().split(",") if len(parts) > 1 and parts[1].strip() else []
                val_gt[eid] = matches

    log(f"Loaded validation set: {len(val_queries):,} entities.")

    # 2. Token IDF & Target Store
    idf_dict, default_idf = compute_token_idf()
    target_files = [
        os.path.join(DATASET_DIR, "train_source2.tsv"),
        os.path.join(DATASET_DIR, "train_source3.tsv")
    ]
    target_store = UnifiedTargetStore(target_files)

    # Fast indexer
    from src.data_driven_blocker import DataDrivenBlocker
    blocker = DataDrivenBlocker(target_store, idf_dict, default_idf, max_df_cutoff=1500, max_cands=65)

    log("Indexing targets...")
    total_indexed = 0
    t0 = time.time()
    for f_idx, tf in enumerate(target_files):
        with open(tf, "rb") as f:
            f.readline()
            byte_pos = f.tell()
            line = f.readline()
            while line:
                parts = line.decode("utf-8", errors="replace").rstrip("\r\n").split("\t")
                if len(parts) >= 4:
                    int_id = target_store.add_target(parts[0].strip(), f_idx, byte_pos)
                    blocker.index_target_line(int_id, parts[1].strip(), parts[2].strip(), parts[3].strip())
                    total_indexed += 1
                byte_pos = f.tell()
                line = f.readline()
    log(f"Indexed {total_indexed:,} targets in {time.time()-t0:.1f}s.")

    # Load EXP_003 model
    booster_path = os.path.join(EXP_003_DIR, "model.txt")
    booster = lgb.Booster(model_file=booster_path)
    log(f"Loaded EXP_003 LightGBM model from {booster_path}")

    # Score validation set
    log("Scoring validation queries...")
    t_val0 = time.time()
    val_cand_predictions = {}
    val_candidate_pairs = {}

    for i, (eid, q_name, q_addr, q_country) in enumerate(val_queries):
        q_rep = build_entity_representation(eid, q_name, q_addr, q_country)
        cands = blocker.retrieve_candidates(q_rep)

        if not cands:
            val_cand_predictions[eid] = []
            val_candidate_pairs[eid] = []
            continue

        c_objs = [target_store.get_target(int_id) for int_id in cands]
        val_candidate_pairs[eid] = [ct.entity_id for ct in c_objs]

        feats_list = [EnhancedFeatureExtractorV3.extract_features(q_rep, ct, idf_dict, default_idf) for ct in c_objs]
        X_val_mat = np.array(feats_list, dtype=np.float32)
        raw_probs = booster.predict(X_val_mat)

        adjusted_scores = []
        for j, prob in enumerate(raw_probs):
            f = feats_list[j]
            exact_both = f[24]
            exact_addr_c = f[25]
            addr_conflict = f[19]
            diff_name_exact = f[27]
            numeric_conflict = f[22]
            rare_tokens = f[33]

            score = float(prob)
            if exact_both == 1.0:
                score = max(score, 0.96)
            elif exact_addr_c == 1.0:
                score = max(score, 0.85)
            elif diff_name_exact == 1.0:
                score = max(score, 0.72)
            elif addr_conflict == 1.0:
                score = score * 0.35
            elif numeric_conflict == 1.0:
                score = score * 0.45

            if rare_tokens >= 2.0:
                score = min(score * 1.10, 0.99)

            adjusted_scores.append((score, c_objs[j].entity_id))

        adjusted_scores.sort(key=lambda x: -x[0])
        val_cand_predictions[eid] = adjusted_scores

    log(f"Scored {len(val_queries):,} validation entities in {time.time()-t_val0:.1f}s.")

    # 3. 2D Grid Sweep over tau_India and tau_US
    log("\nSweeping 2D Country Thresholds (tau_India x tau_US)...")
    tau_vals = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    best_f05 = -1.0
    best_tau_in = 0.30
    best_tau_us = 0.30
    best_m = None

    for t_in in tau_vals:
        for t_us in tau_vals:
            preds = {}
            for eid, scored in val_cand_predictions.items():
                c_norm = val_country_map.get(eid, "other")
                tau = t_in if c_norm == "india" else t_us
                preds[eid] = [cid for s, cid in scored if s >= tau]

            m = evaluate_predictions(
                predictions=preds,
                ground_truth=val_gt,
                candidate_pairs=val_candidate_pairs,
                s1_country_map=val_country_map
            )
            macro_f05 = m["macro_f05"]

            if macro_f05 > best_f05:
                best_f05 = macro_f05
                best_tau_in = t_in
                best_tau_us = t_us
                best_m = m

    log("=" * 80)
    log(f"STEP 1 RESULT: OPTIMAL COUNTRY THRESHOLDS FOUND!")
    log(f"  tau_India:           {best_tau_in:.2f}")
    log(f"  tau_US:              {best_tau_us:.2f}")
    log(f"  Macro F0.5:          {best_m['macro_f05']:.4f} ({best_m['macro_f05']*100:.2f}%)")
    log(f"  Macro Precision:     {best_m['macro_precision']:.4f}")
    log(f"  Macro Recall:        {best_m['macro_recall']:.4f}")
    log(f"  Candidate Recall:    {best_m['candidate_recall']:.4f}")
    log(f"  Singleton Accuracy:  {best_m['singleton_accuracy']:.4f}")
    log("Country Breakdown:")
    for c, c_data in best_m["country_metrics"].items():
        log(f"  * {c}: F0.5 = {c_data['macro_f05']:.4f}, Prec = {c_data['macro_precision']:.4f}, Rec = {c_data['macro_recall']:.4f}")
    log(f"Previous EXP_003 Global tau=0.30 F0.5: 0.6673")
    delta_s1 = best_m['macro_f05'] - 0.6673
    log(f"DELTA from Step 1 Country Tuning: {delta_s1:+.4f} ({delta_s1*100:+.2f}%)")
    log("=" * 80)

    target_store.close()
    return best_tau_in, best_tau_us, best_m


if __name__ == "__main__":
    run_step_1()
