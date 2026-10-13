#!/usr/bin/env python3
"""
HARD-NEGATIVE MINING MODULE
Amazon ML Challenge 2026 - Business Entity Resolution

This module provides modular, non-destructive hard-negative mining around the existing training pipeline:
- Mined candidates must be known non-matches (NOT in ground truth)
- Scored by current model and filtered by score threshold (high model score + known incorrect match)
- Conservative selection (top-k per query, max-per-positive)
- Full protection against false negatives (positive pair in hard-negative set is strictly prevented)
- Deduplication against existing training pairs
- Comprehensive diagnostic statistics and distribution reporting
- Safe persistence of mined hard-negative artifacts
"""

import os
import json
import time
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class HardNegativeMinerConfig:
    """Configuration parameters for hard-negative mining."""
    enabled: bool = True
    top_k: int = 3
    max_per_positive: int = 3
    score_threshold: float = 0.20
    rounds: int = 1
    max_candidate_retrieval: int = 65


def compute_score_distribution(scores: List[float]) -> Dict[str, Any]:
    """Computes quantile distribution and summary statistics of candidate scores."""
    if not scores:
        return {
            "count": 0,
            "min": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "top_5": []
        }
    arr = np.array(scores, dtype=np.float32)
    sorted_scores = sorted(scores, reverse=True)
    return {
        "count": len(scores),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "top_5": [round(float(s), 4) for s in sorted_scores[:5]]
    }


def mine_hard_negatives(
    train_queries: List[Tuple[str, str, str, str]],
    target_store: Any,
    blocker: Any,
    feature_extractor_fn: Any,
    model: Any,
    full_gt: Dict[str, List[str]],
    initial_pair_keys: Set[Tuple[str, str]],
    idf_dict: Dict[str, float],
    default_idf: float,
    config: HardNegativeMinerConfig,
    build_entity_representation_fn: Any,
    log_fn: Any = print
) -> Tuple[List[Dict[str, Any]], List[List[float]], Dict[str, Any]]:
    """
    Identifies high-scoring incorrect candidates across training queries.
    
    CRITICAL SAFETY RULES:
    1. positive pair in hard-negative set MUST NEVER HAPPEN.
    2. Duplicates of existing training negatives are identified and removed.
    3. Operates exclusively on candidates retrieved by the existing blocker.
    """
    log_fn("=" * 80)
    log_fn("STARTING HARD-NEGATIVE MINING STAGE")
    log_fn(f"  Configuration: top_k={config.top_k}, max_per_pos={config.max_per_positive}, "
           f"threshold={config.score_threshold}, rounds={config.rounds}")
    log_fn("=" * 80)

    t0 = time.time()
    mined_records: List[Dict[str, Any]] = []
    mined_features: List[List[float]] = []
    all_mined_scores: List[float] = []

    total_candidates_examined = 0
    total_hard_negatives_found = 0
    duplicate_count = 0
    retained_count = 0
    queries_with_hard_negs = 0

    num_queries = len(train_queries)

    for i, (eid, q_name, q_addr, q_country) in enumerate(train_queries):
        if (i + 1) % 5000 == 0:
            log_fn(f"  Mined hard negatives for {i+1:,}/{num_queries:,} queries "
                   f"(found {retained_count:,} retained, {duplicate_count:,} dupes) in {time.time()-t0:.1f}s...")

        gt_matches = set(full_gt.get(eid, []))
        q_rep = build_entity_representation_fn(eid, q_name, q_addr, q_country)
        
        # Retrieve candidates using existing blocker
        try:
            cands = blocker.retrieve_candidates(q_rep, max_cands=config.max_candidate_retrieval)
        except TypeError:
            cands = blocker.retrieve_candidates(q_rep)[:config.max_candidate_retrieval]
        if not cands:
            continue

        c_objs = [target_store.get_target(int_id) for int_id in cands]
        
        # Filter: STRICTLY EXCLUDE GROUND-TRUTH POSITIVE MATCHES
        # Rule: positive pair in hard-negative set must NEVER happen
        wrong_candidates = []
        for ct in c_objs:
            if ct.entity_id in gt_matches:
                continue  # Never treat a true positive as a negative
            wrong_candidates.append(ct)

        if not wrong_candidates:
            continue

        total_candidates_examined += len(wrong_candidates)

        # Extract features for all wrong candidates
        cand_feats = [feature_extractor_fn(q_rep, ct, idf_dict, default_idf) for ct in wrong_candidates]
        X_mat = np.array(cand_feats, dtype=np.float32)
        
        # Score candidate set using current model
        raw_scores = model.predict(X_mat)

        # Identify difficult wrong candidates (score >= threshold)
        scored_negatives = []
        for j, score in enumerate(raw_scores):
            s_val = float(score)
            if s_val >= config.score_threshold:
                scored_negatives.append((s_val, wrong_candidates[j], cand_feats[j]))

        if not scored_negatives:
            continue

        # Sort descending by model score to get the hardest negatives
        scored_negatives.sort(key=lambda x: -x[0])

        # Select conservative top_k (bounded by max_per_positive)
        max_to_add = min(config.top_k, config.max_per_positive)
        selected_for_query = scored_negatives[:max_to_add]
        
        query_retained = 0
        for rank, (score, ct, feats) in enumerate(selected_for_query):
            total_hard_negatives_found += 1
            all_mined_scores.append(score)

            pair_key = (eid, ct.entity_id)
            if pair_key in initial_pair_keys:
                duplicate_count += 1
                continue  # Duplicate hard negative removed

            # Safety assertion: double-check ct.entity_id is not positive
            assert ct.entity_id not in gt_matches, f"FATAL: True positive {ct.entity_id} found in hard negatives for {eid}"

            retained_count += 1
            query_retained += 1
            mined_records.append({
                "source1_entity_id": eid,
                "target_entity_id": ct.entity_id,
                "model_score": score,
                "rank": rank + 1
            })
            mined_features.append(feats)

        if query_retained > 0:
            queries_with_hard_negs += 1

    elapsed = time.time() - t0
    score_dist = compute_score_distribution(all_mined_scores)

    mining_stats = {
        "queries_examined": num_queries,
        "queries_with_hard_negatives": queries_with_hard_negs,
        "total_wrong_candidates_examined": total_candidates_examined,
        "hard_negatives_mined": total_hard_negatives_found,
        "duplicate_negatives_removed": duplicate_count,
        "hard_negatives_retained": retained_count,
        "score_distribution": score_dist,
        "elapsed_seconds": round(elapsed, 2)
    }

    log_fn("=" * 80)
    log_fn("HARD-NEGATIVE MINING COMPLETE:")
    log_fn(f"  Hard negatives mined:         {total_hard_negatives_found:,}")
    log_fn(f"  Duplicate negatives removed:  {duplicate_count:,}")
    log_fn(f"  Hard negatives retained:      {retained_count:,}")
    log_fn(f"  Score median:                 {score_dist['median']:.4f}")
    log_fn(f"  Score min / max:              {score_dist['min']:.4f} / {score_dist['max']:.4f}")
    log_fn(f"  Top 5 scores:                 {score_dist['top_5']}")
    log_fn(f"  Elapsed mining time:          {elapsed:.1f}s")
    log_fn("=" * 80)

    return mined_records, mined_features, mining_stats


def augment_training_dataset(
    X_initial: List[List[float]],
    y_initial: List[int],
    mined_features: List[List[float]],
    log_fn: Any = print
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Safely expands the training set by appending mined hard negatives without deleting original data.
    """
    initial_pairs = len(X_initial)
    initial_pos = sum(1 for y in y_initial if y == 1)
    initial_neg = initial_pairs - initial_pos

    hard_neg_count = len(mined_features)
    y_hard_neg = [0] * hard_neg_count

    X_expanded = list(X_initial) + list(mined_features)
    y_expanded = list(y_initial) + y_hard_neg

    final_pairs = len(X_expanded)
    final_pos = initial_pos
    final_neg = initial_neg + hard_neg_count

    avg_per_pos = (hard_neg_count / initial_pos) if initial_pos > 0 else 0.0

    aug_stats = {
        "initial_training_pairs": initial_pairs,
        "initial_positives": initial_pos,
        "initial_negatives": initial_neg,
        "hard_negatives_added": hard_neg_count,
        "avg_hard_neg_per_positive": round(avg_per_pos, 4),
        "final_training_pairs": final_pairs,
        "final_positives": final_pos,
        "final_negatives": final_neg
    }

    log_fn(f"Training dataset augmented:")
    log_fn(f"  Initial pairs: {initial_pairs:,} ({initial_pos:,} pos, {initial_neg:,} neg)")
    log_fn(f"  Hard negatives added: {hard_neg_count:,} (avg {avg_per_pos:.2f} per positive)")
    log_fn(f"  Final pairs: {final_pairs:,} ({final_pos:,} pos, {final_neg:,} neg)")

    return np.array(X_expanded, dtype=np.float32), np.array(y_expanded, dtype=np.int32), aug_stats


def save_hard_negatives_artifact(
    output_dir: str,
    mined_records: List[Dict[str, Any]],
    stats: Dict[str, Any],
    log_fn: Any = print
):
    """Saves mined hard negatives to TSV and JSON metadata artifacts."""
    os.makedirs(output_dir, exist_ok=True)
    tsv_path = os.path.join(output_dir, "hard_negatives.tsv")
    json_path = os.path.join(output_dir, "hard_negative_stats.json")

    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\thard_negative_entity_id\tmodel_score\trank\n")
        for rec in mined_records:
            f.write(f"{rec['source1_entity_id']}\t{rec['target_entity_id']}\t{rec['model_score']:.6f}\t{rec['rank']}\n")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    log_fn(f"Saved {len(mined_records):,} hard-negative records to {tsv_path}")
    log_fn(f"Saved mining statistics to {json_path}")
