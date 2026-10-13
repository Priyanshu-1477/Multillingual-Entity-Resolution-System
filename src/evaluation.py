"""
Entity-level Evaluation Module for Amazon ML Challenge 2026.
Calculates macro F0.5 score, precision, recall, singleton accuracy, candidate recall, and country breakdown.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict


def calculate_entity_f05(
    pred_ids: List[str],
    gt_ids: List[str],
    f_beta: float = 0.5
) -> Tuple[float, float, float, int, int, int]:
    """
    Calculates Precision, Recall, F0.5, TP, FP, FN for a single Source 1 entity.
    Singletons:
    - GT empty & Pred empty => P=1.0, R=1.0, F0.5=1.0
    - GT empty & Pred non-empty => P=0.0, R=0.0, F0.5=0.0
    - GT non-empty & Pred empty => P=0.0, R=0.0, F0.5=0.0
    """
    pred_set = set(pred_ids)
    gt_set = set(gt_ids)

    if not gt_set and not pred_set:
        return 1.0, 1.0, 1.0, 0, 0, 0

    if not gt_set and pred_set:
        return 0.0, 0.0, 0.0, 0, len(pred_set), 0

    if gt_set and not pred_set:
        return 0.0, 0.0, 0.0, 0, 0, len(gt_set)

    tp = len(pred_set.intersection(gt_set))
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0

    beta_sq = f_beta ** 2  # 0.25 for beta=0.5
    denom = (beta_sq * precision + recall)

    f05 = ((1.0 + beta_sq) * precision * recall) / denom if denom > 0 else 0.0
    return precision, recall, f05, tp, fp, fn


def evaluate_predictions(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
    candidate_pairs: Optional[Dict[str, List[str]]] = None,
    s1_country_map: Optional[Dict[str, str]] = None,
    f_beta: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluates predictions against ground truth across all Source 1 entities.
    Computes macro F0.5, macro precision, macro recall, candidate recall, singleton metrics, and country breakdown.
    """
    all_s1_ids = list(ground_truth.keys())
    if s1_country_map is None:
        s1_country_map = {}

    p_list, r_list, f05_list = [], [], []
    total_tp, total_fp, total_fn = 0, 0, 0
    singleton_correct, total_singletons = 0, 0
    false_merges, missed_matches = 0, 0

    # Candidate recall tracking
    total_gt_matches = 0
    gt_matches_in_candidates = 0

    country_metrics = defaultdict(lambda: {"f05": [], "prec": [], "rec": []})

    for s1_id in all_s1_ids:
        gt_ids = ground_truth.get(s1_id, [])
        pred_ids = predictions.get(s1_id, [])

        p, r, f05, tp, fp, fn = calculate_entity_f05(pred_ids, gt_ids, f_beta=f_beta)

        p_list.append(p)
        r_list.append(r)
        f05_list.append(f05)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        if fp > 0:
            false_merges += fp
        if fn > 0:
            missed_matches += fn

        # Singleton evaluation
        if not gt_ids:
            total_singletons += 1
            if not pred_ids:
                singleton_correct += 1

        # Candidate recall tracking
        if candidate_pairs is not None:
            cand_set = set(candidate_pairs.get(s1_id, []))
            for gid in gt_ids:
                total_gt_matches += 1
                if gid in cand_set:
                    gt_matches_in_candidates += 1

        # Country breakdown
        country = s1_country_map.get(s1_id, "Unknown")
        country_metrics[country]["f05"].append(f05)
        country_metrics[country]["prec"].append(p)
        country_metrics[country]["rec"].append(r)

    macro_p = float(np.mean(p_list)) if p_list else 0.0
    macro_r = float(np.mean(r_list)) if r_list else 0.0
    macro_f05 = float(np.mean(f05_list)) if f05_list else 0.0
    singleton_acc = (singleton_correct / float(total_singletons)) if total_singletons > 0 else 1.0
    cand_recall = (gt_matches_in_candidates / float(total_gt_matches)) if total_gt_matches > 0 else 0.0

    c_summary = {}
    for c, data in country_metrics.items():
        c_summary[c] = {
            "macro_f05": round(float(np.mean(data["f05"])), 4),
            "macro_precision": round(float(np.mean(data["prec"])), 4),
            "macro_recall": round(float(np.mean(data["rec"])), 4),
            "count": len(data["f05"])
        }

    return {
        "macro_f05": round(macro_f05, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "candidate_recall": round(cand_recall, 4),
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "false_merges": false_merges,
        "missed_matches": missed_matches,
        "total_entities": len(all_s1_ids),
        "total_singletons": total_singletons,
        "singleton_correct": singleton_correct,
        "singleton_accuracy": round(singleton_acc, 4),
        "country_metrics": c_summary
    }
