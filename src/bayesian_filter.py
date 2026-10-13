#!/usr/bin/env python3
"""
BAYESIAN FILTER & CALIBRATION LAYER
Amazon ML Challenge 2026 - Business Entity Resolution

Implements Step 6:
- Bayesian Probability Calibration (Platt / MAP scaling on top of LightGBM margins)
- Bayesian Singleton Prior & Cluster Uncertainty Protection (prevents singleton false merges)
- Bayes-Optimal Decision Filter under Asymmetric F0.5 Loss (Precision weighted 4x over Recall)
"""

import math
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class BayesianCalibrator:
    """
    Step 6: Fits a Bayesian Logistic Calibrator with Gaussian priors on weights:
        P(Match | score) = 1 / (1 + exp(-(w * score + b)))
    """

    def __init__(self, prior_variance: float = 4.0):
        self.prior_variance = prior_variance
        self.w = 1.0
        self.b = 0.0
        self.fitted = False

    def fit(self, scores: np.ndarray, labels: np.ndarray, num_iters: int = 50, lr: float = 0.05):
        """Fit calibration parameters using MAP (Maximum A Posteriori) gradient descent."""
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)

        w = 1.5
        b = -0.5

        N = len(scores)
        if N == 0:
            self.fitted = True
            return

        for _ in range(num_iters):
            # Compute logits
            logits = np.clip(w * scores + b, -25.0, 25.0)
            probs = 1.0 / (1.0 + np.exp(-logits))

            # Gradients with Gaussian prior regularization (L2 towards w=1, b=0)
            err = probs - labels
            grad_w = (np.sum(err * scores) / N) + (w - 1.0) / (self.prior_variance * N)
            grad_b = (np.sum(err) / N) + b / (self.prior_variance * N)

            w -= lr * grad_w
            b -= lr * grad_b

        self.w = float(w)
        self.b = float(b)
        self.fitted = True

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Converts raw LightGBM scores to well-calibrated posterior probabilities."""
        if not self.fitted:
            return np.clip(scores, 0.0, 1.0)
        logits = np.clip(self.w * np.asarray(scores, dtype=np.float64) + self.b, -25.0, 25.0)
        return 1.0 / (1.0 + np.exp(-logits))


class BayesianSingletonFilter:
    """
    Step 6: Evaluates candidate sets under empirical singleton priors.
    If evidence across all retrieved candidates is weak or highly ambiguous,
    the Bayesian filter updates the posterior probability that the entity is an
    unmatched singleton (P_singleton ~ 14.6%) and prunes false positive merges.
    """

    def __init__(
        self,
        singleton_prior: float = 0.146,
        singleton_score_mean: float = 0.15,
        singleton_score_std: float = 0.10,
        match_score_mean: float = 0.70,
        match_score_std: float = 0.18
    ):
        self.p_single = singleton_prior
        self.p_match = 1.0 - singleton_prior
        self.mu_0 = singleton_score_mean
        self.sigma_0 = max(singleton_score_std, 1e-4)
        self.mu_1 = match_score_mean
        self.sigma_1 = max(match_score_std, 1e-4)

    def _gaussian_pdf(self, x: float, mu: float, sigma: float) -> float:
        var = sigma ** 2
        denom = math.sqrt(2.0 * math.pi * var)
        num = math.exp(-((x - mu) ** 2) / (2.0 * var))
        return num / denom

    def compute_singleton_posterior(self, top_score: float) -> float:
        """
        Computes P(singleton | top_score) using Bayes' rule.
        """
        if top_score <= 0.0:
            return 1.0

        lik_0 = self._gaussian_pdf(top_score, self.mu_0, self.sigma_0)
        lik_1 = self._gaussian_pdf(top_score, self.mu_1, self.sigma_1)

        evidence = (self.p_single * lik_0) + (self.p_match * lik_1)
        if evidence <= 1e-12:
            return 0.5
        return (self.p_single * lik_0) / evidence

    def filter_candidates(
        self,
        scored_candidates: List[Tuple[float, str]],
        country_tau: float,
        singleton_threshold: float = 0.65
    ) -> List[str]:
        """
        Applies Bayesian posterior thresholding:
        1. If P(singleton | max_score) > singleton_threshold, reject all candidates.
        2. Otherwise, accept candidates whose calibrated posterior probability >= country_tau.
        """
        if not scored_candidates:
            return []

        top_score = scored_candidates[0][0]
        p_single = self.compute_singleton_posterior(top_score)

        # Singleton guard: suppress false merges if top candidate is uncertain
        if p_single >= singleton_threshold:
            return []

        # Bayes-optimal acceptance for F0.5
        accepted = []
        for score, cand_id in scored_candidates:
            if score >= country_tau:
                accepted.append(cand_id)

        return accepted
