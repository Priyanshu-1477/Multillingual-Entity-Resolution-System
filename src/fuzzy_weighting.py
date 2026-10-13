#!/usr/bin/env python3
"""
LEARNABLE FUZZY INFORMATION-PRESERVING WEIGHTING MODULE
Amazon ML Challenge 2026 - Business Entity Resolution

Implements the continuous soft-weighting pipeline:
  X (Similarity Vector) + Q (Data Quality Vector)
  → Context Decomposition:
      - Reliability (R ∈ [0, 1])
      - Informativeness (I ∈ [0, 1])
      - Ambiguity (A ≥ 0)
      - Uncertainty (U ∈ [0, 1])
  → Analytic Weight Calculation:
      w_i = (R_i * I_i) / (1.0 + A_i)
      w̃_i = w_i * (1.0 - U_i)
  → Modulated Feature Space:
      Z = w̃ ⊙ X
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any


class FuzzyWeightingNet(nn.Module):
    """
    Learnable Fuzzy Information-Preserving Feature Weighting Network.
    
    Dynamically modulates pairwise similarity vector X based on 
    latent reliability (R), informativeness (I), ambiguity (A), and uncertainty (U).
    """

    def __init__(self, num_quality_features: int = 5, num_similarity_features: int = 36):
        super().__init__()
        self.num_quality_features = num_quality_features
        self.num_similarity_features = num_similarity_features

        self.context_mlp = nn.Sequential(
            nn.Linear(num_quality_features + num_similarity_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.ria_mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, num_similarity_features * 3),
            nn.Sigmoid()
        )
        self.uncertainty_mlp = nn.Sequential(
            nn.Linear(32, num_similarity_features),
            nn.Sigmoid()
        )
        self.predictor = nn.Linear(num_similarity_features, 1)

    def forward(self, q: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            q: Data-quality vector tensor [Batch, num_quality_features]
            x: Similarity feature vector tensor [Batch, num_similarity_features]
        Returns:
            out: Logit prediction tensor [Batch, 1]
            z: Modulated feature representation Z = w̃ ⊙ X [Batch, num_similarity_features]
            final_w: Effective fuzzy feature weights w̃ [Batch, num_similarity_features]
            u: Epistemic uncertainty estimates U [Batch, num_similarity_features]
        """
        ctx_input = torch.cat([q, x], dim=1)
        ctx = self.context_mlp(ctx_input)
        ria = self.ria_mlp(ctx)

        feat_dim = x.shape[1]
        R = ria[:, :feat_dim]
        I = ria[:, feat_dim:feat_dim * 2]
        A = ria[:, feat_dim * 2:]

        # Weight calculation: w_i = (R_i * I_i) / (1 + A_i)
        w = (R * I) / (1.0 + A)
        
        # Uncertainty attenuation: w̃_i = w_i * (1 - U_i)
        u = self.uncertainty_mlp(ctx)
        final_w = w * (1.0 - u)

        # Modulated feature representation: Z = w̃ ⊙ X
        z = final_w * x
        out = self.predictor(z)

        return out, z, final_w, u


def extract_quality_features(q_rep: Any, ct: Any) -> np.ndarray:
    """
    Extract pairwise data-quality vector Q from query and candidate entity representations.
    Features:
      1. Missing query name indicator
      2. Missing candidate name indicator
      3. Absolute name length discrepancy / 100.0
      4. Missing query address indicator
      5. Missing candidate address indicator
    """
    q = []
    qn = getattr(q_rep, "name_raw", "") or ""
    tn = getattr(ct, "name_raw", "") or ""
    qa = getattr(q_rep, "address_raw", "") or ""
    ta = getattr(ct, "address_raw", "") or ""

    q.append(1.0 if not qn else 0.0)
    q.append(1.0 if not tn else 0.0)
    q.append(abs(len(qn) - len(tn)) / 100.0)
    q.append(1.0 if not qa else 0.0)
    q.append(1.0 if not ta else 0.0)

    return np.array(q, dtype=np.float32)
