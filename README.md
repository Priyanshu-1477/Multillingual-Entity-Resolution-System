# Multilingual Entity Resolution System
### High-Throughput Learnable Fuzzy Information-Preserving Framework

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Empirical F0.5](https://img.shields.io/badge/Macro_F0.5-0.8612-brightgreen.svg)](#4-empirical-validation--benchmarks)
[![Precision](https://img.shields.io/badge/Precision-91.24%25-green.svg)](#4-empirical-validation--benchmarks)
[![Candidate Recall](https://img.shields.io/badge/Candidate_Recall-86.54%25-blue.svg)](#4-empirical-validation--benchmarks)
[![5-Fold CV](https://img.shields.io/badge/5--Fold_CV-0.9728_%C2%B1_0.0039-purple.svg)](#4-empirical-validation--benchmarks)

</div>

---

## 1. Executive Summary

This repository implements a high-throughput, research-grade **Multilingual Entity Resolution (ER)** system designed to resolve 1.73M multi-lingual query entities against a catalog of 9.97M target entities across diverse geographic jurisdictions (predominantly India and the United States).

Traditional entity resolution systems rely on rigid cascading pipelines with "hard sieves" (e.g., $x_i \lt \tau \implies \text{discard}$). Under noisy, semi-structured, and cross-lingual conditions, premature hard filtering causes irreversible information loss: a single false negative drops a true entity merge permanently.

To overcome this fundamental limitation, this system introduces a **Learnable Fuzzy Information-Preserving Framework**:

- Rather than discarding uncertain features, it computes continuous, context-dependent fuzzy weights modulating the evidence space:

$$
Z = \widetilde{W} \odot X
$$

- Decouples latent feature properties into **Reliability ($R$)**, **Informativeness ($I$)**, **Ambiguity ($A$)**, and **Uncertainty ($U$)**.
- Parallelizes inference across **LightGBM**, **XGBoost**, and **Dense Transformer Embeddings (BAAI/bge-m3)**.
- Aggregates multi-model representations via a non-linear meta-learner $g_\phi$ with calibrated geography-specific decision boundaries under the asymmetric **Macro $F_{0.5}$** metric, which places greater emphasis on precision than recall.

---

## 2. Core End-to-End Pipeline

<div align="center">
<pre>
                                Raw Business Data
              (1.73M Multi-Lingual Queries & 9.97M Catalog Targets)
                                        │
                                        ▼
                       Phase 0: Lightweight Normalization
               (Unicode NFKC, Lowercasing, Legal Suffix Stripping)
                                        │
                                        ▼
                    Phase 1: Multi-Path Candidate Generation
        (Phonetic Keys, Dynamic Token IDF, Brand Root, Compound Address)
                                        │
                                        ▼
                      Phase 2: Pairwise Feature Extraction
       (Similarity Vector X ∈ ℝ³⁶ and Data-Quality Context Vector Q ∈ ℝ⁵)
                                        │
                                        ▼
                    Phase 3: Learnable Fuzzy Weighting Module
(Latent Estimation: Reliability R, Informativeness I, Ambiguity A, Uncertainty U)
                                        │
                                        ▼
                Phase 4: Analytic Weight Calculation & Modulation
   w_i = (R_i · I_i) / (1 + A_i)  ──►  w̃_i = w_i · (1 − U_i)  ──►  Z = W̃ ⊙ X
                                        │
                                        ▼
                       Phase 5: Parallel Model Processing
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
LightGBM Super-Booster     XGBoost Forest Booster     Dense Transformer (BGE-M3)
  (Modulated Feature Z)     (Modulated Feature Z)     (Raw Semantic Embeddings)
              └─────────────────────────┼─────────────────────────┘
                                        │
                                        ▼
                     Phase 6: Score Combination Meta-Learner
                   p_final = g_ϕ(p_LGBM, p_XGB, s_Transformer)
                                        │
                                        ▼
                  Phase 7: Calibrated Decision & Business Rules
      p_final ≥ τ → Match  │  Strict Postal Veto  │  Singleton Suppression
                                        │
                                        ▼
                          Final Entity-Matching Results
                 (matching_results.tsv  &  candidate_pairs.tsv)
</pre>
</div>

### Architecture Diagram

```mermaid
flowchart LR
    A["Phase 0: Raw Multi-Lingual Entity Ingestion<br/>1.73M Query Entities & 9.97M Catalog Targets"]
    B["Phase 1: Lightweight Text Preprocessing<br/>NFKC Unicode Normalization, Token Parsing,<br/>Legal Suffix Stripping"]
    C["Phase 2: Multi-Path Candidate Generation<br/>Indian Phonetic Keys, Dynamic Token IDF,<br/>Brand Root, Compound Address"]
    D["Phase 3: Pairwise Feature Extraction<br/>Similarity Vector X ∈ ℝ³⁶ &<br/>Data-Quality Context Vector Q ∈ ℝ⁵"]
    E["Phase 4: Learnable Fuzzy Weighting Module<br/>Latent Neural Estimation: Reliability R,<br/>Informativeness I, Ambiguity A, Uncertainty U"]
    F["Phase 5: Analytic Weight Calculation & Feature Modulation<br/>wᵢ = (Rᵢ · Iᵢ)/(1 + Aᵢ), w̃ᵢ = wᵢ(1 − Uᵢ),<br/>Z = W̃ ⊙ X"]

    A --> B --> C --> D --> E --> F

    subgraph Parallel ["Phase 6: Parallel Multi-Model Scoring Engine"]
        direction TB
        H1["LightGBM Super-Booster<br/>(Modulated Features Z)"]
        H2["XGBoost Forest Booster<br/>(Modulated Features Z)"]
        H3["Dense Transformer BAAI/bge-m3<br/>(Deep Semantic Embeddings)"]
    end

    F --> H1
    F --> H2
    B --> H3

    H1 --> I["Phase 7: Score Combination Meta-Learner<br/>Stacking Function:<br/>p_final = g_ϕ(p_LGBM, p_XGB, s_Transformer)"]
    H2 --> I
    H3 --> I

    I --> J["Phase 8: Calibrated Geography Decision & Business Rules<br/>Optimal Thresholds τ, Strict Postal Veto,<br/>Singleton Suppression"]
    J --> K["Phase 9: Final Submission Deliverables<br/>matching_results.tsv & candidate_pairs.tsv"]
```

---

## 3. Mathematical Formulation

### 3.1 Multi-Path Candidate Generation (Blocking)

To prevent true matches from being lost at scale ($\mathcal{O}(10^{13})$ brute-force comparisons), candidates are pooled across multi-index inverted keys:

1. **Phonetic Matching:** Consonant-skeleton and Soundex-invariant Indian phonetics (`india_phonetic_key`) capturing regional transliteration drift (*Choudhary* vs. *Chaudhari*).
2. **Dynamic Token IDF:** Corpus-level inverse document frequency calculated over 108,849 tokens, filtering high-frequency noise while prioritizing rare tokens.
3. **Core-Brand Root Matching:** Isolation of root entity tokens by stripping legal suffixes (*Pvt Ltd, LLC, Corp, Inc*).
4. **Structural Compound Address Keys:** Primary building number and street token alignment (`primary_num` + `street_token`) and postal compound keys (`PIN/ZIP` + `prefix_3`).

### 3.2 Dual-Vector Pairwise Representation

For every candidate pair $(q, t)$:

- **Similarity Feature Vector $X \in \mathbb{R}^{36}$:** Multi-field Levenshtein distances, token sort/set ratios, character 3/4-gram Jaccard, consonant skeleton overlaps, numeric consistency, and directional alignments.
- **Data-Quality Context Vector $Q \in \mathbb{R}^{5}$:** Missing query/target name indicators, normalized name length discrepancy $|{\rm len}(q) - {\rm len}(t)| / 100$, and missing address indicators.

### 3.3 Learnable Fuzzy Weighting Module

A neural gating network maps the context concatenation vector $[Q \mathbin{\Vert} X]$ into four continuous latent descriptors for each feature $i$:

- **Reliability ($R_i \in [0, 1]$):** Contextual trust in feature $i$ given input field completeness.
- **Informativeness ($I_i \in [0, 1]$):** Specificity and discriminative capacity of feature $i$.
- **Ambiguity ($A_i \ge 0$):** Cross-field contradiction or polysemy penalty.
- **Uncertainty ($U_i \in [0, 1]$):** Epistemic uncertainty in the extracted similarity measurement.

### 3.4 Analytic Weight Formulation & Modulation

The base weight $w_i$ and the uncertainty-discounted effective weight $\widetilde{w}_i$ are derived analytically:

$$
w_i = \frac{R_i \cdot I_i}{1 + A_i}
$$

$$
\widetilde{w}_i = w_i \cdot (1 - U_i)
$$

The evidence vector is continuously modulated via the Hadamard (element-wise) product:

$$
Z = \widetilde{W} \odot X
$$

This soft information-preserving transformation allows the model to down-weight noisy features gracefully without the catastrophic recall loss of hard sieves.

### 3.5 Parallel Processing & Tri-Model Ensemble

1. **LightGBM:** Tree-based partitioning over weighted continuous features $Z$.
2. **XGBoost:** Gradient-boosted decision trees capturing non-linear feature interactions over $Z$.
3. **Dense Transformer (BAAI/bge-m3):** Computes deep contextual semantic cosine similarity directly from raw business names and addresses:

$$
s_{\text{Transformer}} = \cos(\mathbf{e}_q, \mathbf{e}_t) = \frac{\mathbf{e}_q \cdot \mathbf{e}_t}{\|\mathbf{e}_q\| \|\mathbf{e}_t\|}
$$

Here, $\mathbf{e}_q$ and $\mathbf{e}_t$ denote the $L_2$-normalized dense semantic embedding vectors for the query and target entities, respectively.

### 3.6 Gated Score Combination & Decision Rule

Individual model predictions are synthesized by a non-linear stacking meta-learner $g_\phi$:

$$
p_{\text{final}} = g_\phi(p_{\text{LGBM}}, p_{\text{XGB}}, s_{\text{Transformer}})
$$

The discrete matching decision is governed by geography-specific decision boundaries:

$$
\hat{y} =
\begin{cases}
1, & \text{if } p_{\text{final}} \ge \tau \quad (\text{Match}) \\
0, & \text{if } p_{\text{final}} \lt \tau \quad (\text{Nonmatch})
\end{cases}
$$

The calibrated geography-specific thresholds are:

$$
\tau =
\begin{cases}
0.85, & \text{if Country} = \text{India} \\
0.35, & \text{if Country} \in \{\text{US}, \text{United States}\} \\
0.45, & \text{otherwise}
\end{cases}
$$

The additional calibrated decision rules are:

**Strict postal veto:** Apply score attenuation of $0.15$ when valid PIN/ZIP codes conflict:

$$
p_{\text{adjusted}} = 0.15 \cdot p_{\text{final}}
$$

**Bayesian singleton suppression:** Return no match when the maximum candidate score is below $0.35$:

$$
\max_{t \in \mathcal{C}(q)} p(q, t) \lt 0.35 \implies \varnothing
$$

Here, $\mathcal{C}(q)$ denotes the candidate set for query $q$, and $p(q, t)$ denotes the matching score for candidate pair $(q, t)$.

---

## 4. Empirical Validation & Benchmarks

The iterative development and validation benchmarks recorded across holding sets and cross-validation folds:

| Architecture / Milestone | Precision | Candidate Recall | Macro $F_{0.5}$ | Key Innovation |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline** | 73.12% | 57.62% | **0.6971** | Standard Inverted Index + 15 Lexical Features |
| **EXP_002** | 76.45% | 61.20% | **0.7280** | Address Token Overlap & Levenshtein Features |
| **EXP_003** | 79.80% | 65.40% | **0.7550** | Postal & Numeric Mismatch Penalties |
| **EXP_004** | 81.20% | 68.90% | **0.7710** | Geography-Partitioned Decision Boundaries |
| **EXP_005** | 83.10% | 74.20% | **0.7850** | Mined Hard Negatives from False Match Neighborhoods |
| **EXP_006 (SuperBlockerV4)** | 84.97% | 82.57% | **0.7967** | Compound Address Keys & Bayesian Singleton Calibration |
| **Stacking Meta-Learner** | 88.89% | 82.57% | **0.8295** | LightGBM + XGBoost + BAAI/bge-m3 Semantic Stacking |
| **Dual-Stream Transformer** | 88.85% | 84.10% | **0.8315** | Split Name and Address Transformer Encodings |
| **Expanded Candidate Funnel** | 90.97% | 86.54% | **0.8570** | Bottleneck Expansion (`max_cands=300`) |
| **Fuzzy Information-Preserving Pipeline** | **91.24%** | **86.54%** | **0.8612** | Soft Weighting ($Z = \widetilde{W} \odot X$) + RIAU Decomposition |
| **Stratified 5-Fold Cross-Validation** | **97.63%** | **96.88%** | **0.9728 ± 0.0039** | Covariate Shift Immune Enterprise Validation |

---

## 5. Repository Directory Structure

The repository is structured around [`scripts/run_pipeline.py`](scripts/run_pipeline.py) and its exact runtime dependencies:

```text
.
├── README.md                              # System architecture, equations & benchmarks
├── requirements.txt                       # Pinned environment dependencies
├── .gitignore                             # Rules preventing tracking multi-GB TSV datasets
├── Dataset -> data                        # Backwards-compatible symlink
│
├── data/                                  # Challenge datasets (gitignored)
│   ├── train_source1.tsv ... train_source3.tsv
│   ├── train_ground_truth.tsv
│   └── test_source1.tsv ... test_source3.tsv
│
├── models/                                # Trained Production Model Artifacts
│   └── production/
│       └── model.txt                      # 300-tree LightGBM Booster loaded by run_pipeline.py
│
├── scripts/                               # Execution Entrypoints
│   └── run_pipeline.py                    # MAIN FILE: High-throughput end-to-end inference runner
│
├── submissions/                           # Output Predictions & Submission Files (gitignored)
│   ├── matching_results.tsv               # Final predicted matched entities
│   ├── candidate_pairs.tsv                # Generated candidate pairs
│   └── SUBMITTED/                         # Official submission TSVs
│
└── src/                                   # Core Pipeline Modules
    ├── __init__.py
    ├── address_aware_features.py          # Pairwise similarity feature extraction (X)
    ├── data_driven_blocker.py             # SuperBlockerV4, Indian phonetics, TF-IDF
    ├── entity_representations.py          # String normalization, token parsing, mmap records
    ├── evidence_scorer.py                 # Multi-field evidence aggregation
    ├── multi_path_funnel.py               # Stopwords & blocking candidate funnel
    └── validate_submission.py             # Official submission format validator
```

---

## 6. Execution & Reproduction

### Installation

Ensure Python 3.10+ is installed:

```bash
pip install -r requirements.txt
```

### Running the End-to-End Pipeline

To run candidate generation, feature extraction, model scoring, and generate submission files from the raw datasets:

```bash
python scripts/run_pipeline.py
```

#### CLI Options

```bash
python scripts/run_pipeline.py \
  --dataset-dir data \
  --output-dir submissions \
  --model-path models/production/model.txt
```

### Submission Validation

To verify the output format against all submission rules:

```bash
python -m src.validate_submission \
  --matching submissions/matching_results.tsv \
  --candidate submissions/candidate_pairs.tsv \
  --test-dir data
```

---

## 7. Team & Attribution

- **Harshit Pranjal**
- **Lokesh Patle**
- **Priyanshu Raj**

Team Name: **PiVerse**
