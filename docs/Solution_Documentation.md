# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** PiVerse  
**Team Members:** Harshit Pranjal, Lokesh Patle, Priyanshu Raj 
**Submission Date:** September 27, 2026

---

## 1. Executive Summary
We present a high-throughput, multi-stage Business Entity Resolution architecture engineered to optimize the asymmetric Macro $F_{0.5}$ metric across 1.73M queries and 9.97M target records. Our solution integrates **SuperBlockerV4** (country-partitioned indexing, core-brand extraction, transliteration-invariant Indian phonetics, and compound address keys) with a 40-dimensional **Address-Aware LightGBM Super-Booster**, augmented by strict postal-conflict hard penalties and Bayesian singleton uncertainty filtering. The resulting system pushes candidate recall from 57.62% to 82.57%, lifts precision to 84.97%, and achieves a macro $F_{0.5}$ score of **0.7967** on held-out validation.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis revealed three critical challenges:
1. **Severe Geographic Heterogeneity:** While US addresses exhibit structured numbering (ZIP codes, primary street numbers), Indian addresses frequently use informal landmarks, descriptive localities, transliteration variants (*Choudhary* vs *Chaudhari*), and absent postal codes.
2. **Asymmetric $F_{0.5}$ Objective Penalty:** A single false positive merge severely penalizes an entity, dropping its score to 0.0. In contrast, precision is weighted $4\times$ heavier than recall ($\beta = 0.5$).
3. **Scale and Cardinality Constraints:** Resolving 1.73M queries against 9.97M targets prohibits brute-force comparison ($O(10^{13})$). Candidate retrieval recall serves as an absolute theoretical ceiling on end-to-end performance.

### 2.2 Solution Strategy
We adopted a decoupled, multi-path funnel architecture:
- **Phase 0: Multilingual Preprocessing:** Unicode NFKC normalization, Latin transliteration (`anyascii`), legal suffix stripping, address standardisation, and country normalization across raw inputs.
- **Phase 1: SuperBlockerV4 Candidate Funnel:** Multi-index candidate retrieval with automated dynamic frequency gating and C-style contiguous integer arrays.
- **Phase 2: 40-Dimensional Pairwise Feature Engineering:** Lexical, phonetic, n-gram Jaccard, consonant skeleton, and address-specific conflict metrics.
- **Phase 3: Cost-Sensitive Super-Booster Classification:** 300-tree LightGBM gradient boosted decision forest trained on mined hard negatives.
- **Phase 4: Bayesian Uncertainty Calibration & Country Thresholds:** Geography-specific optimal decision boundaries ($\tau_{\text{India}} = 0.85$, $\tau_{\text{US}} = 0.35$) with a Bayesian singleton probability gate ($P < 0.35 \implies \emptyset$) and strict postal conflict vetoes.

**Approach Type:** Hybrid Multi-Path Inverted Index Blocking + Address-Aware LightGBM Classifier + Bayesian Singleton Calibrator  
**Core Innovation:** Core-brand root extraction and Indian phonetic keys combined with compound address keys (`primary_num + street_token`), backed by strict postal conflict vetoes (`score * 0.15`).

---

## 3. Candidate Generation (Blocking)
To overcome the candidate recall bottleneck (which stood at 57.62% in baseline inverted indexing), we developed **SuperBlockerV4**:
- **Blocking keys used:**
  1. *Country-Partitioned Inverted Indexing*: Prevents cross-border candidate contamination.
  2. *Core-Brand Extraction*: Isolates root business tokens by stripping legal suffixes (*Pvt Ltd*, *LLC*, *Corp*, *Inc*) and generic descriptors.
  3. *Indian Phonetic Transliteration Keys*: Soundex-invariant and consonant-vowel normalizers capturing spelling variants.
  4. *Compound Address Keys*: `(primary_street_num + major_street_token)` uniquely indexing localized multi-tenant locations.
  5. *Postal + Token Keys*: `(PIN/ZIP + name_prefix_3)` guaranteeing localized retrieval.
  6. *Data-Driven Token IDF Weighting*: 108,849 corpus tokens weighted dynamically, filtering non-selective tokens with frequency cutoffs.
- **Candidate pairs generated:** Average of 8–15 candidates per query entity across 1.73M queries (zero empty candidate sets).
- **How true matches were not lost:** Multi-path union ensured that if an entity varied in name spelling, its compound address key retrieved it; conversely, if the address was missing or noisy, core-brand and phonetic keys retrieved the match. Candidate recall jumped from **57.62% to 82.57%** (+24.95% true matches recovered).

---

## 4. Matching Model

**Features used (40 Dimensions):**
- **Name features:** Levenshtein token ratio, partial ratio, token sort ratio, token set ratio, character 2/3/4-gram Jaccard, consonant skeleton Jaccard, core-brand exact match, Soundex token similarity, and rare-token IDF overlap.
- **Address features:** Cleaned address token overlap, address Levenshtein ratio, primary number exact match, street token overlap, 6-digit Indian PIN code equality/conflict, 5-digit US ZIP code equality/conflict, building number consistency, and directional token alignment.
- **Cross-Field Conflict & Interaction Indicators:**
  - `postal_code_conflict`: Binary flag set when both records possess valid postal codes that disagree.
  - `numeric_conflict`: Binary flag indicating mismatched building/suite street numbers.
  - `name_lev_x_addr_jaccard`: Interaction term prioritizing pairs with both strong name and address alignment.

**Model type:** LightGBM Gradient Boosted Decision Forest (300 trees, learning rate 0.05, num_leaves 63, colsample_bytree 0.8, subsample 0.8).  
**Threshold selection method:** 2D grid search optimizing Macro $F_{0.5}$ on held-out validation split:
- $\tau_{\text{India}} = 0.85$: Stringent threshold preventing false merges amidst noisy Indian landmark addresses.
- $\tau_{\text{US}} = 0.35$: Calibrated threshold maximizing recall across structured US business registries.
- **Hard Penalty Veto**: When conflicting postal codes are detected ($q_{\text{postal}} \neq t_{\text{postal}}$ with length $\ge 3$), predicted match confidence is strictly dampened by $0.15\times$.
- **Bayesian Singleton Guard**: When max candidate probability $P_{\max} < 0.35$, all candidates are discarded to output an empty singleton list, maximizing the $F_{0.5}$ reward for true singletons.

---

## 5. Results & Error Analysis

- **$F_{0.5}$ Score (macro):** **0.7967** (Validation) | Precision: **84.97%** | Candidate Recall: **82.57%**
- **Common false positives (wrong merges):**
  - Chain stores and franchise locations (e.g., *Subway*, *State Bank of India*) with near-identical names in adjacent postal sectors. Substantially mitigated by the compound address key and postal conflict veto.
  - Distinct business divisions sharing identical registered corporate addresses (e.g., *Tata Consultancy Services* vs *Tata Motors* at the same office park). Mitigated by core-brand specific token matching.
- **Common false negatives (missed matches):**
  - Radical name acronyms where no common n-grams or brand roots exist (e.g., *SBI* vs *State Bank of India* with completely omitted address details).
  - Out-of-region entities with completely corrupted country fields.

---

## 6. Conclusion
By pairing country-partitioned multi-path blocking (SuperBlockerV4) with an address-aware gradient boosted classifier and Bayesian singleton filtering, Team PiVerse established a robust, production-grade entity resolution system. The architecture maintains sub-second throughput (indexing ~58,600 targets/sec with zero-copy `mmap`) while achieving exceptional precision (84.97%) and a Macro $F_{0.5}$ of 0.7967, fully verified and ready for deployment.

---

## Appendix

### A. Code Artefacts
The reproducible pipeline code is organized under `code/business_entity_resolution/`:
```
code/business_entity_resolution/
├── preprocessing/                  # Multilingual raw data preprocessing stage
│   ├── multilingual_preprocessing/ # Core preprocessing modules, normalization & tests
│   ├── run_preprocessing.py        # Preprocessing entry point
│   └── README.md                   # Preprocessing documentation
├── src/                            # Feature extraction, blocking & modeling
│   ├── address_aware_features.py   # 40-D feature extraction & conflict vetoes
│   ├── bayesian_filter.py          # Bayesian probability calibrator & singleton filter
│   ├── data_driven_blocker.py      # Core-brand extraction, Indian phonetics & TF-IDF blocker
│   ├── entity_representations.py   # Normalization, token parsing, and mmap representations
│   ├── evaluation.py               # Macro F0.5 evaluation metric implementation
│   ├── evidence_scorer.py          # Multi-field evidence scoring
│   ├── hard_negative_miner.py      # Hard negative mining from candidate neighborhoods
│   ├── multi_path_funnel.py        # Multi-path blocking funnel architecture
│   └── validate_submission.py      # Official submission validator
├── models/
│   └── model.txt                   # Pretrained LightGBM Super-Booster (300 trees)
├── run_pipeline.py                 # End-to-end execution runner
├── README.md                       # Comprehensive reproduction guide
└── requirements.txt                # Pinned dependencies
```
**Entry Point to Reproduce:**
```bash
# 1. (Optional) Run preprocessing on raw files:
python preprocessing/run_preprocessing.py --input /path/to/raw --output /path/to/preprocessed

# 2. Run blocking, matching & output generation:
python run_pipeline.py --dataset-dir /path/to/dataset --output-dir ../../output
```

### B. Additional Results & Validation Certification
- Total Test Queries Evaluated: **1,732,544**
- Non-Empty Matches Output: **1,607,260 (92.77%)**
- Singletons Identified: **125,284 (7.23%)**
- Official Validation Result: `PASS — no blocking issues found. Safe to submit.` (100% containment satisfied, zero missing rows, zero formatting errors).
