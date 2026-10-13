# Multilingual Preprocessing Pipeline for Amazon ML Challenge 2026

## 1. Overview
A production-grade, memory-bounded, streaming preprocessing pipeline engineered specifically for the 24+ million record dataset in the Amazon ML Challenge 2026 Business Entity Resolution competition.

### Core Architecture & Guarantees
- **Total Scale Supported**: Full 24.23+ million records processed end-to-end.
- **Strictly Bounded Memory**: Memory usage remains $O(\text{chunk\_size})$ (typically ~80–95 MB RAM total), never loading the entire dataset into memory.
- **Incremental Streaming**: Chunk-based streaming reader and writer with configurable batch sizes (10k, 25k, 50k, 100k, 250k).
- **Exact Row & ID Preservation**:
  - `INPUT ROW COUNT == OUTPUT ROW COUNT` across all files.
  - `INPUT ENTITY_ID == OUTPUT ENTITY_ID` immutable, preserving leading zeros, prefixes, and exact casing.
  - Zero dropped, zero duplicated, and zero synthesized rows.
- **Source & Ground Truth Separation**:
  - Source files (`source1`, `source2`, `source3`) remain strictly separated.
  - Train and test splits remain strictly separated.
  - Ground truth files are preserved verbatim without contaminating or leaking into source records.
- **Offline & Autonomous**: Operates 100% offline with zero external network or API calls.
- **Resumability & Checkpointing**: Maintains atomic state per file in `preprocessing_state/`. Safely restarts from completed chunks.

---

## 2. Multilingual & Script Transformation Philosophy

### Latin Script Policy (Language-Preserving)
Text written in Latin script is **never translated into English**.
- **French**: `Boulangerie Dupont` $\rightarrow$ `Boulangerie Dupont`
- **Spanish**: `Farmacia San José` $\rightarrow$ `Farmacia San Jose` (diacritics normalized to clean Latin-readable)
- **English**: `ABC Trading Corporation` $\rightarrow$ `ABC Trading Corporation`
- **Portuguese**: `Mercado São João` $\rightarrow$ `Mercado Sao Joao`

### Non-Latin Script Policy (Faithful Transliteration, No Semantic Translation)
Non-Latin scripts are faithfully transliterated into Latin script to preserve entity identity:
- **Hindi / Devanagari**:
  - `शिव शक्ति जनरल स्टोर` $\rightarrow$ `Shiv Shakti General Store` *(NOT "Shiva Power General Store")*
  - `राम मार्केटिंग प्राइवेट लिमिटेड` $\rightarrow$ `Ram Marketing Private Limited`
  - `आदित्य प्रॉपर्टीज एलएलपी` $\rightarrow$ `Aditya Properties LLP`
- **Gujarati**:
  - `શ્રી ગણેશ મોબાઇલ` $\rightarrow$ `Shri Ganesh Mobile` *(NOT "Lord Ganesha Mobile")*
  - `ગુજરાત` $\rightarrow$ `Gujarat`
- **Bengali**:
  - `শিব শক্তি জেনারেল স্টোর` $\rightarrow$ `Shiv Shakti General Store`
- **Tamil**:
  - `ஸ்ரீ கணேஷ் மொபைல்` $\rightarrow$ `Shri Ganesh Mobile`
- **Arabic**:
  - `متجر النور` $\rightarrow$ `mtjr lnwr`
- **CJK / Chinese**:
  - `北京贸易公司` $\rightarrow$ `BeiJingMaoYiGongSi`

### Mixed-Script Resolution
Records containing multiple scripts are segmented dynamically:
- `ABC मोबाइल Store` $\rightarrow$ `ABC Mobile Store`
- `Shree राम Traders` $\rightarrow$ `Shree Ram Traders`
- `Paris दुकान` $\rightarrow$ `Paris Dukan`
- `Door No 183, ... Bangalore, ಕರ್ನಾಟಕ` $\rightarrow$ `Door No 183, ... Bangalore, Karnataka`
- `H.no 910 A 3503, Mumbai, महाराष्ट्र` $\rightarrow$ `H.no 910 A 3503, Mumbai, Maharashtra`

---

## 3. Normalization Rules

### Business Names
- **Unicode Sanitization**: Strips zero-width characters, BOMs, non-printable control characters, and collapses whitespace.
- **Noise Cleanup**: Removes leading/trailing noise punctuation (`-- Holloway Peak Inc Seafood` $\rightarrow$ `Holloway Peak Inc Seafood`, `<< Team Ecole` $\rightarrow$ `Team Ecole`).
- **Entity Suffix Preservation**: Preserves all corporate designators (`LLC`, `Inc`, `Corp`, `Private Limited`, `Pvt Ltd`, `S.A.S`, `GmbH`, `LLP`, `OPC`, etc.).
- **Missing Values**: Empty or null strings output as empty string `""`. Never produces `"nan"` or `"None"`.

### Business Addresses
- **Preserved**: House numbers, shop numbers, building numbers, street numbers, road numbers, PIN codes, ZIP codes, localities, landmarks, unit numbers.
- **Leading Zeros**: Always preserved as strings (e.g. `00123`).
- **Rogue Tokens**: Cleans isolated placeholder noise (e.g. `..., <NULL>, ...` or `..., null, ...`).
- **No Hallucination**: No geocoding, coordinate generation, or synthetic address filling.

### Country Normalization (Open-Set)
- Canonicalizes common aliases:
  - `IND`, `IN`, `India`, `Republic of India` $\rightarrow$ `India`
  - `FRA`, `FR`, `France` $\rightarrow$ `France`
  - `USA`, `US`, `United States`, `United States of America` $\rightarrow$ `United States`
- Open-Set: Any other country (e.g. `Germany`, `Canada`, `Japan`) is preserved cleanly, never dropped.

---

## 4. Output Specification & Directory Layout

```
multilingual_preprocessing/
│
├── preprocessed_output/
│   ├── train_source1.tsv           # Primary preprocessed business records
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   ├── train_ground_truth.tsv      # Preserved ground truth
│   ├── test_source1.tsv
│   ├── test_source2.tsv
│   ├── test_source3.tsv
│   │
│   ├── audit/
│   │   ├── preprocessing_summary.json
│   │   ├── script_statistics.json
│   │   ├── language_statistics.json
│   │   ├── country_statistics.json
│   │   ├── performance_statistics.json
│   │   └── examples.tsv
│   │
│   └── errors/
│       └── preprocessing_errors.tsv
│
├── preprocessing_state/             # Atomic checkpoints per source file
├── temporary/                       # Intermediate / scratch data
├── src/                             # Core modular pipeline source code
├── tests/                           # Complete test suite (35 unit tests + 1M stress test)
├── run_preprocessing.py             # CLI entrypoint
├── requirements.txt                 # Dependency manifest with licenses
└── README.md
```

### Primary Output Schema
The cleaned TSV files contain **strictly 4 columns** with zero extraneous diagnostic columns:
```tsv
entity_id	business_name	business_address	country
```

---

## 5. CLI Usage Guide

### Full Dataset Execution
```bash
python run_preprocessing.py --input actual_data --output preprocessed_output --chunk-size 50000
```

### Configurable Chunk Sizes
```bash
# Available options: 10000, 25000, 50000, 100000, 250000
python run_preprocessing.py --input actual_data --output preprocessed_output --chunk-size 100000
```

### Dry Run (Sample Inspection)
```bash
python run_preprocessing.py --input actual_data --dry-run
```

### Benchmark Mode
```bash
python run_preprocessing.py --input actual_data --benchmark --rows 100000
```

### Validation Only
```bash
python run_preprocessing.py --input actual_data --output preprocessed_output --validate-only
```

---

## 6. Testing & Stress Verification

### Unit & Integration Tests
```bash
python -m pytest multilingual_preprocessing/tests/ -v -k "not stress"
```
*35 tests passing in < 0.1s.*

### 1,000,000 Record Stress Test
```bash
python multilingual_preprocessing/tests/test_stress_1m.py
```
- **Scale**: 1,000,000 multilingual records.
- **Throughput**: ~44,300 rows/second.
- **RAM**: Strict bounded memory (RAM delta +36 MB, peak < 85 MB).
- **Validation**: 100% lockstep row count and entity ID matching.

---

## 7. Open-Source Dependencies & Licenses
All libraries operate locally and offline with permissive open-source licenses:
- `anyascii` (v0.3.3) — ISC License (MIT compatible)
- `regex` (v2024+) — Apache 2.0 License
- `psutil` (v5.9+) — BSD-3-Clause License
- `tqdm` (v4.66+) — MPL-2.0 / MIT License
- `pytest` (v7.0+) — MIT License
