# Amazon ML Challenge 2026 - Multilingual Business Entity Resolution Preprocessing

This directory contains the production-grade, large-scale data preprocessing pipeline for the Amazon ML Challenge 2026 Business Entity Resolution dataset (~24.23 million records).

For detailed documentation, architectural specifications, normalization rules, and benchmarks, please refer to [multilingual_preprocessing/README.md](file:///Users/harshitpranjal/preProcessing/multilingual_preprocessing/README.md).

## Quick Start

### 1. Run Complete 24M Dataset Preprocessing
```bash
python run_preprocessing.py --input actual_data --output preprocessed_output --chunk-size 50000
```

### 2. Run Pre-flight & Dry Run
```bash
python run_preprocessing.py --input actual_data --dry-run
```

### 3. Run Benchmark
```bash
python run_preprocessing.py --input actual_data --benchmark --rows 100000
```

### 4. Run Streaming Output Validation
```bash
python run_preprocessing.py --input actual_data --output preprocessed_output --validate-only
```

### 5. Run Unit Tests & Stress Test
```bash
# Run unit & integration tests
python -m pytest multilingual_preprocessing/tests/ -v -k "not stress"

# Run 1,000,000-record stress test
python multilingual_preprocessing/tests/test_stress_1m.py
```

## Directory Structure
- `actual_data/`: Original raw input TSV files (Source 1, 2, 3 for train/test + ground truth).
- `preprocessed_output/`: Final preprocessed TSVs with canonical schema (`entity_id`, `business_name`, `business_address`, `country`), plus `audit/` and `errors/`.
- `preprocessing_state/`: Resumption checkpoints per source file.
- `multilingual_preprocessing/`: Full production codebase and test suite.
