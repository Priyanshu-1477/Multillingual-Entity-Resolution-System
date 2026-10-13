"""End-to-end integration tests for the preprocessing pipeline."""

import csv
from pathlib import Path
import pytest
from src.config import PreprocessingConfig
from src.pipeline import PreprocessingPipeline
from src.validators import validate_tsv_streaming


def test_pipeline_end_to_end(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    state_dir = tmp_path / "state"
    temp_dir = tmp_path / "temp"

    input_dir.mkdir()

    # Create synthetic multilingual test dataset with ground truth
    src1_path = input_dir / "train_source1.tsv"
    gt_path = input_dir / "train_ground_truth.tsv"

    test_rows = [
        ["S1-0001", "Boulangerie Dupont", "12 Rue de Rivoli, Paris", "France"],
        ["S1-0002", "शिव शक्ति जनरल स्टोर", "Shop 12, Main Road, Delhi", "India"],
        ["S1-0003", "Farmacia San José", "Calle Mayor 5, Madrid", "Spain"],
        ["S1-0004", "ABC मोबाइल Store", "Plot 44, Bengaluru, Karnataka", "IN"],
        ["S1-0005", "Acme Corporation LLC", "100 Broadway, New York, NY 10001", "USA"],
        ["S1-0006", "-- Holloway Peak Inc Seafood", "105 ELM ST, MORGANTON, NC", "US"],
        ["S1-0007", "Fractales Amis Groupe S.A.S", "23 Rue Icmre, Gironde", "FRA"],
    ]

    with open(src1_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["entity_id", "business_name", "business_address", "country"])
        for r in test_rows:
            w.writerow(r)

    with open(gt_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        w.writerow(["S1-0001", "S2-0001,S3-0001"])

    config = PreprocessingConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        state_dir=state_dir,
        temp_dir=temp_dir,
        chunk_size=3,  # Small chunk size to test chunk transitions
    )

    pipeline = PreprocessingPipeline(config)
    pipeline.run_pipeline()

    # 1. Output files exist
    out_src1 = output_dir / "train_source1.tsv"
    out_gt = output_dir / "train_ground_truth.tsv"
    assert out_src1.exists()
    assert out_gt.exists()

    # 2. Validate output TSV
    report = validate_tsv_streaming(src1_path, out_src1)
    assert report.is_valid is True
    assert report.output_rows == len(test_rows)

    # 3. Verify specific transformations
    with open(out_src1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # Header
        rows = list(reader)

    # French preserved
    assert rows[0][1] == "Boulangerie Dupont"
    assert rows[0][3] == "France"

    # Hindi faithfully transliterated
    assert rows[1][1] == "Shiv Shakti General Store"
    assert rows[1][3] == "India"

    # Spanish unaccented
    assert rows[2][1] == "Farmacia San Jose"

    # Mixed script transliterated
    assert rows[3][1] == "ABC Mobile Store"
    assert rows[3][3] == "India"

    # Leading noise stripped
    assert rows[5][1] == "Holloway Peak Inc Seafood"
    assert rows[5][3] == "United States"

    # 4. Verify audit reports
    assert (output_dir / "audit" / "preprocessing_summary.json").exists()
    assert (output_dir / "audit" / "script_statistics.json").exists()
    assert (output_dir / "audit" / "examples.tsv").exists()
