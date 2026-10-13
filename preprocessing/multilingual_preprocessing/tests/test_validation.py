"""Unit tests for streaming output validation."""

import csv
import pytest
from src.validators import validate_tsv_streaming


def test_validation_success(tmp_path):
    in_file = tmp_path / "in.tsv"
    out_file = tmp_path / "out.tsv"

    # Write input
    with open(in_file, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["entity_id", "business_name", "business_address", "country"])
        w.writerow(["S1-001", "Acme Corp", "123 Main St", "US"])
        w.writerow(["S1-002", "Dupont Sarl", "10 Rue Paris", "FR"])

    # Write valid output
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["entity_id", "business_name", "business_address", "country"])
        w.writerow(["S1-001", "Acme Corp", "123 Main St", "United States"])
        w.writerow(["S1-002", "Dupont Sarl", "10 Rue Paris", "France"])

    report = validate_tsv_streaming(in_file, out_file)
    assert report.is_valid is True
    assert report.input_rows == 2
    assert report.output_rows == 2
    assert report.entity_ids_matched is True


def test_validation_detects_row_mismatch(tmp_path):
    in_file = tmp_path / "in.tsv"
    out_file = tmp_path / "out.tsv"

    with open(in_file, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["entity_id", "business_name", "business_address", "country"])
        w.writerow(["S1-001", "Acme Corp", "123 Main St", "US"])
        w.writerow(["S1-002", "Dupont Sarl", "10 Rue Paris", "FR"])

    # Output missing one row
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["entity_id", "business_name", "business_address", "country"])
        w.writerow(["S1-001", "Acme Corp", "123 Main St", "United States"])

    report = validate_tsv_streaming(in_file, out_file)
    assert report.is_valid is False
    assert report.input_rows == 2
    assert report.output_rows == 1


def test_validation_detects_forbidden_literals(tmp_path):
    in_file = tmp_path / "in.tsv"
    out_file = tmp_path / "out.tsv"

    with open(in_file, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["entity_id", "business_name", "business_address", "country"])
        w.writerow(["S1-001", "", "123 Main St", "US"])

    # Output has forbidden 'nan'
    with open(out_file, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["entity_id", "business_name", "business_address", "country"])
        w.writerow(["S1-001", "nan", "123 Main St", "United States"])

    report = validate_tsv_streaming(in_file, out_file)
    assert report.is_valid is False
    assert report.no_forbidden_literals is False
