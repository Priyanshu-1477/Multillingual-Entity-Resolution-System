"""Scalable streaming output validation module."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

EXPECTED_OUTPUT_HEADER = ["entity_id", "business_name", "business_address", "country"]


@dataclass
class ValidationReport:
    """Detailed results of streaming output validation."""

    input_file: str
    output_file: str
    is_valid: bool
    input_rows: int
    output_rows: int
    entity_ids_matched: bool
    no_unexpected_columns: bool
    no_forbidden_literals: bool
    delimiter_valid: bool
    errors: List[str]


def validate_tsv_streaming(input_path: Path, output_path: Path) -> ValidationReport:
    """Perform comprehensive lockstep streaming validation between input and output TSVs.

    Guarantees:
    - O(1) memory usage (never loads full files into RAM).
    - Verifies exact header schema.
    - Verifies input row count == output row count.
    - Verifies lockstep entity_id match.
    - Verifies no accidental literal 'nan', 'NaN', 'None'.
    - Verifies exact 4 columns per row.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    errors: List[str] = []

    if not output_path.exists():
        return ValidationReport(
            input_file=input_path.name,
            output_file=output_path.name,
            is_valid=False,
            input_rows=0,
            output_rows=0,
            entity_ids_matched=False,
            no_unexpected_columns=False,
            no_forbidden_literals=False,
            delimiter_valid=False,
            errors=[f"Output file does not exist: {output_path}"],
        )

    in_rows = 0
    out_rows = 0
    entity_ids_matched = True
    no_unexpected_columns = True
    no_forbidden_literals = True
    delimiter_valid = True

    with open(input_path, "r", encoding="utf-8", errors="replace", newline="") as f_in, \
         open(output_path, "r", encoding="utf-8", errors="replace", newline="") as f_out:

        r_in = csv.reader(f_in, delimiter="\t")
        r_out = csv.reader(f_out, delimiter="\t")

        # Check headers
        header_in = next(r_in, None)
        header_out = next(r_out, None)

        if header_out != EXPECTED_OUTPUT_HEADER:
            no_unexpected_columns = False
            errors.append(f"Header mismatch. Expected {EXPECTED_OUTPUT_HEADER}, got {header_out}")

        while True:
            row_in = next(r_in, None)
            row_out = next(r_out, None)

            if row_in is None and row_out is None:
                # Both completed simultaneously
                break

            if row_in is None and row_out is not None:
                out_rows += 1
                errors.append(f"Output has more rows than input (at output row {out_rows})")
                break

            if row_in is not None and row_out is None:
                in_rows += 1
                errors.append(f"Output has fewer rows than input (missing input row {in_rows})")
                break

            in_rows += 1
            out_rows += 1

            # Validate column count
            if len(row_out) != 4:
                delimiter_valid = False
                if len(errors) < 20:
                    errors.append(f"Row {out_rows} has {len(row_out)} columns instead of 4: {row_out}")

            # Validate entity ID lockstep match
            in_eid = row_in[0].strip() if row_in else ""
            out_eid = row_out[0].strip() if row_out else ""
            if in_eid != out_eid:
                entity_ids_matched = False
                if len(errors) < 20:
                    errors.append(f"Entity ID mismatch at row {out_rows}: input={in_eid}, output={out_eid}")

            # Validate forbidden literals ('nan', 'None', 'NaN')
            for col_idx, col_val in enumerate(row_out):
                if col_val.strip() in ("nan", "None", "NaN"):
                    raw_col_val = row_in[col_idx].strip() if col_idx < len(row_in) else ""
                    # Only flag if not present in raw source
                    if raw_col_val not in ("nan", "None", "NaN"):
                        no_forbidden_literals = False
                        if len(errors) < 20:
                            errors.append(
                                f"Forbidden literal '{col_val}' at row {out_rows}, col {col_idx} "
                                f"(raw input was '{raw_col_val}')"
                            )

    row_count_matched = (in_rows == out_rows)
    if not row_count_matched and not any("rows than input" in e for e in errors):
        errors.append(f"Row count mismatch: input had {in_rows} rows, output had {out_rows} rows.")

    is_valid = (
        len(errors) == 0 and
        row_count_matched and
        entity_ids_matched and
        no_unexpected_columns and
        no_forbidden_literals and
        delimiter_valid
    )

    return ValidationReport(
        input_file=input_path.name,
        output_file=output_path.name,
        is_valid=is_valid,
        input_rows=in_rows,
        output_rows=out_rows,
        entity_ids_matched=entity_ids_matched,
        no_unexpected_columns=no_unexpected_columns,
        no_forbidden_literals=no_forbidden_literals,
        delimiter_valid=delimiter_valid,
        errors=errors,
    )
