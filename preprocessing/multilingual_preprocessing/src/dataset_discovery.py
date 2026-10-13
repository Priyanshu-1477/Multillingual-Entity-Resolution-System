"""Dynamic dataset discovery and schema inspection module."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DiscoveredFile:
    """Metadata describing a discovered dataset file."""

    file_path: Path
    filename: str
    file_type: str  # 'source_train', 'source_test', 'ground_truth', 'other'
    file_size_bytes: int
    approx_size_mb: float
    header: List[str]
    delimiter: str
    is_valid_tsv: bool
    estimated_rows: int


@dataclass
class DatasetDiscoveryReport:
    """Summary of all discovered input files."""

    source_files: List[DiscoveredFile]
    ground_truth_files: List[DiscoveredFile]
    other_files: List[DiscoveredFile]
    total_source_bytes: int
    total_estimated_rows: int


def count_file_lines(file_path: Path) -> int:
    """Fast streaming count of lines in a file using a large read buffer."""
    lines = 0
    with open(file_path, "rb") as f:
        buf_size = 1024 * 1024 * 8  # 8MB buffer
        while True:
            chunk = f.read(buf_size)
            if not chunk:
                break
            lines += chunk.count(b"\n")
    return lines


def discover_dataset(input_dir: Path, count_exact_rows: bool = False) -> DatasetDiscoveryReport:
    """Scan input directory, discover files, inspect headers, and determine logical dataset roles."""
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {input_path}")

    source_files: List[DiscoveredFile] = []
    ground_truth_files: List[DiscoveredFile] = []
    other_files: List[DiscoveredFile] = []

    total_bytes = 0
    total_rows = 0

    tsv_files = sorted(list(input_path.glob("*.tsv")) + list(input_path.glob("*.csv")))

    for fpath in tsv_files:
        size_bytes = fpath.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        # Inspect header and delimiter
        header = []
        delim = "\t"
        is_valid = True

        try:
            with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                first_line = f.readline()
                if "\t" in first_line:
                    delim = "\t"
                elif "," in first_line:
                    delim = ","
                header = [col.strip() for col in first_line.split(delim)]
        except Exception:
            is_valid = False

        rows = 0
        if count_exact_rows:
            total_lines = count_file_lines(fpath)
            rows = max(0, total_lines - 1)  # Minus header
        else:
            # Estimate based on average row length ~100 bytes
            rows = int(size_bytes / 100)

        filename_lower = fpath.name.lower()
        has_entity_id = any("entity_id" in col.lower() for col in header)

        if "ground_truth" in filename_lower or "truth" in filename_lower or (len(header) == 2 and "matched" in header[1].lower()):
            file_type = "ground_truth"
        elif "train" in filename_lower:
            file_type = "source_train"
        elif "test" in filename_lower:
            file_type = "source_test"
        elif has_entity_id:
            file_type = "source_generic"
        else:
            file_type = "other"

        discovered = DiscoveredFile(
            file_path=fpath,
            filename=fpath.name,
            file_type=file_type,
            file_size_bytes=size_bytes,
            approx_size_mb=round(size_mb, 2),
            header=header,
            delimiter=delim,
            is_valid_tsv=is_valid,
            estimated_rows=rows,
        )

        if file_type == "ground_truth":
            ground_truth_files.append(discovered)
        elif file_type.startswith("source"):
            source_files.append(discovered)
            total_bytes += size_bytes
            total_rows += rows
        else:
            other_files.append(discovered)

    return DatasetDiscoveryReport(
        source_files=source_files,
        ground_truth_files=ground_truth_files,
        other_files=other_files,
        total_source_bytes=total_bytes,
        total_estimated_rows=total_rows,
    )
