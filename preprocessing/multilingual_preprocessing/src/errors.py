"""Streaming error tracking and threshold monitoring module."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PreprocessingError:
    """Record describing a recoverable transformation error."""

    entity_id: str
    source_file: str
    row_index: int
    field_name: str
    error_type: str
    error_message: str
    raw_value: str


class ErrorLogger:
    """Streams errors to disk in bounded memory and monitors cumulative error rates."""

    def __init__(self, errors_dir: Path, max_error_rate: float = 0.01):
        self.errors_dir = Path(errors_dir)
        self.errors_dir.mkdir(parents=True, exist_ok=True)
        self.error_file_path = self.errors_dir / "preprocessing_errors.tsv"
        self.max_error_rate = max_error_rate

        self._file = None
        self._writer = None
        self.total_errors = 0
        self.total_processed = 0

    def open(self):
        """Open error log TSV writer."""
        if self._file is None:
            write_header = not self.error_file_path.exists() or self.error_file_path.stat().st_size == 0
            self._file = open(self.error_file_path, "a", encoding="utf-8", newline="")
            self._writer = csv.writer(self._file, delimiter="\t", lineterminator="\n")
            if write_header:
                self._writer.writerow([
                    "entity_id",
                    "source_file",
                    "row_index",
                    "field_name",
                    "error_type",
                    "error_message",
                    "raw_value"
                ])
                self._file.flush()

    def log_error(self, error: PreprocessingError):
        """Increment error counter and stream error details to disk."""
        if self._file is None:
            self.open()

        self.total_errors += 1
        safe_raw_val = error.raw_value.replace("\t", " ").replace("\n", " ").replace("\r", " ")
        self._writer.writerow([
            error.entity_id,
            error.source_file,
            error.row_index,
            error.field_name,
            error.error_type,
            error.error_message,
            safe_raw_val[:200]  # Bound logged snippet length
        ])
        self._file.flush()

    def update_processed_count(self, count: int):
        """Update processed record count and verify error rate bounds."""
        self.total_processed += count
        if self.total_processed >= 1000:
            current_rate = self.total_errors / self.total_processed
            if current_rate > self.max_error_rate:
                raise RuntimeError(
                    f"Error rate {current_rate:.2%} exceeded maximum allowed threshold "
                    f"{self.max_error_rate:.2%} ({self.total_errors} errors / {self.total_processed} records)."
                )

    def close(self):
        """Close error log file."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
