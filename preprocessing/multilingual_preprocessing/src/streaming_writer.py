"""Memory-bounded incremental TSV writer."""

import csv
from pathlib import Path
from typing import List, Sequence

EXPECTED_OUTPUT_SCHEMA = ["entity_id", "business_name", "business_address", "country"]


class StreamingTsvWriter:
    """Incrementally writes cleaned records to a TSV file in bounded memory."""

    def __init__(self, output_path: Path, append_mode: bool = False):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = "a" if append_mode else "w"
        self._file = None
        self._writer = None
        self._header_written = append_mode and self.output_path.exists() and self.output_path.stat().st_size > 0

    def open(self):
        """Open output file descriptor with UTF-8 encoding."""
        if self._file is None:
            self._file = open(self.output_path, self.mode, encoding="utf-8", newline="", errors="strict")
            self._writer = csv.writer(
                self._file,
                delimiter="\t",
                quoting=csv.QUOTE_MINIMAL,
                lineterminator="\n"
            )
            if not self._header_written:
                self._writer.writerow(EXPECTED_OUTPUT_SCHEMA)
                self._file.flush()
                self._header_written = True

    def write_chunk(self, rows: Sequence[Sequence[str]]):
        """Write a chunk of records to disk and flush."""
        if self._file is None:
            self.open()

        # Sanitize any accidental newlines or tabs within fields to preserve TSV integrity
        for row in rows:
            sanitized_row = [
                str(val).replace("\t", " ").replace("\r", " ").replace("\n", " ")
                for val in row
            ]
            self._writer.writerow(sanitized_row)

        self._file.flush()

    def close(self):
        """Close output file descriptor cleanly."""
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
