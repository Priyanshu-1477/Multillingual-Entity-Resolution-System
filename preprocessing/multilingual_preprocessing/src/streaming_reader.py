"""Memory-bounded streaming TSV reader."""

import csv
from pathlib import Path
from typing import Iterator, List, Optional, Tuple


def stream_tsv_chunks(
    file_path: Path,
    chunk_size: int = 50_000,
    start_row: int = 0,
) -> Iterator[Tuple[int, List[List[str]]]]:
    """Stream a TSV file in bounded memory chunks.

    Yields:
        (chunk_index, chunk_rows): chunk_index (0-indexed) and list of row string lists.

    Guarantees:
        - Memory bounded by O(chunk_size).
        - Reads identifiers as strings, preserving leading zeros (e.g., '00123').
        - Empty fields remain empty strings, never converted to 'nan' or 'None'.
        - Resumable: can fast-forward past `start_row` rows safely.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        # Read and skip header line
        header = next(reader, None)
        if header is None:
            return

        # Skip rows if resuming from a checkpoint
        skipped = 0
        while skipped < start_row:
            next_row = next(reader, None)
            if next_row is None:
                return
            skipped += 1

        chunk_index = start_row // chunk_size
        current_chunk: List[List[str]] = []

        for row in reader:
            current_chunk.append(row)
            if len(current_chunk) >= chunk_size:
                yield chunk_index, current_chunk
                chunk_index += 1
                current_chunk = []

        if current_chunk:
            yield chunk_index, current_chunk
