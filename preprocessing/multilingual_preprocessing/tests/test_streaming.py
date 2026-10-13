"""Unit tests for streaming reader, writer, and bounded memory chunks."""

import csv
import pytest
from src.streaming_reader import stream_tsv_chunks
from src.streaming_writer import StreamingTsvWriter


def test_streaming_reader_and_writer(tmp_path):
    test_file = tmp_path / "test_input.tsv"
    num_rows = 120

    # Write test input
    with open(test_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["entity_id", "business_name", "business_address", "country"])
        for i in range(num_rows):
            writer.writerow([f"S1-{i:06d}", f"Business {i}", f"{i} Main St", "US"])

    # Stream chunks of 50
    chunks = list(stream_tsv_chunks(test_file, chunk_size=50))
    assert len(chunks) == 3  # 50, 50, 20
    assert len(chunks[0][1]) == 50
    assert len(chunks[1][1]) == 50
    assert len(chunks[2][1]) == 20

    # Test leading zeros preserved
    assert chunks[0][1][0][0] == "S1-000000"

    # Test StreamingTsvWriter
    out_file = tmp_path / "test_output.tsv"
    with StreamingTsvWriter(out_file) as out_writer:
        for _, chunk_rows in chunks:
            out_writer.write_chunk(chunk_rows)

    # Verify output file
    with open(out_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        assert header == ["entity_id", "business_name", "business_address", "country"]
        rows = list(reader)
        assert len(rows) == num_rows
        assert rows[0][0] == "S1-000000"
