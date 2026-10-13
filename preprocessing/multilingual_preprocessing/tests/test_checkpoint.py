"""Unit tests for checkpointing, state management, and resume validation."""

import pytest
from src.checkpoint import CheckpointManager, FileCheckpoint


def test_checkpoint_lifecycle(tmp_path):
    mgr = CheckpointManager(tmp_path)
    filename = "train_source1.tsv"

    # Initially no checkpoint
    assert mgr.load_checkpoint(filename) is None

    # Save checkpoint
    cp = FileCheckpoint(
        input_filename=filename,
        output_filename="train_source1.tsv",
        total_input_rows=100000,
        completed_rows=50000,
        completed_chunks=1,
        chunk_size=50000,
        worker_count=1,
        pipeline_version="1.0.0",
        config_hash="abc123hash",
        timestamp=123456789.0,
        status="in_progress",
    )
    mgr.save_checkpoint(cp)

    # Load checkpoint
    loaded = mgr.load_checkpoint(filename)
    assert loaded is not None
    assert loaded.completed_rows == 50000
    assert loaded.status == "in_progress"

    # Validate resume compatibility
    out_file = tmp_path / "train_source1.tsv"
    out_file.touch()

    # Matching config
    assert mgr.validate_for_resume(
        checkpoint=loaded,
        current_config_hash="abc123hash",
        current_version="1.0.0",
        current_chunk_size=50000,
        output_file_path=out_file,
    ) is True

    # Incompatible config hash
    assert mgr.validate_for_resume(
        checkpoint=loaded,
        current_config_hash="different_hash",
        current_version="1.0.0",
        current_chunk_size=50000,
        output_file_path=out_file,
    ) is False
