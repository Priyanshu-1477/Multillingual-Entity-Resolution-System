"""Checkpointing and resumability management module."""

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class FileCheckpoint:
    """State record for an individual dataset source file."""

    input_filename: str
    output_filename: str
    total_input_rows: int
    completed_rows: int
    completed_chunks: int
    chunk_size: int
    worker_count: int
    pipeline_version: str
    config_hash: str
    timestamp: float
    status: str  # 'not_started', 'in_progress', 'completed', 'failed'
    last_error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FileCheckpoint":
        return cls(**data)


class CheckpointManager:
    """Manages reading, writing, and validating checkpoints in state_dir."""

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _get_checkpoint_path(self, input_filename: str) -> Path:
        stem = Path(input_filename).stem
        return self.state_dir / f"{stem}_checkpoint.json"

    def load_checkpoint(self, input_filename: str) -> Optional[FileCheckpoint]:
        """Load existing checkpoint file if present."""
        cp_path = self._get_checkpoint_path(input_filename)
        if not cp_path.exists():
            return None
        try:
            with open(cp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return FileCheckpoint.from_dict(data)
        except Exception:
            return None

    def save_checkpoint(self, checkpoint: FileCheckpoint):
        """Save checkpoint state atomically to disk."""
        cp_path = self._get_checkpoint_path(checkpoint.input_filename)
        temp_path = cp_path.with_suffix(".tmp")
        checkpoint.timestamp = time.time()
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        temp_path.replace(cp_path)

    def validate_for_resume(
        self,
        checkpoint: FileCheckpoint,
        current_config_hash: str,
        current_version: str,
        current_chunk_size: int,
        output_file_path: Path,
    ) -> bool:
        """Validate whether an existing checkpoint is safe to resume.

        Ensures:
        - Configuration hash matches
        - Pipeline version matches
        - Chunk size matches
        - Output file exists on disk
        """
        if checkpoint.config_hash != current_config_hash:
            return False
        if checkpoint.pipeline_version != current_version:
            return False
        if checkpoint.chunk_size != current_chunk_size:
            return False
        if not output_file_path.exists():
            return False

        return True

    def remove_checkpoint(self, input_filename: str):
        """Delete checkpoint file (e.g., when forced overwrite is requested)."""
        cp_path = self._get_checkpoint_path(input_filename)
        if cp_path.exists():
            cp_path.unlink()
