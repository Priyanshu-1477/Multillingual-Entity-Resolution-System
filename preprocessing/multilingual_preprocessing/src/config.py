"""Configuration management for multilingual preprocessing pipeline."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PreprocessingConfig:
    """Pipeline configuration parameters with strict type safety and defaults."""

    input_dir: Path = Path("actual_data")
    output_dir: Path = Path("preprocessed_output")
    state_dir: Path = Path("preprocessing_state")
    temp_dir: Path = Path("temporary")

    chunk_size: int = 50_000
    workers: int = 1
    max_error_rate: float = 0.01  # Stop if error rate exceeds 1%

    resume: bool = True
    force: bool = False
    dry_run: bool = False
    benchmark: bool = False
    benchmark_rows: int = 100_000

    min_free_disk_gb: float = 5.0
    pipeline_version: str = "1.0.0"

    # Internal directories derived from output_dir
    audit_dir: Path = field(init=False)
    errors_dir: Path = field(init=False)

    def __post_init__(self):
        # Convert string paths to Path objects if necessary
        self.input_dir = Path(self.input_dir)
        self.output_dir = Path(self.output_dir)
        self.state_dir = Path(self.state_dir)
        self.temp_dir = Path(self.temp_dir)

        self.audit_dir = self.output_dir / "audit"
        self.errors_dir = self.output_dir / "errors"

        # Validate chunk size
        if self.chunk_size <= 0:
            raise ValueError(f"Invalid chunk_size: {self.chunk_size}. Must be positive.")

        if self.max_error_rate <= 0 or self.max_error_rate > 1.0:
            raise ValueError(f"Invalid max_error_rate: {self.max_error_rate}. Must be in (0, 1].")

        if self.workers < 1:
            raise ValueError(f"Workers must be >= 1, got {self.workers}.")

    def ensure_directories(self):
        """Create necessary output, state, temporary, audit, and error directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.errors_dir.mkdir(parents=True, exist_ok=True)

    def get_config_hash(self) -> str:
        """Return deterministic SHA256 hash representing transformation configuration."""
        canonical = {
            "chunk_size": self.chunk_size,
            "max_error_rate": self.max_error_rate,
            "pipeline_version": self.pipeline_version,
        }
        serialized = json.dumps(canonical, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        """Convert config to dictionary representation for logging and audits."""
        data = asdict(self)
        for k, v in data.items():
            if isinstance(v, Path):
                data[k] = str(v)
        data["config_hash"] = self.get_config_hash()
        return data
