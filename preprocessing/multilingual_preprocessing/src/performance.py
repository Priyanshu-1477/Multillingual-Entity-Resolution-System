"""Performance monitoring and progress reporting module."""

import os
import time
from typing import Optional
import psutil


class PerformanceTracker:
    """Tracks elapsed time, memory usage, processing rate, and estimates ETA."""

    def __init__(self, total_expected_rows: Optional[int] = None):
        self.total_expected_rows = total_expected_rows
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.processed_rows = 0
        self.chunk_times = []
        self._process = psutil.Process(os.getpid())

    def get_memory_mb(self) -> float:
        """Return current resident set size (RSS) memory in megabytes."""
        try:
            return round(self._process.memory_info().rss / (1024 * 1024), 2)
        except Exception:
            return 0.0

    def update(self, chunk_rows: int, chunk_duration: float):
        """Update tracker with newly completed chunk."""
        self.processed_rows += chunk_rows
        self.chunk_times.append(chunk_duration)
        if len(self.chunk_times) > 50:
            self.chunk_times.pop(0)

    def get_stats(self) -> dict:
        """Return dictionary of current performance metrics."""
        elapsed = max(1e-6, time.time() - self.start_time)
        rate = self.processed_rows / elapsed
        ram_mb = self.get_memory_mb()

        eta_seconds = 0
        if self.total_expected_rows and self.total_expected_rows > self.processed_rows and rate > 0:
            eta_seconds = (self.total_expected_rows - self.processed_rows) / rate

        return {
            "processed_rows": self.processed_rows,
            "total_expected_rows": self.total_expected_rows,
            "elapsed_seconds": round(elapsed, 2),
            "rows_per_second": round(rate, 1),
            "current_ram_mb": ram_mb,
            "eta_seconds": round(eta_seconds, 1),
        }

    def format_duration(self, seconds: float) -> str:
        """Format seconds into human-readable duration (e.g., 4m 32s)."""
        seconds = int(seconds)
        mins, secs = divmod(seconds, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours}h {mins:02d}m {secs:02d}s"
        return f"{mins}m {secs:02d}s"

    def log_progress(self, force: bool = False, min_interval_sec: float = 3.0):
        """Periodically print structured progress update without flooding console."""
        now = time.time()
        if not force and (now - self.last_log_time) < min_interval_sec:
            return

        self.last_log_time = now
        stats = self.get_stats()
        elapsed_str = self.format_duration(stats["elapsed_seconds"])
        rate_str = f"{stats['rows_per_second']:,.0f} rows/sec"
        ram_str = f"RAM: {stats['current_ram_mb']:.1f} MB"

        if self.total_expected_rows:
            pct = (self.processed_rows / self.total_expected_rows) * 100
            eta_str = f"ETA: {self.format_duration(stats['eta_seconds'])}"
            print(
                f"Processed: {self.processed_rows:,} / {self.total_expected_rows:,} "
                f"({pct:5.1f}%) | Rate: {rate_str:<15} | Elapsed: {elapsed_str} | {eta_str} | {ram_str}"
            )
        else:
            print(
                f"Processed: {self.processed_rows:,} | Rate: {rate_str:<15} | Elapsed: {elapsed_str} | {ram_str}"
            )
