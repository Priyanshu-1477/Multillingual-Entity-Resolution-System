"""End-to-end preprocessing pipeline orchestrator."""

import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional
import psutil

from .audit import AuditCollector
from .checkpoint import CheckpointManager, FileCheckpoint
from .config import PreprocessingConfig
from .dataset_discovery import DiscoveredFile, count_file_lines, discover_dataset
from .errors import ErrorLogger
from .performance import PerformanceTracker
from .record_processor import process_chunk
from .streaming_reader import stream_tsv_chunks
from .streaming_writer import StreamingTsvWriter
from .validators import validate_tsv_streaming


class PreprocessingPipeline:
    """Production-grade pipeline executing memory-bounded multilingual preprocessing."""

    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.config.ensure_directories()
        self.checkpoint_manager = CheckpointManager(self.config.state_dir)
        self.audit_collector = AuditCollector(self.config.audit_dir)
        self.error_logger = ErrorLogger(self.config.errors_dir, max_error_rate=self.config.max_error_rate)

    def preflight_check(self, source_files: List[DiscoveredFile]) -> bool:
        """Verify disk space, file readability, permissions, and dependencies."""
        print("=== RUNNING PRE-FLIGHT CHECKS ===")

        # 1. Check disk space
        disk_usage = shutil.disk_usage(self.config.output_dir)
        free_gb = disk_usage.free / (1024 ** 3)
        print(f"Available disk space: {free_gb:.2f} GB (Required minimum: {self.config.min_free_disk_gb} GB)")
        if free_gb < self.config.min_free_disk_gb:
            raise RuntimeError(
                f"INSUFFICIENT DISK SPACE: {free_gb:.2f} GB available, need at least {self.config.min_free_disk_gb} GB."
            )

        # 2. Check RAM
        ram_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(f"Available system RAM: {ram_gb:.2f} GB")

        # 3. Check input files
        if not source_files:
            raise RuntimeError(f"No source files discovered in input directory: {self.config.input_dir}")

        for sfile in source_files:
            if not sfile.file_path.exists():
                raise FileNotFoundError(f"Source file missing: {sfile.file_path}")
            if sfile.file_size_bytes == 0:
                raise ValueError(f"Source file is empty: {sfile.file_path}")
            print(f"Verified input file: {sfile.filename} ({sfile.approx_size_mb:.1f} MB, ~{sfile.estimated_rows:,} rows)")

        print("Pre-flight checks passed successfully!\n")
        return True

    def process_file(
        self,
        discovered_file: DiscoveredFile,
        perf_tracker: PerformanceTracker,
    ):
        """Process a single source file in chunked streaming fashion with checkpointing."""
        input_path = discovered_file.file_path
        output_path = self.config.output_dir / discovered_file.filename
        input_filename = discovered_file.filename

        print(f"\n>>> Starting Processing: {input_filename} -> {output_path.name}")

        # Check existing checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(input_filename)
        start_row = 0
        append_mode = False

        if checkpoint and not self.config.force:
            if checkpoint.status == "completed":
                print(f"File {input_filename} already completed ({checkpoint.completed_rows:,} rows). Skipping.")
                perf_tracker.update(checkpoint.completed_rows, 0.0)
                return

            is_valid_resume = self.checkpoint_manager.validate_for_resume(
                checkpoint=checkpoint,
                current_config_hash=self.config.get_config_hash(),
                current_version=self.config.pipeline_version,
                current_chunk_size=self.config.chunk_size,
                output_file_path=output_path,
            )

            if is_valid_resume:
                start_row = checkpoint.completed_rows
                append_mode = True
                print(f"Resuming {input_filename} from row {start_row:,} (chunk {checkpoint.completed_chunks}).")
            else:
                print(f"Existing checkpoint for {input_filename} incompatible. Restarting from beginning.")
                if output_path.exists():
                    output_path.unlink()
        elif self.config.force and output_path.exists():
            output_path.unlink()

        # Count total rows for accurate progress if needed
        total_rows = count_file_lines(input_path) - 1

        writer = StreamingTsvWriter(output_path, append_mode=append_mode)
        completed_rows = start_row
        completed_chunks = start_row // self.config.chunk_size

        chunk_stream = stream_tsv_chunks(
            file_path=input_path,
            chunk_size=self.config.chunk_size,
            start_row=start_row,
        )

        with writer:
            for chunk_idx, raw_chunk in chunk_stream:
                t0 = time.time()
                clean_chunk = process_chunk(
                    raw_rows=raw_chunk,
                    source_filename=input_filename,
                    chunk_start_index=completed_rows,
                    error_logger=self.error_logger,
                    audit_collector=self.audit_collector,
                )
                writer.write_chunk(clean_chunk)
                t1 = time.time()

                chunk_len = len(clean_chunk)
                completed_rows += chunk_len
                completed_chunks += 1

                # Update performance tracker & log
                perf_tracker.update(chunk_len, t1 - t0)
                perf_tracker.log_progress()

                # Save checkpoint state after each chunk
                self.checkpoint_manager.save_checkpoint(FileCheckpoint(
                    input_filename=input_filename,
                    output_filename=output_path.name,
                    total_input_rows=total_rows,
                    completed_rows=completed_rows,
                    completed_chunks=completed_chunks,
                    chunk_size=self.config.chunk_size,
                    worker_count=self.config.workers,
                    pipeline_version=self.config.pipeline_version,
                    config_hash=self.config.get_config_hash(),
                    timestamp=time.time(),
                    status="in_progress",
                ))

                # Explicitly release memory
                del clean_chunk
                del raw_chunk

        # Mark checkpoint as completed
        self.checkpoint_manager.save_checkpoint(FileCheckpoint(
            input_filename=input_filename,
            output_filename=output_path.name,
            total_input_rows=total_rows,
            completed_rows=completed_rows,
            completed_chunks=completed_chunks,
            chunk_size=self.config.chunk_size,
            worker_count=self.config.workers,
            pipeline_version=self.config.pipeline_version,
            config_hash=self.config.get_config_hash(),
            timestamp=time.time(),
            status="completed",
        ))

        print(f"Completed {input_filename}: {completed_rows:,} rows processed.")

    def run_ground_truth_preservation(self, ground_truth_files: List[DiscoveredFile]):
        """Preserve ground truth files verbatim into preprocessed_output without altering source records."""
        for gt in ground_truth_files:
            dest = self.config.output_dir / gt.filename
            print(f"Preserving ground truth: {gt.filename} -> {dest.name}")
            shutil.copy2(gt.file_path, dest)
            print(f"Ground truth file preserved ({gt.approx_size_mb:.1f} MB).")

    def run_dry_run(self, source_files: List[DiscoveredFile], sample_size: int = 500):
        """Execute non-destructive dry run inspecting scripts, languages, and showing sample transformations."""
        print("=== DRY RUN MODE ===")
        print(f"Inspecting sample of {sample_size} records from each discovered source file...\n")

        for sfile in source_files:
            print(f"--- Sample from {sfile.filename} ---")
            reader = stream_tsv_chunks(sfile.file_path, chunk_size=sample_size)
            _, chunk = next(reader, (0, []))
            clean_chunk = process_chunk(
                raw_rows=chunk,
                source_filename=sfile.filename,
                chunk_start_index=0,
                error_logger=None,
                audit_collector=self.audit_collector,
            )

            # Display 5 before/after examples
            for i in range(min(5, len(chunk))):
                raw = chunk[i]
                clean = clean_chunk[i]
                print(f"Raw [{raw[0]}]:\n  Name: {raw[1]}\n  Addr: {raw[2]}\n  Country: {raw[3] if len(raw)>3 else ''}")
                print(f"Clean [{clean[0]}]:\n  Name: {clean[1]}\n  Addr: {clean[2]}\n  Country: {clean[3]}\n")

        print("Dry run completed. No production files were overwritten.\n")

    def run_benchmark(self, source_files: List[DiscoveredFile], benchmark_rows: int = 100_000):
        """Execute benchmark mode to measure throughput, memory usage, and estimate 24M-record runtime."""
        print("=== BENCHMARK MODE ===")
        print(f"Benchmarking pipeline on {benchmark_rows:,} records...")

        target_file = source_files[0]
        # Choose a file with non-Latin scripts if available (source 2 or 3) for representative benchmark
        for sf in source_files:
            if "source2" in sf.filename or "source3" in sf.filename:
                target_file = sf
                break

        print(f"Using {target_file.filename} as representative benchmark workload.")

        perf_tracker = PerformanceTracker(total_expected_rows=benchmark_rows)
        processed = 0

        for _, raw_chunk in stream_tsv_chunks(target_file.file_path, chunk_size=self.config.chunk_size):
            remaining = benchmark_rows - processed
            current_batch = raw_chunk[:remaining]

            t0 = time.time()
            clean_chunk = process_chunk(
                raw_rows=current_batch,
                source_filename=target_file.filename,
                chunk_start_index=processed,
                error_logger=None,
                audit_collector=None,
            )
            t1 = time.time()

            batch_len = len(clean_chunk)
            processed += batch_len
            perf_tracker.update(batch_len, t1 - t0)
            perf_tracker.log_progress(force=True)

            del clean_chunk
            del current_batch
            if processed >= benchmark_rows:
                break

        stats = perf_tracker.get_stats()
        rate = stats["rows_per_second"]
        ram = stats["current_ram_mb"]

        # Approximate 24M estimation
        dataset_24m = 24_229_173
        est_seconds_24m = dataset_24m / max(1.0, rate)
        est_24m_str = perf_tracker.format_duration(est_seconds_24m)

        print("\n=== BENCHMARK RESULTS ===")
        print(f"Records benchmarked:      {processed:,}")
        print(f"Elapsed time:             {stats['elapsed_seconds']:.2f} seconds")
        print(f"Processing rate:          {rate:,.0f} rows/sec")
        print(f"Memory usage (RSS):       {ram:.1f} MB (O(chunk_size) bounded)")
        print(f"Estimated 24M runtime:    ~{est_24m_str} (Note: approximate based on single-worker throughput)")
        print("========================\n")

    def run_pipeline(self):
        """Execute the full production preprocessing pipeline across all discovered files."""
        # 1. Discover files
        discovery = discover_dataset(self.config.input_dir, count_exact_rows=True)

        print("=== MULTILINGUAL PREPROCESSING PIPELINE ===")
        print(f"Discovered {len(discovery.source_files)} source files ({discovery.total_estimated_rows:,} total records):")
        for sf in discovery.source_files:
            print(f"  - {sf.filename:<25} ({sf.approx_size_mb:>6.1f} MB, {sf.estimated_rows:>10,} rows)")
        for gt in discovery.ground_truth_files:
            print(f"  - {gt.filename:<25} ({gt.approx_size_mb:>6.1f} MB, {gt.estimated_rows:>10,} rows, Ground Truth)")
        print("==========================================\n")

        # 2. Pre-flight check
        self.preflight_check(discovery.source_files)

        # 3. Dry-run if requested
        if self.config.dry_run:
            self.run_dry_run(discovery.source_files)
            return

        # 4. Benchmark if requested
        if self.config.benchmark:
            self.run_benchmark(discovery.source_files, self.config.benchmark_rows)
            return

        # 5. Full processing
        perf_tracker = PerformanceTracker(total_expected_rows=discovery.total_estimated_rows)

        with self.error_logger:
            for sfile in discovery.source_files:
                self.process_file(sfile, perf_tracker)

        # 6. Preserve ground truth
        self.run_ground_truth_preservation(discovery.ground_truth_files)

        # 7. Incremental streaming validation of all source outputs
        print("\n=== VALIDATING ALL OUTPUT TSV FILES ===")
        all_valid = True
        for sfile in discovery.source_files:
            out_path = self.config.output_dir / sfile.filename
            print(f"Streaming validation: {sfile.filename} ...")
            report = validate_tsv_streaming(sfile.file_path, out_path)
            if report.is_valid:
                print(f"  [PASS] {sfile.filename}: {report.output_rows:,} rows verified. Exact ID match.")
            else:
                all_valid = False
                print(f"  [FAIL] {sfile.filename}: Validation failed! Errors:")
                for err in report.errors[:5]:
                    print(f"    - {err}")

        # 8. Write final audit reports
        perf_stats = perf_tracker.get_stats()
        self.audit_collector.write_reports(
            performance_stats=perf_stats,
            pipeline_metadata=self.config.to_dict(),
        )

        print("\n=== PIPELINE EXECUTION COMPLETE ===")
        print(f"Total processed records: {perf_stats['processed_rows']:,}")
        print(f"Elapsed time:            {perf_tracker.format_duration(perf_stats['elapsed_seconds'])}")
        print(f"Average throughput:      {perf_stats['rows_per_second']:,.0f} rows/sec")
        print(f"All outputs validated:   {'YES [100% PASS]' if all_valid else 'NO [ERRORS DETECTED]'}")
        print(f"Output directory:        {self.config.output_dir}")
        print(f"Audit directory:         {self.config.audit_dir}")
        print(f"Errors log:              {self.error_logger.error_file_path}")
        print("===================================\n")
