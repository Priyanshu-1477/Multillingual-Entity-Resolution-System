#!/usr/bin/env python3
"""Main CLI entrypoint for Amazon ML Challenge 2026 Multilingual Preprocessing Pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from src.config import PreprocessingConfig
from src.dataset_discovery import discover_dataset
from src.pipeline import PreprocessingPipeline
from src.validators import validate_tsv_streaming


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production Multilingual Preprocessing Pipeline for Amazon ML Challenge 2026."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("actual_data"),
        help="Input directory containing raw TSV source files (default: actual_data)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("preprocessed_output"),
        help="Output directory for cleaned TSV files (default: preprocessed_output)",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=50_000,
        help="Chunk size for streaming processing (default: 50000; allowed: 10000, 25000, 50000, 100000, 250000)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=1,
        help="Worker process count (default: 1)",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("preprocessing_state"),
        help="Directory to store checkpoint and resume state (default: preprocessing_state)",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=Path("temporary"),
        help="Directory for temporary files (default: temporary)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute dry run inspecting schema and sample without full processing",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark mode to measure throughput, RAM, and estimate full runtime",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100_000,
        help="Number of rows for benchmark or dry run (default: 100000)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable automatic checkpoint resumption",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing outputs instead of skipping or resuming",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run incremental streaming validation against existing output files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = PreprocessingConfig(
        input_dir=args.input,
        output_dir=args.output,
        state_dir=args.state_dir,
        temp_dir=args.temp_dir,
        chunk_size=args.chunk_size,
        workers=args.workers,
        resume=not args.no_resume,
        force=args.force,
        dry_run=args.dry_run,
        benchmark=args.benchmark,
        benchmark_rows=args.rows,
    )

    # Validate-only mode
    if args.validate_only:
        print("=== VALIDATION-ONLY MODE ===")
        discovery = discover_dataset(config.input_dir)
        all_passed = True
        for sf in discovery.source_files:
            out_file = config.output_dir / sf.filename
            report = validate_tsv_streaming(sf.file_path, out_file)
            if report.is_valid:
                print(f"  [PASS] {sf.filename}: {report.output_rows:,} rows verified.")
            else:
                all_passed = False
                print(f"  [FAIL] {sf.filename}:")
                for err in report.errors:
                    print(f"    - {err}")
        sys.exit(0 if all_passed else 1)

    pipeline = PreprocessingPipeline(config)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
