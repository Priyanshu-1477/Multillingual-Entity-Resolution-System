import csv
import os
import shutil
import sys
import time
from pathlib import Path

# Add project root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import psutil
import pytest

from src.config import PreprocessingConfig
from src.pipeline import PreprocessingPipeline
from src.validators import validate_tsv_streaming

SYNTHETIC_TEMPLATES = [
    # (name, address, country)
    ("Boulangerie Dupont", "12 Rue de Rivoli, Paris 75001", "France"),
    ("शिव शक्ति जनरल स्टोर", "Shop 004, Main Bazar Road, Delhi 110006", "India"),
    ("Farmacia San José", "Calle Mayor 5, Madrid 28013", "Spain"),
    ("ABC मोबाइल Store", "Plot 44, Bengaluru, Karnataka 560001", "IN"),
    ("Acme Corporation LLC", "100 Broadway, New York, NY 10001", "USA"),
    ("-- Holloway Peak Inc Seafood", "105 ELM ST, MORGANTON, NC 28655", "US"),
    ("Fractales Amis Groupe S.A.S", "23 Rue Icmre, Gironde 33260", "FRA"),
    ("શ્રી ગણેશ મોબાઇલ સેન્ટર", "CRYSTAL PLAZA, VADODARA, ગુજરાત 390001", "IND"),
    ("শিব শক্তি জেনারেল স্টোর", "Park Street, Kolkata, West Bengal 700016", "India"),
    ("ஸ்ரீ கணேஷ் மொபைல்", "Anna Salai, Chennai, Tamil Nadu 600002", "India"),
    ("శ్రీ గణేష్ ఎంటర్‌ప్రైజెస్", "Banjara Hills, Hyderabad, Telangana 500034", "India"),
    ("متجر النور التجاري", "14 Avenue des Champs-Élysées, Paris", "France"),
    ("北京宏达贸易有限公司", "Chaoyang District, Beijing", "China"),
    ("LLC Moncada Léarning Center", "5780 Fawn Ct, Fort Worth, Texas 76101", "US"),
    ("Pvt. EFS Print Ventures Ltd.", "Door No 183, 41St Cross, Bangalore, ಕರ್ನಾಟಕ", "India"),
    ("Nagpur Financial Corp", "H.NO 4/A, NAGPUR, महाराष्ट्र 440001", "India"),
    ("Chordia & Partners Company", "1038 SECTOR 9, FARIDABAD, हरियाणा 121006", "India"),
    ("Crystal Staffing Solutions LLC", "8706 Kentucky Derby Drive, Waxhaw, NC", "US"),
    ("Dick Regional Armada", "33466 WARWICK HILLS ROAD, <NULL>, YUCAIPA, CA", "US"),
    ("", "", "US"),  # Empty record edge case
]


def generate_synthetic_stress_file(file_path: Path, num_records: int = 1_000_000):
    """Streamingly write 1,000,000 synthetic multilingual records to disk."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_templates = len(SYNTHETIC_TEMPLATES)

    print(f"Generating {num_records:,} synthetic records into {file_path.name}...")
    t0 = time.time()
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["entity_id", "business_name", "business_address", "country"])

        buf = []
        for i in range(num_records):
            eid = f"SYNTH-{i:08d}"
            name, addr, ctry = SYNTHETIC_TEMPLATES[i % num_templates]
            buf.append([eid, name, addr, ctry])

            if len(buf) >= 50_000:
                writer.writerows(buf)
                buf = []

        if buf:
            writer.writerows(buf)

    print(f"Generated {num_records:,} records in {time.time() - t0:.2f}s ({file_path.stat().st_size / (1024*1024):.1f} MB).")


def run_1m_stress_test(base_dir: Path):
    """Run full 1,000,000 record stress test verifying bounded memory and exact validation."""
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    state_dir = base_dir / "state"
    temp_dir = base_dir / "temp"

    input_file = input_dir / "stress_1m.tsv"
    generate_synthetic_stress_file(input_file, num_records=1_000_000)

    config = PreprocessingConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        state_dir=state_dir,
        temp_dir=temp_dir,
        chunk_size=50_000,
    )

    proc = psutil.Process(os.getpid())
    ram_before = proc.memory_info().rss / (1024 * 1024)

    t0 = time.time()
    pipeline = PreprocessingPipeline(config)
    pipeline.run_pipeline()
    t1 = time.time()

    ram_after = proc.memory_info().rss / (1024 * 1024)
    ram_delta = ram_after - ram_before

    print(f"\n1M Stress Test Performance:")
    print(f"  Duration: {t1 - t0:.2f} seconds")
    print(f"  Throughput: {1_000_000 / (t1 - t0):,.0f} rows/second")
    print(f"  RAM Before: {ram_before:.1f} MB, RAM After: {ram_after:.1f} MB (Delta: {ram_delta:+.1f} MB)")

    # Assert bounded memory: RAM delta must remain well below 500 MB (bounded by O(chunk_size))
    assert ram_after < 500, f"RAM usage too high: {ram_after:.1f} MB"

    # Streaming validation
    out_file = output_dir / "stress_1m.tsv"
    report = validate_tsv_streaming(input_file, out_file)
    assert report.is_valid is True, f"Validation errors: {report.errors}"
    assert report.input_rows == 1_000_000
    assert report.output_rows == 1_000_000
    assert report.entity_ids_matched is True


if __name__ == "__main__":
    stress_dir = Path("temporary/stress_test_run")
    stress_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_1m_stress_test(stress_dir)
        print("\n>>> 1,000,000 RECORD STRESS TEST PASSED PERFECTLY! <<<")
    finally:
        shutil.rmtree(stress_dir, ignore_errors=True)
