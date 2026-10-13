"""Bounded-memory audit reporting module."""

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class AuditCollector:
    """Collects aggregate counters and bounded representative examples in O(1) memory."""

    def __init__(self, audit_dir: Path, max_examples_per_category: int = 20):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.max_examples = max_examples_per_category

        # Aggregate counters
        self.total_records = 0
        self.total_transliterated_names = 0
        self.total_transliterated_addresses = 0
        self.script_counts: Counter = Counter()
        self.language_counts: Counter = Counter()
        self.country_counts: Counter = Counter()
        self.source_file_counts: Counter = Counter()

        # Bounded examples reservoir: category -> list of (raw, processed) tuples
        self.examples: Dict[str, List[Dict[str, str]]] = {
            "pure_latin": [],
            "devanagari": [],
            "gujarati": [],
            "bengali": [],
            "tamil": [],
            "telugu": [],
            "kannada": [],
            "malayalam": [],
            "gurmukhi": [],
            "odia": [],
            "arabic": [],
            "cjk": [],
            "mixed_script": [],
            "accented_latin": [],
        }

    def record_metrics(
        self,
        source_file: str,
        entity_id: str,
        raw_name: str,
        clean_name: str,
        raw_addr: str,
        clean_addr: str,
        raw_country: str,
        clean_country: str,
        detected_scripts: set,
        probable_lang: str,
    ):
        """Update aggregate counters and capture bounded examples."""
        self.total_records += 1
        self.source_file_counts[source_file] += 1
        self.country_counts[clean_country or "Unknown"] += 1
        self.language_counts[probable_lang] += 1

        for s in detected_scripts:
            self.script_counts[s] += 1

        if raw_name != clean_name:
            self.total_transliterated_names += 1
        if raw_addr != clean_addr:
            self.total_transliterated_addresses += 1

        # Determine category for bounded examples
        category = None
        if len(detected_scripts) > 1 and "Latin" in detected_scripts:
            category = "mixed_script"
        elif "Devanagari" in detected_scripts:
            category = "devanagari"
        elif "Gujarati" in detected_scripts:
            category = "gujarati"
        elif "Bengali" in detected_scripts:
            category = "bengali"
        elif "Tamil" in detected_scripts:
            category = "tamil"
        elif "Telugu" in detected_scripts:
            category = "telugu"
        elif "Kannada" in detected_scripts:
            category = "kannada"
        elif "Malayalam" in detected_scripts:
            category = "malayalam"
        elif "Gurmukhi" in detected_scripts:
            category = "gurmukhi"
        elif "Odia" in detected_scripts:
            category = "odia"
        elif "Arabic" in detected_scripts:
            category = "arabic"
        elif "CJK" in detected_scripts:
            category = "cjk"
        elif raw_name != clean_name and detected_scripts == {"Latin"}:
            category = "accented_latin"
        elif detected_scripts == {"Latin"}:
            category = "pure_latin"

        if category and len(self.examples.get(category, [])) < self.max_examples:
            self.examples.setdefault(category, []).append({
                "entity_id": entity_id,
                "category": category,
                "raw_name": raw_name,
                "clean_name": clean_name,
                "raw_address": raw_addr,
                "clean_address": clean_addr,
                "country": clean_country,
            })

    def write_reports(self, performance_stats: Optional[dict] = None, pipeline_metadata: Optional[dict] = None):
        """Flush final JSON summary reports and examples TSV to audit directory."""
        # 1. script_statistics.json
        with open(self.audit_dir / "script_statistics.json", "w", encoding="utf-8") as f:
            json.dump(dict(self.script_counts.most_common()), f, indent=2)

        # 2. language_statistics.json
        with open(self.audit_dir / "language_statistics.json", "w", encoding="utf-8") as f:
            json.dump(dict(self.language_counts.most_common()), f, indent=2)

        # 3. country_statistics.json
        with open(self.audit_dir / "country_statistics.json", "w", encoding="utf-8") as f:
            json.dump(dict(self.country_counts.most_common()), f, indent=2)

        # 4. performance_statistics.json
        if performance_stats:
            with open(self.audit_dir / "performance_statistics.json", "w", encoding="utf-8") as f:
                json.dump(performance_stats, f, indent=2)

        # 5. examples.tsv
        examples_file = self.audit_dir / "examples.tsv"
        with open(examples_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator="\n")
            writer.writerow([
                "category",
                "entity_id",
                "raw_name",
                "clean_name",
                "raw_address",
                "clean_address",
                "country"
            ])
            for cat, items in self.examples.items():
                for item in items:
                    writer.writerow([
                        item["category"],
                        item["entity_id"],
                        item["raw_name"],
                        item["clean_name"],
                        item["raw_address"],
                        item["clean_address"],
                        item["country"],
                    ])

        # 6. preprocessing_summary.json
        summary = {
            "metadata": pipeline_metadata or {},
            "total_records_processed": self.total_records,
            "transliterated_names_count": self.total_transliterated_names,
            "transliterated_addresses_count": self.total_transliterated_addresses,
            "source_file_counts": dict(self.source_file_counts),
            "top_scripts": dict(self.script_counts.most_common(10)),
            "top_languages": dict(self.language_counts.most_common(10)),
            "country_distribution": dict(self.country_counts.most_common(10)),
        }
        with open(self.audit_dir / "preprocessing_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
