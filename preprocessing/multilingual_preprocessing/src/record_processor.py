"""Single record and batch processing logic."""

from typing import List, Optional, Tuple
from .address_normalizer import normalize_address
from .audit import AuditCollector
from .country_normalizer import normalize_country
from .errors import ErrorLogger, PreprocessingError
from .language_detector import detect_probable_language
from .name_normalizer import normalize_business_name
from .script_detector import detect_scripts


def process_record(
    raw_row: List[str],
    source_filename: str,
    row_index: int,
    error_logger: Optional[ErrorLogger] = None,
    audit_collector: Optional[AuditCollector] = None,
) -> List[str]:
    """Process a single raw business record into the canonical schema.

    Returns:
        [entity_id, business_name, business_address, country]
    """
    # Defensive row parsing
    num_cols = len(raw_row)
    if num_cols >= 4:
        entity_id = raw_row[0].strip()
        raw_name = raw_row[1]
        raw_addr = raw_row[2]
        raw_country = raw_row[3]
    elif num_cols == 3:
        entity_id = raw_row[0].strip()
        raw_name = raw_row[1]
        raw_addr = raw_row[2]
        raw_country = ""
    elif num_cols == 2:
        entity_id = raw_row[0].strip()
        raw_name = raw_row[1]
        raw_addr = ""
        raw_country = ""
    elif num_cols == 1:
        entity_id = raw_row[0].strip()
        raw_name = ""
        raw_addr = ""
        raw_country = ""
    else:
        entity_id = f"UNKNOWN_{row_index}"
        raw_name = ""
        raw_addr = ""
        raw_country = ""

    # Process business name
    try:
        clean_name = normalize_business_name(raw_name)
    except Exception as e:
        clean_name = raw_name.strip()
        if error_logger:
            error_logger.log_error(PreprocessingError(
                entity_id=entity_id,
                source_file=source_filename,
                row_index=row_index,
                field_name="business_name",
                error_type=type(e).__name__,
                error_message=str(e),
                raw_value=raw_name,
            ))

    # Process business address
    try:
        clean_addr = normalize_address(raw_addr)
    except Exception as e:
        clean_addr = raw_addr.strip()
        if error_logger:
            error_logger.log_error(PreprocessingError(
                entity_id=entity_id,
                source_file=source_filename,
                row_index=row_index,
                field_name="business_address",
                error_type=type(e).__name__,
                error_message=str(e),
                raw_value=raw_addr,
            ))

    # Process country
    try:
        clean_country = normalize_country(raw_country)
    except Exception as e:
        clean_country = raw_country.strip()
        if error_logger:
            error_logger.log_error(PreprocessingError(
                entity_id=entity_id,
                source_file=source_filename,
                row_index=row_index,
                field_name="country",
                error_type=type(e).__name__,
                error_message=str(e),
                raw_value=raw_country,
            ))

    # Audit tracking (sampling/counter aggregation)
    if audit_collector:
        combined_text = f"{raw_name} {raw_addr}"
        detected_scripts = detect_scripts(combined_text) or {"Latin"}
        probable_lang = detect_probable_language(raw_name, clean_country)
        audit_collector.record_metrics(
            source_file=source_filename,
            entity_id=entity_id,
            raw_name=raw_name,
            clean_name=clean_name,
            raw_addr=raw_addr,
            clean_addr=clean_addr,
            raw_country=raw_country,
            clean_country=clean_country,
            detected_scripts=detected_scripts,
            probable_lang=probable_lang,
        )

    return [entity_id, clean_name, clean_addr, clean_country]


def process_chunk(
    raw_rows: List[List[str]],
    source_filename: str,
    chunk_start_index: int,
    error_logger: Optional[ErrorLogger] = None,
    audit_collector: Optional[AuditCollector] = None,
) -> List[List[str]]:
    """Process an entire chunk of rows with bounded memory and error tracking."""
    processed = []
    for i, raw_row in enumerate(raw_rows):
        row_idx = chunk_start_index + i
        clean_row = process_record(
            raw_row=raw_row,
            source_filename=source_filename,
            row_index=row_idx,
            error_logger=error_logger,
            audit_collector=audit_collector,
        )
        processed.append(clean_row)

    if error_logger:
        error_logger.update_processed_count(len(raw_rows))

    return processed
