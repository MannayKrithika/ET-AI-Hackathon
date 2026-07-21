"""
Industrial Entity Extraction — new module (added on top of Step 11).

Purpose
-------
Whenever a document is uploaded, pull out the structured industrial
entities it contains (Asset IDs, Equipment names, Maintenance IDs, dates,
etc.) and save them as clean JSON under backend/entities/.

This is deliberately SEPARATE from entity_extractor.py:
  - entity_extractor.py calls Gemini and is scoped to a small generic
    vocabulary (Equipment, Status, Operator, ...). It runs over free text
    (chat answers, arbitrary strings) via /extract_entities, /ask,
    /build_graph, /compliance.
  - This module is rule-based / regex-based (no LLM calls, no added
    latency), and is scoped to the 9 specific industrial document types
    below with their own field lists. It runs automatically once per
    uploaded document, right after text extraction.

Nothing in the existing pipeline (upload, chunking, embeddings, FAISS,
Gemini RAG, entity_extractor.py) is modified or removed by this module.

Flow
----
    extract_industrial_entities(original_path, extracted_text_path)
      -> classify_document_type(...)          filename keywords first,
                                                column-name overlap as a
                                                fallback for spreadsheets
      -> _extract_tabular(...)  (xlsx/csv)     one structured record per row
         or
      -> _extract_pdf(...)     (pdf/docx)      one structured record for
                                                the whole document, regex/
                                                line-based over the text
      -> {"document": ..., "document_type": ..., "entities": ...}
"""

import json
import re
from pathlib import Path

import pandas as pd

from tabular_utils import load_sheet_with_detected_header
from text_cleaning import clean_text_encoding

# ---------------------------------------------------------------------------
# Document type classification
# ---------------------------------------------------------------------------

# Order matters: checked top-to-bottom, first keyword match wins. Keeps
# "Breakdown History" distinct from the more general "Maintenance Log",
# unlike chunking.py's guess_document_type() which lumps them together —
# this module needs the finer-grained split because each type has its own
# field schema below.
FILENAME_TYPE_KEYWORDS = [
    (["asset_register", "asset register", "asset_master", "asset list"], "Asset Register"),
    (["breakdown"], "Breakdown History"),
    (["preventive", "pm_schedule", "pm schedule"], "Preventive Maintenance Schedule"),
    (["inspection"], "Inspection Report"),
    (["maintenance", "maintenance_log", "maintenance log"], "Maintenance Log"),
    (["spare", "spare_parts", "inventory"], "Spare Parts Inventory"),
    (["daily", "operating_log", "operating log", "daily_log"], "Daily Operating Log"),
    (["manual"], "Equipment Manual"),
    (["sop", "procedure", "safety"], "Safety Procedure"),
]

# Canonical field -> possible source column names (normalized: lowercase,
# spaces/underscores interchangeable). Used both to classify a spreadsheet
# (best alias-overlap wins) and to pull the right column for each field.
TABULAR_SCHEMAS: dict[str, dict[str, list[str]]] = {
    "Asset Register": {
        "Asset_ID": ["asset_id", "asset_no", "asset_number", "tag_id"],
        "Equipment_Name": ["equipment_name", "equipment", "asset_name"],
        "Category": ["category", "asset_category", "type"],
        "Manufacturer": ["manufacturer", "make", "oem"],
        "Model_Number": ["model_number", "model", "model_no"],
        "Serial_Number": ["serial_number", "serial_no", "s_n", "sn"],
        "Location": ["location", "plant_location", "site"],
        "Department": ["department", "dept"],
        "Installation_Date": ["installation_date", "install_date", "date_installed", "commissioning_date"],
        "Current_Status": ["current_status", "status"],
        "Condition": ["condition", "asset_condition"],
        "Responsible_Engineer": ["responsible_engineer", "engineer", "owner", "responsible_person"],
    },
    "Maintenance Log": {
        "Maintenance_ID": ["maintenance_id", "log_id", "work_order_id", "wo_id"],
        "Asset_ID": ["asset_id", "equipment_id", "tag_id"],
        "Equipment_Name": ["equipment_name", "equipment", "asset_name"],
        "Maintenance_Type": ["maintenance_type", "type", "work_type"],
        "Maintenance_Date": ["maintenance_date", "date", "service_date"],
        "Technician": ["technician", "engineer", "performed_by"],
        "Issue_Reported": ["issue_reported", "issue", "problem", "complaint"],
        "Action_Taken": ["action_taken", "action", "work_done"],
        "Parts_Replaced": ["parts_replaced", "parts_used", "spares_used"],
        "Downtime": ["downtime", "downtime_hours", "total_downtime_hours"],
        "Status": ["status"],
    },
    "Preventive Maintenance Schedule": {
        "Schedule_ID": ["schedule_id", "pm_id", "plan_id"],
        "Asset_ID": ["asset_id", "equipment_id", "tag_id"],
        "Equipment_Name": ["equipment_name", "equipment", "asset_name"],
        "PM_Frequency": ["pm_frequency", "frequency", "interval"],
        "Last_Maintenance": ["last_maintenance", "last_serviced", "last_pm_date"],
        "Next_Maintenance": ["next_maintenance", "next_due", "next_pm_date", "due_date"],
        "Assigned_Technician": ["assigned_technician", "technician", "engineer"],
        "Status": ["status"],
    },
    "Inspection Report": {
        "Inspection_ID": ["inspection_id", "report_id"],
        "Asset_ID": ["asset_id", "equipment_id", "tag_id"],
        "Equipment_Name": ["equipment_name", "equipment", "asset_name"],
        "Inspection_Date": ["inspection_date", "date"],
        "Inspector": ["inspector", "performed_by", "engineer"],
        "Observation": ["observation", "finding", "issue", "description", "defect"],
        "Risk_Level": ["risk_level", "risk", "severity", "priority"],
        "Recommendation": ["recommendation", "action", "corrective_action"],
        "Status": ["status"],
    },
    "Breakdown History": {
        "Ticket_ID": ["ticket_id", "breakdown_id", "incident_id"],
        "Asset_ID": ["asset_id", "equipment_id", "tag_id"],
        "Equipment_Name": ["equipment_name", "equipment", "asset_name"],
        "Failure_Date": ["failure_date", "failure_datetime", "breakdown_date", "date"],
        "Root_Cause": ["root_cause", "root_cause_analysis", "cause"],
        "Downtime": ["downtime", "downtime_hours", "total_downtime_hours"],
        "Corrective_Action": ["corrective_action", "action_taken", "action"],
        "Resolution_Status": ["resolution_status", "status"],
    },
    "Spare Parts Inventory": {
        "Part_SKU": ["part_sku", "sku", "part_no", "part_number"],
        "Part_Name": ["part_name", "name", "description"],
        "Compatible_Asset_ID": ["compatible_asset_id", "asset_id", "equipment_id"],
        "Target_Equipment_Model": ["target_equipment_model", "equipment_model", "model"],
        "Quantity_On_Hand": ["quantity_on_hand", "qty_on_hand", "stock", "quantity"],
        "Minimum_Required_Stock": ["minimum_required_stock", "min_stock", "reorder_level"],
        "Unit_Price": ["unit_price", "price"],
        "Supplier_Name": ["supplier_name", "supplier", "vendor"],
    },
    "Daily Operating Log": {
        "Asset_ID": ["asset_id", "equipment_id", "tag_id"],
        "Equipment_Name": ["equipment_name", "equipment", "asset_name"],
        "Date": ["date", "log_date"],
        "Temperature": ["temperature", "temp"],
        "Pressure": ["pressure"],
        "Running_Hours": ["running_hours", "operating_hours", "hours_run"],
        "Status": ["status"],
    },
}

TABULAR_EXTENSIONS = {".csv", ".xlsx"}
PDF_LIKE_EXTENSIONS = {".pdf", ".docx"}


def _normalize_col(col: str) -> str:
    """'Total Downtime (Hours)' -> 'total_downtime_hours' — used only for
    alias matching, not for the field names written out to JSON."""
    key = str(col).strip().lower()
    key = re.sub(r"[/()]+", "", key)
    key = re.sub(r"[\s_]+", "_", key).strip("_")
    return key


def classify_document_type(filename: str, columns: list[str] | None = None) -> str:
    """
    Filename keywords take priority (fast, usually reliable). If nothing
    matches and this is a spreadsheet, fall back to whichever schema's
    aliases overlap the most with the sheet's actual column names — a
    breakdown log named "Q3_Export.xlsx" still classifies correctly this
    way.
    """
    lower_name = filename.lower()
    for keywords, label in FILENAME_TYPE_KEYWORDS:
        if any(k in lower_name for k in keywords):
            return label

    if columns:
        normalized_cols = {_normalize_col(c) for c in columns}
        best_label, best_score = None, 0
        for label, schema in TABULAR_SCHEMAS.items():
            score = 0
            for aliases in schema.values():
                if any(alias in normalized_cols for alias in aliases):
                    score += 1
            if score > best_score:
                best_label, best_score = label, score
        if best_label and best_score >= 2:
            return best_label

    return "Unclassified Document"


# ---------------------------------------------------------------------------
# Regex helpers — IDs, dates, serial numbers
# ---------------------------------------------------------------------------

# General-purpose "code-like" ID: letters/prefix, a dash, digits — matches
# Asset IDs, Equipment IDs, Ticket IDs, Maintenance IDs, Inspection IDs,
# SKUs (EQ-2001, BRK-8012, PM-045, SKU-1123, ...).
ID_PATTERN = re.compile(r"\b[A-Z]{2,6}-\d{2,8}\b")

# Serial numbers are messier (mixed letters/digits, no fixed dash position)
# — e.g. "SN2394871", "A7X-99231-B".
SERIAL_PATTERN = re.compile(r"\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9-]{5,20}\b")

_DATE_FORMATS = (
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y", "%d-%m-%y", "%d/%m/%y",
    "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y",
    "%Y-%m-%d",
)


def normalize_date(value: str) -> str:
    """Best-effort convert a date string to ISO 8601 (YYYY-MM-DD). Leaves
    the value untouched if it doesn't match a known format, rather than
    guessing wrong."""
    import datetime

    stripped = value.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", stripped):
        return stripped[:10]
    for fmt in _DATE_FORMATS:
        try:
            return datetime.datetime.strptime(stripped, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return stripped


def _normalize_id(value: str) -> str:
    match = ID_PATTERN.search(value.upper())
    return match.group(0) if match else value.strip()


# ---------------------------------------------------------------------------
# Tabular extraction (Excel / CSV) — uses the sheet's own column names
# rather than NLP, per the spec.
# ---------------------------------------------------------------------------

_ID_FIELDS = {
    "Asset_ID", "Equipment_ID", "Ticket_ID", "Maintenance_ID", "Schedule_ID",
    "Inspection_ID", "Part_SKU", "Compatible_Asset_ID",
}
_DATE_FIELDS = {
    "Installation_Date", "Maintenance_Date", "Last_Maintenance",
    "Next_Maintenance", "Inspection_Date", "Failure_Date", "Date",
}


def _find_column(row_lookup: dict[str, str], aliases: list[str]) -> str | None:
    for alias in aliases:
        val = row_lookup.get(alias)
        if val:
            return val
    return None


def _extract_row_entities(row: pd.Series, schema: dict[str, list[str]]) -> dict:
    row_lookup: dict[str, str] = {}
    for col, val in row.items():
        val_str = "" if pd.isna(val) else clean_text_encoding(str(val)).strip()
        if val_str:
            row_lookup[_normalize_col(col)] = val_str

    record = {}
    for field, aliases in schema.items():
        value = _find_column(row_lookup, aliases)
        if not value:
            continue
        if field in _DATE_FIELDS:
            value = normalize_date(value)
        elif field in _ID_FIELDS:
            value = _normalize_id(value)
        record[field] = value
    return record


def _load_sheets(file_path: Path) -> dict[str, pd.DataFrame]:
    if file_path.suffix.lower() == ".csv":
        raw_sheets = {"Sheet1": pd.read_csv(file_path, header=None, dtype=str)}
    else:
        raw_sheets = pd.read_excel(file_path, sheet_name=None, header=None, dtype=str)
    return {name: load_sheet_with_detected_header(raw) for name, raw in raw_sheets.items()}


def _extract_tabular(file_path: Path, document_type: str) -> tuple[str, list[dict]]:
    sheets = _load_sheets(file_path)

    if document_type == "Unclassified Document":
        # Try to reclassify now that we can see real column names.
        all_columns = [c for df in sheets.values() for c in df.columns]
        document_type = classify_document_type(file_path.name, all_columns)

    schema = TABULAR_SCHEMAS.get(document_type)
    if schema is None:
        # Still unclassified (or a spreadsheet type without a known
        # schema) — return raw rows rather than silently dropping data.
        records = []
        for df in sheets.values():
            for _, row in df.iterrows():
                records.append({
                    _normalize_col(c): ("" if pd.isna(v) else clean_text_encoding(str(v)).strip())
                    for c, v in row.items()
                })
        return document_type, records

    records = []
    for df in sheets.values():
        for _, row in df.iterrows():
            record = _extract_row_entities(row, schema)
            if record:
                records.append(record)
    return document_type, records


# ---------------------------------------------------------------------------
# PDF / DOCX extraction — rule-based first; only reaches for very
# lightweight pattern matching for procedures/warnings, never a
# transformer model.
# ---------------------------------------------------------------------------

_PDF_LINE_PATTERNS: dict[str, re.Pattern] = {
    "Manufacturer": re.compile(r"^\s*Manufacturer\s*[:\-]\s*(.+)$", re.IGNORECASE),
    "Equipment_Model": re.compile(r"^\s*(?:Model|Equipment\s*Model)\s*[:\-]\s*(.+)$", re.IGNORECASE),
    "Equipment_Name": re.compile(r"^\s*Equipment\s*(?:Name)?\s*[:\-]\s*(.+)$", re.IGNORECASE),
    "Maintenance_Interval": re.compile(r"^\s*(?:Maintenance|Service)\s*Interval\s*[:\-]\s*(.+)$", re.IGNORECASE),
    "Inspection_Requirement": re.compile(r"^\s*Inspection\s*(?:Requirement|Frequency)\s*[:\-]\s*(.+)$", re.IGNORECASE),
}

# Multi-hit patterns: several warnings/hazards/PPE lines can appear in one
# document, so these are collected as lists rather than a single value.
_WARNING_LINE = re.compile(r"^\s*(?:WARNING|CAUTION|DANGER)\s*[:\-]?\s*(.+)$", re.IGNORECASE)
_HAZARD_LINE = re.compile(r"^\s*Hazard\s*[:\-]\s*(.+)$", re.IGNORECASE)
_PPE_LINE = re.compile(r"^\s*PPE(?:\s*Requirement)?\s*[:\-]\s*(.+)$", re.IGNORECASE)
_SAFETY_PROC_LINE = re.compile(r"^\s*Safety\s*Procedure\s*[:\-]\s*(.+)$", re.IGNORECASE)
_OPERATING_PROC_LINE = re.compile(r"^\s*Operating\s*Procedure\s*[:\-]\s*(.+)$", re.IGNORECASE)

# Numeric limits with units, anywhere in a line, e.g. "Max Temperature: 250°C"
_TEMP_LIMIT = re.compile(r"(\d+(?:\.\d+)?)\s*°?\s*(?:deg\s*)?C\b", re.IGNORECASE)
_PRESSURE_LIMIT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:psi|bar|kpa|mpa)\b", re.IGNORECASE)


def _extract_pdf(text: str) -> dict:
    entities: dict = {}
    warnings, hazards, ppe, safety_procs, operating_procs = [], [], [], [], []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        for field, pattern in _PDF_LINE_PATTERNS.items():
            if field in entities:
                continue
            match = pattern.match(line)
            if match:
                entities[field] = match.group(1).strip()

        m = _WARNING_LINE.match(line)
        if m:
            warnings.append(m.group(1).strip())
            continue
        m = _HAZARD_LINE.match(line)
        if m:
            hazards.append(m.group(1).strip())
            continue
        m = _PPE_LINE.match(line)
        if m:
            ppe.append(m.group(1).strip())
            continue
        m = _SAFETY_PROC_LINE.match(line)
        if m:
            safety_procs.append(m.group(1).strip())
            continue
        m = _OPERATING_PROC_LINE.match(line)
        if m:
            operating_procs.append(m.group(1).strip())

    if warnings:
        entities["Warning"] = warnings
    if hazards:
        entities["Hazard"] = hazards
    if ppe:
        entities["PPE_Requirement"] = ppe
    if safety_procs:
        entities["Safety_Procedure"] = safety_procs
    if operating_procs:
        entities["Operating_Procedure"] = operating_procs

    temp_matches = _TEMP_LIMIT.findall(text)
    if temp_matches:
        entities["Temperature_Limit"] = sorted({f"{t}°C" for t in temp_matches})
    pressure_matches = _PRESSURE_LIMIT.findall(text)
    if pressure_matches:
        entities["Pressure_Limit"] = sorted({p for p in pressure_matches})

    # Equipment IDs mentioned anywhere (asset tags stamped in a manual, etc.)
    id_matches = sorted(set(ID_PATTERN.findall(text.upper())))
    if id_matches:
        entities["Equipment_ID"] = id_matches

    return entities


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_industrial_entities(
    original_path: Path,
    extracted_text_path: Path | None,
) -> dict | None:
    """
    Runs entity extraction for one uploaded document.

    original_path: file as saved in uploads/ (used directly for CSV/XLSX).
    extracted_text_path: the .txt file in extracted_text/ (used for PDF/DOCX).

    Returns {"document": ..., "document_type": ..., "entities": ...} or
    None if the file type isn't supported yet (e.g. images).
    """
    extension = original_path.suffix.lower()
    clean_name = re.sub(r"^[0-9a-f]{8}_", "", original_path.name)

    if extension in TABULAR_EXTENSIONS:
        document_type = classify_document_type(clean_name)
        document_type, records = _extract_tabular(original_path, document_type)
        return {
            "document": clean_name,
            "document_type": document_type,
            "entities": records,
        }

    if extension in PDF_LIKE_EXTENSIONS:
        if extracted_text_path is None or not extracted_text_path.exists():
            return None
        document_type = classify_document_type(clean_name)
        text = extracted_text_path.read_text(encoding="utf-8")
        entities = _extract_pdf(text)
        return {
            "document": clean_name,
            "document_type": document_type,
            "entities": entities,
        }

    return None  # PNG/JPG etc. — nothing to extract yet


def save_entities(result: dict, original_path: Path, entities_dir: Path) -> Path:
    """Saves one document's extraction result as backend/entities/<slug>_entities.json."""
    from chunking import slugify  # local import avoids a circular import at module load time

    clean_stem = re.sub(r"^[0-9a-f]{8}_", "", original_path.stem)
    out_path = entities_dir / f"{slugify(clean_stem)}_entities.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out_path


def load_all_entities(entities_dir: Path) -> list[dict]:
    """Powers GET /entities — every extracted-entities file, grouped by document."""
    results = []
    for path in sorted(entities_dir.glob("*_entities.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    return results
