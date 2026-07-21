"""
Chunking — Step 4.

Different document types get chunked differently (semantic chunking, not a
blind "every 500 words" split):

    PDF manuals / DOCX SOPs   -> split by headings; any section over
                                 ~500 words gets sub-split into ~300-500
                                 word pieces
    CSV inspection reports    -> one row = one chunk
    XLSX maintenance logs     -> one row = one chunk (every sheet)

Every chunk, regardless of source, has the same shape:

    {
        "chunk_id": "pump_manual_001",
        "document_name": "Pump_Manual.pdf",
        "document_type": "Equipment Manual",
        "chunk_number": 1,
        "section_title": "Lubrication" | None,
        "source_page": 15 | None,
        "text": "...",
        "word_count": 214,
    }

document_type is a human label (used later for citations in the chatbot),
guessed from the filename against the categories in the problem statement.
It does NOT change chunking strategy — extension does that.

source_page is only populated for PDFs (where "page" is a real, fixed
concept). DOCX/CSV/XLSX have no reliable page boundary, so it's None there —
section_title (sheet name, heading) carries the equivalent context instead.

Quality filter: heading-based chunking (PDF/DOCX) throws away two kinds of
junk before saving —
  1. Chunks with too few words to carry standalone meaning (page numbers,
     stray fragments).
  2. Chunks whose text repeats verbatim across many chunks of the same
     document — running headers/footers ("MOTOR GUIDE | JULY 2019" on every
     page) look exactly like this: identical text, once per page.
Row-based chunking (CSV/XLSX) is NOT filtered — a short row is still a
complete, meaningful record, not a fragment.
"""

import datetime
import json
import re
from pathlib import Path

import pandas as pd

from text_cleaning import clean_text_encoding
from tabular_utils import load_sheet_with_detected_header


# ---------------------------------------------------------------------------
# Quality filter (PDF / DOCX heading-based chunks only)
# ---------------------------------------------------------------------------

MIN_MEANINGFUL_WORDS = 5   # below this, a chunk is almost certainly a
                           # page number, stray heading, or fragment
REPEAT_THRESHOLD = 3       # text seen this many times across one document's
                           # chunks is treated as a running header/footer

PURE_NUMBER_PATTERN = re.compile(r"^[\d\s.\-–—]+$")  # "2", "- 3 -", "12.3"


def _normalize_for_dedup(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_too_short_or_numeric(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if PURE_NUMBER_PATTERN.match(stripped):
        return True
    return len(stripped.split()) < MIN_MEANINGFUL_WORDS


def filter_low_value_chunks(chunks: list[dict]) -> tuple[list[dict], int]:
    """
    Drops page numbers / fragments / repeated headers-footers from a list
    of heading-based chunks. Returns (kept_chunks, num_dropped).
    """
    # Count how often each normalized text appears in this document's chunks.
    freq: dict[str, int] = {}
    for c in chunks:
        key = _normalize_for_dedup(c["text"])
        freq[key] = freq.get(key, 0) + 1

    kept = []
    for c in chunks:
        key = _normalize_for_dedup(c["text"])

        if _is_too_short_or_numeric(c["text"]):
            continue
        if freq[key] >= REPEAT_THRESHOLD:
            # Same exact text repeated across several chunks -> boilerplate
            # header/footer, not content.
            continue

        kept.append(c)

    return kept, len(chunks) - len(kept)


# ---------------------------------------------------------------------------
# Document type labelling (for metadata / citations only)
# ---------------------------------------------------------------------------

TYPE_KEYWORDS = [
    (["inspection"], "Inspection Report"),
    (["breakdown", "maintenance", "log"], "Maintenance Log"),
    (["manual"], "Equipment Manual"),
    (["sop", "procedure", "safety"], "Safety Procedure"),
    (["datasheet", "spec"], "Technical Datasheet"),
    (["drawing", "p&id", "pid"], "Engineering Drawing"),
]


def guess_document_type(filename: str, extension: str) -> str:
    lower_name = filename.lower()
    for keywords, label in TYPE_KEYWORDS:
        if any(k in lower_name for k in keywords):
            return label

    # Fall back to something generic based on extension alone.
    return {
        ".pdf": "PDF Document",
        ".docx": "Word Document",
        ".csv": "CSV Record Set",
        ".xlsx": "Excel Record Set",
    }.get(extension, "Document")


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "document"


# ---------------------------------------------------------------------------
# Field name normalization — messy source headers -> clean, parseable keys
# ---------------------------------------------------------------------------

# Exact renames for known ugly names (seen in real maintenance-log exports).
_FIELD_RENAME_MAP = {
    "failure_date/time": "Failure_Date",
    "resolution_date/time": "Resolution_Date",
    "total_downtime_(hours)": "Downtime_Hours",
    "total_downtime_hours": "Downtime_Hours",
    "root_cause_analysis": "Root_Cause",
}


def _normalize_field_name(col: str) -> str:
    """'Total_Downtime_(Hours)' -> 'Downtime_Hours', 'Failure_Date/Time' ->
    'Failure_Date'. Falls back to stripping /, (, ) from unrecognized
    columns so every key stays easy to parse programmatically."""
    key = str(col).strip().lower()
    if key in _FIELD_RENAME_MAP:
        return _FIELD_RENAME_MAP[key]
    cleaned = re.sub(r"[/()]+", "", str(col))
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned


# Equipment-type inference from equipment name, for filtering ("show all
# pump failures") without depending on the exact model string.
_EQUIPMENT_TYPE_KEYWORDS = [
    (["pump"], "Pump"),
    (["motor"], "Motor"),
    (["valve"], "Valve"),
    (["compressor"], "Compressor"),
    (["conveyor"], "Conveyor"),
    (["transformer"], "Transformer"),
    (["fan", "blower"], "Fan/Blower"),
    (["gearbox", "gear box"], "Gearbox"),
    (["boiler"], "Boiler"),
    (["turbine"], "Turbine"),
]


def _infer_equipment_type(equipment_name: str | None) -> str | None:
    if not equipment_name:
        return None
    lower = equipment_name.lower()
    for keywords, label in _EQUIPMENT_TYPE_KEYWORDS:
        if any(k in lower for k in keywords):
            return label
    return None


def _slugify_id(value: str) -> str:
    """Light sanitizer for using a real ID (e.g. 'PMP-KS-29') as a chunk_id
    component — keeps it human-readable, unlike full slugify()."""
    return re.sub(r"[^A-Za-z0-9\-]+", "_", value.strip()).strip("_")


def _load_tabular_with_header_detection(file_path: Path) -> dict[str, pd.DataFrame]:
    """Returns {sheet_name: cleaned_dataframe}. CSV gets one entry, "Sheet1"."""
    if file_path.suffix.lower() == ".csv":
        raw_sheets = {"Sheet1": pd.read_csv(file_path, header=None, dtype=str)}
    else:
        raw_sheets = pd.read_excel(file_path, sheet_name=None, header=None, dtype=str)

    return {name: load_sheet_with_detected_header(raw) for name, raw in raw_sheets.items()}


# Formats we'll try, in order. Ambiguous DD/MM vs MM/DD cases are read
# day-first, since the source data is Indian industrial records.
_DATE_ONLY_FORMATS = (
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
    "%d-%m-%y", "%d/%m/%y",
    "%d %b %Y", "%d %B %Y",
    "%b %d, %Y", "%B %d, %Y",
)
# Same date formats, each with a trailing time component — "23-04-2025 00:00"
_DATETIME_FORMATS = tuple(f"{fmt} %H:%M" for fmt in _DATE_ONLY_FORMATS) + tuple(
    f"{fmt} %H:%M:%S" for fmt in _DATE_ONLY_FORMATS
)
_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ISO_DATETIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


def _normalize_date(value: str) -> str:
    """Best-effort convert a date (or date+time) string to ISO 8601.
    'YYYY-MM-DD' for date-only input, 'YYYY-MM-DDTHH:MM:SS' when a time
    component is present. Leaves the value alone (rather than guessing
    wrong) if it doesn't match a known format."""
    stripped = value.strip()
    if not stripped or _ISO_DATE_PATTERN.match(stripped) or _ISO_DATETIME_PATTERN.match(stripped):
        return stripped

    for fmt in _DATETIME_FORMATS:
        try:
            return datetime.datetime.strptime(stripped, fmt).strftime("%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue

    for fmt in _DATE_ONLY_FORMATS:
        try:
            return datetime.datetime.strptime(stripped, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    return value


# Column-name aliases used to find the right field regardless of how the
# source spreadsheet happened to label it.
_SUMMARY_FIELD_ALIASES = {
    "equipment": ["equipment_name", "equipment", "asset_name", "asset"],
    "asset_id": ["asset_id", "asset_no", "asset_number", "tag_id", "equipment_id"],
    "observation": ["observation", "finding", "issue", "description", "defect"],
    "risk": ["risk_level", "risk", "severity", "priority"],
    "recommendation": ["recommendation", "action", "corrective_action", "action_taken"],
    "root_cause": ["root_cause", "root_cause_analysis", "cause"],
    "action_required": ["action_required", "action_taken", "corrective_action"],
    "downtime_hours": ["downtime_hours", "total_downtime_hours", "downtime"],
    "failure_date": ["failure_date", "failure_datetime"],
    "status": ["status"],
}


def _find_field(row_lookup: dict[str, str], aliases: list[str]) -> str | None:
    for alias in aliases:
        val = row_lookup.get(alias)
        if val:
            return val
    return None


def _generate_summary(row_lookup: dict[str, str]) -> str | None:
    """
    Builds a one-line natural-language summary from whatever recognizable
    fields exist in this row. Handles two shapes:
      - Breakdown/maintenance-log rows (Equipment + Root_Cause + Downtime)
      - Inspection rows (Observation + Recommendation)
    Returns None if neither shape matches — a summary should never be
    fabricated from data that isn't there.
    """
    equipment = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["equipment"])
    asset_id = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["asset_id"])
    root_cause = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["root_cause"])
    action = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["action_required"])
    downtime = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["downtime_hours"])
    status = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["status"])

    # --- Breakdown / maintenance-log shape ---------------------------------
    if root_cause or downtime:
        asset_ref = f" ({asset_id})" if asset_id else ""
        subject = f"{equipment}{asset_ref}" if equipment else "The equipment"
        cause_clause = f" experienced {root_cause[0].lower() + root_cause[1:]}" if root_cause else " had a reported failure"
        downtime_clause = f", resulting in {downtime} hours of downtime" if downtime else ""
        sentence = f"{subject}{cause_clause}{downtime_clause}.".strip()

        parts = [sentence]
        if action:
            parts.append(f"{action}.")
        if status:
            parts.append(f"Ticket {status.lower()}.")
        return " ".join(parts)

    # --- Inspection shape ----------------------------------------------------
    observation = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["observation"])
    recommendation = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["recommendation"])
    risk = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["risk"])

    if not observation and not recommendation:
        return None  # row doesn't match either known shape; nothing to summarize

    sentences = []
    if observation:
        risk_prefix = f"{risk}-risk " if risk else ""
        location = f" on {equipment}" if equipment else ""
        obs_lower = observation[0].lower() + observation[1:] if len(observation) > 1 else observation.lower()
        sentence = f"{risk_prefix}{obs_lower}{location}.".strip()
        sentence = sentence[0].upper() + sentence[1:]  # capitalize first letter only —
        # .capitalize() would also lowercase "Grundfos CR32 Pump" etc.
        sentences.append(sentence)

    if recommendation:
        if status and status.lower() in {"closed", "completed", "resolved", "done"}:
            sentences.append(f"{recommendation} completed.")
        elif status:
            sentences.append(f"Recommended: {recommendation}. Status: {status}.")
        else:
            sentences.append(f"Recommended: {recommendation}.")
    elif status:
        sentences.append(f"Status: {status}.")

    return " ".join(sentences)


def _extract_metadata(row_lookup: dict[str, str]) -> dict:
    """
    Pulls typed, filterable fields out of the row separately from the chunk
    text — so a vector DB that supports metadata filtering can answer
    "pump failures with downtime > 12 hours" without re-parsing prose.
    Only includes keys that actually have a value.
    """
    equipment = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["equipment"])
    asset_id = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["asset_id"])
    status = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["status"])
    downtime = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["downtime_hours"])
    root_cause = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["root_cause"])
    failure_date = _find_field(row_lookup, _SUMMARY_FIELD_ALIASES["failure_date"])

    metadata = {}
    if equipment:
        metadata["equipment"] = equipment
    if asset_id:
        metadata["asset_id"] = asset_id
    if status:
        metadata["status"] = status
    if downtime:
        try:
            metadata["downtime_hours"] = float(downtime) if "." in downtime else int(downtime)
        except ValueError:
            metadata["downtime_hours"] = downtime
    if root_cause:
        metadata["root_cause"] = root_cause
    if failure_date:
        year_match = re.match(r"^(\d{4})", failure_date)
        if year_match:
            metadata["year"] = int(year_match.group(1))
    equipment_type = _infer_equipment_type(equipment)
    if equipment_type:
        metadata["equipment_type"] = equipment_type

    return metadata


def _row_to_text(row: pd.Series) -> tuple[str, dict, str | None]:
    """Returns (chunk_text, metadata, asset_id_or_None)."""
    lines = []
    row_lookup = {}
    for col, val in row.items():
        val_str = "" if pd.isna(val) else clean_text_encoding(val)
        clean_col = _normalize_field_name(col)
        if val_str and "date" in clean_col.lower():
            val_str = _normalize_date(val_str)
        lines.append(f"{clean_col}: {val_str}")
        if val_str:
            row_lookup[clean_col.lower()] = val_str

    summary = _generate_summary(row_lookup)
    if summary:
        lines.append("")
        lines.append("Summary:")
        lines.append(summary)

    metadata = _extract_metadata(row_lookup)
    asset_id = metadata.get("asset_id")

    return "\n".join(lines), metadata, asset_id


def chunk_csv(file_path: Path, document_name: str, document_type: str) -> list[dict]:
    sheets = _load_tabular_with_header_detection(file_path)
    df = sheets["Sheet1"]
    slug = slugify(file_path.stem)

    chunks = []
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        text, metadata, asset_id = _row_to_text(row)
        chunk_id = f"{_slugify_id(asset_id)}_{i:03d}" if asset_id else f"{slug}_{i:03d}"
        chunks.append({
            "chunk_id": chunk_id,
            "document_name": document_name,
            "document_type": document_type,
            "chunk_number": i,
            "section_title": None,
            "source_page": None,  # CSV rows have no page concept
            "text": text,
            "metadata": metadata,
            "word_count": len(text.split()),
        })
    return chunks


def chunk_xlsx(file_path: Path, document_name: str, document_type: str) -> list[dict]:
    sheets = _load_tabular_with_header_detection(file_path)
    slug = slugify(file_path.stem)

    chunks = []
    counter = 0
    for sheet_name, df in sheets.items():
        for _, row in df.iterrows():
            counter += 1
            text, metadata, asset_id = _row_to_text(row)
            chunk_id = f"{_slugify_id(asset_id)}_{counter:03d}" if asset_id else f"{slug}_{counter:03d}"
            chunks.append({
                "chunk_id": chunk_id,
                "document_name": document_name,
                "document_type": document_type,
                "chunk_number": counter,
                "section_title": sheet_name,
                "source_page": None,  # Excel rows have no page concept; sheet name stands in
                "text": text,
                "metadata": metadata,
                "word_count": len(text.split()),
            })
    return chunks




# ---------------------------------------------------------------------------
# Strategy 2: heading-based chunking (PDF / DOCX)
# ---------------------------------------------------------------------------

# A line counts as a heading if it's short, has no sentence-ending
# punctuation, and looks like a title rather than a sentence. This is a
# heuristic, not a perfect parser — good enough for typical manuals/SOPs
# with numbered or titled sections.
HEADING_PATTERN = re.compile(
    r"^(chapter|section|part)\s+\d+[:.]?\s*.*$|"      # "Chapter 3: Maintenance"
    r"^\d+(\.\d+)*\s+[A-Z].{0,60}$|"                    # "2.3 Installation"
    r"^[A-Z][A-Za-z0-9 /&\-]{2,60}$",                   # "LUBRICATION" / "Troubleshooting"
    re.IGNORECASE,
)

WORDS_PER_SUBCHUNK = 400  # midpoint of the requested 300-500 word range

PAGE_SEPARATOR_LINES = {"—", "-", "–", "_", "*", "•"}


def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip().lower())


def _is_page_separator_line(line: str) -> bool:
    """Bare page numbers or divider glyphs — never content on their own."""
    stripped = line.strip()
    if not stripped:
        return True
    if PURE_NUMBER_PATTERN.match(stripped):
        return True
    return stripped in PAGE_SEPARATOR_LINES


def _strip_boilerplate_lines(text: str) -> str:
    """
    Removes running headers/footers and page numbers BEFORE section
    splitting, so they can't glue themselves onto the tail of an otherwise
    good chunk — e.g. a "MOTOR GUIDE | JULY 2019" line printed on every page
    doesn't get caught by heading detection (the "|" breaks that regex), so
    without this pass it silently becomes trailing noise on the previous
    section's body instead of its own filterable chunk.

    Two things get removed:
      1. Bare page numbers / divider lines ("2", "—") — always, regardless
         of how often they occur, since a lone number is never content.
      2. Short lines (<= 8 words) that repeat 3+ times identically across
         the document — the running-header/footer signature.
    """
    lines = text.splitlines()

    freq: dict[str, int] = {}
    for line in lines:
        if line.strip():
            freq[_normalize_line(line)] = freq.get(_normalize_line(line), 0) + 1

    cleaned = []
    for line in lines:
        if _is_page_separator_line(line):
            continue
        norm = _normalize_line(line)
        if freq.get(norm, 0) >= REPEAT_THRESHOLD and len(norm.split()) <= 8:
            continue
        cleaned.append(line)

    return "\n".join(cleaned)


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 70:
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    return bool(HEADING_PATTERN.match(stripped))


def _split_into_sections(text: str) -> list[tuple[str | None, str, int]]:
    """
    Returns [(heading_or_None, section_body, body_start_word_offset), ...].

    body_start_word_offset is the word index (0-based, same tokenization as
    str.split()) where this section's body begins in the full document —
    used afterwards to figure out which PDF page a chunk came from.
    """
    lines = text.splitlines()
    sections = []
    current_heading = None
    current_body: list[str] = []
    current_body_start_word = 0
    word_counter = 0

    for line in lines:
        if _looks_like_heading(line):
            if current_body or current_heading:
                sections.append((current_heading, "\n".join(current_body).strip(), current_body_start_word))
            current_heading = line.strip()
            current_body = []
            word_counter += len(line.split())
            current_body_start_word = word_counter
        else:
            current_body.append(line)
            word_counter += len(line.split())

    if current_body or current_heading:
        sections.append((current_heading, "\n".join(current_body).strip(), current_body_start_word))

    # No headings detected at all -> treat the whole document as one section.
    if not sections:
        sections = [(None, text.strip(), 0)]

    return [(h, b, s) for h, b, s in sections if b]


def _split_by_word_count(text: str, start_offset: int, size: int = WORDS_PER_SUBCHUNK) -> list[tuple[str, int]]:
    """Returns [(piece_text, piece_start_word_offset), ...]."""
    words = text.split()
    if len(words) <= size:
        return [(text, start_offset)]
    return [
        (" ".join(words[i:i + size]), start_offset + i)
        for i in range(0, len(words), size)
    ]


def _build_page_boundaries(page_word_counts: list[int] | None) -> list[int] | None:
    """Cumulative word count after each page: page 1 ends at boundaries[0], etc."""
    if not page_word_counts:
        return None
    cumulative = []
    running = 0
    for count in page_word_counts:
        running += count
        cumulative.append(running)
    return cumulative


def _word_offset_to_page(offset: int, boundaries: list[int]) -> int:
    for page_index, boundary in enumerate(boundaries):
        if offset < boundary:
            return page_index + 1  # 1-indexed
    return len(boundaries)  # fell past the end -> clamp to last page


def chunk_by_headings(
    text: str,
    document_name: str,
    document_type: str,
    page_word_counts: list[int] | None = None,
) -> tuple[list[dict], int]:
    """Returns (chunks, num_filtered_out)."""
    slug = slugify(Path(document_name).stem)
    text = clean_text_encoding(text)
    text = _strip_boilerplate_lines(text)
    sections = _split_into_sections(text)
    boundaries = _build_page_boundaries(page_word_counts)

    raw_chunks = []
    for heading, body, body_start_word in sections:
        # If a section is very long, split further into ~300-500 word pieces.
        if len(body.split()) > 500:
            pieces = _split_by_word_count(body, body_start_word)
        else:
            pieces = [(body, body_start_word)]

        for piece_text, piece_start_word in pieces:
            source_page = None
            if boundaries:
                piece_word_count = len(piece_text.split())
                midpoint = piece_start_word + piece_word_count // 2
                source_page = _word_offset_to_page(midpoint, boundaries)

            raw_chunks.append({
                "document_name": document_name,
                "document_type": document_type,
                "section_title": heading,
                "source_page": source_page,
                "text": piece_text,
                "word_count": len(piece_text.split()),
            })

    kept_chunks, num_filtered = filter_low_value_chunks(raw_chunks)

    # Renumber sequentially now that low-value chunks have been dropped —
    # keeps chunk_number/chunk_id continuous rather than skipping gaps.
    chunks = []
    for i, c in enumerate(kept_chunks, start=1):
        chunks.append({
            "chunk_id": f"{slug}_{i:03d}",
            "document_name": c["document_name"],
            "document_type": c["document_type"],
            "chunk_number": i,
            "section_title": c["section_title"],
            "source_page": c["source_page"],
            "text": c["text"],
            "metadata": None,  # row-level metadata only applies to CSV/XLSX
            "word_count": c["word_count"],
        })

    return chunks, num_filtered


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def create_chunks(
    original_path: Path,
    extracted_text_path: Path | None,
    page_word_counts: list[int] | None = None,
) -> tuple[list[dict], int]:
    """
    original_path: the file as saved in uploads/ (used directly for CSV/XLSX,
                   so we chunk real rows rather than re-parsing flattened text)
    extracted_text_path: the .txt file in extracted_text/ (used for PDF/DOCX)
    page_word_counts: per-page word counts from extraction (PDF only) — used
                       to populate source_page on each chunk.

    Returns (chunks, num_filtered_out). num_filtered_out is always 0 for
    row-based chunking (CSV/XLSX), since that path has no quality filter.
    """
    extension = original_path.suffix.lower()
    document_name = original_path.name
    # Strip the uuid prefix (e.g. "05516934_Pump_Manual.pdf" -> "Pump_Manual.pdf")
    # so citations later show a clean, human-readable name.
    clean_name = re.sub(r"^[0-9a-f]{8}_", "", document_name)
    document_type = guess_document_type(clean_name, extension)

    if extension == ".csv":
        return chunk_csv(original_path, clean_name, document_type), 0

    if extension == ".xlsx":
        return chunk_xlsx(original_path, clean_name, document_type), 0

    if extension in (".pdf", ".docx"):
        if extracted_text_path is None or not extracted_text_path.exists():
            return [], 0
        text = extracted_text_path.read_text(encoding="utf-8")
        # page_word_counts only makes sense for PDF; DOCX has no page concept.
        pwc = page_word_counts if extension == ".pdf" else None
        return chunk_by_headings(text, clean_name, document_type, pwc)

    # PNG/JPG etc. — nothing to chunk yet, no extracted text exists.
    return [], 0


def save_chunks(chunks: list[dict], original_path: Path, chunks_dir: Path) -> Path:
    clean_stem = re.sub(r"^[0-9a-f]{8}_", "", original_path.stem)
    out_path = chunks_dir / f"{slugify(clean_stem)}_chunks.json"
    out_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    return out_path
