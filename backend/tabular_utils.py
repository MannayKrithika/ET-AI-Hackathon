"""
Shared spreadsheet utilities — used by both extraction.py (builds the human-
readable .txt preview) and chunking.py (builds row-based chunks straight from
the original file). Kept in one place so the two can't quietly drift apart,
which is exactly what happened before: extraction.py had header-row
detection, chunking.py didn't, so chunks still showed "Unnamed: 1" even
though the .txt preview looked fine.
"""

import pandas as pd

from text_cleaning import clean_text_encoding


def find_header_row(raw_df: pd.DataFrame, max_scan: int = 10) -> int:
    """
    Some spreadsheets have a merged title row (and sometimes a blank row)
    sitting above the real header row, e.g.:

        Row 0: "Equipment Breakdown History & RCA"   <- title, mostly empty cells
        Row 1: (blank)
        Row 2: "Ticket ID", "Asset ID", "Equipment Name", ...   <- real header

    Naively treating row 0 as the header produces "Unnamed: 1", "Unnamed: 2",
    etc. for every real column.

    Heuristic: the real header row and the data rows below it both use
    (roughly) the full column width. A title row, by contrast, only has one
    or two non-null cells (the rest are NaN from merged cells). So we take
    the most common "non-null cell count" across all rows as the sheet's
    real width, then scan top-down for the first row that matches it.
    """
    non_null_counts = raw_df.notna().sum(axis=1)
    if len(non_null_counts) == 0:
        return 0

    full_width = non_null_counts.mode().iloc[0]

    scan_limit = min(max_scan, len(raw_df))
    for i in range(scan_limit):
        if non_null_counts.iloc[i] >= full_width and full_width >= 2:
            return i

    return 0  # fallback: nothing matched, assume row 0 is the header


def sanitize_column_name(col, fallback_index: int = 0) -> str:
    """'Total Downtime Hours' -> 'Total_Downtime_Hours'."""
    if pd.isna(col):
        return f"Column_{fallback_index + 1}"
    name = clean_text_encoding(col).strip()
    name = name.replace(" ", "_")
    return name or f"Column_{fallback_index + 1}"


def load_sheet_with_detected_header(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a header=None-loaded DataFrame, finds the real header row, and
    returns a clean DataFrame with sanitized column names and blank rows
    dropped.
    """
    if raw_df.empty:
        return raw_df

    header_idx = find_header_row(raw_df)
    columns = [sanitize_column_name(c, i) for i, c in enumerate(raw_df.iloc[header_idx])]

    data_df = raw_df.iloc[header_idx + 1:].copy()
    data_df.columns = columns
    data_df = data_df.dropna(how="all")  # drop fully-empty rows (trailing blanks etc.)
    return data_df.reset_index(drop=True)
