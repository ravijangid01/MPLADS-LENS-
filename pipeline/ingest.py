"""
pipeline/ingest.py
Loads user CSV/Excel using a YAML column map, validates columns,
parses Indian number formats and dd-mm-yyyy dates, and reports
dropped rows.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

CANONICAL_COLUMNS = [
    "work_id", "mp_ref", "constituency", "state", "district",
    "district_class", "block_or_village", "work_title", "work_description",
    "work_category", "implementing_agency", "recommended_date",
    "sanctioned_date", "sanctioned_amount", "released_amount",
    "expenditure_amount", "status", "last_progress_date",
    "completion_date", "completion_percent",
]

REQUIRED_COLUMNS = [
    "work_id", "state", "district", "work_title",
    "sanctioned_amount", "sanctioned_date",
]

AMOUNT_COLUMNS = ["sanctioned_amount", "released_amount", "expenditure_amount"]

DATE_COLUMNS = [
    "recommended_date", "sanctioned_date", "completion_date", "last_progress_date"
]

# eSAKSHI portal — district_class lookup (subset; extended in clean.py)
DISTRICT_CLASS_MAP = {
    "Jaisalmer": "desert", "Barmer": "desert", "Bikaner": "desert",
    "Lahaul and Spiti": "hill", "Kinnaur": "hill", "Chamba": "hill",
    "Kullu": "hill", "Shimla": "hill", "Mandi": "hill", "Kangra": "hill",
    "Solan": "plain", "Hamirpur": "plain", "Una": "plain",
    "Puri": "coastal", "Kendrapara": "coastal", "Jagatsinghpur": "coastal",
    "Balasore": "coastal", "Koraput": "tribal", "Nabarangpur": "tribal",
    "Rayagada": "tribal", "Gadchiroli": "tribal", "Nandurbar": "tribal",
    "Ratnagiri": "coastal", "Sindhudurg": "coastal",
    "Jodhpur": "plain", "Jaipur": "plain", "Ajmer": "plain",
    "Udaipur": "hill", "Sirohi": "hill", "Chittorgarh": "hill",
    "Dungarpur": "tribal",
}


def load_column_map(path: str) -> dict[str, Optional[str]]:
    """Load a YAML column map file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # Strip comments / private keys
    return {k: v for k, v in raw.items() if not k.startswith("#")}


def parse_indian_number(value: str | float | int) -> float:
    """
    Parse Indian number format strings to float.
    Handles: '1,23,456', '12.50 Lakh', '3.2 Crore', plain floats.
    Returns NaN for unparseable values.
    """
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace(",", "")
    # Check for lakh / crore suffix
    multiplier = 1.0
    lower = s.lower()
    if "crore" in lower:
        multiplier = 1e7
        s = re.sub(r"crore", "", lower, flags=re.IGNORECASE).strip()
    elif "lakh" in lower:
        multiplier = 1e5
        s = re.sub(r"lakh", "", lower, flags=re.IGNORECASE).strip()
    # Remove currency symbols
    s = re.sub(r"[₹rs\.]+", "", s, flags=re.IGNORECASE).strip()
    try:
        return float(s) * multiplier
    except ValueError:
        return float("nan")


def parse_indian_date(value: str | pd.Timestamp | None) -> pd.Timestamp:
    """
    Parse dd-mm-yyyy, dd/mm/yyyy, or ISO dates. Returns NaT on failure.
    """
    if pd.isna(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    s = str(value).strip()
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y"):
        try:
            return pd.Timestamp(pd.to_datetime(s, format=fmt))
        except Exception:
            pass
    try:
        return pd.Timestamp(pd.to_datetime(s, dayfirst=True))
    except Exception:
        return pd.NaT


def load_user_file(
    file_path: str,
    column_map_path: Optional[str] = None,
    sheet_name: int | str = 0,
) -> tuple[pd.DataFrame, dict]:
    """
    Load a user-supplied CSV or Excel file, apply column map, and
    return (canonical_df, report_dict).

    Parameters
    ----------
    file_path : str
        Path to CSV or Excel file.
    column_map_path : str, optional
        Path to YAML column map. If None, uses example map.
    sheet_name : int | str
        Sheet name/index for Excel files.

    Returns
    -------
    (DataFrame, report)
        Canonical DataFrame and a dict with ingestion report.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Load raw data
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw_df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str)
    else:
        raw_df = pd.read_csv(file_path, dtype=str, encoding="utf-8-sig")

    original_rows = len(raw_df)
    logger.info("Loaded %d rows from %s", original_rows, file_path)

    # Load column map
    if column_map_path is None:
        column_map_path = str(
            Path(__file__).parent.parent / "config" / "column_map.example.yaml"
        )
    col_map = load_column_map(column_map_path)

    # Build mapping: canonical → raw header
    rename_map: dict[str, str] = {}
    missing_canonical: list[str] = []
    for canonical, raw_header in col_map.items():
        if canonical.startswith("_"):
            continue  # optional portal-specific columns
        if raw_header is None:
            missing_canonical.append(canonical)
            continue
        if raw_header in raw_df.columns:
            rename_map[raw_header] = canonical
        else:
            missing_canonical.append(canonical)

    df = raw_df.rename(columns=rename_map)

    # Add missing canonical columns as NaN
    for col in CANONICAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Keep only canonical columns
    df = df[CANONICAL_COLUMNS].copy()

    # ── Parse amounts ───────────────────────────────────────────────────────
    for col in AMOUNT_COLUMNS:
        df[col] = df[col].apply(parse_indian_number)

    # ── Parse dates ─────────────────────────────────────────────────────────
    for col in DATE_COLUMNS:
        df[col] = df[col].apply(parse_indian_date)

    # ── Derive sanction_year ─────────────────────────────────────────────────
    df["sanction_year"] = df["sanctioned_date"].dt.year.astype("Int64")

    # ── Derive district_class if missing ────────────────────────────────────
    if df["district_class"].isna().all():
        df["district_class"] = df["district"].map(DISTRICT_CLASS_MAP).fillna("plain")

    # ── Normalise status ────────────────────────────────────────────────────
    status_map = {
        "completed": "completed",
        "complete": "completed",
        "in progress": "in_progress",
        "inprogress": "in_progress",
        "ongoing": "in_progress",
        "not started": "not_started",
        "notstarted": "not_started",
        "yet to start": "not_started",
    }
    df["status"] = (
        df["status"]
        .str.strip()
        .str.lower()
        .map(status_map)
        .fillna("in_progress")
    )

    # ── Normalise completion_percent ─────────────────────────────────────────
    df["completion_percent"] = pd.to_numeric(df["completion_percent"], errors="coerce")
    df["completion_percent"] = df["completion_percent"].clip(0, 100)

    # ── Drop rows missing required fields ───────────────────────────────────
    drop_report: dict[str, int] = {}
    for req in REQUIRED_COLUMNS:
        mask = df[req].isna()
        n_drop = mask.sum()
        if n_drop > 0:
            drop_report[req] = int(n_drop)
            df = df[~mask]

    # ── Add hidden column placeholder ───────────────────────────────────────
    if "_planted_type" not in df.columns:
        df["_planted_type"] = ""

    rows_kept = len(df)
    rows_dropped = original_rows - rows_kept

    report = {
        "source": str(file_path),
        "original_rows": original_rows,
        "rows_kept": rows_kept,
        "rows_dropped": rows_dropped,
        "drop_reasons": drop_report,
        "missing_canonical_columns": missing_canonical,
    }
    logger.info("Ingestion complete: %d rows kept, %d dropped", rows_kept, rows_dropped)
    return df, report
