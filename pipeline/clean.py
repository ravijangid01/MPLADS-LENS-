"""
pipeline/clean.py
Standardises agency and category text, fixes units (lakh/crore → rupees),
removes exact-duplicate rows, and computes a data-quality score per work.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ─── Category normalisation map ───────────────────────────────────────────────
CATEGORY_ALIASES: dict[str, str] = {
    # road
    "road": "road", "roads": "road", "cc road": "road", "village road": "road",
    "approach road": "road", "rural road": "road", "internal road": "road",
    "culvert": "road", "bridge": "road",
    # community hall
    "community hall": "community hall", "panchayat bhavan": "community hall",
    "multipurpose hall": "community hall", "community centre": "community hall",
    "panchayat ghar": "community hall",
    # drinking water
    "drinking water": "drinking water", "water supply": "drinking water",
    "water tank": "drinking water", "hand pump": "drinking water",
    "piped water": "drinking water", "tube well": "drinking water",
    "borewell": "drinking water",
    # school
    "school": "school", "education": "school", "classroom": "school",
    "primary school": "school", "higher secondary": "school",
    # health
    "health": "health", "health centre": "health", "hospital": "health",
    "phc": "health", "sub centre": "health", "chc": "health",
    "dispensary": "health",
    # sanitation
    "sanitation": "sanitation", "toilet": "sanitation", "latrine": "sanitation",
    "drainage": "sanitation", "sewage": "sanitation", "waste": "sanitation",
    # other
    "anganwadi": "other", "solar": "other", "cremation": "other",
    "park": "other", "playground": "other", "street light": "other",
}

# Key fields for data quality scoring
QUALITY_FIELDS = [
    "mp_ref", "constituency", "district", "block_or_village",
    "work_title", "work_description", "work_category", "implementing_agency",
    "recommended_date", "sanctioned_date", "sanctioned_amount",
    "released_amount", "expenditure_amount", "status",
    "last_progress_date", "completion_percent",
]


def normalise_category(raw: str) -> str:
    """Map free-text work category to one of the 7 canonical categories."""
    if pd.isna(raw) or str(raw).strip() == "":
        return "other"
    text = str(raw).lower().strip()
    # Exact match first
    if text in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[text]
    # Partial match
    for alias, canonical in CATEGORY_ALIASES.items():
        if alias in text:
            return canonical
    return "other"


def normalise_agency(raw: str) -> str:
    """Standardise agency name: titlecase, collapse whitespace, strip punctuation."""
    if pd.isna(raw) or str(raw).strip() == "":
        return "Unknown Agency"
    text = re.sub(r"\s+", " ", str(raw).strip())
    # Strip trailing slashes, commas, dots
    text = re.sub(r"[/,\.]+$", "", text)
    # Expand common abbreviations
    abbrev = {
        r"\bPWD\b": "Public Works Department",
        r"\bGP\b": "Gram Panchayat",
        r"\bZP\b": "Zila Parishad",
        r"\bRES\b": "Rural Engineering Services",
        r"\bBDO\b": "Block Development Office",
        r"\bDRDA\b": "District Rural Development Agency",
        r"\bMC\b": "Municipal Corporation",
        r"\bULB\b": "Urban Local Body",
    }
    for pattern, replacement in abbrev.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text.title()


def compute_data_quality_score(row: pd.Series) -> float:
    """
    Return a 0–1 data quality score for a single row based on the
    proportion of key fields that are non-null and non-empty.
    """
    filled = 0
    for col in QUALITY_FIELDS:
        val = row.get(col, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        if isinstance(val, str) and val.strip() == "":
            continue
        if pd.isna(val):
            continue
        filled += 1
    return filled / len(QUALITY_FIELDS)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean and standardise a canonical works DataFrame.

    Steps
    -----
    1. Normalise work_category text.
    2. Standardise implementing_agency text.
    3. Remove exact-duplicate rows (all canonical columns).
    4. Clip negative amounts to 0.
    5. Compute data_quality_score per work.
    6. Forward-fill district_class for rows missing it.

    Returns
    -------
    (cleaned_df, report)
    """
    n_before = len(df)
    report: dict = {}

    # 1. Category normalisation
    df = df.copy()
    df["work_category"] = df["work_category"].apply(normalise_category)

    # 2. Agency standardisation
    df["implementing_agency"] = df["implementing_agency"].apply(normalise_agency)

    # 3. Remove exact-duplicate rows on canonical columns
    canonical_cols = [
        c for c in df.columns if not c.startswith("_")
    ]
    n_before_dup = len(df)
    df = df.drop_duplicates(subset=canonical_cols, keep="first")
    n_exact_dups = n_before_dup - len(df)
    report["exact_duplicates_removed"] = n_exact_dups

    # 4. Clip negative amounts
    for col in ["sanctioned_amount", "released_amount", "expenditure_amount"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    # 5. Data quality score
    df["data_quality_score"] = df.apply(compute_data_quality_score, axis=1)

    # 6. Fill missing district_class
    from pipeline.ingest import DISTRICT_CLASS_MAP
    mask_missing = df["district_class"].isna() | (df["district_class"] == "")
    df.loc[mask_missing, "district_class"] = (
        df.loc[mask_missing, "district"].map(DISTRICT_CLASS_MAP).fillna("plain")
    )

    # 7. Ensure sanction_year is present
    if "sanction_year" not in df.columns or df["sanction_year"].isna().all():
        df["sanction_year"] = df["sanctioned_date"].dt.year.astype("Int64")

    n_after = len(df)
    report["rows_before_clean"] = n_before
    report["rows_after_clean"] = n_after
    report["mean_data_quality"] = float(df["data_quality_score"].mean())

    logger.info(
        "Clean complete: %d → %d rows, %d exact dups removed, mean quality=%.2f",
        n_before, n_after, n_exact_dups, report["mean_data_quality"],
    )
    return df.reset_index(drop=True), report
