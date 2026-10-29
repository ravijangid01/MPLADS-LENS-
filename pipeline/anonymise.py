"""
pipeline/anonymise.py
Stable hash-based anonymisation: the same entity always gets the same alias.
Uses SHA-256(entity_type + raw_value + salt) truncated to 3 digits.

The anonymised view is ON by default; toggled off via sidebar for user's
own local data only. Real names are never hard-coded.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SALT = "mplads-lens-demo-2026"


@lru_cache(maxsize=4096)
def _hash_alias(entity_type: str, raw_value: str, salt: str) -> str:
    """Return a stable 3-digit numeric alias for an entity."""
    h = hashlib.sha256(f"{entity_type}|{raw_value}|{salt}".encode()).hexdigest()
    # Take first 6 hex chars → 0..16777215, mod 900 + 100 → 100..999
    num = int(h[:6], 16) % 900 + 100
    return str(num)


def anonymise_constituency(raw: str, salt: str = DEFAULT_SALT) -> str:
    """Return 'Constituency C-NNN' for a raw constituency name."""
    return f"Constituency C-{_hash_alias('constituency', raw, salt)}"


def anonymise_member(raw: str, salt: str = DEFAULT_SALT) -> str:
    """Return 'Member M-NNN' for a raw MP reference."""
    return f"Member M-{_hash_alias('member', raw, salt)}"


def anonymise_agency(raw: str, salt: str = DEFAULT_SALT) -> str:
    """Return 'Agency A-NNN' for a raw agency name."""
    return f"Agency A-{_hash_alias('agency', raw, salt)}"


def anonymise_district(raw: str, salt: str = DEFAULT_SALT) -> str:
    """Return 'District D-NNN' for a raw district name."""
    return f"District D-{_hash_alias('district', raw, salt)}"


def apply_anonymisation(
    df: pd.DataFrame,
    salt: str = DEFAULT_SALT,
) -> pd.DataFrame:
    """
    Return a copy of the DataFrame with anonymised display columns added.
    Originals are preserved; anonymised versions are suffixed with ``_anon``.

    Parameters
    ----------
    df : pd.DataFrame
        Canonical works DataFrame.
    salt : str
        Anonymisation salt.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional ``*_anon`` columns.
    """
    df = df.copy()
    df["constituency_anon"] = df["constituency"].apply(
        lambda x: anonymise_constituency(str(x), salt) if pd.notna(x) else "Unknown"
    )
    df["mp_ref_anon"] = df["mp_ref"].apply(
        lambda x: anonymise_member(str(x), salt) if pd.notna(x) else "Unknown"
    )
    df["implementing_agency_anon"] = df["implementing_agency"].apply(
        lambda x: anonymise_agency(str(x), salt) if pd.notna(x) else "Unknown"
    )
    df["district_anon"] = df["district"].apply(
        lambda x: anonymise_district(str(x), salt) if pd.notna(x) else "Unknown"
    )
    return df


def get_display_columns(anon: bool = True) -> dict[str, str]:
    """
    Return a dict mapping canonical column → display column name
    based on the anonymisation toggle.

    Parameters
    ----------
    anon : bool
        If True, return anonymised column names.

    Returns
    -------
    dict
        Mapping used by UI pages to pick the right column.
    """
    if anon:
        return {
            "constituency": "constituency_anon",
            "mp_ref": "mp_ref_anon",
            "implementing_agency": "implementing_agency_anon",
            "district": "district_anon",
        }
    return {
        "constituency": "constituency",
        "mp_ref": "mp_ref",
        "implementing_agency": "implementing_agency",
        "district": "district",
    }
