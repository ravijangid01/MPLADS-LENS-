"""
pipeline/peer_groups.py
Assigns peer group labels to each work with a fallback hierarchy:
  Level 0 (tightest): work_category × district_class × sanction_year
  Level 1 (fallback): work_category × state
  Level 2 (widest):   work_category only

Records the fallback level used for each work in the column `peer_level`.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def assign_peer_groups(
    df: pd.DataFrame,
    min_group_size: int = 8,
) -> pd.DataFrame:
    """
    Assign peer group keys and the fallback level used.

    Adds two columns to the DataFrame:
    - ``peer_group_key``: string key identifying the peer group
    - ``peer_level``: 0 = tightest, 1 = category×state, 2 = category only

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned canonical works DataFrame.
    min_group_size : int
        Minimum works in a group before falling back to a wider group.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``peer_group_key`` and ``peer_level`` columns added.
    """
    df = df.copy()

    # Pre-compute group sizes at each level
    df["_pg_0"] = (
        df["work_category"].astype(str)
        + "||"
        + df["district_class"].astype(str)
        + "||"
        + df["sanction_year"].astype(str)
    )
    df["_pg_1"] = df["work_category"].astype(str) + "||" + df["state"].astype(str)
    df["_pg_2"] = df["work_category"].astype(str)

    sizes_0 = df.groupby("_pg_0")["_pg_0"].transform("count")
    sizes_1 = df.groupby("_pg_1")["_pg_1"].transform("count")

    peer_group_key: list[str] = []
    peer_level: list[int] = []

    for i, row in df.iterrows():
        if sizes_0.loc[i] >= min_group_size:
            peer_group_key.append(row["_pg_0"])
            peer_level.append(0)
        elif sizes_1.loc[i] >= min_group_size:
            peer_group_key.append(row["_pg_1"])
            peer_level.append(1)
        else:
            peer_group_key.append(row["_pg_2"])
            peer_level.append(2)

    df["peer_group_key"] = peer_group_key
    df["peer_level"] = peer_level

    # Clean up temp columns
    df = df.drop(columns=["_pg_0", "_pg_1", "_pg_2"])

    level_counts = pd.Series(peer_level).value_counts().to_dict()
    logger.info("Peer group levels assigned: %s", level_counts)
    return df


def describe_peer_group(peer_group_key: str, peer_level: int) -> str:
    """
    Return a human-readable description of a peer group.

    Parameters
    ----------
    peer_group_key : str
        The peer group key string.
    peer_level : int
        The fallback level (0, 1, or 2).

    Returns
    -------
    str
        Plain English description.
    """
    parts = peer_group_key.split("||")
    if peer_level == 0 and len(parts) == 3:
        cat, dist_class, year = parts
        return f"{cat.title()} works in {dist_class} districts ({year})"
    elif peer_level == 1 and len(parts) == 2:
        cat, state = parts
        return f"{cat.title()} works in {state} (state-level fallback)"
    else:
        return f"{parts[0].title()} works (category-level fallback)"
