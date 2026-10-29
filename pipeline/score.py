"""
pipeline/score.py
Combines detector scores into a final weighted score.
Weights are configurable via config.yaml and sidebar sliders.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DETECTOR_NAMES = ["cost", "duplicate", "stalled", "agency", "delay"]
DEFAULT_WEIGHTS = {"cost": 0.30, "duplicate": 0.25, "stalled": 0.25, "agency": 0.10, "delay": 0.10}
MULTI_DETECTOR_BONUS = 5.0


def compute_final_scores(
    scored_df: pd.DataFrame,
    weights: Optional[dict[str, float]] = None,
    multi_bonus: float = MULTI_DETECTOR_BONUS,
) -> pd.DataFrame:
    """
    Compute final composite score for each work.

    Parameters
    ----------
    scored_df : pd.DataFrame
        DataFrame with columns ``{detector}_score`` for each detector.
    weights : dict, optional
        Weight for each detector. Must sum ≤ 1. Defaults to config values.
    multi_bonus : float
        Bonus added when ≥2 detectors fire (score > 0).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``final_score`` and ``severity`` columns added.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Normalise weights so they sum to 1
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weights must sum to a positive number")
    w = {k: v / total for k, v in weights.items()}

    df = scored_df.copy()

    # Ensure all detector score columns exist
    for det in DETECTOR_NAMES:
        col = f"{det}_score"
        if col not in df.columns:
            df[col] = 0.0

    # Weighted sum
    final = sum(df[f"{det}_score"] * w.get(det, 0.0) for det in DETECTOR_NAMES)

    # Multi-detector bonus
    detectors_fired = sum(
        (df[f"{det}_score"] > 0).astype(int) for det in DETECTOR_NAMES
    )
    bonus = np.where(detectors_fired >= 2, multi_bonus, 0.0)

    df["final_score"] = np.clip(final + bonus, 0, 100).round(2)
    df["detectors_fired"] = detectors_fired

    # Severity chips
    df["severity"] = pd.cut(
        df["final_score"],
        bins=[-0.001, 40, 70, 100.001],
        labels=["Low", "Medium", "High"],
    )

    return df


def build_reason_summary(row: pd.Series) -> str:
    """
    Build a brief combined reason string from all firing detectors.

    Parameters
    ----------
    row : pd.Series
        A row from the scored DataFrame containing ``{det}_reason`` columns.

    Returns
    -------
    str
        Pipe-separated reason string.
    """
    reasons = []
    for det in DETECTOR_NAMES:
        reason_col = f"{det}_reason"
        score_col = f"{det}_score"
        if row.get(score_col, 0) > 0 and reason_col in row.index:
            r = row[reason_col]
            if r and str(r).strip():
                reasons.append(str(r).strip())
    return " | ".join(reasons) if reasons else "No specific flag raised."
