"""
detectors/base.py
Abstract base class for all MPLADS-Lens detectors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseDetector(ABC):
    """
    Abstract base class for outlier detectors.

    Each detector must implement ``detect(df)`` which returns a DataFrame
    with columns:
    - ``work_id``
    - ``score``: float 0–100
    - ``reason``: plain-English explanation string
    - ``evidence``: dict with structured evidence for the UI
    """

    name: str = "base"

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    @abstractmethod
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run detection on the works DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned canonical works DataFrame (with peer group columns).

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per work, columns:
            work_id, score (0–100), reason (str), evidence (dict).
        """
        ...

    def _empty_result(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a zero-score result for all works (used when detector cannot run)."""
        return pd.DataFrame({
            "work_id": df["work_id"],
            "score": 0.0,
            "reason": "",
            "evidence": [{}] * len(df),
        })
