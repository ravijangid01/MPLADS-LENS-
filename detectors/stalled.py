"""
detectors/stalled.py
Stalled Work Detector.

Flags works that:
  1. Are not completed (status != 'completed')
  2. Were sanctioned more than N days ago (default 730)
  3. Have released funds > 0 (configurable)
  4. Have no progress update for more than M days, OR no progress recorded

NOTE: This detector NEVER reads the _planted_type column.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import numpy as np

from detectors.base import BaseDetector

logger = logging.getLogger(__name__)

TODAY = pd.Timestamp.today().normalize()


class StalledWorkDetector(BaseDetector):
    """Detect funded works that have stalled with no recent progress."""

    name = "stalled"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = config.get("detectors", {}).get("stalled", {})
        self.days_since_sanction: int = cfg.get("days_since_sanction", 730)
        self.days_no_progress: int = cfg.get("days_no_progress", 365)
        self.require_released: bool = cfg.get("require_released_funds", True)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect stalled works.

        Returns
        -------
        pd.DataFrame
            Columns: work_id, score, reason, evidence.
        """
        df = df.copy()

        results = []
        for _, row in df.iterrows():
            score, reason, evidence = self._check_row(row)
            results.append({
                "work_id": row["work_id"],
                "score": score,
                "reason": reason,
                "evidence": evidence,
            })

        out = pd.DataFrame(results)
        logger.info(
            "StalledWorkDetector: %d flagged out of %d works (%.1f%%)",
            (out["score"] > 0).sum(), len(out), 100 * (out["score"] > 0).mean(),
        )
        return out

    def _check_row(self, row: pd.Series) -> tuple[float, str, dict]:
        """Evaluate a single work row for stalled status."""
        # Must not be completed
        if row.get("status") == "completed":
            return 0.0, "", {}

        sanctioned_date = row.get("sanctioned_date")
        if pd.isna(sanctioned_date):
            return 0.0, "", {}

        days_since = (TODAY - pd.Timestamp(sanctioned_date)).days
        if days_since < self.days_since_sanction:
            return 0.0, "", {}

        # Released funds check
        released = row.get("released_amount", 0)
        if pd.isna(released):
            released = 0.0
        if self.require_released and released <= 0:
            return 0.0, "", {}

        # Progress check
        last_progress = row.get("last_progress_date")
        if pd.isna(last_progress):
            days_no_progress = days_since  # no progress ever recorded
            no_progress_str = "no progress date recorded"
        else:
            days_no_progress = (TODAY - pd.Timestamp(last_progress)).days
            no_progress_str = f"last progress {days_no_progress} days ago"

        if days_no_progress < self.days_no_progress and not pd.isna(last_progress):
            return 0.0, "", {}

        # Score: higher if older + more funds released
        staleness = min(1.0, (days_since - self.days_since_sanction) / (365 * 3))
        release_weight = min(1.0, float(released) / max(1.0, float(row.get("sanctioned_amount", 1))))
        raw_score = float(np.clip(50 + 30 * staleness + 20 * release_weight, 50, 100))

        released_lakh = float(released) / 1e5
        years_since = round(days_since / 365, 1)

        reason = (
            f"Funded ₹{released_lakh:.1f} lakh and {no_progress_str} "
            f"({years_since} years since sanction)."
        )
        evidence = {
            "days_since_sanction": days_since,
            "days_no_progress": days_no_progress,
            "released_amount": float(released),
            "sanctioned_amount": float(row.get("sanctioned_amount", 0)),
            "completion_percent": float(row.get("completion_percent", 0)),
            "sanctioned_date": str(sanctioned_date),
            "last_progress_date": str(last_progress),
            "status": row.get("status", ""),
            "recommended_date": str(row.get("recommended_date", "")),
        }
        return raw_score, reason, evidence
