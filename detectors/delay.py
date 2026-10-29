"""
detectors/delay.py
Process Delay Detector.

Flags works with:
1. Sanction delay > guideline threshold (default 75 days from recommendation
   to sanction). NOTE: verify against current MPLADS guidelines (April 2023).
2. Completion time above peer group's 90th percentile.

NOTE: This detector NEVER reads the _planted_type column.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from detectors.base import BaseDetector

logger = logging.getLogger(__name__)

TODAY = pd.Timestamp.today().normalize()


class ProcessDelayDetector(BaseDetector):
    """Detect works with unusually long sanction or completion delays."""

    name = "delay"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = config.get("detectors", {}).get("delay", {})
        self.sanction_threshold: int = cfg.get("sanction_delay_threshold", 75)
        self.completion_pct: int = cfg.get("completion_percentile_threshold", 90)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect process delays.

        Returns
        -------
        pd.DataFrame
            Columns: work_id, score, reason, evidence.
        """
        df = df.copy()

        # Compute sanction delay in days
        valid_dates = df["recommended_date"].notna() & df["sanctioned_date"].notna()
        df["_sanction_delay"] = np.where(
            valid_dates,
            (df["sanctioned_date"] - df["recommended_date"]).dt.days,
            np.nan,
        )

        # Completion time (for completed works)
        comp_mask = (
            df["status"] == "completed"
        ) & df["sanctioned_date"].notna() & df["completion_date"].notna()
        df["_completion_days"] = np.where(
            comp_mask,
            (df["completion_date"] - df["sanctioned_date"]).dt.days,
            np.nan,
        )

        # Compute peer-group 90th percentile for completion time
        if "peer_group_key" in df.columns:
            p90_map = (
                df.groupby("peer_group_key")["_completion_days"]
                .quantile(self.completion_pct / 100)
                .to_dict()
            )
            df["_peer_p90_completion"] = df["peer_group_key"].map(p90_map)
        else:
            global_p90 = df["_completion_days"].quantile(self.completion_pct / 100)
            df["_peer_p90_completion"] = global_p90

        results = []
        for _, row in df.iterrows():
            score, reason, evidence = self._score_row(row)
            results.append({
                "work_id": row["work_id"],
                "score": score,
                "reason": reason,
                "evidence": evidence,
            })

        out = pd.DataFrame(results)
        logger.info(
            "ProcessDelayDetector: %d flagged out of %d works (%.1f%%)",
            (out["score"] > 0).sum(), len(out), 100 * (out["score"] > 0).mean(),
        )
        return out

    def _score_row(self, row: pd.Series) -> tuple[float, str, dict]:
        sanction_delay = row.get("_sanction_delay")
        completion_days = row.get("_completion_days")
        p90 = row.get("_peer_p90_completion")

        flags = []
        score_components = []
        evidence: dict = {}

        # Flag 1: Long sanction delay
        if pd.notna(sanction_delay) and sanction_delay > self.sanction_threshold:
            excess = sanction_delay - self.sanction_threshold
            s1 = float(np.clip(50 + (excess / self.sanction_threshold) * 30, 50, 80))
            score_components.append(s1)
            flags.append(
                f"Sanction delay {int(sanction_delay)}d "
                f"(guideline: {self.sanction_threshold}d)"
            )
            evidence["sanction_delay_days"] = int(sanction_delay)
            evidence["sanction_threshold"] = self.sanction_threshold

        # Flag 2: Abnormally long completion time vs peer p90 (> 1.25x p90)
        if pd.notna(completion_days) and pd.notna(p90) and p90 > 0 and completion_days > p90 * 1.25:
            ratio = completion_days / p90
            s2 = float(np.clip(50 + (ratio - 1) * 40, 50, 90))
            score_components.append(s2)
            flags.append(
                f"Completion time {int(completion_days)}d "
                f"substantially exceeds peer 90th percentile ({int(p90)}d)"
            )
            evidence["completion_days"] = int(completion_days)
            evidence["peer_p90_completion_days"] = float(p90)

        if not flags:
            return 0.0, "", {}

        final_score = float(np.clip(max(score_components), 0, 100))
        reason = "; ".join(flags) + "."
        evidence["sanction_delay_days"] = evidence.get("sanction_delay_days", float("nan"))
        return final_score, reason, evidence
