"""
detectors/cost.py
Cost Outlier Detector.

Peer group: work_category × district_class × sanction_year (falls back to
category × state, then category only if group too small).
Uses log(sanctioned_amount) with robust z-score (median + MAD).
Flags upward outliers above a configurable threshold.

NOTE: This detector NEVER reads the _planted_type column.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

from detectors.base import BaseDetector
from pipeline.peer_groups import describe_peer_group

logger = logging.getLogger(__name__)

EPSILON = 1e-10  # avoid log(0)


class CostOutlierDetector(BaseDetector):
    """Detect works whose cost is a statistical outlier within their peer group."""

    name = "cost"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = config.get("detectors", {}).get("cost", {})
        self.threshold: float = cfg.get("robust_z_threshold", 3.5)
        self.min_group_size: int = cfg.get("min_peer_group_size", 8)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect cost outliers using robust z-score within peer groups.

        Returns
        -------
        pd.DataFrame
            Columns: work_id, score, reason, evidence.
        """
        if "peer_group_key" not in df.columns:
            logger.warning("peer_group_key missing; returning zero scores")
            return self._empty_result(df)

        df = df.copy()
        df["_log_cost"] = np.log(df["sanctioned_amount"].clip(lower=EPSILON))

        results = []
        for pg_key, group in df.groupby("peer_group_key"):
            n = len(group)
            log_costs = group["_log_cost"].values
            med = float(np.median(log_costs))
            mad = float(median_abs_deviation(log_costs, scale="normal"))
            if mad < EPSILON:
                mad = EPSILON  # prevent division by zero for uniform groups

            for i, row in group.iterrows():
                z = (row["_log_cost"] - med) / mad
                # Only flag upward outliers
                if z > self.threshold:
                    raw_score = min(100.0, ((z - self.threshold) / self.threshold) * 50 + 50)
                    raw_score = float(np.clip(raw_score, 50, 100))
                else:
                    raw_score = 0.0

                peer_level = int(row.get("peer_level", 0))
                pg_desc = describe_peer_group(str(pg_key), peer_level)

                # Ratio of this work's cost to peer median
                cost_ratio = np.exp(row["_log_cost"] - med) if med != 0 else 1.0
                cost_ratio = float(cost_ratio)

                reason = ""
                if raw_score > 0:
                    reason = (
                        f"Costs {cost_ratio:.1f}× the median of {n} similar works "
                        f"({pg_desc}). Robust z-score: {z:.1f}."
                    )

                # Nearest peers (same group, sorted by cost)
                peer_sample = (
                    group[["work_id", "sanctioned_amount", "work_category", "district"]]
                    .sort_values("sanctioned_amount")
                    .head(10)
                    .to_dict("records")
                )

                evidence = {
                    "peer_group_key": str(pg_key),
                    "peer_group_description": pg_desc,
                    "peer_level": peer_level,
                    "peer_group_size": n,
                    "peer_median_cost": float(np.exp(med)),
                    "peer_mad": float(mad),
                    "robust_z": float(z),
                    "cost_ratio": cost_ratio,
                    "peer_costs": group["sanctioned_amount"].tolist(),
                    "peer_sample": peer_sample,
                }

                results.append({
                    "work_id": row["work_id"],
                    "score": raw_score,
                    "reason": reason,
                    "evidence": evidence,
                })

        out = pd.DataFrame(results)
        logger.info(
            "CostOutlierDetector: %d flagged out of %d works (%.1f%%)",
            (out["score"] > 0).sum(), len(out), 100 * (out["score"] > 0).mean(),
        )
        return out
