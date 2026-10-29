"""
detectors/agency.py
Agency and District Pattern Detector.

Aggregates per-agency features (completion rate, stall share, mean delay,
spend-to-release ratio, mean cost z-score) and runs IsolationForest.
Explains each flag by naming the top 2 most deviant features.

NOTE: This detector NEVER reads the _planted_type column.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from detectors.base import BaseDetector

logger = logging.getLogger(__name__)

TODAY = pd.Timestamp.today().normalize()

AGENCY_FEATURES = [
    "completion_rate",
    "stall_share",
    "mean_sanction_delay_days",
    "spend_to_release_ratio",
    "mean_cost_z",
]

FEATURE_LABELS = {
    "completion_rate": "Completion rate",
    "stall_share": "Stalled works share",
    "mean_sanction_delay_days": "Average sanction delay (days)",
    "spend_to_release_ratio": "Spend-to-release ratio",
    "mean_cost_z": "Average cost z-score",
}


def _compute_agency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-agency aggregate features."""
    today = TODAY

    rows = []
    for agency, grp in df.groupby("implementing_agency"):
        n = len(grp)
        completion_rate = float((grp["status"] == "completed").mean())
        stall_share = float(
            (
                (grp["status"] != "completed")
                & (grp["released_amount"].fillna(0) > 0)
                & (
                    (today - grp["sanctioned_date"].fillna(today)).dt.days > 730
                )
            ).mean()
        )
        # Sanction delay
        valid_delay = grp.dropna(subset=["recommended_date", "sanctioned_date"])
        if len(valid_delay) > 0:
            delays = (valid_delay["sanctioned_date"] - valid_delay["recommended_date"]).dt.days
            mean_delay = float(delays.clip(lower=0).mean())
        else:
            mean_delay = float("nan")

        # Spend-to-release ratio
        released = grp["released_amount"].fillna(0).sum()
        expenditure = grp["expenditure_amount"].fillna(0).sum()
        spend_ratio = float(expenditure / released) if released > 0 else float("nan")

        # Mean cost z (within agency)
        log_costs = np.log(grp["sanctioned_amount"].clip(lower=1e-10))
        all_log_costs = np.log(df["sanctioned_amount"].clip(lower=1e-10).values)
        global_med = float(np.median(all_log_costs))
        global_mad = float(median_abs_deviation(all_log_costs, scale="normal"))
        if global_mad < 1e-10:
            global_mad = 1.0
        mean_cost_z = float(((log_costs - global_med) / global_mad).mean())

        rows.append({
            "implementing_agency": agency,
            "n_works": n,
            "completion_rate": completion_rate,
            "stall_share": stall_share,
            "mean_sanction_delay_days": mean_delay,
            "spend_to_release_ratio": spend_ratio,
            "mean_cost_z": mean_cost_z,
        })

    return pd.DataFrame(rows)


class AgencyPatternDetector(BaseDetector):
    """Detect anomalous agencies and districts using IsolationForest."""

    name = "agency"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = config.get("detectors", {}).get("agency", {})
        self.contamination: float = cfg.get("contamination", 0.08)
        self.random_state: int = cfg.get("random_state", 42)
        self.min_works: int = cfg.get("min_works_per_agency", 5)

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect agency-level outliers using IsolationForest.

        Returns
        -------
        pd.DataFrame
            Columns: work_id, score, reason, evidence.
        """
        agency_df = _compute_agency_features(df)
        eligible = agency_df[agency_df["n_works"] >= self.min_works].copy()

        if len(eligible) < 2:
            logger.warning("Too few agencies for IsolationForest; returning zero scores")
            return self._empty_result(df)

        # Impute NaN with median
        feature_matrix = eligible[AGENCY_FEATURES].copy()
        for col in AGENCY_FEATURES:
            med = feature_matrix[col].median()
            feature_matrix[col] = feature_matrix[col].fillna(med)

        # Scale and run IsolationForest
        scaler = RobustScaler()
        X = scaler.fit_transform(feature_matrix.values)

        iso = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
        )
        iso.fit(X)
        anomaly_scores = -iso.score_samples(X)  # higher = more anomalous
        # Normalise to 0-100
        min_s, max_s = anomaly_scores.min(), anomaly_scores.max()
        if max_s > min_s:
            norm_scores = (anomaly_scores - min_s) / (max_s - min_s) * 80 + 20
        else:
            norm_scores = np.zeros(len(anomaly_scores))
        predictions = iso.predict(X)  # -1 = outlier

        eligible = eligible.copy()
        eligible["_iso_score"] = norm_scores
        eligible["_iso_pred"] = predictions

        # Peer medians for explanation
        peer_medians = feature_matrix.median().to_dict()

        # Build agency → score map
        agency_scores: dict[str, float] = {}
        agency_reasons: dict[str, str] = {}
        agency_evidences: dict[str, dict] = {}

        for _, ag_row in eligible.iterrows():
            agency = ag_row["implementing_agency"]
            is_outlier = ag_row["_iso_pred"] == -1
            raw_score = float(ag_row["_iso_score"]) if is_outlier else 0.0

            # Top 2 most deviant features (vs peer median)
            deviations = {}
            for feat in AGENCY_FEATURES:
                val = feature_matrix.loc[ag_row.name, feat]
                med = peer_medians[feat]
                if not np.isnan(val) and med != 0:
                    deviations[feat] = abs(val - med) / (abs(med) + 1e-10)
                elif not np.isnan(val):
                    deviations[feat] = abs(val)

            top2 = sorted(deviations, key=deviations.get, reverse=True)[:2]

            reason_parts = []
            for feat in top2:
                val = feature_matrix.loc[ag_row.name, feat]
                med = peer_medians[feat]
                label = FEATURE_LABELS.get(feat, feat)
                if feat in ("completion_rate",):
                    reason_parts.append(
                        f"{label} {val:.0%} vs peer median {med:.0%}"
                    )
                elif feat == "stall_share":
                    reason_parts.append(
                        f"{label} {val:.0%} vs peer median {med:.0%}"
                    )
                elif feat == "mean_sanction_delay_days":
                    reason_parts.append(
                        f"{label} {val:.0f}d vs peer median {med:.0f}d"
                    )
                else:
                    reason_parts.append(
                        f"{label} {val:.2f} vs peer median {med:.2f}"
                    )

            reason = "; ".join(reason_parts) if (is_outlier and reason_parts) else ""

            evidence = {
                "agency": agency,
                "n_works": int(ag_row["n_works"]),
                "features": feature_matrix.loc[ag_row.name].to_dict(),
                "peer_medians": peer_medians,
                "top_deviant_features": top2,
                "isolation_score": float(ag_row["_iso_score"]),
            }

            agency_scores[agency] = raw_score
            agency_reasons[agency] = reason
            agency_evidences[agency] = evidence

        # Map agency scores back to works
        results = []
        for _, row in df.iterrows():
            agency = row["implementing_agency"]
            score = agency_scores.get(agency, 0.0)
            reason = agency_reasons.get(agency, "")
            evidence = agency_evidences.get(agency, {})
            results.append({
                "work_id": row["work_id"],
                "score": score,
                "reason": reason,
                "evidence": evidence,
            })

        # Persist agency-level DataFrame for the Agencies page
        self.agency_df = eligible[
            ["implementing_agency", "n_works"] + AGENCY_FEATURES + ["_iso_score", "_iso_pred"]
        ].copy()

        out = pd.DataFrame(results)
        logger.info(
            "AgencyPatternDetector: %d works flagged (from %d anomalous agencies)",
            (out["score"] > 0).sum(),
            (eligible["_iso_pred"] == -1).sum(),
        )
        return out
