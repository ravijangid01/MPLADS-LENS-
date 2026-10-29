"""tests/test_cost.py"""
import pandas as pd
import pytest
from detectors.cost import CostOutlierDetector


def test_cost_detector_output_columns(peered_df, config):
    det = CostOutlierDetector(config)
    result = det.detect(peered_df)
    assert "work_id" in result.columns
    assert "score" in result.columns
    assert "reason" in result.columns
    assert "evidence" in result.columns


def test_cost_detector_score_bounds(peered_df, config):
    det = CostOutlierDetector(config)
    result = det.detect(peered_df)
    assert result["score"].between(0, 100).all(), "Scores must be 0-100"


def test_cost_detector_flag_rate(peered_df, config):
    """Flag rate should be ≤ 8% overall."""
    det = CostOutlierDetector(config)
    result = det.detect(peered_df)
    flag_rate = (result["score"] > 0).mean()
    assert flag_rate <= 0.08, f"Flag rate too high: {flag_rate:.2%}"


def test_cost_detector_recall(peered_df, config):
    """Should recover ≥ 80% of planted cost_outlier anomalies."""
    det = CostOutlierDetector(config)
    result = det.detect(peered_df)
    merged = peered_df.merge(result[["work_id", "score"]], on="work_id", how="left")
    planted = merged[merged["_planted_type"] == "cost_outlier"]
    if len(planted) == 0:
        pytest.skip("No planted cost_outlier anomalies in this dataset")
    recall = (planted["score"] > 0).mean()
    assert recall >= 0.80, f"Cost recall {recall:.0%} below 80% target"


def test_cost_detector_reason_for_flagged(peered_df, config):
    """Flagged works should have non-empty reasons."""
    det = CostOutlierDetector(config)
    result = det.detect(peered_df)
    flagged = result[result["score"] > 0]
    assert (flagged["reason"].str.len() > 0).all(), "All flagged works must have reasons"


def test_cost_detector_fallback_peer_groups(config):
    """Fallback to wider peer group when group is too small."""
    import numpy as np
    # Create tiny dataset where level-0 groups have < 8 members
    data = []
    for i in range(20):
        data.append({
            "work_id": f"W{i}", "work_category": "road", "district_class": "plain",
            "state": "TestState", "district": "TestDist", "block_or_village": "B1",
            "sanctioned_amount": 1e6 * (i + 1),
            "recommended_date": pd.Timestamp("2023-01-01"),
            "sanctioned_date": pd.Timestamp("2023-03-01"),
            "completion_date": pd.NaT, "last_progress_date": pd.NaT,
            "status": "in_progress", "completion_percent": 0,
            "released_amount": 5e5, "expenditure_amount": 2e5,
            "implementing_agency": "PWD", "mp_ref": "MP001",
            "constituency": "C1", "sanction_year": 2015 + (i % 5),  # 4 items per year, < 8
            "_planted_type": "", "data_quality_score": 0.9,
        })
    # Add a big cost outlier
    data[0]["sanctioned_amount"] = 1e9  # extreme outlier
    data[0]["_planted_type"] = "cost_outlier"
    df = pd.DataFrame(data)
    from pipeline.peer_groups import assign_peer_groups
    df = assign_peer_groups(df, min_group_size=8)
    # All works should use level 1 or 2 (too few for level 0)
    assert (df["peer_level"] > 0).all()
    det = CostOutlierDetector(config)
    result = det.detect(df)
    # Outlier should still be detected
    outlier_row = result[result["work_id"] == "W0"]
    assert len(outlier_row) == 1 and outlier_row.iloc[0]["score"] > 0
