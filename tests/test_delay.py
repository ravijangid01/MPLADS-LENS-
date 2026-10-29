"""tests/test_delay.py"""
import pandas as pd
import pytest
from detectors.delay import ProcessDelayDetector


def test_delay_detector_output_columns(peered_df, config):
    det = ProcessDelayDetector(config)
    result = det.detect(peered_df.head(500))
    assert "work_id" in result.columns
    assert "score" in result.columns
    assert "reason" in result.columns
    assert "evidence" in result.columns


def test_delay_score_bounds(peered_df, config):
    det = ProcessDelayDetector(config)
    result = det.detect(peered_df.head(500))
    assert result["score"].between(0, 100).all()


def test_delay_flag_rate(peered_df, config):
    det = ProcessDelayDetector(config)
    result = det.detect(peered_df)
    flag_rate = (result["score"] > 0).mean()
    assert flag_rate <= 0.08, f"Delay flag rate too high: {flag_rate:.2%}"


def test_delay_detector_recall(peered_df, config):
    """Should recover ≥ 80% of planted delay anomalies."""
    det = ProcessDelayDetector(config)
    result = det.detect(peered_df)
    merged = peered_df.merge(result[["work_id", "score"]], on="work_id", how="left")
    planted = merged[merged["_planted_type"] == "delay"]
    if len(planted) == 0:
        pytest.skip("No planted delay anomalies")
    recall = (planted["score"] > 0).mean()
    assert recall >= 0.80, f"Delay recall {recall:.0%} below 80%"


def test_delay_within_guideline_not_flagged(config):
    """Works with sanction delay ≤ threshold should not be flagged for delay."""
    data = [{
        "work_id": "NODELAY001", "work_category": "road",
        "block_or_village": "B1", "district": "D1", "state": "S1",
        "district_class": "plain",
        "work_title": "Road", "work_description": "Road",
        "implementing_agency": "PWD", "mp_ref": "MP001", "constituency": "C1",
        "recommended_date": pd.Timestamp("2023-01-01"),
        "sanctioned_date": pd.Timestamp("2023-02-15"),  # 45 days < 75
        "completion_date": pd.Timestamp("2023-10-01"),
        "last_progress_date": pd.Timestamp("2023-10-01"),
        "sanctioned_amount": 1e6, "released_amount": 1e6, "expenditure_amount": 1e6,
        "status": "completed", "completion_percent": 100,
        "sanction_year": 2023, "_planted_type": "", "data_quality_score": 1.0,
        "peer_group_key": "road||plain||2023", "peer_level": 0,
    }]
    df = pd.DataFrame(data)
    det = ProcessDelayDetector(config)
    result = det.detect(df)
    assert result.iloc[0]["score"] == 0.0, "Work within guideline should not be flagged for delay"
