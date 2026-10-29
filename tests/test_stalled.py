"""tests/test_stalled.py"""
import pandas as pd
import pytest
from detectors.stalled import StalledWorkDetector


def test_stalled_detector_output_columns(peered_df, config):
    det = StalledWorkDetector(config)
    result = det.detect(peered_df.head(200))
    assert "work_id" in result.columns
    assert "score" in result.columns
    assert "reason" in result.columns
    assert "evidence" in result.columns


def test_stalled_detector_score_bounds(peered_df, config):
    det = StalledWorkDetector(config)
    result = det.detect(peered_df.head(500))
    assert result["score"].between(0, 100).all()


def test_stalled_detector_flag_rate(peered_df, config):
    det = StalledWorkDetector(config)
    result = det.detect(peered_df)
    flag_rate = (result["score"] > 0).mean()
    assert flag_rate <= 0.08, f"Stalled flag rate too high: {flag_rate:.2%}"


def test_stalled_detector_recall(peered_df, config):
    """Should recover ≥ 80% of planted stalled anomalies."""
    det = StalledWorkDetector(config)
    result = det.detect(peered_df)
    merged = peered_df.merge(result[["work_id", "score"]], on="work_id", how="left")
    planted = merged[merged["_planted_type"] == "stalled"]
    if len(planted) == 0:
        pytest.skip("No planted stalled anomalies in this dataset")
    recall = (planted["score"] > 0).mean()
    assert recall >= 0.80, f"Stalled recall {recall:.0%} below 80%"


def test_stalled_completed_works_not_flagged(config):
    """Completed works should never be flagged as stalled."""
    data = [{
        "work_id": "COMP001", "work_category": "road", "district": "D1",
        "block_or_village": "B1", "state": "S1", "district_class": "plain",
        "work_title": "Road work", "work_description": "Road work",
        "implementing_agency": "PWD", "mp_ref": "MP001", "constituency": "C1",
        "sanctioned_date": pd.Timestamp("2020-01-01"),  # old
        "recommended_date": pd.Timestamp("2019-11-01"),
        "completion_date": pd.Timestamp("2021-01-01"),
        "last_progress_date": pd.Timestamp("2021-01-01"),
        "sanctioned_amount": 1e6, "released_amount": 1e6, "expenditure_amount": 1e6,
        "status": "completed", "completion_percent": 100,
        "sanction_year": 2020, "_planted_type": "", "data_quality_score": 1.0,
    }]
    df = pd.DataFrame(data)
    det = StalledWorkDetector(config)
    result = det.detect(df)
    assert result.iloc[0]["score"] == 0.0, "Completed works must not be flagged as stalled"


def test_stalled_no_funds_not_flagged(config):
    """Works with no released funds should not be flagged (when require_released=True)."""
    data = [{
        "work_id": "NOFUND001", "work_category": "road", "district": "D1",
        "block_or_village": "B1", "state": "S1", "district_class": "plain",
        "work_title": "Road work", "work_description": "Road work",
        "implementing_agency": "PWD", "mp_ref": "MP001", "constituency": "C1",
        "sanctioned_date": pd.Timestamp("2020-01-01"),
        "recommended_date": pd.Timestamp("2019-11-01"),
        "completion_date": pd.NaT, "last_progress_date": pd.NaT,
        "sanctioned_amount": 1e6, "released_amount": 0,  # no release
        "expenditure_amount": 0,
        "status": "not_started", "completion_percent": 0,
        "sanction_year": 2020, "_planted_type": "", "data_quality_score": 0.7,
    }]
    df = pd.DataFrame(data)
    det = StalledWorkDetector(config)
    result = det.detect(df)
    assert result.iloc[0]["score"] == 0.0, "Unfunded works should not be stalled-flagged"
