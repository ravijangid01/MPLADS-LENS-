"""tests/test_duplicate.py"""
import pandas as pd
import pytest
from detectors.duplicate import NearDuplicateDetector


def test_duplicate_detector_output_columns(peered_df, config):
    det = NearDuplicateDetector(config)
    result = det.detect(peered_df.head(500))  # use subset for speed
    assert "work_id" in result.columns
    assert "score" in result.columns
    assert "reason" in result.columns
    assert "evidence" in result.columns


def test_duplicate_detector_score_bounds(peered_df, config):
    det = NearDuplicateDetector(config)
    result = det.detect(peered_df.head(500))
    assert result["score"].between(0, 100).all()


def test_duplicate_detector_flag_rate(peered_df, config):
    """Flag rate should be ≤ 8%."""
    det = NearDuplicateDetector(config)
    result = det.detect(peered_df)
    flag_rate = (result["score"] > 0).mean()
    assert flag_rate <= 0.08, f"Duplicate flag rate too high: {flag_rate:.2%}"


def test_duplicate_detector_obvious_pair(config):
    """Two nearly identical works at same location should be detected."""
    data = []
    for i in range(20):
        data.append({
            "work_id": f"BG{i}", "work_category": "road",
            "block_or_village": "TestBlock", "district": "D1",
            "work_title": f"Misc road work {i}",
            "work_description": f"Misc road work {i}",
            "sanctioned_date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=i * 30),
            "recommended_date": pd.Timestamp("2022-11-01"),
            "sanctioned_amount": 1e6, "released_amount": 5e5,
            "expenditure_amount": 2e5, "status": "in_progress",
            "completion_percent": 30, "completion_date": pd.NaT,
            "last_progress_date": pd.Timestamp("2023-06-01"),
            "mp_ref": "MP001", "constituency": "C1",
            "state": "TestState", "district_class": "plain",
            "implementing_agency": "PWD", "sanction_year": 2023,
            "_planted_type": "", "data_quality_score": 0.9,
        })
    # Plant a near-duplicate pair
    data[0]["work_title"] = "Construction of CC Road in TestBlock"
    data[0]["work_description"] = "Construction of CC Road in TestBlock"
    data[0]["sanctioned_date"] = pd.Timestamp("2023-01-01")
    data[1]["work_title"] = "Construction of CC Road in TestBlock Phase II"
    data[1]["work_description"] = "Construction of CC Road in TestBlock Phase II"
    data[1]["sanctioned_date"] = pd.Timestamp("2023-02-15")  # 45 days apart

    df = pd.DataFrame(data)
    det = NearDuplicateDetector(config)
    result = det.detect(df)

    flagged = result[result["score"] > 0]
    # At least one of the planted pair should be flagged
    assert len(flagged) >= 1, "Obvious near-duplicate pair should be detected"


def test_duplicate_evidence_has_pair(peered_df, config):
    """Flagged duplicate should have match_work_id in evidence."""
    det = NearDuplicateDetector(config)
    result = det.detect(peered_df.head(300))
    flagged = result[result["score"] > 0]
    if len(flagged) == 0:
        pytest.skip("No duplicates flagged in small subset")
    for _, row in flagged.iterrows():
        ev = row["evidence"]
        assert isinstance(ev, dict)
        assert "match_work_id" in ev or "work_a" in ev
