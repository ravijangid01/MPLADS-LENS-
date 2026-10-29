"""tests/test_agency.py"""
import pandas as pd
import pytest
from detectors.agency import AgencyPatternDetector


def test_agency_detector_output_columns(peered_df, config):
    det = AgencyPatternDetector(config)
    result = det.detect(peered_df)
    assert "work_id" in result.columns
    assert "score" in result.columns
    assert "reason" in result.columns
    assert "evidence" in result.columns


def test_agency_detector_score_bounds(peered_df, config):
    det = AgencyPatternDetector(config)
    result = det.detect(peered_df)
    assert result["score"].between(0, 100).all()


def test_agency_detector_flag_rate(peered_df, config):
    det = AgencyPatternDetector(config)
    result = det.detect(peered_df)
    flag_rate = (result["score"] > 0).mean()
    assert flag_rate <= 0.15, f"Agency flag rate too high: {flag_rate:.2%}"


def test_agency_detector_deterministic(peered_df, config):
    """Two runs should produce identical results (fixed seed)."""
    det1 = AgencyPatternDetector(config)
    det2 = AgencyPatternDetector(config)
    r1 = det1.detect(peered_df)
    r2 = det2.detect(peered_df)
    assert (r1["score"].round(4) == r2["score"].round(4)).all()


def test_agency_flagged_works_have_reasons(peered_df, config):
    det = AgencyPatternDetector(config)
    result = det.detect(peered_df)
    flagged = result[result["score"] > 0]
    if len(flagged) == 0:
        pytest.skip("No agency anomalies detected")
    assert (flagged["reason"].str.len() > 0).all(), "Flagged agency works must have reasons"


def test_agency_evidence_has_features(peered_df, config):
    det = AgencyPatternDetector(config)
    result = det.detect(peered_df)
    flagged = result[result["score"] > 0]
    if len(flagged) == 0:
        pytest.skip("No agency anomalies")
    for _, row in flagged.head(5).iterrows():
        ev = row["evidence"]
        assert isinstance(ev, dict)
        assert "features" in ev
        assert "peer_medians" in ev
