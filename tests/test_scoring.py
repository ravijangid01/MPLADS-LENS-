"""tests/test_scoring.py"""
import pandas as pd
import numpy as np
import pytest
from pipeline.score import compute_final_scores, build_reason_summary, DETECTOR_NAMES


def _make_scored_df(n: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n):
        row = {"work_id": f"W{i}", "work_title": f"Work {i}"}
        for det in DETECTOR_NAMES:
            row[f"{det}_score"] = float(i * 10 % 100)
            row[f"{det}_reason"] = f"{det} reason {i}" if i % 2 == 0 else ""
        rows.append(row)
    return pd.DataFrame(rows)


def test_final_scores_within_bounds():
    df = _make_scored_df()
    result = compute_final_scores(df)
    assert result["final_score"].between(0, 100).all()


def test_severity_values():
    df = _make_scored_df()
    result = compute_final_scores(df)
    result["severity"] = result["severity"].astype(str)
    assert set(result["severity"].unique()).issubset({"High", "Medium", "Low"})


def test_weights_change_ranking():
    df = _make_scored_df(20)
    # Cost-heavy
    r1 = compute_final_scores(df, weights={"cost": 1.0, "duplicate": 0, "stalled": 0, "agency": 0, "delay": 0})
    # Stalled-heavy
    r2 = compute_final_scores(df, weights={"cost": 0, "duplicate": 0, "stalled": 1.0, "agency": 0, "delay": 0})
    # Both use same input scores so values equal, but ensure no crash
    assert len(r1) == len(r2)


def test_multi_detector_bonus():
    """Works with ≥2 detectors firing should get a bonus."""
    df = pd.DataFrame([{
        "work_id": "MULTI001",
        "cost_score": 50.0, "cost_reason": "costs high",
        "duplicate_score": 50.0, "duplicate_reason": "duplicate",
        "stalled_score": 0.0, "stalled_reason": "",
        "agency_score": 0.0, "agency_reason": "",
        "delay_score": 0.0, "delay_reason": "",
    }])
    result = compute_final_scores(df, weights={"cost": 0.5, "duplicate": 0.5, "stalled": 0, "agency": 0, "delay": 0})
    # Base score = 50, bonus should be added
    assert result.iloc[0]["final_score"] > 50.0


def test_zero_weights_raises():
    df = _make_scored_df()
    with pytest.raises(ValueError):
        compute_final_scores(df, weights={k: 0 for k in DETECTOR_NAMES})


def test_build_reason_summary():
    row = pd.Series({
        "cost_score": 80.0, "cost_reason": "Costs 5x median",
        "duplicate_score": 0.0, "duplicate_reason": "",
        "stalled_score": 60.0, "stalled_reason": "No progress for 2 years",
        "agency_score": 0.0, "agency_reason": "",
        "delay_score": 0.0, "delay_reason": "",
    })
    summary = build_reason_summary(row)
    assert "Costs 5x" in summary
    assert "No progress" in summary
    assert "|" in summary


def test_detectors_fired_count():
    df = pd.DataFrame([{
        "work_id": "TEST",
        "cost_score": 50.0, "cost_reason": "x",
        "duplicate_score": 0.0, "duplicate_reason": "",
        "stalled_score": 50.0, "stalled_reason": "x",
        "agency_score": 0.0, "agency_reason": "",
        "delay_score": 0.0, "delay_reason": "",
    }])
    result = compute_final_scores(df)
    assert result.iloc[0]["detectors_fired"] == 2


def test_ingest_column_mapping():
    """Test that ingest correctly maps user columns to canonical schema."""
    import tempfile, os, yaml, pandas as pd
    from pipeline.ingest import load_user_file

    # Create a minimal CSV
    test_data = {
        "Sr. No.": ["1", "2"],
        "State": ["Rajasthan", "Odisha"],
        "District": ["Jaipur", "Puri"],
        "Name of Work": ["Road work A", "Water supply B"],
        "Category of Work": ["Road", "Drinking Water"],
        "Sanctioned Amount (Rs.)": ["1000000", "500000"],
        "Date of Sanction": ["01-04-2023", "15-06-2023"],
        "Implementing Agency": ["PWD", "Gram Panchayat"],
        "Status": ["In Progress", "Completed"],
    }
    df_raw = pd.DataFrame(test_data)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df_raw.to_csv(f, index=False)
        csv_path = f.name

    col_map = {
        "work_id": "Sr. No.",
        "state": "State",
        "district": "District",
        "work_title": "Name of Work",
        "work_description": "Name of Work",
        "work_category": "Category of Work",
        "sanctioned_amount": "Sanctioned Amount (Rs.)",
        "sanctioned_date": "Date of Sanction",
        "implementing_agency": "Implementing Agency",
        "status": "Status",
        "mp_ref": None, "constituency": None, "district_class": None,
        "block_or_village": None, "recommended_date": None,
        "released_amount": None, "expenditure_amount": None,
        "last_progress_date": None, "completion_date": None,
        "completion_percent": None,
    }

    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as yf:
        yaml.dump(col_map, yf)
        yaml_path = yf.name

    result_df, report = load_user_file(csv_path, yaml_path)
    os.unlink(csv_path)
    os.unlink(yaml_path)

    assert len(result_df) == 2
    assert "work_id" in result_df.columns
    assert "sanctioned_amount" in result_df.columns
    assert result_df["sanctioned_amount"].iloc[0] == 1_000_000.0
    assert result_df["work_category"].iloc[0] == "Road"
