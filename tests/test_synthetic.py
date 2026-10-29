"""tests/test_synthetic.py"""
import pandas as pd
import pytest
from pipeline.synthetic import generate_synthetic_data

CANONICAL_COLUMNS_CHECK = [
    "work_id", "mp_ref", "constituency", "state", "district", "district_class",
    "block_or_village", "work_title", "work_description", "work_category",
    "implementing_agency", "recommended_date", "sanctioned_date",
    "sanctioned_amount", "released_amount", "expenditure_amount",
    "status", "last_progress_date", "completion_date", "completion_percent",
    "_planted_type",
]


def test_row_count(synthetic_df):
    """Should generate approximately n_works rows (duplicates may add a few more)."""
    assert len(synthetic_df) >= 20_000, "Should have at least 20,000 rows"
    assert len(synthetic_df) <= 21_000, "Should not balloon beyond reasonable range"


def test_canonical_columns_present(synthetic_df):
    for col in CANONICAL_COLUMNS_CHECK:
        assert col in synthetic_df.columns, f"Missing column: {col}"


def test_anomaly_rate(synthetic_df):
    """Overall anomaly rate should be between 1% and 6%."""
    n_anomalies = (synthetic_df["_planted_type"] != "").sum()
    rate = n_anomalies / len(synthetic_df)
    assert 0.01 <= rate <= 0.06, f"Anomaly rate {rate:.2%} out of expected range"


def test_planted_types_present(synthetic_df):
    """All 5 planted anomaly types should be present."""
    types = set(synthetic_df[synthetic_df["_planted_type"] != ""]["_planted_type"].unique())
    expected = {"cost_outlier", "duplicate", "stalled", "agency_pattern", "delay"}
    missing = expected - types
    assert not missing, f"Missing planted types: {missing}"


def test_sanctioned_amounts_positive(synthetic_df):
    """All sanctioned amounts should be positive."""
    assert (synthetic_df["sanctioned_amount"] > 0).all()


def test_status_values(synthetic_df):
    """Status should only have canonical values."""
    valid = {"completed", "in_progress", "not_started"}
    actual = set(synthetic_df["status"].unique())
    assert actual.issubset(valid), f"Unknown statuses: {actual - valid}"


def test_reproducibility():
    """Same seed should produce identical results."""
    df1 = generate_synthetic_data(n_works=500, random_seed=7)
    df2 = generate_synthetic_data(n_works=500, random_seed=7)
    assert df1.shape == df2.shape
    assert (df1["work_id"] == df2["work_id"]).all()
