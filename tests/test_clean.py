"""tests/test_clean.py"""
import pandas as pd
import numpy as np
from pipeline.clean import clean, normalise_category, normalise_agency


def test_normalise_category_road():
    assert normalise_category("CC Road") == "road"
    assert normalise_category("Village Road") == "road"


def test_normalise_category_health():
    assert normalise_category("PHC") == "health"
    assert normalise_category("Sub Centre") == "health"


def test_normalise_category_unknown():
    assert normalise_category("") == "other"
    assert normalise_category(None) == "other"


def test_normalise_agency():
    result = normalise_agency("  PWD  ")
    assert "Public Works Department" in result


def test_clean_removes_exact_dups(synthetic_df):
    # Create a dataframe with an artificial exact dup
    import copy
    df = synthetic_df.copy().head(100)
    dup = df.iloc[[0]].copy()
    dup_df = pd.concat([df, dup], ignore_index=True)
    cleaned, report = clean(dup_df)
    assert report["exact_duplicates_removed"] >= 1


def test_clean_data_quality_score(synthetic_df):
    df, report = clean(synthetic_df.copy())
    assert "data_quality_score" in df.columns
    assert 0 <= report["mean_data_quality"] <= 1.0


def test_clean_negative_amounts(synthetic_df):
    df = synthetic_df.copy()
    df.loc[0, "sanctioned_amount"] = -100000
    cleaned, _ = clean(df)
    assert cleaned["sanctioned_amount"].min() >= 0


def test_clean_row_count(synthetic_df):
    df, report = clean(synthetic_df.copy())
    assert report["rows_after_clean"] <= report["rows_before_clean"]
