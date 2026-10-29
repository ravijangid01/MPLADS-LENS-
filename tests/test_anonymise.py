"""tests/test_anonymise.py"""
from pipeline.anonymise import (
    anonymise_constituency, anonymise_member, anonymise_agency,
    anonymise_district, apply_anonymisation, get_display_columns,
)
import pandas as pd


def test_stable_hash_constituency():
    """Same input always produces same output."""
    a = anonymise_constituency("Jaipur Urban", "testsalt")
    b = anonymise_constituency("Jaipur Urban", "testsalt")
    assert a == b


def test_different_entities_different_alias():
    a = anonymise_constituency("Jaipur Urban", "salt")
    b = anonymise_constituency("Mumbai North", "salt")
    assert a != b


def test_alias_format_constituency():
    result = anonymise_constituency("Test", "salt")
    assert result.startswith("Constituency C-")
    num = result.split("-")[-1]
    assert 100 <= int(num) <= 999


def test_alias_format_member():
    result = anonymise_member("MP1234", "salt")
    assert result.startswith("Member M-")


def test_alias_format_agency():
    result = anonymise_agency("PWD Rajasthan", "salt")
    assert result.startswith("Agency A-")


def test_apply_anonymisation(synthetic_df):
    result = apply_anonymisation(synthetic_df.head(50))
    assert "constituency_anon" in result.columns
    assert "mp_ref_anon" in result.columns
    assert "implementing_agency_anon" in result.columns
    assert "district_anon" in result.columns
    # Anonymised values should not be raw values
    for _, row in result.iterrows():
        assert row["constituency_anon"] != row["constituency"]


def test_get_display_columns_anon():
    m = get_display_columns(anon=True)
    assert m["district"] == "district_anon"
    assert m["mp_ref"] == "mp_ref_anon"


def test_get_display_columns_raw():
    m = get_display_columns(anon=False)
    assert m["district"] == "district"
    assert m["mp_ref"] == "mp_ref"
