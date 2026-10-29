"""tests/conftest.py — shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
import pandas as pd

from pipeline.synthetic import generate_synthetic_data
from pipeline.clean import clean
from pipeline.peer_groups import assign_peer_groups


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """Full synthetic dataset (20k works). Session-scoped for speed."""
    return generate_synthetic_data(n_works=20_000, random_seed=42, anomaly_rate=0.035)


@pytest.fixture(scope="session")
def small_df() -> pd.DataFrame:
    """Smaller synthetic dataset for unit tests (1,000 works)."""
    return generate_synthetic_data(n_works=1_000, random_seed=99, anomaly_rate=0.05)


@pytest.fixture(scope="session")
def cleaned_df(synthetic_df) -> pd.DataFrame:
    df, _ = clean(synthetic_df)
    return df


@pytest.fixture(scope="session")
def peered_df(cleaned_df) -> pd.DataFrame:
    return assign_peer_groups(cleaned_df, min_group_size=8)


@pytest.fixture(scope="session")
def config() -> dict:
    import yaml
    with open(ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)
