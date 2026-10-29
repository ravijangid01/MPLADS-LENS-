"""
pipeline/synthetic.py
Generates ~20,000 synthetic MPLADS works across 4 states and 40 districts
with realistic cost distributions, terrain effects, and planted anomalies.

The planted anomaly type is stored in a hidden column `_planted_type` which
is NEVER read by any detector — it is used only by tests and the Method page.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

STATES = {
    "Rajasthan": {
        "districts": [
            ("Jaisalmer", "desert"), ("Barmer", "desert"), ("Bikaner", "desert"),
            ("Jodhpur", "plain"), ("Jaipur", "plain"), ("Ajmer", "plain"),
            ("Udaipur", "hill"), ("Sirohi", "hill"), ("Chittorgarh", "hill"),
            ("Dungarpur", "tribal"),
        ]
    },
    "Himachal Pradesh": {
        "districts": [
            ("Lahaul and Spiti", "hill"), ("Kinnaur", "hill"), ("Chamba", "hill"),
            ("Kullu", "hill"), ("Shimla", "hill"), ("Mandi", "hill"),
            ("Kangra", "hill"), ("Solan", "plain"), ("Hamirpur", "plain"),
            ("Una", "plain"),
        ]
    },
    "Odisha": {
        "districts": [
            ("Puri", "coastal"), ("Kendrapara", "coastal"), ("Jagatsinghpur", "coastal"),
            ("Balasore", "coastal"), ("Koraput", "tribal"), ("Nabarangpur", "tribal"),
            ("Rayagada", "tribal"), ("Bhubaneswar", "plain"), ("Cuttack", "plain"),
            ("Ganjam", "plain"),
        ]
    },
    "Maharashtra": {
        "districts": [
            ("Ratnagiri", "coastal"), ("Sindhudurg", "coastal"), ("Nashik", "plain"),
            ("Pune", "plain"), ("Nagpur", "plain"), ("Amravati", "plain"),
            ("Gadchiroli", "tribal"), ("Nandurbar", "tribal"), ("Satara", "hill"),
            ("Kolhapur", "hill"),
        ]
    },
}

CATEGORIES = [
    "road", "community hall", "drinking water", "school",
    "health", "sanitation", "other",
]

# Base cost medians in rupees (lakh) per category
CATEGORY_COST_MEDIANS = {
    "road": 35.0,
    "community hall": 25.0,
    "drinking water": 18.0,
    "school": 22.0,
    "health": 28.0,
    "sanitation": 12.0,
    "other": 20.0,
}

# Terrain multipliers (hill/desert legitimately cost more)
TERRAIN_MULTIPLIERS = {
    "plain": 1.0,
    "hill": 1.45,
    "desert": 1.35,
    "tribal": 1.20,
    "coastal": 1.15,
}

DISTRICT_CLASSES = list(TERRAIN_MULTIPLIERS.keys())

AGENCIES = [
    "Public Works Department", "Gram Panchayat", "Zila Parishad",
    "NHAI", "State Road Corporation", "Municipal Corporation",
    "Rural Engineering Services", "Block Development Office",
    "District Rural Development Agency", "State Water Board",
    "School Education Department", "District Health Mission",
    "Pradhan Mantri Gram Sadak Yojana", "National Rural Infrastructure",
    "Urban Local Body",
]

STATUSES = ["completed", "in_progress", "not_started"]

# Planted anomaly breakdown (proportions must sum to 1.0)
PLANTED_TYPES = {
    "cost_outlier": 0.28,
    "duplicate": 0.22,
    "stalled": 0.26,
    "agency_pattern": 0.12,
    "delay": 0.12,
}


def _stable_id(prefix: str, index: int) -> str:
    """Generate a stable work ID."""
    return f"{prefix}-{index:06d}"


def _lognormal_cost(median_lakh: float, sigma: float, rng: np.random.Generator) -> float:
    """Sample from log-normal with given median (in lakhs), return rupees."""
    mu = np.log(median_lakh)
    return float(np.exp(rng.normal(mu, sigma))) * 1e5  # convert lakh → rupees


def generate_synthetic_data(
    n_works: int = 20_000,
    random_seed: int = 42,
    anomaly_rate: float = 0.035,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate synthetic MPLADS works dataset.

    Parameters
    ----------
    n_works : int
        Number of works to generate.
    random_seed : int
        Fixed seed for reproducibility.
    anomaly_rate : float
        Approximate fraction of works with planted anomalies.
    output_path : str, optional
        If provided, save as parquet at this path.

    Returns
    -------
    pd.DataFrame
        Canonical works DataFrame including hidden ``_planted_type`` column.
    """
    rng = np.random.default_rng(random_seed)
    logger.info("Generating %d synthetic works (seed=%d)", n_works, random_seed)

    rows = []
    # Flatten states / districts
    all_districts: list[tuple[str, str, str, str]] = []  # (state, district, class, abbr)
    for state, info in STATES.items():
        for dist_name, dist_class in info["districts"]:
            all_districts.append((state, dist_name, dist_class, state[:2].upper()))

    # Build MP refs per district (2–3 MPs each)
    mp_pool: dict[str, list[str]] = {}
    mp_counter = 1
    for _, dist_name, _, _ in all_districts:
        n_mp = rng.integers(2, 4)
        mp_pool[dist_name] = [f"MP{mp_counter + i:04d}" for i in range(n_mp)]
        mp_counter += int(n_mp)

    # Build agency→district "bad agency" pattern (for planted agency anomaly)
    bad_agencies = rng.choice(AGENCIES, size=2, replace=False).tolist()

    ref_date = pd.Timestamp("2023-04-01")

    for idx in range(n_works):
        # Geography
        state, district, dist_class, state_abbr = all_districts[
            rng.integers(0, len(all_districts))
        ]
        mp_ref = rng.choice(mp_pool[district])
        constituency = f"{district}-CONST"
        block_or_village = f"Block-{rng.integers(1, 20):02d}"

        # Category
        cat_idx = rng.integers(0, len(CATEGORIES))
        category = CATEGORIES[cat_idx]

        # Implementing agency
        agency = rng.choice(AGENCIES)

        # Dates
        rec_offset = int(rng.integers(0, 365 * 3))  # 0–3 years after ref_date
        recommended_date = ref_date + pd.Timedelta(days=rec_offset)

        # Sanction delay (guideline: ≤75 days from MP recommendation; normal is 20-55 days)
        normal_delay = int(rng.integers(20, 55))
        sanctioned_date = recommended_date + pd.Timedelta(days=normal_delay)

        # Financial
        median_lakh = CATEGORY_COST_MEDIANS[category]
        terrain_mult = TERRAIN_MULTIPLIERS[dist_class]
        sigma = 0.45  # log-normal spread
        sanctioned_amount = _lognormal_cost(median_lakh * terrain_mult, sigma, rng)
        released_frac = float(rng.uniform(0.5, 1.0))
        released_amount = sanctioned_amount * released_frac
        expenditure_frac = float(rng.uniform(0.3, released_frac))
        expenditure_amount = sanctioned_amount * expenditure_frac

        # Status and progress
        status_probs = [0.45, 0.45, 0.10]
        status = rng.choice(STATUSES, p=status_probs)

        today_ts = pd.Timestamp.today().normalize()

        if status == "completed":
            completion_percent = 100.0
            work_duration = int(rng.integers(120, 450))
            completion_date = sanctioned_date + pd.Timedelta(days=work_duration)
            last_progress_date = completion_date
        elif status == "in_progress":
            completion_percent = float(rng.uniform(20, 95))
            completion_date = pd.NaT
            # Active ongoing works have recent progress updates (within last 30-180 days)
            days_ago = int(rng.integers(15, 150))
            last_progress_date = today_ts - pd.Timedelta(days=days_ago)
            if last_progress_date < sanctioned_date:
                last_progress_date = sanctioned_date + pd.Timedelta(days=30)
        else:  # not_started
            completion_percent = 0.0
            completion_date = pd.NaT
            last_progress_date = pd.NaT
            released_amount = 0.0
            expenditure_amount = 0.0

        sanction_year = sanctioned_date.year

        work_title = _generate_work_title(category, district, block_or_village, rng)
        work_description = work_title  # in real data these are often the same

        planted_type = ""

        row = {
            "work_id": _stable_id(f"{state_abbr}{district[:3].upper()}", idx),
            "mp_ref": mp_ref,
            "constituency": constituency,
            "state": state,
            "district": district,
            "district_class": dist_class,
            "block_or_village": block_or_village,
            "work_title": work_title,
            "work_description": work_description,
            "work_category": category,
            "implementing_agency": agency,
            "recommended_date": recommended_date,
            "sanctioned_date": sanctioned_date,
            "sanctioned_amount": sanctioned_amount,
            "released_amount": released_amount,
            "expenditure_amount": expenditure_amount,
            "status": status,
            "last_progress_date": last_progress_date,
            "completion_date": completion_date,
            "completion_percent": completion_percent,
            "sanction_year": sanction_year,
            "_planted_type": planted_type,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Plant anomalies ────────────────────────────────────────────────────
    df = _plant_anomalies(df, rng, anomaly_rate)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        logger.info("Saved synthetic data to %s", output_path)

    logger.info(
        "Generated %d works; anomalies planted: %s",
        len(df),
        df["_planted_type"].value_counts().to_dict(),
    )
    return df


LANDMARKS = [
    "Hanuman Temple", "Primary School", "Canal Bridge", "Main Chowk", "Bus Stand",
    "Panchayat Bhawan", "Community Well", "Railway Crossing", "Post Office",
    "Hospital Road", "Weekly Haat", "Gandhi Nagar", "Shanti Colony", "River Ghat",
    "Shiv Mandir", "Veterinary Hospital", "Krishi Mandi", "Govt High School",
    "Bazaar Chowk", "Ambedkar Bhawan", "Durga Mandir", "Sub Health Center",
    "Talab Bund", "Police Outpost", "Village Entry Gate", "Market Complex",
]


def _generate_work_title(
    category: str, district: str, block: str, rng: np.random.Generator
) -> str:
    """Generate a realistic-sounding work title with specific local location identifier."""
    loc1 = rng.choice(LANDMARKS)
    loc2 = rng.choice(LANDMARKS)
    while loc2 == loc1:
        loc2 = rng.choice(LANDMARKS)

    templates = {
        "road": [
            f"Construction of CC Road from {loc1} to {loc2} in {block}",
            f"Widening of village road near {loc1} in {block}, {district}",
            f"Repair and renovation of approach road from {loc1} to {loc2}",
            f"Construction of Internal Roads connecting {loc1} in {block}",
        ],
        "community hall": [
            f"Construction of Community Hall near {loc1} in {block}",
            f"Construction of Panchayat Bhavan adjacent to {loc1}, {district}",
            f"Building of Multi-purpose Community Centre near {loc1} in {block}",
        ],
        "drinking water": [
            f"Supply of Drinking Water from {loc1} to {loc2} in {block}",
            f"Construction of Overhead Water Tank near {loc1} in {block}",
            f"Installation of Deep Tube Wells and Hand Pumps near {loc1}",
            f"Piped Water Supply Scheme covering {loc1} in {block}",
        ],
        "school": [
            f"Construction of Additional Classroom at {loc1} in {block}",
            f"Renovation and roofing work at {loc1}, {district}",
            f"Construction of Toilet Block and Drinking Station at {loc1}",
        ],
        "health": [
            f"Construction of Sub-Health Centre near {loc1} in {block}",
            f"Renovation of Dispensary and Ward at {loc1}, {block}",
            f"Construction of Mother and Child Care Unit at {loc1}",
        ],
        "sanitation": [
            f"Construction of Community Sanitary Complex near {loc1} in {block}",
            f"Construction of Waste Segregation Shed at {loc1} in {block}",
            f"Solid and Liquid Waste Management facility near {loc1}, {district}",
        ],
        "other": [
            f"Construction of Aanganwadi Centre near {loc1} in {block}",
            f"Installation of High Mast Solar Lights at {loc1} in {block}",
            f"Construction of Cremation Shed and Boundary Wall at {loc1}",
        ],
    }
    choices = templates.get(category, [f"Community development work near {loc1} in {block}"])
    return str(rng.choice(choices))


def _plant_anomalies(
    df: pd.DataFrame,
    rng: np.random.Generator,
    anomaly_rate: float,
) -> pd.DataFrame:
    """
    Plant labelled anomalies into the DataFrame.
    The ``_planted_type`` column is NEVER read by detectors.
    """
    total_anomalies = int(len(df) * anomaly_rate)
    # Distribute across types
    type_counts = {
        t: max(1, int(total_anomalies * frac))
        for t, frac in PLANTED_TYPES.items()
    }

    used_indices: set[int] = set()

    def _pick(n: int) -> list[int]:
        available = list(set(df.index.tolist()) - used_indices)
        chosen = rng.choice(available, size=min(n, len(available)), replace=False).tolist()
        used_indices.update(chosen)
        return chosen

    # 1. Cost outliers — boost sanctioned_amount 5–15x the median
    for i in _pick(type_counts["cost_outlier"]):
        cat = df.at[i, "work_category"]
        dc = df.at[i, "district_class"]
        median = CATEGORY_COST_MEDIANS[cat] * TERRAIN_MULTIPLIERS[dc] * 1e5
        multiplier = float(rng.uniform(5, 15))
        df.at[i, "sanctioned_amount"] = median * multiplier
        df.at[i, "released_amount"] = df.at[i, "sanctioned_amount"] * 0.8
        df.at[i, "_planted_type"] = "cost_outlier"

    # 2. Duplicate pairs — copy a row, same block, slightly different title
    for i in _pick(type_counts["duplicate"]):
        src = df.iloc[i].copy()
        # Modify slightly
        j = len(df)
        new_row = src.to_dict()
        new_row["work_id"] = f"DUP-{i:06d}"
        new_row["work_title"] = src["work_title"] + " (Phase II)"
        new_row["work_description"] = new_row["work_title"]
        days_apart = int(rng.integers(10, 60))
        new_row["sanctioned_date"] = src["sanctioned_date"] + pd.Timedelta(days=days_apart)
        new_row["recommended_date"] = src["recommended_date"] + pd.Timedelta(days=days_apart)
        new_row["sanction_year"] = new_row["sanctioned_date"].year
        new_row["_planted_type"] = "duplicate"
        df.at[i, "_planted_type"] = "duplicate"
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # 3. Stalled works — old sanction, funds released, no progress
    for i in _pick(type_counts["stalled"]):
        old_date = pd.Timestamp("2022-01-01") - pd.Timedelta(days=int(rng.integers(0, 365)))
        df.at[i, "sanctioned_date"] = old_date
        df.at[i, "recommended_date"] = old_date - pd.Timedelta(days=60)
        df.at[i, "sanction_year"] = old_date.year
        df.at[i, "status"] = "in_progress"
        df.at[i, "last_progress_date"] = pd.NaT  # no progress recorded
        df.at[i, "released_amount"] = df.at[i, "sanctioned_amount"] * 0.7
        df.at[i, "completion_percent"] = float(rng.uniform(0, 20))
        df.at[i, "_planted_type"] = "stalled"

    # 4. Agency pattern — cluster several works under one agency with low completion
    bad_agency = rng.choice(AGENCIES)
    for i in _pick(type_counts["agency_pattern"]):
        df.at[i, "implementing_agency"] = bad_agency
        df.at[i, "status"] = "in_progress"
        df.at[i, "completion_percent"] = float(rng.uniform(0, 25))
        old_date = pd.Timestamp("2022-06-01") - pd.Timedelta(days=int(rng.integers(0, 300)))
        df.at[i, "sanctioned_date"] = old_date
        df.at[i, "sanction_year"] = old_date.year
        df.at[i, "last_progress_date"] = old_date + pd.Timedelta(days=int(rng.integers(30, 90)))
        df.at[i, "_planted_type"] = "agency_pattern"

    # 5. Process delay — very long sanction delay
    for i in _pick(type_counts["delay"]):
        delay_days = int(rng.integers(200, 500))
        rec_date = df.at[i, "recommended_date"]
        df.at[i, "sanctioned_date"] = rec_date + pd.Timedelta(days=delay_days)
        df.at[i, "sanction_year"] = df.at[i, "sanctioned_date"].year
        df.at[i, "_planted_type"] = "delay"

    return df.reset_index(drop=True)
