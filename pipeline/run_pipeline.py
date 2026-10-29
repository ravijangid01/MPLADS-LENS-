"""
pipeline/run_pipeline.py
Orchestrates the full MPLADS-Lens pipeline:
  ingest → clean → peer groups → detectors → score → persist

Returns scored DataFrame and stores results in DuckDB.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

import duckdb
import pandas as pd
import yaml

from pipeline.synthetic import generate_synthetic_data
from pipeline.clean import clean
from pipeline.peer_groups import assign_peer_groups
from pipeline.anonymise import apply_anonymisation
from pipeline.score import compute_final_scores, build_reason_summary, DETECTOR_NAMES
from detectors.cost import CostOutlierDetector
from detectors.duplicate import NearDuplicateDetector
from detectors.stalled import StalledWorkDetector
from detectors.agency import AgencyPatternDetector
from detectors.delay import ProcessDelayDetector

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(
    df: Optional[pd.DataFrame] = None,
    config_path: str = "config/config.yaml",
    progress_callback: Optional[Callable[[str, float], None]] = None,
    weights: Optional[dict[str, float]] = None,
    data_source: str = "Synthetic demo data",
) -> tuple[pd.DataFrame, dict]:
    """
    Run the complete MPLADS-Lens pipeline.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-loaded canonical DataFrame. If None, generates synthetic data.
    config_path : str
        Path to config.yaml.
    progress_callback : callable, optional
        Called with (step_name, fraction_complete) for progress bars.
    weights : dict, optional
        Override scoring weights from config.
    data_source : str
        Badge label for this data run.

    Returns
    -------
    (scored_df, pipeline_report)
    """
    def _progress(step: str, pct: float) -> None:
        if progress_callback:
            progress_callback(step, pct)
        logger.info("[Pipeline] %s (%.0f%%)", step, pct * 100)

    config = load_config(config_path)
    report: dict = {"data_source": data_source, "steps": {}}

    # ── Step 1: Load data ────────────────────────────────────────────────────
    _progress("Loading data", 0.05)
    if df is None:
        syn_cfg = config.get("synthetic", {})
        output_path = syn_cfg.get("output_path", "data/synthetic_works.parquet")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(output_path).exists():
            df = pd.read_parquet(output_path)
            logger.info("Loaded cached synthetic data from %s", output_path)
        else:
            df = generate_synthetic_data(
                n_works=syn_cfg.get("n_works", 20_000),
                random_seed=syn_cfg.get("random_seed", 42),
                anomaly_rate=syn_cfg.get("anomaly_rate", 0.035),
                output_path=output_path,
            )
    report["steps"]["load"] = {"rows": len(df)}

    # ── Step 2: Clean ────────────────────────────────────────────────────────
    _progress("Cleaning data", 0.15)
    df, clean_report = clean(df)
    report["steps"]["clean"] = clean_report

    # ── Step 3: Peer groups ───────────────────────────────────────────────────
    _progress("Assigning peer groups", 0.25)
    min_pg_size = config.get("detectors", {}).get("cost", {}).get("min_peer_group_size", 8)
    df = assign_peer_groups(df, min_group_size=min_pg_size)

    # ── Step 4: Run detectors ─────────────────────────────────────────────────
    detectors = [
        CostOutlierDetector(config),
        NearDuplicateDetector(config),
        StalledWorkDetector(config),
        AgencyPatternDetector(config),
        ProcessDelayDetector(config),
    ]

    detector_results: dict[str, pd.DataFrame] = {}
    for i, detector in enumerate(detectors):
        _progress(f"Running {detector.name} detector", 0.30 + i * 0.10)
        try:
            result = detector.detect(df)
            # Rename columns to namespaced versions
            result = result.rename(columns={
                "score": f"{detector.name}_score",
                "reason": f"{detector.name}_reason",
                "evidence": f"{detector.name}_evidence",
            })
            detector_results[detector.name] = result
            report["steps"][detector.name] = {
                "flagged": int((result[f"{detector.name}_score"] > 0).sum())
            }
        except Exception as e:
            logger.error("Detector %s failed: %s", detector.name, e)
            # Return zero scores for failed detector
            empty = pd.DataFrame({
                "work_id": df["work_id"],
                f"{detector.name}_score": 0.0,
                f"{detector.name}_reason": "",
                f"{detector.name}_evidence": [{}] * len(df),
            })
            detector_results[detector.name] = empty
            report["steps"][detector.name] = {"flagged": 0, "error": str(e)}

    # Store agency_df if available
    agency_det = next((d for d in detectors if d.name == "agency"), None)
    agency_df = getattr(agency_det, "agency_df", None)

    # ── Step 5: Merge detector results ───────────────────────────────────────
    _progress("Merging scores", 0.80)
    scored_df = df.copy()
    for det_name, result_df in detector_results.items():
        scored_df = scored_df.merge(
            result_df[["work_id", f"{det_name}_score", f"{det_name}_reason", f"{det_name}_evidence"]],
            on="work_id",
            how="left",
        )
        # Fill NaN scores (works not returned by detector)
        scored_df[f"{det_name}_score"] = scored_df[f"{det_name}_score"].fillna(0.0)
        scored_df[f"{det_name}_reason"] = scored_df[f"{det_name}_reason"].fillna("")
        scored_df[f"{det_name}_evidence"] = scored_df[f"{det_name}_evidence"].apply(
            lambda x: x if isinstance(x, dict) else {}
        )

    # ── Step 6: Final scoring ─────────────────────────────────────────────────
    _progress("Computing final scores", 0.88)
    w_cfg = config.get("scoring", {}).get("weights", {})
    bonus = config.get("scoring", {}).get("multi_detector_bonus", 5.0)
    if weights is None:
        weights = w_cfg
    scored_df = compute_final_scores(scored_df, weights=weights, multi_bonus=bonus)
    scored_df["reason_summary"] = scored_df.apply(build_reason_summary, axis=1)

    # ── Step 7: Anonymise ─────────────────────────────────────────────────────
    _progress("Applying anonymisation", 0.93)
    anon_salt = config.get("anonymise", {}).get("salt", "mplads-lens-demo-2026")
    scored_df = apply_anonymisation(scored_df, salt=anon_salt)

    # ── Step 8: Persist to DuckDB ────────────────────────────────────────────
    _progress("Persisting to database", 0.96)
    db_path = config.get("pipeline", {}).get("db_path", "data/mplads_lens.duckdb")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    _persist_to_duckdb(scored_df, agency_df, db_path)

    report["steps"]["scoring"] = {
        "total_works": len(scored_df),
        "flagged_any": int((scored_df["final_score"] > 0).sum()),
        "high_severity": int((scored_df["severity"] == "High").sum()),
        "medium_severity": int((scored_df["severity"] == "Medium").sum()),
        "low_severity": int((scored_df["severity"] == "Low").sum()),
    }
    report["data_source"] = data_source

    _progress("Done", 1.0)
    logger.info(
        "Pipeline complete: %d works, %d flagged",
        len(scored_df),
        (scored_df["final_score"] > 0).sum(),
    )
    return scored_df, report


def _persist_to_duckdb(
    scored_df: pd.DataFrame,
    agency_df: Optional[pd.DataFrame],
    db_path: str,
) -> None:
    """Persist scored works and agency results to DuckDB."""
    # Evidence columns (dicts) are not directly storable; serialise to JSON string
    import json

    df_to_store = scored_df.copy()
    for det in DETECTOR_NAMES:
        col = f"{det}_evidence"
        if col in df_to_store.columns:
            df_to_store[col] = df_to_store[col].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else "{}"
            )

    # Convert severity (Categorical) to string for DuckDB
    if "severity" in df_to_store.columns:
        df_to_store["severity"] = df_to_store["severity"].astype(str)

    con = duckdb.connect(db_path)
    con.execute("DROP TABLE IF EXISTS scored_works")
    con.execute("CREATE TABLE scored_works AS SELECT * FROM df_to_store")

    # Review log table (create if not exists)
    con.execute("""
        CREATE TABLE IF NOT EXISTS review_log (
            work_id VARCHAR,
            reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewer_note TEXT,
            is_reviewed BOOLEAN DEFAULT TRUE
        )
    """)

    if agency_df is not None:
        con.execute("DROP TABLE IF EXISTS agency_scores")
        agency_df_store = agency_df.copy()
        if "_iso_pred" in agency_df_store.columns:
            agency_df_store["is_anomalous"] = agency_df_store["_iso_pred"] == -1
            agency_df_store = agency_df_store.drop(columns=["_iso_pred"])
        con.execute("CREATE TABLE agency_scores AS SELECT * FROM agency_df_store")

    con.close()
    logger.info("Persisted results to DuckDB: %s", db_path)


def load_scored_df(db_path: str = "data/mplads_lens.duckdb") -> pd.DataFrame:
    """Load scored works from DuckDB."""
    import json
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("SELECT * FROM scored_works").df()
    con.close()

    # Deserialise evidence columns
    for det in DETECTOR_NAMES:
        col = f"{det}_evidence"
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) else {}
            )
    return df


def load_agency_df(db_path: str = "data/mplads_lens.duckdb") -> pd.DataFrame:
    """Load agency scores from DuckDB."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute("SELECT * FROM agency_scores").df()
    except Exception:
        df = pd.DataFrame()
    con.close()
    return df


def mark_reviewed(
    work_id: str,
    note: str,
    db_path: str = "data/mplads_lens.duckdb",
) -> None:
    """Log a review action for a work."""
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE TABLE IF NOT EXISTS review_log (
            work_id VARCHAR,
            reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            reviewer_note TEXT,
            is_reviewed BOOLEAN DEFAULT TRUE
        )
    """)
    # Remove prior review for same work_id
    con.execute("DELETE FROM review_log WHERE work_id = ?", [work_id])
    con.execute(
        "INSERT INTO review_log (work_id, reviewed_at, reviewer_note, is_reviewed) "
        "VALUES (?, CURRENT_TIMESTAMP, ?, TRUE)",
        [work_id, note],
    )
    con.close()


def load_review_log(db_path: str = "data/mplads_lens.duckdb") -> pd.DataFrame:
    """Load the review log from DuckDB."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute("SELECT * FROM review_log ORDER BY reviewed_at DESC").df()
    except Exception:
        df = pd.DataFrame(columns=["work_id", "reviewed_at", "reviewer_note", "is_reviewed"])
    con.close()
    return df
