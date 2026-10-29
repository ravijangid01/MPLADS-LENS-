"""
pages/5_Data.py
Data management page: source badge, quality summary, CSV/Excel upload with
column mapping wizard, regenerate synthetic data, rerun pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
import tempfile
import pandas as pd
import yaml
import streamlit as st

st.set_page_config(page_title="Data | MPLADS-Lens", page_icon="📁", layout="wide")

css_path = ROOT / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "anon_mode" not in st.session_state:
    st.session_state["anon_mode"] = True
if "data_source" not in st.session_state:
    st.session_state["data_source"] = "Synthetic demo data"

st.markdown(
    '<div class="review-banner">'
    '⚠️ <strong>Outliers for review, not findings of fraud.</strong>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("## 📁 Data Management")

DB_PATH = "data/mplads_lens.duckdb"
SYNTH_PATH = "data/synthetic_works.parquet"

# ── Source badge ───────────────────────────────────────────────────────────────
source = st.session_state.get("data_source", "Synthetic demo data")
is_user = "User" in source
badge_class = "user-data" if is_user else ""
st.markdown(
    f'<span class="source-badge {badge_class}">📊 Current data: {source}</span>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ── Current data quality summary ──────────────────────────────────────────────
st.markdown("### Current Data Summary")

if Path(DB_PATH).exists():
    try:
        from pipeline.run_pipeline import load_scored_df
        df = load_scored_df(DB_PATH)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Works", f"{len(df):,}")
        with col2:
            st.metric("States", df["state"].nunique() if "state" in df.columns else "N/A")
        with col3:
            st.metric("Districts", df["district"].nunique() if "district" in df.columns else "N/A")
        with col4:
            dq = df["data_quality_score"].mean() if "data_quality_score" in df.columns else 0
            st.metric("Mean Data Quality", f"{dq:.0%}")

        with st.expander("Column-level quality"):
            from pipeline.clean import QUALITY_FIELDS
            quality_data = []
            for col in QUALITY_FIELDS:
                if col in df.columns:
                    filled = df[col].notna().sum()
                    pct = filled / len(df)
                    quality_data.append({"Column": col, "Filled": filled, "Coverage": f"{pct:.0%}"})
            if quality_data:
                st.dataframe(pd.DataFrame(quality_data), use_container_width=True)
    except Exception as e:
        st.info(f"No data loaded yet or error: {e}")
else:
    st.info("No database found. Generate synthetic data or upload a file below.")

st.markdown("---")

# ── Regenerate synthetic data ──────────────────────────────────────────────────
st.markdown("### 🔄 Synthetic Data")
st.markdown(
    "Generate fresh synthetic MPLADS works (20,000 records across 4 states and 40 districts) "
    "with planted anomalies at ~3.5%."
)

col_regen, col_rerun = st.columns(2)
with col_regen:
    if st.button("🔁 Regenerate synthetic data", use_container_width=True):
        from pipeline.synthetic import generate_synthetic_data
        with st.spinner("Generating 20,000 synthetic works…"):
            Path(SYNTH_PATH).parent.mkdir(parents=True, exist_ok=True)
            if Path(SYNTH_PATH).exists():
                Path(SYNTH_PATH).unlink()
            generate_synthetic_data(output_path=SYNTH_PATH)
        st.success("Synthetic data regenerated. Click **Rerun pipeline** to score it.")
        st.session_state["data_source"] = "Synthetic demo data"

with col_rerun:
    if st.button("▶️ Rerun full pipeline", use_container_width=True):
        progress_bar = st.progress(0, text="Starting pipeline…")
        status_text = st.empty()

        def cb(step: str, pct: float) -> None:
            progress_bar.progress(pct, text=f"Pipeline: {step}")
            status_text.text(f"▶ {step} ({pct:.0%})")

        from pipeline.run_pipeline import run_pipeline
        with st.spinner("Running pipeline…"):
            _, report = run_pipeline(
                progress_callback=cb,
                data_source=st.session_state.get("data_source", "Synthetic demo data"),
            )
        progress_bar.progress(1.0, text="Done!")
        status_text.text("")
        st.success(
            f"Pipeline complete: {report['steps']['scoring']['total_works']:,} works analysed, "
            f"{report['steps']['scoring']['flagged_any']:,} flagged."
        )
        # Clear caches
        st.cache_data.clear()
        st.rerun()

st.markdown("---")

# ── Upload user data ───────────────────────────────────────────────────────────
st.markdown("### ⬆️ Upload eSAKSHI Export (CSV or Excel)")
st.markdown("""
1. Export your MPLADS works data from the **eSAKSHI portal** as CSV or Excel.
2. Upload it below.
3. Map your column headers to the canonical schema using the wizard.
4. Click **Run pipeline** to analyse your data.

> **Privacy:** Your file is processed locally and is never sent to any server.
""")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx", "xls"],
    help="eSAKSHI works export, or any CSV/Excel with MPLADS works data.",
)

if uploaded_file is not None:
    st.markdown("#### Column Mapping Wizard")
    st.markdown("Map your file's column headers to the MPLADS-Lens canonical schema.")

    # Load example column map
    col_map_path = ROOT / "config" / "column_map.example.yaml"
    with open(col_map_path, encoding="utf-8") as f:
        example_map = yaml.safe_load(f)

    # Read uploaded file headers
    try:
        if uploaded_file.name.endswith((".xlsx", ".xls")):
            raw_preview = pd.read_excel(uploaded_file, nrows=3, dtype=str)
        else:
            raw_preview = pd.read_csv(uploaded_file, nrows=3, dtype=str)
        uploaded_file.seek(0)  # reset for later use
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    raw_headers = list(raw_preview.columns)
    st.markdown("**Preview of uploaded file (first 3 rows):**")
    st.dataframe(raw_preview, use_container_width=True)

    # Canonical columns that need mapping
    from pipeline.ingest import CANONICAL_COLUMNS
    CANONICAL_TO_DESC = {
        "work_id": "Work ID / Serial No.",
        "mp_ref": "MP Reference",
        "constituency": "Constituency",
        "state": "State",
        "district": "District",
        "district_class": "District Class (plain/hill/desert/tribal/coastal)",
        "block_or_village": "Block or Village Name",
        "work_title": "Work Title / Name of Work",
        "work_description": "Work Description",
        "work_category": "Category of Work",
        "implementing_agency": "Implementing Agency",
        "recommended_date": "Date of Recommendation (dd-mm-yyyy)",
        "sanctioned_date": "Date of Sanction (dd-mm-yyyy)",
        "sanctioned_amount": "Sanctioned Amount (₹ or in lakh/crore)",
        "released_amount": "Released Amount (₹ or in lakh/crore)",
        "expenditure_amount": "Expenditure Amount (₹ or in lakh/crore)",
        "status": "Status (Completed/In Progress/Not Started)",
        "last_progress_date": "Last Progress Date (dd-mm-yyyy)",
        "completion_date": "Completion Date (dd-mm-yyyy)",
        "completion_percent": "% Completion (0–100)",
    }

    # Auto-suggest mappings (simple fuzzy match on header names)
    def _auto_suggest(canonical: str, headers: list[str]) -> str | None:
        """Simple substring match for auto-suggestion."""
        keywords = {
            "work_id": ["sr", "serial", "work id", "work_id"],
            "mp_ref": ["member", "mp", "parliament"],
            "constituency": ["constituency"],
            "state": ["state"],
            "district": ["district"],
            "block_or_village": ["block", "village"],
            "work_title": ["name of work", "work name", "title"],
            "work_description": ["description", "name of work"],
            "work_category": ["category"],
            "implementing_agency": ["agency", "implementing"],
            "recommended_date": ["recommend", "date of rec"],
            "sanctioned_date": ["sanction", "date of sanc"],
            "sanctioned_amount": ["sanctioned amount", "sanctioned amt"],
            "released_amount": ["released", "release"],
            "expenditure_amount": ["expenditure", "expend"],
            "status": ["status"],
            "last_progress_date": ["progress", "last updat"],
            "completion_date": ["completion date", "date of comp"],
            "completion_percent": ["percent", "%", "completion %"],
        }
        kws = keywords.get(canonical, [canonical])
        for h in headers:
            for kw in kws:
                if kw.lower() in h.lower():
                    return h
        return None

    user_map: dict[str, str | None] = {}
    with st.expander("📋 Column Mapping (click to expand)", expanded=True):
        for canon, desc in CANONICAL_TO_DESC.items():
            suggestion = _auto_suggest(canon, raw_headers)
            options = ["(not available in my file)"] + raw_headers
            default_idx = raw_headers.index(suggestion) + 1 if suggestion in raw_headers else 0
            sel = st.selectbox(
                f"{desc} → `{canon}`",
                options=options,
                index=default_idx,
                key=f"colmap_{canon}",
            )
            user_map[canon] = None if sel == "(not available in my file)" else sel

    # Build and display the resulting YAML
    final_map = {k: v for k, v in user_map.items()}
    yaml_str = yaml.dump(final_map, default_flow_style=False, allow_unicode=True)

    st.download_button(
        "⬇️ Download column map YAML",
        data=yaml_str,
        file_name="my_column_map.yaml",
        mime="text/yaml",
    )

    if st.button("▶️ Ingest & Run Pipeline with this mapping", use_container_width=True, type="primary"):
        with st.spinner("Ingesting your data…"):
            # Save uploaded file to temp location
            suffix = ".xlsx" if uploaded_file.name.endswith((".xlsx", ".xls")) else ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Save column map to temp YAML
            with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as ytmp:
                yaml.dump(final_map, ytmp)
                yaml_path = ytmp.name

            from pipeline.ingest import load_user_file
            try:
                user_df, ingest_report = load_user_file(tmp_path, yaml_path)
                st.success(
                    f"Ingested {ingest_report['rows_kept']:,} rows "
                    f"({ingest_report['rows_dropped']} dropped)."
                )
                if ingest_report.get("drop_reasons"):
                    st.warning(f"Dropped rows reasons: {ingest_report['drop_reasons']}")

                st.session_state["data_source"] = f"User-supplied data: {uploaded_file.name}"

                # Run pipeline
                progress_bar2 = st.progress(0, text="Running pipeline…")

                def cb2(step: str, pct: float) -> None:
                    progress_bar2.progress(pct, text=f"Pipeline: {step}")

                from pipeline.run_pipeline import run_pipeline
                _, report = run_pipeline(
                    df=user_df,
                    progress_callback=cb2,
                    data_source=f"User-supplied data: {uploaded_file.name}",
                )
                progress_bar2.progress(1.0, text="Done!")
                st.success(
                    f"Pipeline complete: {report['steps']['scoring']['total_works']:,} works analysed, "
                    f"{report['steps']['scoring']['flagged_any']:,} flagged."
                )
                st.cache_data.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline failed: {e}")
                import traceback
                st.code(traceback.format_exc())

st.markdown("""
<div class="app-footer">
  MPLADS-Lens · Your data is processed locally and never transmitted.
</div>
""", unsafe_allow_html=True)
