"""
pages/2_Flagged_Works.py
Filterable table of flagged works with CSV export.
Clicking a row navigates to Work Detail (via query param).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
import pandas as pd
import streamlit as st

from pipeline.anonymise import get_display_columns
from pipeline.run_pipeline import load_scored_df, load_review_log
from pipeline.score import compute_final_scores, build_reason_summary, DETECTOR_NAMES

st.set_page_config(page_title="Flagged Works | MPLADS-Lens", page_icon="🚩", layout="wide")

css_path = ROOT / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "anon_mode" not in st.session_state:
    st.session_state["anon_mode"] = True
if "weights" not in st.session_state:
    st.session_state["weights"] = {
        "cost": 0.30, "duplicate": 0.25, "stalled": 0.25, "agency": 0.10, "delay": 0.10
    }

st.markdown(
    '<div class="review-banner">'
    '⚠️ <strong>Outliers for review, not findings of fraud.</strong>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("## 🚩 Flagged Works")

DB_PATH = "data/mplads_lens.duckdb"
if not Path(DB_PATH).exists():
    st.warning("No pipeline data found. Please visit the **📁 Data** page to run the pipeline.")
    st.stop()


@st.cache_data(ttl=60)
def get_data(db_path: str) -> pd.DataFrame:
    return load_scored_df(db_path)


@st.cache_data(ttl=30)
def get_review_log(db_path: str) -> pd.DataFrame:
    return load_review_log(db_path)


df = get_data(DB_PATH)
review_log = get_review_log(DB_PATH)

# Rescore with sidebar weights
weights = st.session_state.get("weights")
df = compute_final_scores(df, weights=weights)
df["reason_summary"] = df.apply(build_reason_summary, axis=1)
df["severity"] = df["severity"].astype(str)

# Mark reviewed works
reviewed_ids = set(review_log["work_id"].tolist()) if len(review_log) > 0 else set()
df["reviewed"] = df["work_id"].isin(reviewed_ids)

anon = st.session_state.get("anon_mode", True)
col_map = get_display_columns(anon)
district_col = col_map["district"]
agency_col = col_map["implementing_agency"]

# Ensure anon columns
if anon and district_col not in df.columns:
    from pipeline.anonymise import apply_anonymisation
    df = apply_anonymisation(df)

# Only flagged works
flagged_df = df[df["final_score"] > 0].copy()

if len(flagged_df) == 0:
    st.info("No works flagged with current thresholds. Try adjusting the detector weights in the sidebar.")
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────────────
with st.expander("🔽 Filters", expanded=True):
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        all_states = sorted(flagged_df["state"].dropna().unique().tolist())
        sel_states = st.multiselect("State", all_states, default=[])
    with fc2:
        all_cats = sorted(flagged_df["work_category"].dropna().unique().tolist())
        sel_cats = st.multiselect("Category", all_cats, default=[])
    with fc3:
        all_sev = ["High", "Medium", "Low"]
        sel_sev = st.multiselect("Severity", all_sev, default=all_sev)

    fc4, fc5, fc6 = st.columns(3)
    with fc4:
        sel_detectors = st.multiselect(
            "Detector fired",
            DETECTOR_NAMES,
            default=[],
            help="Show works where at least one selected detector fired.",
        )
    with fc5:
        min_score, max_score = st.slider(
            "Score range", 0.0, 100.0, (0.0, 100.0), step=1.0
        )
    with fc6:
        free_text = st.text_input("Search (work ID or title)", "")

    top_n = st.selectbox("Show top N works", [10, 25, 50, 100, 200], index=0)

# Apply filters
filt = flagged_df.copy()
if sel_states:
    filt = filt[filt["state"].isin(sel_states)]
if sel_cats:
    filt = filt[filt["work_category"].isin(sel_cats)]
if sel_sev:
    filt = filt[filt["severity"].isin(sel_sev)]
if sel_detectors:
    mask = pd.Series([False] * len(filt), index=filt.index)
    for det in sel_detectors:
        if f"{det}_score" in filt.columns:
            mask = mask | (filt[f"{det}_score"] > 0)
    filt = filt[mask]
filt = filt[(filt["final_score"] >= min_score) & (filt["final_score"] <= max_score)]
if free_text.strip():
    q = free_text.strip().lower()
    title_mask = filt["work_title"].str.lower().str.contains(q, na=False)
    id_mask = filt["work_id"].str.lower().str.contains(q, na=False)
    filt = filt[title_mask | id_mask]

filt = filt.sort_values("final_score", ascending=False).head(top_n)

st.markdown(f"**{len(filt)} works** shown (filtered from {len(flagged_df)} flagged)")

# ── Display columns ────────────────────────────────────────────────────────────
display_cols = [
    "work_id", "work_title", "work_category", district_col,
    "sanctioned_amount", "final_score", "severity", "reason_summary", "reviewed"
]
display_cols = [c for c in display_cols if c in filt.columns]

show_df = filt[display_cols].copy()
show_df["sanctioned_amount"] = show_df["sanctioned_amount"].apply(lambda x: f"₹{x/1e5:.1f}L")
show_df["final_score"] = show_df["final_score"].apply(lambda x: f"{x:.1f}")
show_df["reviewed"] = show_df["reviewed"].map({True: "✅ Reviewed", False: ""})

rename_map = {
    "work_id": "Work ID", "work_title": "Title", "work_category": "Category",
    district_col: "District", "sanctioned_amount": "Amount",
    "final_score": "Score", "severity": "Severity",
    "reason_summary": "Reason", "reviewed": "Reviewed",
}
show_df = show_df.rename(columns=rename_map)


def _color_row(row: pd.Series):
    sev = row.get("Severity", "")
    if sev == "High":
        return ["background-color:#FFF5F5"] * len(row)
    elif sev == "Medium":
        return ["background-color:#FFFDE7"] * len(row)
    return [""] * len(row)


def _color_sev(val: str) -> str:
    c = {"High": "color:#C62828;font-weight:700",
         "Medium": "color:#E65100;font-weight:700",
         "Low": "color:#455A64"}
    return c.get(str(val), "")


styled = show_df.style.apply(_color_row, axis=1).applymap(_color_sev, subset=["Severity"])
st.dataframe(styled, use_container_width=True, height=480)

# ── Work Detail navigation ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Open work detail:** enter a Work ID from the table above.")
sel_work_id = st.text_input("Work ID", placeholder="e.g. RAJAIS-000042")
if st.button("🔎 Open Work Detail", use_container_width=False):
    if sel_work_id.strip():
        st.session_state["selected_work_id"] = sel_work_id.strip()
        st.switch_page("pages/3_Work_Detail.py")
    else:
        st.warning("Please enter a Work ID.")

# ── CSV Export ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Export filtered results**")
export_df = filt[display_cols].copy()
export_df = export_df.rename(columns=rename_map)
csv_buf = io.StringIO()
export_df.to_csv(csv_buf, index=False)
st.download_button(
    label="⬇️ Download filtered CSV",
    data=csv_buf.getvalue(),
    file_name="mplads_lens_flagged_works.csv",
    mime="text/csv",
    use_container_width=False,
)

st.markdown("""
<div class="app-footer">
  MPLADS-Lens · All flags are for review only — no finding of wrongdoing is implied.
</div>
""", unsafe_allow_html=True)
