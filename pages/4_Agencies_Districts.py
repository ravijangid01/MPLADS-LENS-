"""
pages/4_Agencies_Districts.py
Scatter of completion rate vs average delay with anomalous agencies
highlighted (anonymised), plus a ranked table.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline.anonymise import anonymise_agency, anonymise_district
from pipeline.run_pipeline import load_agency_df, load_scored_df

st.set_page_config(page_title="Agencies & Districts | MPLADS-Lens", page_icon="🏢", layout="wide")

css_path = ROOT / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if "anon_mode" not in st.session_state:
    st.session_state["anon_mode"] = True

st.markdown(
    '<div class="review-banner">'
    '⚠️ <strong>Outliers for review, not findings of fraud.</strong>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("## 🏢 Agencies & Districts")

DB_PATH = "data/mplads_lens.duckdb"
if not Path(DB_PATH).exists():
    st.warning("No pipeline data found. Please run the pipeline from the **📁 Data** page.")
    st.stop()


@st.cache_data(ttl=120)
def get_agency_data(db_path: str) -> pd.DataFrame:
    return load_agency_df(db_path)


@st.cache_data(ttl=120)
def get_works_data(db_path: str) -> pd.DataFrame:
    return load_scored_df(db_path)


agency_df = get_agency_data(DB_PATH)
works_df = get_works_data(DB_PATH)

anon = st.session_state.get("anon_mode", True)

if len(agency_df) == 0:
    st.info("Agency scores not yet computed. Please re-run the pipeline from the **📁 Data** page.")
    st.stop()

# Anonymise agency names
if anon:
    agency_df = agency_df.copy()
    agency_df["agency_display"] = agency_df["implementing_agency"].apply(anonymise_agency)
else:
    agency_df["agency_display"] = agency_df["implementing_agency"]

# Mark anomalous
if "is_anomalous" not in agency_df.columns:
    agency_df["is_anomalous"] = False

# ── Scatter: Completion rate vs Mean sanction delay ────────────────────────────
st.markdown("### Completion Rate vs Average Sanction Delay")
st.caption("Each point represents one implementing agency. Flagged agencies (IsolationForest) are highlighted in red.")

scatter_df = agency_df.dropna(subset=["completion_rate", "mean_sanction_delay_days"]).copy()
scatter_df["completion_rate_pct"] = scatter_df["completion_rate"] * 100
scatter_df["color"] = scatter_df["is_anomalous"].map({True: "Flagged", False: "Normal"})
scatter_df["size"] = scatter_df["n_works"].clip(5, 100)

color_map = {"Flagged": "#C62828", "Normal": "#0070C0"}

fig = px.scatter(
    scatter_df,
    x="mean_sanction_delay_days",
    y="completion_rate_pct",
    color="color",
    color_discrete_map=color_map,
    size="size",
    hover_name="agency_display",
    hover_data={
        "n_works": True,
        "stall_share": ":.1%",
        "color": False,
        "size": False,
    },
    labels={
        "mean_sanction_delay_days": "Avg Sanction Delay (days)",
        "completion_rate_pct": "Completion Rate (%)",
    },
    template="plotly_white",
)
fig.update_layout(
    paper_bgcolor="white",
    font_color="#1F3864",
    height=450,
    margin=dict(l=10, r=10, t=20, b=10),
    legend_title="Agency Status",
)
# Add guideline line
fig.add_vline(
    x=75, line_dash="dash", line_color="#C55A11",
    annotation_text="75d guideline", annotation_position="top right",
)
st.plotly_chart(fig, use_container_width=True)

# ── District-level summary ─────────────────────────────────────────────────────
st.markdown("### District Summary")

if "district" in works_df.columns:
    dist_summary = (
        works_df.groupby("district")
        .agg(
            n_works=("work_id", "count"),
            completion_rate=("status", lambda x: (x == "completed").mean()),
            mean_final_score=("final_score", "mean"),
            flagged=("final_score", lambda x: (x > 0).sum()),
        )
        .reset_index()
        .sort_values("mean_final_score", ascending=False)
    )

    if anon:
        dist_summary["district_display"] = dist_summary["district"].apply(anonymise_district)
    else:
        dist_summary["district_display"] = dist_summary["district"]

    dist_summary["completion_rate"] = dist_summary["completion_rate"].apply(lambda x: f"{x:.0%}")
    dist_summary["mean_final_score"] = dist_summary["mean_final_score"].apply(lambda x: f"{x:.1f}")

    show_dist = dist_summary[["district_display", "n_works", "flagged", "completion_rate", "mean_final_score"]]
    show_dist.columns = ["District", "Works", "Flagged", "Completion Rate", "Avg Score"]
    st.dataframe(show_dist.reset_index(drop=True), use_container_width=True, height=400)

# ── Agency ranked table ────────────────────────────────────────────────────────
st.markdown("### Agency Ranked Table")

show_agency = agency_df[[
    "agency_display", "n_works", "completion_rate", "stall_share",
    "mean_sanction_delay_days", "spend_to_release_ratio", "_iso_score" if "_iso_score" in agency_df.columns else "n_works",
    "is_anomalous",
]].copy()

show_agency["completion_rate"] = show_agency["completion_rate"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
show_agency["stall_share"] = show_agency["stall_share"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
show_agency["mean_sanction_delay_days"] = show_agency["mean_sanction_delay_days"].apply(
    lambda x: f"{x:.0f}d" if pd.notna(x) else "N/A"
)
show_agency["spend_to_release_ratio"] = show_agency["spend_to_release_ratio"].apply(
    lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
)
show_agency["is_anomalous"] = show_agency["is_anomalous"].map({True: "🔴 Flagged", False: "✅ Normal"})

col_rename = {
    "agency_display": "Agency",
    "n_works": "Works",
    "completion_rate": "Completion",
    "stall_share": "Stall Share",
    "mean_sanction_delay_days": "Avg Delay",
    "spend_to_release_ratio": "Spend/Release",
    "is_anomalous": "Status",
}
if "_iso_score" in show_agency.columns:
    col_rename["_iso_score"] = "Anomaly Score"

show_agency = show_agency.rename(columns=col_rename)
st.dataframe(show_agency.reset_index(drop=True), use_container_width=True, height=400)

st.markdown("""
<div class="app-footer">
  MPLADS-Lens · Agencies flagged here are statistical outliers for further review, not confirmed findings.
</div>
""", unsafe_allow_html=True)
