"""
pages/1_Overview.py
Overview dashboard: KPI cards, flag counts by detector, score distribution,
flags by district, and top-10 highest-scored works.
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

from pipeline.anonymise import get_display_columns
from pipeline.run_pipeline import load_scored_df, run_pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Overview | MPLADS-Lens", page_icon="📊", layout="wide")

css_path = ROOT / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "anon_mode" not in st.session_state:
    st.session_state["anon_mode"] = True
if "weights" not in st.session_state:
    st.session_state["weights"] = {
        "cost": 0.30, "duplicate": 0.25, "stalled": 0.25, "agency": 0.10, "delay": 0.10
    }

# ── Plotly palette ────────────────────────────────────────────────────────────
PALETTE = ["#0070C0", "#1F3864", "#C55A11", "#7CB9E8", "#DEEAF6", "#5A6A8A"]
SEVERITY_COLORS = {"High": "#C62828", "Medium": "#E65100", "Low": "#455A64"}


@st.cache_data(ttl=300)
def get_data(db_path: str) -> pd.DataFrame:
    return load_scored_df(db_path)


# ── Persistent banner ──────────────────────────────────────────────────────────
st.markdown(
    '<div class="review-banner">'
    '⚠️ <strong>Outliers for review, not findings of fraud.</strong>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("## 📊 Overview")

DB_PATH = "data/mplads_lens.duckdb"

# Ensure pipeline has run
if not Path(DB_PATH).exists():
    with st.spinner("Running pipeline on synthetic data…"):
        run_pipeline()
    st.rerun()

try:
    df = get_data(DB_PATH)
except Exception as e:
    st.error(f"Could not load data: {e}. Please visit the Data page to regenerate.")
    st.stop()

# Apply weights rescoring if changed
from pipeline.score import compute_final_scores, build_reason_summary
weights = st.session_state.get("weights")
df = compute_final_scores(df, weights=weights)
df["reason_summary"] = df.apply(build_reason_summary, axis=1)
df["severity"] = df["severity"].astype(str)

anon = st.session_state.get("anon_mode", True)
col_map = get_display_columns(anon)
district_col = col_map["district"]

# Ensure anon columns exist
if anon and district_col not in df.columns:
    from pipeline.anonymise import apply_anonymisation
    df = apply_anonymisation(df)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
flagged_df = df[df["final_score"] > 0]
total_amount_flagged = flagged_df["sanctioned_amount"].sum() / 1e7  # crore
mean_quality = df["data_quality_score"].mean() if "data_quality_score" in df.columns else 0.0

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value">{len(df):,}</div>
  <div class="kpi-label">Works Analysed</div>
</div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value">{len(flagged_df):,}</div>
  <div class="kpi-label">Works Flagged</div>
</div>""", unsafe_allow_html=True)
with c3:
    pct = 100 * len(flagged_df) / max(len(df), 1)
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value">{pct:.1f}%</div>
  <div class="kpi-label">Flagged Share</div>
</div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value">₹{total_amount_flagged:.1f} Cr</div>
  <div class="kpi-label">Amount in Flagged Works</div>
</div>""", unsafe_allow_html=True)
with c5:
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value">{mean_quality:.0%}</div>
  <div class="kpi-label">Data Quality Score</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row ────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Flags by Detector")
    from pipeline.score import DETECTOR_NAMES
    det_counts = {
        det: int((df[f"{det}_score"] > 0).sum())
        for det in DETECTOR_NAMES
        if f"{det}_score" in df.columns
    }
    det_df = pd.DataFrame({
        "Detector": list(det_counts.keys()),
        "Count": list(det_counts.values()),
    }).sort_values("Count", ascending=True)
    fig_bar = px.bar(
        det_df, x="Count", y="Detector", orientation="h",
        color_discrete_sequence=PALETTE,
        template="plotly_white",
    )
    fig_bar.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        font_color="#1F3864",
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Number of Works Flagged",
        yaxis_title="",
        height=280,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_b:
    st.markdown("#### Score Distribution")
    fig_hist = px.histogram(
        df[df["final_score"] > 0],
        x="final_score",
        nbins=30,
        color_discrete_sequence=["#0070C0"],
        template="plotly_white",
        labels={"final_score": "Composite Score (0–100)"},
    )
    fig_hist.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        font_color="#1F3864",
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=280,
        yaxis_title="Count",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ── Flags by district ─────────────────────────────────────────────────────────
st.markdown("#### Flags by District (top 15)")
if district_col in df.columns:
    district_flags = (
        flagged_df.groupby(district_col)
        .size()
        .reset_index(name="Flagged Works")
        .sort_values("Flagged Works", ascending=False)
        .head(15)
    )
    district_flags.columns = ["District", "Flagged Works"]
    fig_dist = px.bar(
        district_flags, x="Flagged Works", y="District", orientation="h",
        color_discrete_sequence=["#C55A11"],
        template="plotly_white",
    )
    fig_dist.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        font_color="#1F3864",
        paper_bgcolor="white",
        plot_bgcolor="white",
        yaxis={"categoryorder": "total ascending"},
        height=360,
        yaxis_title="",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ── Severity breakdown ────────────────────────────────────────────────────────
col_s1, col_s2, col_s3 = st.columns(3)
high_n = int((df["severity"] == "High").sum())
med_n = int((df["severity"] == "Medium").sum())
low_n = int((df["severity"] == "Low").sum())

with col_s1:
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value" style="color:#C62828">{high_n:,}</div>
  <div class="kpi-label">🔴 High Severity (≥70)</div>
</div>""", unsafe_allow_html=True)
with col_s2:
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value" style="color:#E65100">{med_n:,}</div>
  <div class="kpi-label">🟡 Medium Severity (40–69)</div>
</div>""", unsafe_allow_html=True)
with col_s3:
    st.markdown(f"""
<div class="kpi-card">
  <div class="kpi-value" style="color:#455A64">{low_n:,}</div>
  <div class="kpi-label">🔵 Low Severity (&lt;40)</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Top 10 highest-scored works ───────────────────────────────────────────────
st.markdown("#### Top 10 Highest-Scored Works")
disp_cols = ["work_id", "work_category", district_col, "sanctioned_amount", "final_score", "severity", "reason_summary"]
disp_cols = [c for c in disp_cols if c in df.columns]

top10 = (
    df.sort_values("final_score", ascending=False)
    .head(10)[disp_cols]
    .copy()
)
top10["sanctioned_amount"] = top10["sanctioned_amount"].apply(lambda x: f"₹{x/1e5:.1f}L")
top10["final_score"] = top10["final_score"].apply(lambda x: f"{x:.1f}")

rename_map = {
    "work_id": "Work ID",
    "work_category": "Category",
    district_col: "District",
    "sanctioned_amount": "Amount",
    "final_score": "Score",
    "severity": "Severity",
    "reason_summary": "Reason",
}
top10 = top10.rename(columns=rename_map)

def _color_severity(val: str) -> str:
    colors = {"High": "background-color:#FDECEA;color:#C62828",
              "Medium": "background-color:#FFF8E1;color:#E65100",
              "Low": "background-color:#E8EEF7;color:#1F3864"}
    return colors.get(str(val), "")

styled = top10.style.applymap(_color_severity, subset=["Severity"])
st.dataframe(styled, use_container_width=True, height=380)

st.markdown("""
<div class="app-footer">
  MPLADS-Lens · All flags are for review only — no finding of wrongdoing is implied.
</div>
""", unsafe_allow_html=True)
