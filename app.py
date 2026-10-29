"""
app.py — MPLADS-Lens landing page and global layout.
Entry point: streamlit run app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on the Python path and is the working directory
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MPLADS-Lens | SIH26102",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load CSS ──────────────────────────────────────────────────────────────────
css_path = ROOT / "assets" / "style.css"
if css_path.exists():
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
if "anon_mode" not in st.session_state:
    st.session_state["anon_mode"] = True  # ON by default
if "data_source" not in st.session_state:
    st.session_state["data_source"] = "Synthetic demo data"
if "pipeline_run" not in st.session_state:
    st.session_state["pipeline_run"] = False
if "weights" not in st.session_state:
    st.session_state["weights"] = {
        "cost": 0.30, "duplicate": 0.25, "stalled": 0.25,
        "agency": 0.10, "delay": 0.10,
    }

# ── Header ────────────────────────────────────────────────────────────────────
source_label = st.session_state.get("data_source", "Synthetic demo data")
source_class = "user-data" if "User" in source_label else ""
st.markdown(f"""
<div class="app-header">
  <div>
    <h1>🔍 MPLADS-Lens</h1>
    <div class="subtitle">SIH26102 · Ministry of Statistics &amp; Programme Implementation</div>
  </div>
  <div>
    <span class="source-badge {source_class}">{source_label}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Persistent review banner ─────────────────────────────────────────────────
st.markdown(
    '<div class="review-banner">'
    '⚠️ <strong>Outliers for review, not findings of fraud.</strong> '
    'All flags are statistical signals for further investigation. '
    'No confirmation of wrongdoing is implied for any work, agency, or person.'
    '</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 MPLADS-Lens")
    st.markdown("**SIH26102 · MoSPI**")
    st.divider()

    # Anonymisation toggle
    st.markdown("### 🔒 Privacy")
    anon = st.toggle(
        "Anonymised view",
        value=st.session_state["anon_mode"],
        help="When ON: constituencies, members and agencies are shown as coded aliases. "
             "Toggle off only for your own locally-supplied data.",
    )
    st.session_state["anon_mode"] = anon
    if anon:
        st.caption("✅ Anonymised · Real names hidden")
    else:
        st.caption("⚠️ Showing raw names (local data only)")

    st.divider()

    # Scoring weight sliders
    st.markdown("### ⚖️ Detector Weights")
    st.caption("Adjust weights and click **Rescore** to re-rank.")
    w_cost = st.slider("Cost outlier", 0.0, 1.0, st.session_state["weights"]["cost"], 0.05)
    w_dup  = st.slider("Near-duplicate", 0.0, 1.0, st.session_state["weights"]["duplicate"], 0.05)
    w_stall = st.slider("Stalled work", 0.0, 1.0, st.session_state["weights"]["stalled"], 0.05)
    w_agency = st.slider("Agency pattern", 0.0, 1.0, st.session_state["weights"]["agency"], 0.05)
    w_delay = st.slider("Process delay", 0.0, 1.0, st.session_state["weights"]["delay"], 0.05)

    total_w = w_cost + w_dup + w_stall + w_agency + w_delay
    if total_w > 0:
        st.caption(f"Sum: {total_w:.2f} (auto-normalised to 1.0)")
    else:
        st.error("At least one weight must be > 0")

    if st.button("♻️ Rescore with these weights", use_container_width=True):
        st.session_state["weights"] = {
            "cost": w_cost, "duplicate": w_dup, "stalled": w_stall,
            "agency": w_agency, "delay": w_delay,
        }
        st.rerun()

    st.divider()
    st.markdown("### 🗺️ Navigation")
    try:
        st.page_link("pages/1_Overview.py", label="📊 Overview", icon="📊")
        st.page_link("pages/2_Flagged_Works.py", label="🚩 Flagged Works", icon="🚩")
        st.page_link("pages/3_Work_Detail.py", label="🔎 Work Detail", icon="🔎")
        st.page_link("pages/4_Agencies_Districts.py", label="🏢 Agencies & Districts", icon="🏢")
        st.page_link("pages/5_Data.py", label="📁 Data", icon="📁")
        st.page_link("pages/6_Method_and_Limits.py", label="📖 Method & Limits", icon="📖")
    except Exception:
        pass

# ── Landing page content ──────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
## Welcome to MPLADS-Lens

**MPLADS-Lens** is an explainable outlier-detection system for the
*Members of Parliament Local Area Development Scheme* (MPLADS).
It helps parliamentary oversight committees and MoSPI analysts identify
works that may warrant closer review — cost anomalies, near-duplicate
entries, stalled funded works, and process delays.

### What this system does
- 🔴 **Cost outliers** — works priced far outside their peer group
- 🟡 **Near-duplicates** — similar works at the same location, close in time
- 🟠 **Stalled works** — funded but with no progress for an extended period
- 🟣 **Agency patterns** — implementing agencies with unusual performance profiles
- 🔵 **Process delays** — abnormal time from recommendation to sanction or completion

> **Important**: Every flag is a statistical signal, not a confirmed finding.
> Legitimate reasons — terrain, remoteness, revised scope — must be checked
> before any action. See the **Method & Limits** page for full details.

### Quick Start
1. The app loads with **synthetic demonstration data** (20,000 works).
2. Navigate to **📊 Overview** to see the summary.
3. Open **🚩 Flagged Works** to browse and filter flagged entries.
4. Click any row to open the **🔎 Work Detail** page with full evidence.
5. Upload your own eSAKSHI export on the **📁 Data** page.
    """)

with col2:
    st.markdown("""
### 📊 At a glance
""")
    # Run pipeline if not yet done
    from pathlib import Path
    db_path = "data/mplads_lens.duckdb"
    
    if not Path(db_path).exists():
        with st.spinner("Running pipeline on synthetic data (first launch)…"):
            from pipeline.run_pipeline import run_pipeline
            _, rep = run_pipeline()
            st.session_state["pipeline_run"] = True
            st.session_state["pipeline_report"] = rep
            st.rerun()
    else:
        st.session_state["pipeline_run"] = True

    if st.session_state.get("pipeline_run"):
        try:
            from pipeline.run_pipeline import load_scored_df
            df = load_scored_df(db_path)
            flagged = (df["final_score"] > 0).sum()
            high = (df["severity"] == "High").sum()
            total_amt = df.loc[df["final_score"] > 0, "sanctioned_amount"].sum() / 1e7

            st.markdown(f"""
<div class="kpi-card" style="margin-bottom:12px">
  <div class="kpi-value">{len(df):,}</div>
  <div class="kpi-label">Works Analysed</div>
</div>
<div class="kpi-card" style="margin-bottom:12px">
  <div class="kpi-value">{flagged:,}</div>
  <div class="kpi-label">Works Flagged</div>
</div>
<div class="kpi-card" style="margin-bottom:12px">
  <div class="kpi-value">{high:,}</div>
  <div class="kpi-label">High Severity</div>
</div>
<div class="kpi-card">
  <div class="kpi-value">₹{total_amt:.1f} Cr</div>
  <div class="kpi-label">Amount in Flagged Works</div>
</div>
""", unsafe_allow_html=True)
        except Exception as e:
            st.info("Pipeline data not yet loaded. Navigate to 📊 Overview to start.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
  MPLADS-Lens · SIH26102 · MoSPI · Built for Smart India Hackathon 2026 &nbsp;|&nbsp;
  All flags are for review only — no finding of wrongdoing is implied. &nbsp;|&nbsp;
  <strong>Data: Synthetic demo data (not real MPLADS data)</strong>
</div>
""", unsafe_allow_html=True)
