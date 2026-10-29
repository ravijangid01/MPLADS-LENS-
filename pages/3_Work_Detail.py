"""
pages/3_Work_Detail.py
Detailed view for a single flagged work: headline reason, score breakdown,
evidence panels (cost/dup/stalled/agency/delay), peer table,
Legitimate Reasons box, and Mark-as-Reviewed control.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from pipeline.anonymise import get_display_columns, apply_anonymisation
from pipeline.run_pipeline import load_scored_df, load_review_log, mark_reviewed
from pipeline.score import compute_final_scores, build_reason_summary, DETECTOR_NAMES
from pipeline.peer_groups import describe_peer_group

st.set_page_config(page_title="Work Detail | MPLADS-Lens", page_icon="🔎", layout="wide")

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

st.markdown("## 🔎 Work Detail")

DB_PATH = "data/mplads_lens.duckdb"
if not Path(DB_PATH).exists():
    st.warning("No pipeline data found. Please run the pipeline from the **📁 Data** page.")
    st.stop()


@st.cache_data(ttl=60)
def get_data(db_path: str) -> pd.DataFrame:
    return load_scored_df(db_path)


df_all = get_data(DB_PATH)
weights = st.session_state.get("weights")
df_all = compute_final_scores(df_all, weights=weights)
df_all["reason_summary"] = df_all.apply(build_reason_summary, axis=1)
df_all["severity"] = df_all["severity"].astype(str)

anon = st.session_state.get("anon_mode", True)
if anon and "district_anon" not in df_all.columns:
    df_all = apply_anonymisation(df_all)

# ── Work selection ─────────────────────────────────────────────────────────────
presel = st.session_state.get("selected_work_id", "")
work_id_input = st.text_input(
    "Enter Work ID",
    value=presel,
    placeholder="e.g. RAJAIS-000042",
    help="Enter the Work ID from the Flagged Works table.",
)

if not work_id_input.strip():
    # Default to top-scored work
    if len(df_all) > 0:
        work_id_input = str(df_all.sort_values("final_score", ascending=False).iloc[0]["work_id"])
        st.info(f"Showing highest-scored work: **{work_id_input}**")

row_matches = df_all[df_all["work_id"] == work_id_input.strip()]
if len(row_matches) == 0:
    st.error(f"Work ID `{work_id_input}` not found. Check spelling or return to Flagged Works.")
    st.stop()

row = row_matches.iloc[0]

# ── Headline ──────────────────────────────────────────────────────────────────
sev = str(row.get("severity", "Low"))
sev_colors = {"High": "#C62828", "Medium": "#E65100", "Low": "#455A64"}
sev_bg = {"High": "#FDECEA", "Medium": "#FFF8E1", "Low": "#E8EEF7"}
col_map = get_display_columns(anon)

st.markdown(f"""
<div class="section-card">
  <h3 style="margin-top:0">{row.get('work_title','N/A')}</h3>
  <p>
    <strong>Work ID:</strong> {row['work_id']} &nbsp;|&nbsp;
    <strong>Category:</strong> {row.get('work_category','').title()} &nbsp;|&nbsp;
    <strong>State:</strong> {row.get('state','N/A')} &nbsp;|&nbsp;
    <strong>District:</strong> {row.get(col_map['district'], row.get('district','N/A'))} &nbsp;|&nbsp;
    <strong>Status:</strong> {row.get('status','N/A').replace('_',' ').title()}
  </p>
  <p>
    <strong>Sanctioned:</strong> ₹{row.get('sanctioned_amount',0)/1e5:.1f}L &nbsp;|&nbsp;
    <strong>Released:</strong> ₹{row.get('released_amount',0)/1e5:.1f}L &nbsp;|&nbsp;
    <strong>Expenditure:</strong> ₹{row.get('expenditure_amount',0)/1e5:.1f}L &nbsp;|&nbsp;
    <strong>Completion:</strong> {row.get('completion_percent',0):.0f}%
  </p>
  <p>
    <span style="background:{sev_bg[sev]};color:{sev_colors[sev]};padding:3px 12px;
    border-radius:20px;font-weight:700;font-size:0.85rem">{sev} Severity</span>
    &nbsp; <strong>Composite Score: {row.get('final_score',0):.1f} / 100</strong>
  </p>
</div>
""", unsafe_allow_html=True)

# ── Score Breakdown ────────────────────────────────────────────────────────────
st.markdown("### 📊 Score Breakdown by Detector")
det_labels = {
    "cost": "Cost Outlier",
    "duplicate": "Near-Duplicate",
    "stalled": "Stalled Work",
    "agency": "Agency Pattern",
    "delay": "Process Delay",
}
det_scores = {d: float(row.get(f"{d}_score", 0)) for d in DETECTOR_NAMES}

bar_fig = go.Figure(go.Bar(
    x=list(det_labels.values()),
    y=list(det_scores.values()),
    marker_color=["#C62828" if v > 70 else "#E65100" if v > 40 else "#0070C0" for v in det_scores.values()],
    text=[f"{v:.1f}" for v in det_scores.values()],
    textposition="outside",
))
bar_fig.update_layout(
    yaxis=dict(range=[0, 110], title="Score (0–100)"),
    template="plotly_white",
    paper_bgcolor="white",
    font_color="#1F3864",
    margin=dict(l=10, r=10, t=20, b=10),
    height=260,
    showlegend=False,
)
st.plotly_chart(bar_fig, use_container_width=True)

# ── Evidence panels ────────────────────────────────────────────────────────────
st.markdown("### 🔍 Evidence")

tabs = st.tabs(["Cost Outlier", "Near-Duplicate", "Stalled Work", "Agency Pattern", "Process Delay"])

# ── Tab 1: Cost Outlier ────────────────────────────────────────────────────────
with tabs[0]:
    ev = row.get("cost_evidence", {})
    if not isinstance(ev, dict):
        ev = {}
    score_val = det_scores.get("cost", 0)

    if score_val == 0:
        st.info("No cost outlier flag for this work.")
    else:
        st.markdown(f"**{row.get('cost_reason','No reason recorded.')}**")
        pg_desc = ev.get("peer_group_description", "N/A")
        pg_size = ev.get("peer_group_size", 0)
        pg_level = ev.get("peer_level", 0)
        peer_med = ev.get("peer_median_cost", 0)
        fallback_labels = {0: "Category × District Class × Year", 1: "Category × State (fallback)", 2: "Category only (widest fallback)"}
        st.markdown(f"""
<div class="evidence-panel">
  <strong>Peer group:</strong> {pg_desc} ({pg_size} works) <br>
  <strong>Peer group level:</strong> {fallback_labels.get(pg_level, str(pg_level))} <br>
  <strong>Peer median cost:</strong> ₹{peer_med/1e5:.1f}L &nbsp;|&nbsp;
  <strong>This work:</strong> ₹{row.get('sanctioned_amount',0)/1e5:.1f}L &nbsp;|&nbsp;
  <strong>Ratio:</strong> {ev.get('cost_ratio',1):.1f}×
</div>
""", unsafe_allow_html=True)

        # Strip / box plot of peer group costs
        peer_costs = ev.get("peer_costs", [])
        if peer_costs:
            peer_df = pd.DataFrame({"Cost (₹ Lakh)": [c / 1e5 for c in peer_costs], "Group": "Peer works"})
            this_cost = row.get("sanctioned_amount", 0) / 1e5
            fig = go.Figure()
            fig.add_trace(go.Box(
                y=peer_df["Cost (₹ Lakh)"],
                name="Peer Works",
                marker_color="#0070C0",
                boxpoints="all",
                jitter=0.3,
                pointpos=0,
            ))
            fig.add_trace(go.Scatter(
                x=["Peer Works"],
                y=[this_cost],
                mode="markers",
                name="This work",
                marker=dict(color="#C62828", size=14, symbol="diamond"),
            ))
            fig.add_hline(y=peer_med / 1e5, line_dash="dash", line_color="#C55A11",
                          annotation_text="Peer median", annotation_position="right")
            fig.update_layout(
                template="plotly_white", paper_bgcolor="white", font_color="#1F3864",
                yaxis_title="Cost (₹ Lakh)", height=350, margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        # 10 nearest peers table
        peer_sample = ev.get("peer_sample", [])
        if peer_sample:
            st.markdown("**10 nearest peers:**")
            peer_tbl = pd.DataFrame(peer_sample)
            if "sanctioned_amount" in peer_tbl.columns:
                peer_tbl["sanctioned_amount"] = peer_tbl["sanctioned_amount"].apply(lambda x: f"₹{x/1e5:.1f}L")
            st.dataframe(peer_tbl, use_container_width=True)

# ── Tab 2: Near-Duplicate ──────────────────────────────────────────────────────
with tabs[1]:
    ev = row.get("duplicate_evidence", {})
    if not isinstance(ev, dict):
        ev = {}
    score_val = det_scores.get("duplicate", 0)

    if score_val == 0:
        st.info("No near-duplicate flag for this work.")
    else:
        st.markdown(f"**{row.get('duplicate_reason','No reason recorded.')}**")
        work_a = ev.get("work_a", {})
        work_b = ev.get("work_b", {})

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"""
<div class="evidence-panel">
  <strong>Work A (this work)</strong><br>
  <strong>ID:</strong> {work_a.get('work_id','')}<br>
  <strong>Title:</strong> {work_a.get('title','')}<br>
  <strong>Sanctioned:</strong> {work_a.get('date','')}<br>
  <strong>Amount:</strong> ₹{float(work_a.get('amount',0))/1e5:.1f}L
</div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
<div class="evidence-panel">
  <strong>Work B (potential duplicate)</strong><br>
  <strong>ID:</strong> {work_b.get('work_id','')}<br>
  <strong>Title:</strong> {work_b.get('title','')}<br>
  <strong>Sanctioned:</strong> {work_b.get('date','')}<br>
  <strong>Amount:</strong> ₹{float(work_b.get('amount',0))/1e5:.1f}L
</div>""", unsafe_allow_html=True)

        st.markdown(f"""
**Cosine similarity:** {ev.get('cosine_similarity',0):.2f} &nbsp;|&nbsp;
**Fuzzy score:** {ev.get('fuzzy_score',0):.0f}/100 &nbsp;|&nbsp;
**Days apart:** {ev.get('days_apart',0)} days
""")

# ── Tab 3: Stalled Work ────────────────────────────────────────────────────────
with tabs[2]:
    ev = row.get("stalled_evidence", {})
    if not isinstance(ev, dict):
        ev = {}
    score_val = det_scores.get("stalled", 0)

    if score_val == 0:
        st.info("No stalled work flag for this work.")
    else:
        st.markdown(f"**{row.get('stalled_reason','No reason recorded.')}**")

        # Timeline
        timeline_events = [
            ("Recommendation", ev.get("recommended_date", row.get("recommended_date", "N/A"))),
            ("Sanction", ev.get("sanctioned_date", str(row.get("sanctioned_date", "N/A")))),
            ("Last Progress", ev.get("last_progress_date", str(row.get("last_progress_date", "Not recorded")))),
            ("Today", str(pd.Timestamp.today().date())),
        ]

        st.markdown('<div style="margin-top:12px">', unsafe_allow_html=True)
        for label, date_val in timeline_events:
            is_stalled = label in ("Last Progress", "Today") and str(date_val) in ("Not recorded", "NaT", "nan", "None")
            css_class = "stalled" if label == "Last Progress" and (str(date_val) in ("Not recorded", "NaT", "nan", "None")) else ""
            st.markdown(f"""
<div class="timeline-item {css_class}">
  <strong>{label}:</strong> {date_val}
</div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
**Released funds:** ₹{ev.get('released_amount',0)/1e5:.1f}L of ₹{ev.get('sanctioned_amount',0)/1e5:.1f}L sanctioned &nbsp;|&nbsp;
**Completion:** {ev.get('completion_percent',0):.0f}% &nbsp;|&nbsp;
**Days since sanction:** {ev.get('days_since_sanction',0)}
""")

# ── Tab 4: Agency Pattern ──────────────────────────────────────────────────────
with tabs[3]:
    ev = row.get("agency_evidence", {})
    if not isinstance(ev, dict):
        ev = {}
    score_val = det_scores.get("agency", 0)

    if score_val == 0:
        st.info("No agency pattern flag for this work.")
    else:
        st.markdown(f"**{row.get('agency_reason','No reason recorded.')}**")
        agency_name = ev.get("agency", row.get("implementing_agency", "N/A"))
        if anon:
            from pipeline.anonymise import anonymise_agency
            agency_display = anonymise_agency(str(agency_name))
        else:
            agency_display = agency_name

        st.markdown(f"**Agency under review:** {agency_display} ({ev.get('n_works',0)} works)")

        features = ev.get("features", {})
        medians = ev.get("peer_medians", {})

        if features and medians:
            feature_labels = {
                "completion_rate": "Completion Rate",
                "stall_share": "Stall Share",
                "mean_sanction_delay_days": "Avg Sanction Delay (days)",
                "spend_to_release_ratio": "Spend-to-Release Ratio",
                "mean_cost_z": "Mean Cost Z-Score",
            }
            comp_df = pd.DataFrame({
                "Feature": [feature_labels.get(k, k) for k in features],
                "This Agency": list(features.values()),
                "Peer Median": [medians.get(k, 0) for k in features],
            })
            fig = go.Figure()
            fig.add_trace(go.Bar(name="This Agency", x=comp_df["Feature"],
                                  y=comp_df["This Agency"], marker_color="#C55A11"))
            fig.add_trace(go.Bar(name="Peer Median", x=comp_df["Feature"],
                                  y=comp_df["Peer Median"], marker_color="#0070C0"))
            fig.update_layout(
                barmode="group", template="plotly_white",
                paper_bgcolor="white", font_color="#1F3864",
                height=320, margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 5: Process Delay ───────────────────────────────────────────────────────
with tabs[4]:
    ev = row.get("delay_evidence", {})
    if not isinstance(ev, dict):
        ev = {}
    score_val = det_scores.get("delay", 0)

    if score_val == 0:
        st.info("No process delay flag for this work.")
    else:
        st.markdown(f"**{row.get('delay_reason','No reason recorded.')}**")
        st.markdown(f"""
<div class="evidence-panel">
  <strong>Sanction delay:</strong> {ev.get('sanction_delay_days','N/A')} days
  (guideline: {ev.get('sanction_threshold', 75)} days) <br>
  <strong>Completion time:</strong> {ev.get('completion_days','N/A')} days <br>
  <strong>Peer 90th percentile:</strong> {ev.get('peer_p90_completion_days','N/A')} days
</div>
""", unsafe_allow_html=True)

# ── Legitimate Reasons Box ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div class="legit-box">
  <h4>✅ Legitimate reasons to check before escalation</h4>
  <ul>
    <li><strong>Terrain &amp; remoteness:</strong> Hill, desert, and tribal districts legitimately
        cost 20–45% more due to access, material transport, and labour.</li>
    <li><strong>Revised scope:</strong> The district authority may have sanctioned a revised or
        enhanced scope not reflected in the original recommendation.</li>
    <li><strong>Force majeure:</strong> Natural disasters, pandemic, or security situations
        can legitimately delay progress without any fault.</li>
    <li><strong>Two-phase works:</strong> What appears as a near-duplicate may be a legitimately
        sanctioned second phase of the same work.</li>
    <li><strong>Data entry lag:</strong> Progress updates may not have been uploaded yet;
        check the eSAKSHI portal directly for the latest status.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

# ── Mark as Reviewed ───────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📝 Mark as Reviewed")

# Check if already reviewed
review_log = load_review_log(DB_PATH)
existing = review_log[review_log["work_id"] == row["work_id"]]

if len(existing) > 0:
    st.markdown(f"""
<span class="reviewed-badge">✅ Reviewed</span>
&nbsp; Reviewed on {existing.iloc[0]['reviewed_at']} — {existing.iloc[0]['reviewer_note']}
""", unsafe_allow_html=True)

with st.form("review_form"):
    note = st.text_area(
        "Review note",
        placeholder="Enter your review findings, e.g. 'Verified with district office — delay due to land acquisition. No further action required.'",
        height=100,
    )
    submitted = st.form_submit_button("✅ Save Review")
    if submitted:
        if note.strip():
            mark_reviewed(str(row["work_id"]), note.strip(), DB_PATH)
            st.success("Review saved. This work is now marked as reviewed.")
            get_data.clear()
            st.rerun()
        else:
            st.warning("Please enter a review note before saving.")

st.markdown("""
<div class="app-footer">
  MPLADS-Lens · All flags are for review only — no finding of wrongdoing is implied.
</div>
""", unsafe_allow_html=True)
