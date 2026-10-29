"""
pages/6_Method_and_Limits.py
Plain-language explanation of each detector, thresholds, ethics statement,
limitations list, and live validation table (recall/precision vs planted anomalies).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Method & Limits | MPLADS-Lens", page_icon="📖", layout="wide")

css_path = ROOT / "assets" / "style.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    '<div class="review-banner">'
    '⚠️ <strong>Outliers for review, not findings of fraud.</strong>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown("## 📖 Method & Limits")

# ── Ethics statement ───────────────────────────────────────────────────────────
st.markdown("""
<div class="section-card" style="border-left:5px solid #C55A11">
<h3 style="color:#C55A11;margin-top:0">⚠️ Important Ethics Statement</h3>

<strong>MPLADS-Lens is an outlier detection system, NOT a fraud detection system.</strong>
We do not claim, imply, or infer that any flagged work, implementing agency, district,
or Member of Parliament has engaged in corruption, fraud, or misconduct.

Statistical outliers can arise from perfectly legitimate causes including terrain,
remoteness, revised project scope, data entry lags, force majeure events, or measurement
artefacts in the source data.

Every flag produced by this system is a <em>signal for further investigation</em> by the
appropriate authority (district administration, state government, parliamentary committee,
or MoSPI). Investigation and verification — not automated scoring — determines whether
any action is warranted.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Detector explanations ──────────────────────────────────────────────────────
st.markdown("### How Each Detector Works")

with st.expander("🔴 Cost Outlier Detector", expanded=True):
    st.markdown("""
**What it does:** Identifies works whose sanctioned amount is statistically
unusually high compared to similar works.

**Method:**
1. Works are grouped into *peer groups* by `work_category × district_class × sanction_year`.
2. If a peer group has fewer than **8 works**, we fall back to `category × state`;
   if still too small, to `category` only. The fallback level is always shown in the evidence.
3. We take `log(sanctioned_amount)` (log-normal costs are standard in public works).
4. We compute a **robust z-score** using the group median and MAD
   (median absolute deviation), which is resistant to the very outliers we seek.
5. Works above the threshold **z = 3.5** (configurable) are flagged.
6. Only **upward** outliers are flagged; below-median costs are not.

**Legitimate reasons for high cost:** Hill/desert terrain (45% markup is typical),
exceptionally large scope, revised design, price escalation clauses.

**Threshold:** Robust z-score ≥ 3.5 (configurable in `config.yaml`).
    """)

with st.expander("🟡 Near-Duplicate Detector"):
    st.markdown("""
**What it does:** Finds pairs of works at the same location with very similar
descriptions sanctioned close in time.

**Method:**
1. Work descriptions are normalised (lowercase, numbers removed, stopwords removed).
2. **Blocking** by `block_or_village × work_category` so we only compare within
   geographically close works of the same type — this keeps the algorithm fast on 20k records.
3. Within each block, we compute:
   - **TF-IDF cosine similarity** (captures semantic similarity)
   - **rapidfuzz `token_set_ratio`** (handles reordered or partially repeated text)
4. A pair is flagged if cosine ≥ 0.75 OR fuzzy ≥ 80 (configurable; "or" mode by default)
   AND the sanction dates are within **180 days** (configurable).
5. Each work in a flagged pair receives a score proportional to the similarity.

**Legitimate reasons for similarity:** Two-phase works, adjacent sections of
the same road, works for different beneficiary groups in the same location.

**Thresholds:** cosine ≥ 0.75, fuzzy ≥ 80/100, date window ≤ 180 days (all configurable).
    """)

with st.expander("🟠 Stalled Work Detector"):
    st.markdown("""
**What it does:** Identifies funded works that have been sanctioned for over 2 years
with no reported progress.

**Method:**
Works are flagged if **all** of the following are true:
1. Status is not `completed`.
2. Sanctioned more than **730 days ago** (configurable).
3. `released_amount > 0` (public money has been disbursed).
4. `last_progress_date` is more than **365 days ago** OR is not recorded.

The score is higher for older works with more funds released.

**Legitimate reasons for stalling:** Land acquisition disputes, contractor default,
inter-departmental approval delays, court orders.

**Thresholds:** 730 days since sanction, 365 days no progress (both configurable).
    """)

with st.expander("🟣 Agency Pattern Detector"):
    st.markdown("""
**What it does:** Identifies implementing agencies with unusual performance
profiles across their portfolio of works.

**Method:**
1. For each agency with ≥ 5 works (configurable), we compute 5 features:
   - **Completion rate** (share of works with status = completed)
   - **Stall share** (funded works inactive for > 2 years)
   - **Mean sanction delay** (days from MP recommendation to DM sanction)
   - **Spend-to-release ratio** (expenditure ÷ released funds)
   - **Mean cost z-score** (average cost deviation vs all works)
2. These features are scaled with `RobustScaler` and fed to
   **IsolationForest** (contamination = 0.08, seed = 42).
3. Flagged agencies have their top 2 most deviant features reported.
4. All works under a flagged agency receive the agency's anomaly score.

**Important:** Agency flags reflect the *portfolio* profile, not individual works.
A flagged agency may have legitimate explanations (specialised in hard terrains, etc.).

**Threshold:** IsolationForest contamination = 0.08 (configurable).
    """)

with st.expander("🔵 Process Delay Detector"):
    st.markdown("""
**What it does:** Flags works with abnormally long sanction delays or completion times.

**Method:**
1. **Sanction delay** = days from `recommended_date` to `sanctioned_date`.
   - Flagged if > **75 days** (current MPLADS guidelines, April 2023 revision).
   - ⚠️ *This threshold must be verified against the latest MPLADS guidelines.*
2. **Completion time** = days from `sanctioned_date` to `completion_date` (completed works).
   - Flagged if above the **90th percentile** of the same peer group.

**Legitimate reasons for delay:** Ministerial approval for enhanced cost,
design revisions, contractor procurement timelines, seasonal restrictions.

**Thresholds:** 75-day sanction guideline (verify current guidelines), peer p90
completion time (configurable).
    """)

st.markdown("---")

# ── Scoring explanation ────────────────────────────────────────────────────────
st.markdown("### Composite Scoring")
st.markdown("""
Each detector produces a score from **0 to 100** (0 = no flag, 100 = strongest signal).

The **final composite score** is:

```
final_score = Σ (weight_i × score_i) + multi_detector_bonus
```

Where:
- Default weights: Cost 0.30, Duplicate 0.25, Stalled 0.25, Agency 0.10, Delay 0.10
- **Multi-detector bonus**: +5 points when ≥2 detectors fire (capped at 100)
- Weights are editable in `config/config.yaml` and via the sidebar sliders

**Severity thresholds:**
- 🔴 **High**: score ≥ 70
- 🟡 **Medium**: 40 ≤ score < 70
- 🔵 **Low**: score < 40
""")

st.markdown("---")

# ── Peer group fallback ────────────────────────────────────────────────────────
st.markdown("### Peer Group Fallback Logic")
st.markdown("""
| Level | Group Definition | Used when |
|-------|-----------------|-----------|
| 0 (tightest) | Category × District Class × Sanction Year | Group ≥ 8 works |
| 1 (fallback) | Category × State | Level 0 < 8 works |
| 2 (widest) | Category only | Level 1 < 8 works |

The fallback level is always shown in the cost outlier evidence panel for transparency.
""")

st.markdown("---")

# ── Live validation table ─────────────────────────────────────────────────────
st.markdown("### Validation Against Planted Anomalies")
st.markdown("""
The synthetic dataset contains **planted anomalies** in a hidden `_planted_type` column
that no detector reads. This allows us to measure detector performance on known ground truth.

The table below is computed **live** from the current synthetic data.
""")

DB_PATH = "data/mplads_lens.duckdb"
SYNTH_PATH = "data/synthetic_works.parquet"

if not Path(DB_PATH).exists() or not Path(SYNTH_PATH).exists():
    st.info("Run the pipeline from the **📁 Data** page to see live validation metrics.")
else:
    try:
        from pipeline.run_pipeline import load_scored_df
        scored = load_scored_df(DB_PATH)
        synthetic = pd.read_parquet(SYNTH_PATH)

        # Merge on work_id
        merged = scored.merge(
            synthetic[["work_id", "_planted_type"]],
            on="work_id",
            how="left",
        )
        merged["_planted_type"] = merged["_planted_type"].fillna("")

        # Planted type → detector mapping
        planted_to_det = {
            "cost_outlier": "cost",
            "duplicate": "duplicate",
            "stalled": "stalled",
            "agency_pattern": "agency",
            "delay": "delay",
        }

        rows = []
        for planted_type, det in planted_to_det.items():
            planted = merged[merged["_planted_type"] == planted_type]
            n_planted = len(planted)
            if n_planted == 0:
                continue
            detected = planted[planted[f"{det}_score"] > 0]
            recall = len(detected) / n_planted if n_planted > 0 else 0.0
            # Precision: of all works flagged by this detector, what fraction were planted?
            all_flagged_by_det = merged[merged[f"{det}_score"] > 0]
            n_flagged = len(all_flagged_by_det)
            true_positives = all_flagged_by_det[all_flagged_by_det["_planted_type"] == planted_type]
            precision = len(true_positives) / n_flagged if n_flagged > 0 else 0.0
            flag_rate = n_flagged / len(merged)

            recall_pass = "✅" if recall >= 0.80 else "❌"
            flag_pass = "✅" if flag_rate <= 0.08 else "⚠️"

            rows.append({
                "Detector": det.title(),
                "Planted Anomalies": n_planted,
                "Detected": len(detected),
                "Recall": f"{recall:.0%} {recall_pass}",
                "Precision": f"{precision:.0%}",
                "Flag Rate": f"{flag_rate:.1%} {flag_pass}",
                "Target Recall": "≥ 80%",
                "Max Flag Rate": "≤ 8%",
            })

        if rows:
            val_df = pd.DataFrame(rows)
            st.dataframe(val_df, use_container_width=True)
            st.caption(
                "✅ = meets target threshold. ❌ = below target. ⚠️ = flag rate slightly above target. "
                "Duplicate recall is expected to be lower since the planted duplicate pairs "
                "may not always share the same block, limiting blocking efficiency."
            )
        else:
            st.info("Could not match planted anomalies — ensure you are using synthetic data.")
    except Exception as e:
        st.warning(f"Could not compute validation metrics: {e}")

st.markdown("---")

# ── Limitations ────────────────────────────────────────────────────────────────
st.markdown("### Known Limitations")
st.markdown("""
1. **Legitimate cost variation**: Terrain effects, material prices, and scope
   differences explain most cost outliers. The system cannot automatically
   distinguish these from true anomalies.

2. **Incomplete data**: Many real MPLADS exports have missing progress dates,
   incomplete financial figures, or inconsistent category labels. This system
   degrades gracefully but flags more conservatively when data is sparse.

3. **No ground truth**: There is no confirmed set of "fraudulent" or "erroneous"
   works to train on. Planted anomalies in synthetic data provide a proxy but
   real data may have different distributional properties.

4. **Agency-level inference**: Agency flags are based on portfolio statistics,
   not individual works. A flagged agency may have entirely legitimate
   explanations for its profile.

5. **Guideline evolution**: MPLADS guidelines change periodically. The sanction
   delay threshold (currently 75 days) must be verified against the most recent
   guidelines before use in a formal review process.

6. **Text similarity limits**: The near-duplicate detector works best when
   work descriptions are detailed. Very short or generic descriptions (e.g.,
   "Road construction") will not be distinguishable.

7. **Date parsing**: Dates in formats other than dd-mm-yyyy or ISO 8601 may
   be misread. Always validate date-dependent flags against source documents.
""")

st.markdown("""
<div class="app-footer">
  MPLADS-Lens · Method page · Statistical outlier detection, not a finding of fraud or misconduct.
</div>
""", unsafe_allow_html=True)
