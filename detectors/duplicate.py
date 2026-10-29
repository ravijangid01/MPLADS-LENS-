"""
detectors/duplicate.py
Near-Duplicate Detector.

Uses TF-IDF cosine similarity + rapidfuzz token_set_ratio on normalised
work descriptions, with blocking by block_or_village × work_category and
a configurable date window. Efficient blocking keeps O(n) tractable.

NOTE: This detector NEVER reads the _planted_type column.
"""

from __future__ import annotations

import logging
import re
import string
from typing import Any

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from detectors.base import BaseDetector

logger = logging.getLogger(__name__)

# Basic grammatical stopwords
STOPWORDS = {
    "of", "in", "at", "the", "a", "an", "and", "for", "to",
    "under", "by", "with", "from", "on", "into", "over",
}


def _normalise_text(text: str) -> str:
    """Lowercase, remove numbers, punctuation, and basic grammar stopwords."""
    if pd.isna(text) or str(text).strip() == "":
        return ""
    t = str(text).lower()
    t = re.sub(r"\d+", " ", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in t.split() if w not in STOPWORDS and len(w) >= 2]
    return " ".join(tokens)


class NearDuplicateDetector(BaseDetector):
    """Detect near-duplicate works within the same location and category."""

    name = "duplicate"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        cfg = config.get("detectors", {}).get("duplicate", {})
        self.cosine_threshold: float = cfg.get("cosine_threshold", 0.75)
        self.fuzzy_threshold: float = cfg.get("fuzzy_threshold", 80)
        self.date_window: int = cfg.get("date_window_days", 180)
        self.combine_mode: str = cfg.get("combine_mode", "or")

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect near-duplicate works using TF-IDF + rapidfuzz with blocking.

        Returns
        -------
        pd.DataFrame
            Columns: work_id, score, reason, evidence.
        """
        df = df.copy()
        df["_norm_desc"] = df["work_description"].apply(_normalise_text)

        # Blocking key: district × block × category
        df["_block_key"] = (
            df["district"].astype(str)
            + "||"
            + df["block_or_village"].fillna("").astype(str)
            + "||"
            + df["work_category"].astype(str)
        )

        # Track max score per work_id
        max_scores: dict[str, float] = {wid: 0.0 for wid in df["work_id"]}
        reasons: dict[str, str] = {wid: "" for wid in df["work_id"]}
        evidences: dict[str, dict] = {wid: {} for wid in df["work_id"]}

        for block_key, block_df in df.groupby("_block_key"):
            if len(block_df) < 2:
                continue

            indices = block_df.index.tolist()
            texts = block_df["_norm_desc"].tolist()
            work_ids = block_df["work_id"].tolist()
            dates = block_df["sanctioned_date"].tolist()
            amounts = block_df["sanctioned_amount"].tolist()
            titles = block_df["work_title"].tolist()

            # TF-IDF cosine within block
            non_empty = [t if t.strip() else "NA" for t in texts]
            try:
                vect = TfidfVectorizer(min_df=1)
                tfidf_mat = vect.fit_transform(non_empty)
                cos_sim = cosine_similarity(tfidf_mat)
            except Exception:
                cos_sim = np.zeros((len(texts), len(texts)))

            n = len(block_df)
            for i in range(n):
                for j in range(i + 1, n):
                    wid_i, wid_j = work_ids[i], work_ids[j]
                    # Date window check
                    d_i, d_j = dates[i], dates[j]
                    if pd.notna(d_i) and pd.notna(d_j):
                        days_apart = abs((d_i - d_j).days)
                        if days_apart > self.date_window:
                            continue
                    else:
                        days_apart = 0

                    # Require non-trivial text length
                    if len(texts[i].strip()) < 5 or len(texts[j].strip()) < 5:
                        continue

                    cosine = float(cos_sim[i, j])
                    fuzzy = float(fuzz.token_set_ratio(texts[i], texts[j]))

                    if self.combine_mode == "or":
                        fired = (cosine >= self.cosine_threshold) or (fuzzy >= self.fuzzy_threshold)
                    else:
                        fired = (cosine >= self.cosine_threshold) and (fuzzy >= self.fuzzy_threshold)

                    if not fired:
                        continue

                    # Score: average of normalised similarities
                    sim_score = max(cosine, fuzzy / 100.0)
                    pair_score = float(np.clip(sim_score * 100, 50, 100))

                    reason = (
                        f"Similar description at same location "
                        f"(cosine={cosine:.2f}, fuzzy={fuzzy:.0f}/100), "
                        f"sanctioned {days_apart} days apart."
                    )
                    evidence = {
                        "match_work_id": wid_j,
                        "cosine_similarity": cosine,
                        "fuzzy_score": fuzzy,
                        "days_apart": days_apart,
                        "block_key": block_key,
                        "work_a": {
                            "work_id": wid_i,
                            "title": titles[i],
                            "date": str(dates[i]),
                            "amount": amounts[i],
                        },
                        "work_b": {
                            "work_id": wid_j,
                            "title": titles[j],
                            "date": str(dates[j]),
                            "amount": amounts[j],
                        },
                    }

                    for wid in [wid_i, wid_j]:
                        if pair_score > max_scores.get(wid, 0):
                            max_scores[wid] = pair_score
                            reasons[wid] = reason
                            evidences[wid] = evidence

        results = []
        for _, row in df.iterrows():
            wid = row["work_id"]
            results.append({
                "work_id": wid,
                "score": max_scores.get(wid, 0.0),
                "reason": reasons.get(wid, ""),
                "evidence": evidences.get(wid, {}),
            })

        out = pd.DataFrame(results)
        logger.info(
            "NearDuplicateDetector: %d flagged out of %d works (%.1f%%)",
            (out["score"] > 0).sum(), len(out), 100 * (out["score"] > 0).mean(),
        )
        return out
