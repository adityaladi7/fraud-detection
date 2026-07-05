"""
Almost no portfolio project includes this, which is exactly why it's
worth having. In production, a fraud model's input distribution shifts
constantly - new merchant categories, holiday spending spikes, a
promotion that changes typical transaction amounts - and a model that
looked great at training time silently degrades.

This module computes the Population Stability Index (PSI) between a
reference distribution (training data) and a live batch, per feature.
PSI is the industry-standard metric banks and card networks actually
use for this - not something invented for this project.

Rule of thumb (also industry-standard):
  PSI < 0.1  -> no significant drift
  0.1 - 0.25 -> moderate drift, monitor
  > 0.25     -> significant drift, investigate / retrain
"""

import numpy as np
import pandas as pd


def _psi_for_feature(reference: pd.Series, live: pd.Series, buckets=10) -> float:
    breakpoints = np.unique(
        np.percentile(reference, np.linspace(0, 100, buckets + 1))
    )
    if len(breakpoints) < 3:
        return 0.0  # not enough variance to bucket meaningfully

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    live_counts, _ = np.histogram(live, bins=breakpoints)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / max(ref_counts.sum(), 1))
    live_pct = np.where(live_counts == 0, 1e-4, live_counts / max(live_counts.sum(), 1))

    psi = np.sum((live_pct - ref_pct) * np.log(live_pct / ref_pct))
    return float(psi)


def check_drift(reference_df: pd.DataFrame, live_df: pd.DataFrame, feature_columns) -> pd.DataFrame:
    """Returns a per-feature PSI score plus a severity label, so the
    dashboard can surface exactly which features are drifting and by
    how much - not just a single opaque 'drift detected' flag."""
    results = []
    for col in feature_columns:
        if col not in reference_df.columns or col not in live_df.columns:
            continue
        psi = _psi_for_feature(reference_df[col], live_df[col])
        severity = (
            "stable" if psi < 0.1
            else "moderate" if psi < 0.25
            else "significant"
        )
        results.append({"feature": col, "psi": round(psi, 4), "severity": severity})

    return pd.DataFrame(results).sort_values("psi", ascending=False)


if __name__ == "__main__":
    # quick sanity check with synthetic shifted data
    rng = np.random.default_rng(0)
    reference = pd.DataFrame({"amount": rng.normal(50, 15, 5000)})
    live_stable = pd.DataFrame({"amount": rng.normal(50, 15, 500)})
    live_shifted = pd.DataFrame({"amount": rng.normal(120, 40, 500)})

    print(check_drift(reference, live_stable, ["amount"]))
    print(check_drift(reference, live_shifted, ["amount"]))
