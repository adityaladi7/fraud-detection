"""
Run the entire fraud detection pipeline end to end with ONE command:

    python run_demo.py

No Streamlit, no Colab, no server. This script:
  1. Generates synthetic transaction data (if not already present)
  2. Trains the fraud model + SHAP explainer (if not already present)
  3. Streams a fresh batch of transactions through the live feature
     engineering + scoring pipeline
  4. Checks for drift against the training baseline
  5. Writes everything to report.html - a single static file you can
     open directly in any browser, or host for free on GitHub Pages.
"""

import os
import pickle
import webbrowser

import pandas as pd

from data.stream_simulator import generate_batch_csv, stream
from features.feature_engineer import FeatureEngineer, FEATURE_COLUMNS
from monitor.drift_monitor import check_drift
from model.train_model import train

DATA_PATH = "synthetic_transactions.csv"
MODEL_PATH = "fraud_model.pkl"
REPORT_PATH = "report.html"

SAMPLE_SIZE = 300


def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    if not os.path.exists(DATA_PATH):
        generate_batch_csv(path=DATA_PATH, n=50000)
    model, explainer = train(data_path=DATA_PATH, model_out=MODEL_PATH)
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def score_live_batch(model, n=SAMPLE_SIZE):
    fe = FeatureEngineer()
    rows = []
    for txn in stream(n=n, fraud_rate=0.05, seed=7):
        feat = fe.transform(txn.__dict__)
        X = pd.DataFrame([feat])[FEATURE_COLUMNS]
        feat["fraud_score"] = round(float(model.predict_proba(X)[0, 1]), 4)
        rows.append(feat)
    return pd.DataFrame(rows)


def explain_top_flags(explainer, df, top_n=5):
    flagged = df[df["fraud_score"] > 0.5].sort_values("fraud_score", ascending=False).head(top_n)
    explanations = []
    for _, row in flagged.iterrows():
        X_row = pd.DataFrame([row])[FEATURE_COLUMNS]
        shap_vals = explainer.shap_values(X_row)[0]
        contrib = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "value": X_row.iloc[0].values,
            "shap_contribution": shap_vals,
        }).sort_values("shap_contribution", key=abs, ascending=False).head(3)
        explanations.append((row, contrib))
    return explanations


def build_html_report(df, flagged, explanations, drift_df):
    def df_to_html(d, extra_class=""):
        return d.to_html(index=False, classes=f"tbl {extra_class}", border=0)

    explanation_blocks = ""
    for row, contrib in explanations:
        explanation_blocks += f"""
        <div class="card">
          <h4>Transaction {row['txn_id'][:8]}... (user {row['user_id']}, score {row['fraud_score']})</h4>
          <p class="muted">Amount: {row['amount']} &middot; Merchant: {row['merchant_category']} &middot; Country: {row['merchant_country']}</p>
          {df_to_html(contrib)}
        </div>"""

    flag_rate = len(flagged) / len(df) if len(df) else 0
    significant_drift = drift_df[drift_df["severity"] == "significant"]

    drift_banner = ""
    if not significant_drift.empty:
        feats = ", ".join(significant_drift["feature"].tolist())
        drift_banner = f'<div class="banner warn">Significant drift detected in: {feats}</div>'
    else:
        drift_banner = '<div class="banner ok">No significant drift detected.</div>'

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Real-time fraud detection report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #222; background: #fafafa; }}
  h1 {{ font-size: 24px; }}
  h2 {{ font-size: 18px; margin-top: 40px; border-bottom: 1px solid #ddd; padding-bottom: 6px; }}
  h4 {{ margin-bottom: 4px; }}
  .muted {{ color: #666; font-size: 13px; }}
  .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
  .metric {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px 24px; text-align: center; }}
  .metric .num {{ font-size: 28px; font-weight: 600; }}
  .metric .label {{ font-size: 13px; color: #666; }}
  table.tbl {{ border-collapse: collapse; width: 100%; margin: 10px 0 20px; font-size: 13px; }}
  table.tbl th, table.tbl td {{ border: 1px solid #e0e0e0; padding: 6px 10px; text-align: left; }}
  table.tbl th {{ background: #f0f0f0; }}
  .card {{ background: white; border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }}
  .banner {{ padding: 12px 16px; border-radius: 8px; margin: 16px 0; font-size: 14px; }}
  .banner.ok {{ background: #e6f4ea; color: #1e7e34; }}
  .banner.warn {{ background: #fdecea; color: #a13a2e; }}
</style>
</head>
<body>
  <h1>Real-time fraud detection report</h1>
  <p class="muted">Generated from a fresh {len(df)}-transaction batch scored by the trained model.</p>

  <div class="metrics">
    <div class="metric"><div class="num">{len(df)}</div><div class="label">Transactions scored</div></div>
    <div class="metric"><div class="num">{len(flagged)}</div><div class="label">Flagged as fraud</div></div>
    <div class="metric"><div class="num">{flag_rate:.1%}</div><div class="label">Flag rate</div></div>
  </div>

  <h2>Top flagged transactions</h2>
  {df_to_html(flagged[["txn_id","user_id","amount","merchant_category","merchant_country","fraud_score"]].head(10))}

  <h2>Why were they flagged? (SHAP explanations)</h2>
  {explanation_blocks if explanation_blocks else "<p>No transactions were flagged in this batch.</p>"}

  <h2>Feature drift vs training baseline</h2>
  {drift_banner}
  {df_to_html(drift_df)}

</body>
</html>"""
    return html


def main():
    print("Loading or training model...")
    bundle = load_or_train_model()
    model, explainer = bundle["model"], bundle["explainer"]

    print(f"Scoring a fresh batch of {SAMPLE_SIZE} transactions...")
    df = score_live_batch(model)
    flagged = df[df["fraud_score"] > 0.5].sort_values("fraud_score", ascending=False)

    print("Generating SHAP explanations for top flagged transactions...")
    explanations = explain_top_flags(explainer, df)

    print("Checking for feature drift...")
    drift_df = check_drift(bundle["reference_features"], df[FEATURE_COLUMNS], FEATURE_COLUMNS)

    print("Writing report.html...")
    html = build_html_report(df, flagged, explanations, drift_df)
    with open(REPORT_PATH, "w") as f:
        f.write(html)

    print(f"\nDone. Open {os.path.abspath(REPORT_PATH)} in your browser.")
    try:
        webbrowser.open(f"file://{os.path.abspath(REPORT_PATH)}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
