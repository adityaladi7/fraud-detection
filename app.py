"""
Live demo dashboard. This is what you screen-share in an interview.

Run with: streamlit run dashboard/app.py

Flow: pull a transaction from the simulated stream -> compute rolling
features -> score with the model -> generate a SHAP-based explanation
for flagged transactions -> periodically check the live feature
distribution against the training baseline for drift.
"""

import os
import sys
import pickle
import time

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data.stream_simulator import stream
from features.feature_engineer import FeatureEngineer, FEATURE_COLUMNS
from monitor.drift_monitor import check_drift

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "fraud_model.pkl")

st.set_page_config(page_title="Real-time fraud detection", layout="wide")
st.title("Real-time fraud detection engine")
st.caption("Streaming transactions -> feature engineering -> model + SHAP -> drift monitoring")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


if not os.path.exists(MODEL_PATH):
    st.error(
        "No trained model found. Run `python model/train_model.py` first "
        "to generate data and train the model."
    )
    st.stop()

bundle = load_model()
model = bundle["model"]
explainer = bundle["explainer"]
reference_features = bundle["reference_features"]

if "fe" not in st.session_state:
    st.session_state.fe = FeatureEngineer()
if "scored_rows" not in st.session_state:
    st.session_state.scored_rows = []
if "stream_gen" not in st.session_state:
    st.session_state.stream_gen = stream(n=None, delay=0, fraud_rate=0.03, seed=None)

col_controls, col_metric = st.columns([3, 1])
with col_controls:
    batch_size = st.slider("Transactions per refresh", 5, 100, 25)
with col_metric:
    run = st.button("Score next batch")

if run:
    new_rows = []
    for _ in range(batch_size):
        raw_txn = next(st.session_state.stream_gen)
        feat_row = st.session_state.fe.transform(raw_txn.__dict__)
        X = pd.DataFrame([feat_row])[FEATURE_COLUMNS]
        prob = model.predict_proba(X)[0, 1]
        feat_row["fraud_score"] = round(float(prob), 4)
        new_rows.append(feat_row)

    st.session_state.scored_rows.extend(new_rows)
    # keep last 2000 rows so the app doesn't grow unbounded
    st.session_state.scored_rows = st.session_state.scored_rows[-2000:]

df = pd.DataFrame(st.session_state.scored_rows)

if df.empty:
    st.info("Click 'Score next batch' to start the stream.")
    st.stop()

flagged = df[df["fraud_score"] > 0.5].sort_values("fraud_score", ascending=False)

m1, m2, m3 = st.columns(3)
m1.metric("Transactions scored", len(df))
m2.metric("Flagged as fraud", len(flagged))
m3.metric("Flag rate", f"{len(flagged) / len(df):.1%}")

st.subheader("Flagged transactions")
if flagged.empty:
    st.write("No transactions flagged yet.")
else:
    top_flagged = flagged.head(10)
    st.dataframe(
        top_flagged[["txn_id", "user_id", "amount", "merchant_category",
                     "merchant_country", "fraud_score"]],
        use_container_width=True,
    )

    st.subheader("Why was the top transaction flagged?")
    top_row = top_flagged.iloc[0]
    X_row = pd.DataFrame([top_row])[FEATURE_COLUMNS]
    shap_values = explainer.shap_values(X_row)
    contributions = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "value": X_row.iloc[0].values,
        "shap_contribution": shap_values[0],
    }).sort_values("shap_contribution", key=abs, ascending=False)
    st.dataframe(contributions, use_container_width=True)
    st.caption("Positive SHAP contribution = pushed the score toward fraud.")

st.subheader("Feature drift vs training baseline")
if len(df) >= 30:
    drift_df = check_drift(reference_features, df[FEATURE_COLUMNS], FEATURE_COLUMNS)
    st.dataframe(drift_df, use_container_width=True)
    significant = drift_df[drift_df["severity"] == "significant"]
    if not significant.empty:
        st.warning(
            f"Significant drift detected in: {', '.join(significant['feature'].tolist())}"
        )
else:
    st.write("Score at least 30 transactions to see drift analysis.")
