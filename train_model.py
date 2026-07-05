"""
Trains a gradient-boosted fraud classifier and attaches a SHAP explainer.

The point isn't the AUC number - it's that every prediction this model
makes later can be explained in plain language ("flagged because amount
is 6 std devs above this user's normal spend, and this is a new merchant
country for them"). That explainability layer is what makes this look
like a production fraud system instead of a Kaggle submission.
"""

import sys
import os
import pickle

import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from features.feature_engineer import FeatureEngineer, FEATURE_COLUMNS
from data.stream_simulator import generate_batch_csv


def build_training_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Replay the raw transaction log through the SAME feature engineer
    that will run online, in order, per user. This guarantees training
    and serving features are computed identically - a very common source
    of bugs in real fraud systems (train/serve skew) that you're avoiding
    by design here."""
    fe = FeatureEngineer()
    raw_df = raw_df.sort_values("timestamp")
    rows = [fe.transform(row.to_dict()) for _, row in raw_df.iterrows()]
    return pd.DataFrame(rows)


def train(data_path="synthetic_transactions.csv", model_out="fraud_model.pkl"):
    if not os.path.exists(data_path):
        generate_batch_csv(path=data_path, n=50000)

    raw_df = pd.read_csv(data_path)
    feat_df = build_training_frame(raw_df)

    X = feat_df[FEATURE_COLUMNS]
    y = feat_df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    ap = average_precision_score(y_test, probs)
    print(f"ROC-AUC: {auc:.4f}   Average Precision: {ap:.4f}")

    explainer = shap.TreeExplainer(model)

    with open(model_out, "wb") as f:
        pickle.dump({
            "model": model,
            "explainer": explainer,
            "feature_columns": FEATURE_COLUMNS,
            "reference_features": X_train,  # kept for drift comparison baseline
        }, f)

    print(f"Saved model + explainer to {model_out}")
    return model, explainer


if __name__ == "__main__":
    train()
