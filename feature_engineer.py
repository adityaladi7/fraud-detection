"""
This is the part that separates a real fraud engine from a notebook.
Instead of static columns in a CSV, features are computed from a rolling
window of each user's recent transaction history - the same shape of
computation you'd do with a Flink/Kafka Streams state store in production.

We keep it in-memory here (a dict of deques) so the whole thing runs
without external infra, but the logic - "what happened in the last N
transactions / last T minutes for this user" - is the real pattern.
"""

from collections import defaultdict, deque
from datetime import datetime, timedelta

WINDOW_SIZE = 10          # last N transactions per user
VELOCITY_WINDOW_MIN = 5   # minutes, for "transactions in last 5 min"


class FeatureEngineer:
    def __init__(self, window_size=WINDOW_SIZE, velocity_window_min=VELOCITY_WINDOW_MIN):
        self.window_size = window_size
        self.velocity_window_min = velocity_window_min
        # per-user rolling history of recent transactions
        self.history = defaultdict(lambda: deque(maxlen=window_size))
        # per-user set of merchant categories seen historically (long-term)
        self.merchant_profile = defaultdict(lambda: defaultdict(int))
        self.country_profile = defaultdict(lambda: defaultdict(int))

    def transform(self, txn: dict) -> dict:
        """Given a raw transaction dict, return it enriched with rolling
        features, THEN update internal state. Order matters: we compute
        features using history BEFORE this transaction, so we're not
        leaking the current transaction into its own baseline."""
        user_id = txn["user_id"]
        ts = datetime.fromisoformat(txn["timestamp"])
        hist = self.history[user_id]

        amounts = [h["amount"] for h in hist]
        avg_amount = sum(amounts) / len(amounts) if amounts else txn["amount"]
        amount_zscore = self._safe_zscore(txn["amount"], amounts)

        recent_count = sum(
            1 for h in hist
            if ts - datetime.fromisoformat(h["timestamp"]) <= timedelta(minutes=self.velocity_window_min)
        )

        merchant_counts = self.merchant_profile[user_id]
        total_merchant_obs = sum(merchant_counts.values())
        merchant_familiarity = (
            merchant_counts[txn["merchant_category"]] / total_merchant_obs
            if total_merchant_obs > 0 else 0.0
        )

        country_counts = self.country_profile[user_id]
        total_country_obs = sum(country_counts.values())
        country_familiarity = (
            country_counts[txn["merchant_country"]] / total_country_obs
            if total_country_obs > 0 else 0.0
        )

        features = {
            **txn,
            "avg_amount_last_n": round(avg_amount, 2),
            "amount_zscore": round(amount_zscore, 3),
            "txn_count_last_5min": recent_count,
            "merchant_familiarity": round(merchant_familiarity, 3),
            "country_familiarity": round(country_familiarity, 3),
            "is_new_user": int(len(hist) == 0),
        }

        # update state AFTER computing features
        hist.append(txn)
        merchant_counts[txn["merchant_category"]] += 1
        country_counts[txn["merchant_country"]] += 1

        return features

    @staticmethod
    def _safe_zscore(value, history_values):
        if len(history_values) < 2:
            return 0.0
        mean = sum(history_values) / len(history_values)
        variance = sum((v - mean) ** 2 for v in history_values) / len(history_values)
        std = variance ** 0.5
        if std == 0:
            return 0.0
        return (value - mean) / std


FEATURE_COLUMNS = [
    "amount",
    "avg_amount_last_n",
    "amount_zscore",
    "txn_count_last_5min",
    "merchant_familiarity",
    "country_familiarity",
    "is_new_user",
]
