"""
Simulates a real-time transaction stream, the way you'd receive events
from Kafka / Kinesis in a real payments system.

Each transaction has a `user_id` so downstream feature engineering can
maintain per-user rolling state, and an injected `is_fraud` label so we
can train + evaluate the model. In production you would NOT have this
label at inference time - it's only here to build the training set and
to score your demo against ground truth.
"""

import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from faker import Faker

fake = Faker()

MERCHANT_CATEGORIES = [
    "grocery", "electronics", "travel", "restaurant",
    "fuel", "online_retail", "jewelry", "utilities",
]

NUM_USERS = 200


@dataclass
class Transaction:
    txn_id: str
    user_id: str
    timestamp: str
    amount: float
    merchant_category: str
    merchant_country: str
    is_fraud: int


def _make_user_pool(n=NUM_USERS):
    """Give each synthetic user a 'home' country and typical spend level,
    so fraud (foreign country, abnormal amount) has something to deviate from."""
    return [
        {
            "user_id": f"user_{i:04d}",
            "home_country": fake.country_code(),
            "avg_amount": random.uniform(15, 250),
        }
        for i in range(n)
    ]


def generate_transaction(user_pool, fraud_rate=0.02):
    user = random.choice(user_pool)
    is_fraud = random.random() < fraud_rate

    if is_fraud:
        # Fraud pattern: amount spikes, merchant category is atypical,
        # country doesn't match the user's home country.
        amount = round(user["avg_amount"] * random.uniform(4, 12), 2)
        merchant_country = fake.country_code()
        while merchant_country == user["home_country"]:
            merchant_country = fake.country_code()
        merchant_category = random.choice(["jewelry", "electronics", "online_retail"])
    else:
        amount = round(max(1, random.gauss(user["avg_amount"], user["avg_amount"] * 0.3)), 2)
        merchant_country = user["home_country"]
        merchant_category = random.choice(MERCHANT_CATEGORIES)

    return Transaction(
        txn_id=fake.uuid4(),
        user_id=user["user_id"],
        timestamp=datetime.utcnow().isoformat(),
        amount=amount,
        merchant_category=merchant_category,
        merchant_country=merchant_country,
        is_fraud=int(is_fraud),
    )


def stream(n=None, delay=0.0, fraud_rate=0.02, seed=None):
    """Generator that yields Transaction objects one at a time.
    Set delay > 0 to simulate real wall-clock arrival for a live demo.
    Set n=None for an infinite stream (e.g. feeding the dashboard)."""
    if seed is not None:
        random.seed(seed)
        Faker.seed(seed)

    user_pool = _make_user_pool()
    count = 0
    while n is None or count < n:
        yield generate_transaction(user_pool, fraud_rate=fraud_rate)
        count += 1
        if delay:
            time.sleep(delay)


def generate_batch_csv(path="synthetic_transactions.csv", n=50000, fraud_rate=0.02, seed=42):
    """Generate a static training set - this is what you'll use for
    model training since it's faster than streaming for that step."""
    import pandas as pd

    rows = [asdict(t) for t in stream(n=n, delay=0, fraud_rate=fraud_rate, seed=seed)]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path} ({df['is_fraud'].mean():.2%} fraud rate)")
    return df


if __name__ == "__main__":
    generate_batch_csv()
