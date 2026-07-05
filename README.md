# Real-time fraud detection engine

A miniature version of the kind of system fintech and payments companies
(Mastercard, Visa, Stripe) run in production: transactions are scored in
real time, every flagged transaction comes with a plain-language reason,
and a monitor watches for the model going stale.

## Why this project (not just "trained a classifier")

Most portfolio fraud projects stop at a static CSV and an AUC score. This
one demonstrates the parts that actually distinguish an ML engineer from
an ML student:

1. **Streaming-shaped features, not batch features** - rolling per-user
   state (spend velocity, merchant familiarity) computed the same way
   online and offline, avoiding train/serve skew.
2. **Explainability, not just a score** - every flagged transaction gets
   a SHAP-based breakdown of *why*.
3. **Drift monitoring** - a PSI-based module that flags when live data
   diverges from what the model was trained on. Almost nobody builds
   this, and it's the single thing that signals "this person has
   thought about what happens after deployment."

## Architecture

```
Transaction stream -> Feature engineering -> Fraud model + SHAP -> Drift monitor -> Live dashboard
```

- `data/stream_simulator.py` - generates synthetic transactions with an
  injected fraud pattern (amount spikes, unfamiliar merchant country).
- `features/feature_engineer.py` - maintains rolling per-user state and
  computes features online, one transaction at a time.
- `model/train_model.py` - trains an XGBoost classifier and attaches a
  SHAP TreeExplainer.
- `monitor/drift_monitor.py` - Population Stability Index (PSI) drift
  detection, the same metric banks use in production.
- `dashboard/app.py` - Streamlit app that ties it all together for a
  live demo.

## Setup - one command, no server needed

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python run_demo.py
```

That's it. `run_demo.py` generates the data, trains the model, scores a
fresh batch of transactions, checks for drift, and writes everything to
`report.html` - a single static file that opens automatically in your
browser. No Streamlit, no server, no Colab needed.

Re-run `python run_demo.py` any time to score a new batch against the
same trained model.

### Optional: a live interactive dashboard instead of a static report

If you want the interactive version (sliders, live re-scoring) for a
screen-recorded demo, `dashboard/app.py` is a Streamlit app you can run
locally with `streamlit run dashboard/app.py`. This is optional -
`run_demo.py` is the simplest path and is what you'd point a recruiter to
on GitHub.

### Hosting the report on GitHub

Since `report.html` is a single self-contained static file, you can:
1. Push the repo (including the generated `report.html`) to GitHub.
2. Enable GitHub Pages (repo Settings -> Pages -> deploy from the
   branch/root), and your report is live at
   `https://yourusername.github.io/fraud-detection-engine/report.html`
   with zero extra setup.

## What to put in your resume / interview

Don't describe this as "built a fraud detection model." Describe the
system:

> Built an end-to-end real-time fraud detection pipeline with online
> feature engineering, SHAP-based explainability, and PSI-based drift
> monitoring to detect model staleness - demoed via a live Streamlit
> dashboard.

Be ready to talk about:
- **Latency**: how the rolling-window feature computation stays O(1)
  per transaction instead of rescanning history.
- **Train/serve skew**: why replaying data through the same
  `FeatureEngineer` class for training avoids a very common production bug.
- **What you'd change for real production**: swap the in-memory
  `FeatureEngineer` state for a Kafka Streams / Flink state store or
  Redis, put the model behind a low-latency serving endpoint, and
  replace the PSI check with a scheduled job that alerts on-call.

## Extending it (if you want to go beyond the weekend scope)

- Swap the synthetic simulator for a real public fraud dataset (e.g.
  IEEE-CIS Fraud Detection on Kaggle) to strengthen the "real data" story.
- Add a feedback loop: let flagged transactions be labeled "confirmed
  fraud" / "false positive" in the dashboard, and periodically retrain.
- Containerize with Docker and add a docker-compose that spins up the
  simulator, model service, and dashboard as separate processes -
  closer to how this would actually be deployed.
