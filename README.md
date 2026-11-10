# Fraud Anomaly Intelligence

[![CI](https://github.com/Lokeshroy2/fraud-anomaly-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Lokeshroy2/fraud-anomaly-intelligence/actions/workflows/ci.yml)

An end-to-end fraud detection system on the [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection)
dataset (590K labeled transactions, ~3.5% fraud rate): time-based
train/val/test split, leakage-safe feature engineering, a supervised
model compared against an unsupervised anomaly detector, cost-based
decision thresholds, SHAP explainability, a FastAPI scoring service,
and drift monitoring on genuinely out-of-time data.

## Why this isn't just "ran XGBoost on Kaggle"

- **Time-based split, not random.** Fraud rate visibly drifts week to
  week ([notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)), so a random
  split would leak future patterns into training. Train/val/test are
  carved chronologically instead.
- **Leakage-safe feature pipeline.** Frequency encodings and `card1 +
  addr1` user aggregates are fit on train only, with an explicit
  fallback for cold-start keys never seen in val/test/production.
- **Supervised vs. unsupervised, honestly compared.** LightGBM
  massively outperforms a zero-label Isolation Forest (PR-AUC 0.61 vs
  0.13) — but the Isolation Forest still beats random by ~4x with no
  label information at all, which is the whole point: it's a fallback
  signal for fraud typologies the supervised model has never seen.
- **A real business framing, not just AUC.** A cost-based threshold
  (missed fraud = transaction amount, false alarm = fixed review cost)
  cuts test-set cost by **56.6%** vs the naive 0.5 cutoff.
- **Drift monitoring on real future data**, not a synthetic holdout:
  the Kaggle competition's unlabeled test set is chronologically later
  than the labeled data, so it's reused here as genuine out-of-time
  comparison data for PSI/KS drift checks — no labels needed.

## Results

| Step | Headline number |
|---|---|
| EDA | 3.50% fraud rate (~28:1 imbalance); 214/434 columns >50% missing |
| Feature engineering | V1-V339 reduced to 80 via missingness-pattern + correlation clustering |
| Baselines (val) | LightGBM (no rebalancing) PR-AUC 0.610 beats class-weighted (0.559) and SMOTE (0.576-0.587) |
| Baselines (test) | LightGBM PR-AUC 0.547, ROC-AUC 0.912, recall@top-1% 25.4% |
| Isolation Forest (val) | PR-AUC 0.134 - ~3.9x random baseline, zero labels used |
| Cost threshold | Optimal threshold 0.02 vs default 0.5 saves **$212K (56.6%)** on test |
| Drift | No week crosses PSI 0.2 alert threshold; visible bump (~0.10) in the unlabeled future period |

Val and test PR-AUC differ (0.610 vs 0.547) - reported as-is rather than
cherry-picked, since that gap is itself a realistic signal about
generalization on a time-based split.

Cost assumptions (stated, not measured - see
[src/models/threshold.py](src/models/threshold.py)): missed fraud =
full transaction amount; manual review = $5 flat; wrongly auto-blocking
a legit transaction = $25 flat.

## Architecture

```
ieee-fraud-detection/*.csv (gitignored, download from Kaggle)
        |
src/data/load.py        -> merge transaction + identity tables
src/data/split.py        -> time-based 70/15/15 train/val/test
src/data/features.py     -> leakage-safe FeatureEngineer (fit on train only)
        |
src/models/train.py       -> LightGBM vs Logistic Regression, MLflow-tracked
src/models/anomaly.py     -> Isolation Forest (unsupervised) comparison
src/models/threshold.py   -> cost-based approve/review/block thresholds
src/explain/shap_utils.py -> global importance + per-transaction reasons
        |
src/api/main.py  --------> FastAPI service (/score, /score/batch, /model/info)
src/drift/monitor.py -----> PSI/KS drift report vs the unlabeled future period
```

## Running it

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt

# Download the IEEE-CIS Fraud Detection dataset from Kaggle into ieee-fraud-detection/
# (train_transaction.csv, train_identity.csv, test_transaction.csv, test_identity.csv)

python -m src.models.train        # trains + saves models/lightgbm_best.joblib, feature_engineer.joblib
python -m src.models.anomaly      # Isolation Forest comparison
python -m src.models.threshold    # cost-based thresholds -> models/thresholds.json
python -m src.explain.shap_utils  # SHAP report -> reports/
python -m src.drift.monitor       # drift report -> reports/drift_report.html

pytest tests/                     # unit + API tests (synthetic fixtures, no dataset needed)
docker compose up                 # serves the API (uses the committed model artifacts) + MLflow UI
```

The trained model, feature pipeline, and thresholds (`models/lightgbm_best.joblib`,
`models/feature_engineer.joblib`, `models/thresholds.json`) are committed
since they're small (~5MB total) and let the API/Docker image run
without retraining. Everything else under `models/` (other experiments,
the raw dataset) is gitignored.

## Project layout

- `notebooks/01_eda.ipynb` - fraud rate, missingness, time-drift EDA
- `src/data/` - loading, time-based split, feature engineering
- `src/models/` - training, Isolation Forest, cost-based thresholds
- `src/explain/` - SHAP explainability
- `src/api/` - FastAPI scoring service
- `src/drift/` - PSI/KS drift monitoring
- `tests/` - unit tests (synthetic data) + API tests (real committed artifacts)
- `reports/` - generated plots and the static drift HTML report
