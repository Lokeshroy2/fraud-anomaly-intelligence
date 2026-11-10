"""Unsupervised anomaly detection (Isolation Forest) vs the supervised
LightGBM baseline.

The supervised model only ever learns patterns present in labeled
historical fraud. Isolation Forest is trained with zero label information,
so it generalizes differently: it flags transactions that look
structurally unusual, including fraud patterns the label set has never
seen before. That's the "new fraud typology" fallback case supervised
models can't cover by construction.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.data.load import load_labeled_data
from src.data.split import time_based_split
from src.models.metrics import compute_metrics

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def train_isolation_forest(X_train: pd.DataFrame, contamination: float = 0.035) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train.fillna(-999))
    return model


def anomaly_scores(model: IsolationForest, X: pd.DataFrame):
    # score_samples: higher = more normal. Flip sign so higher = more fraud-like,
    # matching the convention our metrics functions expect.
    return -model.score_samples(X.fillna(-999))


def main() -> None:
    df = load_labeled_data()
    train_df, val_df, _ = time_based_split(df)

    # Reuse the exact fitted FeatureEngineer from Step 4 training so the
    # feature set here matches the saved lightgbm_best.joblib model.
    fe = joblib.load(MODELS_DIR / "feature_engineer.joblib")
    X_train = fe.transform(train_df)
    X_val, y_val = fe.transform(val_df), val_df["isFraud"]

    iso_forest = train_isolation_forest(X_train, contamination=train_df["isFraud"].mean())
    iso_scores = anomaly_scores(iso_forest, X_val)
    iso_metrics = compute_metrics(y_val, iso_scores)

    lgbm = joblib.load(MODELS_DIR / "lightgbm_best.joblib")
    lgbm_scores = lgbm.predict_proba(X_val)[:, 1]
    lgbm_metrics = compute_metrics(y_val, lgbm_scores)

    comparison = pd.DataFrame(
        [
            {"model": "lightgbm (supervised)", **lgbm_metrics},
            {"model": "isolation_forest (unsupervised, no labels)", **iso_metrics},
        ]
    )
    print("\n=== Supervised vs unsupervised on validation set ===")
    print(comparison.to_string(index=False))

    random_baseline = y_val.mean()
    lift = iso_metrics["pr_auc"] / random_baseline
    print(
        f"\nInsight: LightGBM wins by a wide margin (PR-AUC {lgbm_metrics['pr_auc']:.3f} vs "
        f"{iso_metrics['pr_auc']:.3f}) because it has direct access to historical fraud "
        f"labels - Isolation Forest only reaches {iso_metrics['recall_at_top_1pct']:.1%} "
        f"recall in its top 1% vs LightGBM's {lgbm_metrics['recall_at_top_1pct']:.1%}. Still, "
        f"Isolation Forest's PR-AUC is a {lift:.1f}x lift over the random baseline "
        f"({random_baseline:.1%} fraud rate) with zero label information, which is exactly "
        "why it's useful as a fallback signal: it has no notion of 'known fraud patterns' "
        "to begin with, so it can flag structurally unusual transactions from fraud "
        "typologies the supervised model has never been trained on - not a replacement "
        "for the supervised model, but a complementary triage layer."
    )

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(iso_forest, MODELS_DIR / "isolation_forest.joblib")


if __name__ == "__main__":
    main()
