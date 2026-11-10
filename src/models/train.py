"""Baseline models with MLflow tracking: Logistic Regression -> LightGBM,
comparing three imbalance-handling strategies on LightGBM (none / class
weights / SMOTE). PR-AUC is the primary metric since fraud is ~3.5% of
the data; ROC-AUC and recall-at-precision/top-k% are logged alongside it.
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data.features import FeatureEngineer
from src.data.load import load_labeled_data
from src.data.split import time_based_split
from src.models.metrics import compute_metrics

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
MLFLOW_EXPERIMENT = "fraud-detection"


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, imbalance: str) -> lgb.LGBMClassifier:
    params = dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    if imbalance == "none":
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
    elif imbalance == "class_weight":
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = lgb.LGBMClassifier(scale_pos_weight=scale_pos_weight, **params)
        model.fit(X_train, y_train)
    elif imbalance == "smote":
        X_filled = X_train.fillna(-999)
        X_res, y_res = SMOTE(sampling_strategy=0.3, k_neighbors=5, random_state=42).fit_resample(
            X_filled, y_train
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(X_res, y_res)
    else:
        raise ValueError(f"Unknown imbalance strategy: {imbalance}")

    return model


def main() -> None:
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    df = load_labeled_data()
    train_df, val_df, test_df = time_based_split(df)

    fe = FeatureEngineer().fit(train_df)
    X_train, y_train = fe.transform(train_df), train_df["isFraud"]
    X_val, y_val = fe.transform(val_df), val_df["isFraud"]

    results = []

    with mlflow.start_run(run_name="logistic_regression_baseline"):
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("imbalance", "class_weight")
        lr = train_logistic_regression(X_train, y_train)
        val_scores = lr.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, val_scores)
        mlflow.log_metrics(metrics)
        results.append({"model": "logistic_regression", "imbalance": "class_weight", **metrics})

    lightgbm_models = {}
    for imbalance in ["none", "class_weight", "smote"]:
        with mlflow.start_run(run_name=f"lightgbm_{imbalance}"):
            mlflow.log_param("model", "lightgbm")
            mlflow.log_param("imbalance", imbalance)
            model = train_lightgbm(X_train, y_train, imbalance)
            val_scores = model.predict_proba(X_val)[:, 1]
            metrics = compute_metrics(y_val, val_scores)
            mlflow.log_metrics(metrics)
            mlflow.lightgbm.log_model(model, "model")
            results.append({"model": "lightgbm", "imbalance": imbalance, **metrics})
            lightgbm_models[imbalance] = (model, metrics["pr_auc"])

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print("\n=== Validation results (sorted by PR-AUC) ===")
    print(results_df.to_string(index=False))

    best_imbalance, (best_model, best_pr_auc) = max(
        lightgbm_models.items(), key=lambda kv: kv[1][1]
    )
    print(f"\nBest LightGBM variant: imbalance={best_imbalance} (PR-AUC={best_pr_auc:.4f})")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "lightgbm_best.joblib")
    joblib.dump(fe, MODELS_DIR / "feature_engineer.joblib")


if __name__ == "__main__":
    main()
