"""FastAPI fraud scoring service.

Loads the trained LightGBM model, its fitted FeatureEngineer, a SHAP
TreeExplainer, and the cost-based decision thresholds exactly once at
startup (not per-request), then serves /score and /score/batch.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI

from src.api.schemas import BatchScoreIn, ModelInfo, Reason, ScoreOut, TransactionIn
from src.data.features import M_COLS
from src.explain.shap_utils import load_explainer, top_reasons

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
CATEGORICAL_COLS = {
    "card4", "card6", "ProductCD", "DeviceType", "DeviceInfo",
    "P_emaildomain", "R_emaildomain", *M_COLS,
}

ml_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_state["feature_engineer"] = joblib.load(MODELS_DIR / "feature_engineer.joblib")
    ml_state["model"] = joblib.load(MODELS_DIR / "lightgbm_best.joblib")
    ml_state["explainer"] = load_explainer(ml_state["model"])
    with open(MODELS_DIR / "thresholds.json") as f:
        ml_state["thresholds"] = json.load(f)
    ml_state["expected_raw_columns"] = ml_state["feature_engineer"].expected_raw_columns()
    yield
    ml_state.clear()


app = FastAPI(title="Fraud Anomaly Intelligence API", lifespan=lifespan)


def _transaction_to_row(txn: TransactionIn) -> pd.DataFrame:
    flat = txn.model_dump(exclude={"C", "D", "M", "V"})
    flat.update(txn.C)
    flat.update(txn.D)
    flat.update(txn.M)
    flat.update(txn.V)
    row = pd.DataFrame([flat])
    row = row.reindex(columns=ml_state["expected_raw_columns"], fill_value=np.nan)
    numeric_cols = [c for c in row.columns if c not in CATEGORICAL_COLS]
    row[numeric_cols] = row[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return row


def _decide(score: float) -> str:
    t = ml_state["thresholds"]
    if score >= t["block_threshold"]:
        return "block"
    if score >= t["review_threshold"]:
        return "review"
    return "approve"


def score_transaction(txn: TransactionIn) -> ScoreOut:
    row = _transaction_to_row(txn)
    X = ml_state["feature_engineer"].transform(row)
    score = float(ml_state["model"].predict_proba(X)[:, 1][0])
    reasons = top_reasons(ml_state["explainer"], X, n=3)
    return ScoreOut(
        TransactionID=txn.TransactionID,
        fraud_probability=score,
        decision=_decide(score),
        top_reasons=[Reason(**r) for r in reasons],
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    model = ml_state["model"]
    t = ml_state["thresholds"]
    return ModelInfo(
        model_type=type(model).__name__,
        n_features=model.n_features_,
        review_threshold=t["review_threshold"],
        block_threshold=t["block_threshold"],
        review_cost=t["review_cost"],
        block_fp_cost=t["block_fp_cost"],
    )


@app.post("/score", response_model=ScoreOut)
def score(txn: TransactionIn) -> ScoreOut:
    return score_transaction(txn)


@app.post("/score/batch", response_model=list[ScoreOut])
def score_batch(batch: BatchScoreIn) -> list[ScoreOut]:
    return [score_transaction(txn) for txn in batch.transactions]
