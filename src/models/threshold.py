"""Cost-based decision thresholds, tuned on validation and reported on test.

Business cost assumptions (explicit, since these drive every number below):
  - A missed fraud transaction (false negative) costs its full TransactionAmt.
  - A false alarm sent to manual review (false positive, or any true positive
    that still needs confirming) costs a fixed REVIEW_COST per transaction -
    this is the analyst's time, not the transaction amount.
  - An auto-blocked transaction that turns out legit (false positive in the
    "block" tier) costs BLOCK_FP_COST - a flat customer-friction estimate,
    higher than a review since there's no human check before the decision.

These are stated assumptions for a portfolio project, not measured figures -
in a real deployment they'd come from finance/ops data.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.load import load_labeled_data
from src.data.split import time_based_split
from src.models.metrics import compute_metrics

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

REVIEW_COST = 5.0
BLOCK_FP_COST = 25.0
DEFAULT_THRESHOLD = 0.5


def total_cost(y_true: np.ndarray, amount: np.ndarray, flagged: np.ndarray) -> float:
    """Cost of a binary flag/no-flag decision.

    Flagged transactions (fraud or not) go to manual review: REVIEW_COST each.
    Unflagged fraud is a missed loss: full transaction amount.
    """
    missed_fraud_cost = amount[(~flagged) & (y_true == 1)].sum()
    review_cost = REVIEW_COST * flagged.sum()
    return float(missed_fraud_cost + review_cost)


def find_optimal_threshold(y_val, amount_val, score_val, grid=None) -> tuple[float, float]:
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    costs = [total_cost(y_val.values, amount_val.values, score_val >= t) for t in grid]
    best_idx = int(np.argmin(costs))
    return float(grid[best_idx]), costs[best_idx]


def three_tier_policy(score: np.ndarray, t_review: float, t_block: float) -> np.ndarray:
    """approve (score < t_review) / review (t_review <= score < t_block) / block (score >= t_block)."""
    decision = np.full(score.shape, "approve", dtype=object)
    decision[score >= t_review] = "review"
    decision[score >= t_block] = "block"
    return decision


def find_block_threshold(y_val, score_val, min_precision: float = 0.95) -> float:
    """Highest-confidence cutoff above which we trust an automatic block
    (no human review) - requires precision >= min_precision among flagged txns.
    """
    order = np.argsort(-score_val)
    y_sorted = y_val.values[order]
    score_sorted = score_val[order]
    cum_precision = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
    eligible = np.where(cum_precision >= min_precision)[0]
    if len(eligible) == 0:
        return float(score_val.max()) + 1e-6  # nothing qualifies; block tier empty
    return float(score_sorted[eligible.max()])


def policy_cost(y_true, amount, decision: np.ndarray) -> float:
    cost = 0.0
    cost += amount[(decision == "approve") & (y_true == 1)].sum()
    cost += REVIEW_COST * (decision == "review").sum()
    cost += BLOCK_FP_COST * ((decision == "block") & (y_true == 0)).sum()
    return float(cost)


def main() -> None:
    df = load_labeled_data()
    train_df, val_df, test_df = time_based_split(df)

    fe = joblib.load(MODELS_DIR / "feature_engineer.joblib")
    model = joblib.load(MODELS_DIR / "lightgbm_best.joblib")

    X_val, y_val, amt_val = fe.transform(val_df), val_df["isFraud"], val_df["TransactionAmt"]
    X_test, y_test, amt_test = fe.transform(test_df), test_df["isFraud"], test_df["TransactionAmt"]

    score_val = model.predict_proba(X_val)[:, 1]
    score_test = model.predict_proba(X_test)[:, 1]

    # --- Single optimal threshold, tuned on val, reported on test ---
    optimal_t, val_cost_at_optimal = find_optimal_threshold(y_val, amt_val, score_val)
    cost_default_test = total_cost(y_test.values, amt_test.values, score_test >= DEFAULT_THRESHOLD)
    cost_optimal_test = total_cost(y_test.values, amt_test.values, score_test >= optimal_t)
    savings = cost_default_test - cost_optimal_test

    print("=== Single-threshold cost optimization ===")
    print(f"Optimal threshold (tuned on val): {optimal_t:.2f}")
    print(f"Test cost @ default threshold (0.50): ${cost_default_test:,.2f}")
    print(f"Test cost @ optimal threshold ({optimal_t:.2f}): ${cost_optimal_test:,.2f}")
    print(f"Savings from using optimal vs default threshold: ${savings:,.2f} "
          f"({savings / cost_default_test:.1%} reduction)")

    # --- Three-tier approve/review/block policy ---
    t_block = find_block_threshold(y_val, score_val, min_precision=0.95)
    t_review = optimal_t

    decision_test = three_tier_policy(score_test, t_review, t_block)
    tier_counts = pd.Series(decision_test).value_counts()
    policy_cost_test = policy_cost(y_test.values, amt_test.values, decision_test)

    print("\n=== Three-tier policy (approve / review / block) ===")
    print(f"Review threshold: {t_review:.2f} | Block threshold: {t_block:.2f}")
    print(tier_counts.to_string())
    print(f"Test cost under 3-tier policy: ${policy_cost_test:,.2f}")

    print("\n=== Model discrimination on test (for reference) ===")
    print(compute_metrics(y_test, score_test))


if __name__ == "__main__":
    main()
