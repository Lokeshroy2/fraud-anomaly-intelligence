"""Shared evaluation metrics for imbalanced fraud classification."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def recall_at_precision(y_true, y_score, target_precision: float = 0.5) -> float:
    """Best recall achievable while keeping precision >= target_precision.

    More actionable than a single accuracy/F1 number: "if we insist reviewed
    transactions are right at least half the time, how much fraud do we
    actually catch?"
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    eligible = recall[precision >= target_precision]
    return float(eligible.max()) if len(eligible) else 0.0


def recall_at_top_k_pct(y_true, y_score, k_pct: float = 1.0) -> float:
    """Of the top k% highest-scored transactions, what fraction of all fraud is caught."""
    n = len(y_score)
    k = max(1, int(n * k_pct / 100))
    top_k_idx = np.argsort(y_score)[-k:]
    y_true = np.asarray(y_true)
    total_fraud = y_true.sum()
    if total_fraud == 0:
        return 0.0
    return float(y_true[top_k_idx].sum() / total_fraud)


def compute_metrics(y_true, y_score) -> dict:
    return {
        "pr_auc": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "recall_at_precision_50": recall_at_precision(y_true, y_score, 0.5),
        "recall_at_top_1pct": recall_at_top_k_pct(y_true, y_score, 1.0),
    }
