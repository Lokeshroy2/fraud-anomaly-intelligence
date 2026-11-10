import numpy as np
import pandas as pd

from src.models.threshold import (
    REVIEW_COST,
    find_optimal_threshold,
    policy_cost,
    three_tier_policy,
    total_cost,
)


def test_total_cost_matches_manual_calculation():
    y_true = np.array([1, 0, 1, 0])
    amount = np.array([100.0, 50.0, 200.0, 10.0])
    flagged = np.array([False, True, False, False])

    # Missed fraud (y=1, not flagged): txn 0 (100) + txn 2 (200) = 300
    # Flagged (review cost): txn 1 -> 1 * REVIEW_COST
    expected = 300.0 + REVIEW_COST * 1
    assert total_cost(y_true, amount, flagged) == expected


def test_total_cost_zero_when_all_fraud_flagged_and_nothing_else():
    y_true = np.array([1, 0, 0])
    amount = np.array([100.0, 50.0, 50.0])
    flagged = np.array([True, False, False])

    assert total_cost(y_true, amount, flagged) == REVIEW_COST


def test_find_optimal_threshold_prefers_flagging_cheap_review_over_expensive_miss():
    # One expensive fraud case scored just above 0.5; review is cheap ($5)
    # relative to the fraud amount, so the optimal threshold should flag it.
    y_val = pd.Series([1, 0, 0, 0])
    amount_val = pd.Series([1000.0, 10.0, 10.0, 10.0])
    score_val = np.array([0.6, 0.1, 0.1, 0.1])

    best_t, best_cost = find_optimal_threshold(y_val, amount_val, score_val)
    assert best_t <= 0.6
    assert best_cost == REVIEW_COST


def test_three_tier_policy_buckets_correctly():
    score = np.array([0.01, 0.1, 0.5, 0.9])
    decision = three_tier_policy(score, t_review=0.05, t_block=0.8)

    assert list(decision) == ["approve", "review", "review", "block"]


def test_policy_cost_counts_each_tier_correctly():
    y_true = np.array([0, 1, 0])
    amount = np.array([10.0, 500.0, 10.0])
    decision = np.array(["approve", "approve", "block"])

    # approve+fraud (idx 1) = missed cost 500; block+legit (idx 2) = BLOCK_FP_COST
    from src.models.threshold import BLOCK_FP_COST

    assert policy_cost(y_true, amount, decision) == 500.0 + BLOCK_FP_COST
