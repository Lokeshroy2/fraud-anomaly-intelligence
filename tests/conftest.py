"""Shared test fixtures. Uses small synthetic data, never the real
1.3GB dataset - tests must run in CI, which doesn't have it."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_transactions():
    def _make(n: int, seed: int = 0) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        df = pd.DataFrame({
            "TransactionID": np.arange(n),
            "TransactionDT": np.sort(rng.integers(0, 30 * 86400, n)),
            "TransactionAmt": rng.uniform(1, 500, n),
            "ProductCD": rng.choice(["W", "C", "R"], n),
            "card1": rng.integers(1000, 1010, n),
            "card2": rng.choice([100.0, 200.0, np.nan], n),
            "card3": rng.choice([150.0, np.nan], n),
            "card4": rng.choice(["visa", "mastercard"], n),
            "card5": rng.choice([100.0, 200.0], n),
            "card6": rng.choice(["debit", "credit"], n),
            "addr1": rng.choice([100.0, 200.0, np.nan], n),
            "addr2": rng.choice([10.0, np.nan], n),
            "dist1": rng.choice([5.0, np.nan], n),
            "dist2": rng.choice([np.nan], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", np.nan], n),
            "R_emaildomain": rng.choice(["gmail.com", np.nan], n),
            "DeviceType": rng.choice(["mobile", "desktop", np.nan], n),
            "DeviceInfo": rng.choice(["iOS Device", "Windows", np.nan], n),
            "isFraud": rng.choice([0, 1], n, p=[0.9, 0.1]),
        })
        for i in range(1, 15):
            df[f"C{i}"] = rng.integers(0, 10, n).astype(float)
        for i in range(1, 16):
            df[f"D{i}"] = rng.choice([1.0, 5.0, np.nan], n)
        for i in range(1, 10):
            df[f"M{i}"] = rng.choice(["T", "F", np.nan], n)

        # A few correlated V-column "blocks" sharing a missingness pattern,
        # mirroring the real dataset structure that _select_v_columns relies on.
        v_base_1 = rng.normal(0, 1, n)
        missing_mask_1 = rng.random(n) < 0.3
        for i in range(1, 5):
            col = v_base_1 + rng.normal(0, 0.01, n)
            col[missing_mask_1] = np.nan
            df[f"V{i}"] = col

        v_base_2 = rng.normal(0, 1, n)
        missing_mask_2 = rng.random(n) < 0.6
        for i in range(5, 8):
            col = v_base_2 + rng.normal(0, 0.01, n)
            col[missing_mask_2] = np.nan
            df[f"V{i}"] = col

        return df

    return _make
