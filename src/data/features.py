"""Leakage-safe feature engineering.

`FeatureEngineer.fit` is called on the train split only. Everything it
learns (frequency maps, user aggregates, global fallbacks, selected V
columns) is stored on the object and reused as-is by `.transform()` on
val/test/inference — nothing is ever recomputed on data the model
shouldn't have seen yet. The same fitted object is serialized alongside
the model and reused by the API (Step 8) to featurize a single incoming
transaction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FREQ_ENCODE_COLS = [
    "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
    "ProductCD", "DeviceType", "DeviceInfo",
]
M_COLS = [f"M{i}" for i in range(1, 10)]
BASE_NUMERIC_COLS = [
    "TransactionAmt", "card1", "card2", "card3", "card5",
    "addr1", "addr2", "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
    "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9",
    "D10", "D11", "D12", "D13", "D14", "D15",
]
HIGH_MISSING_THRESHOLD = 0.90
V_CORR_THRESHOLD = 0.90


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    seconds = df["TransactionDT"]
    out["hour_of_day"] = (seconds // 3600) % 24
    out["day_of_week"] = (seconds // 86400) % 7
    return out


def _amount_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    amt = df["TransactionAmt"]
    out["log_amount"] = np.log1p(amt)
    out["amount_cents"] = np.round((amt - np.floor(amt)) * 100).astype(int)
    return out


def _m_col_features(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {"T": 1, "F": 0}
    out = pd.DataFrame(index=df.index)
    for col in M_COLS:
        if col in df.columns:
            out[col] = df[col].map(mapping).fillna(-1).astype(int)
    return out


def _select_v_columns(train_df: pd.DataFrame, v_cols: list[str]) -> list[str]:
    """Pick one representative V column per (missingness-pattern, correlation) cluster."""
    missing_frac = train_df[v_cols].isna().mean()
    keep_candidates = missing_frac[missing_frac < HIGH_MISSING_THRESHOLD].index.tolist()
    if not keep_candidates:
        return []

    # Group by rounded missingness pattern first — columns from the same
    # underlying source table tend to share an identical missing fraction.
    groups: dict[float, list[str]] = {}
    for col in keep_candidates:
        key = round(missing_frac[col], 6)
        groups.setdefault(key, []).append(col)

    selected: list[str] = []
    for cols in groups.values():
        if len(cols) == 1:
            selected.append(cols[0])
            continue
        # Within a missingness-pattern group, cluster by correlation and
        # keep one column per cluster (greedy, deterministic column order —
        # a plain `set` here would pick differently each run due to Python's
        # per-process string hash randomization).
        sub = train_df[cols].fillna(-999)
        corr = sub.corr().abs()
        remaining = sorted(cols, key=lambda c: int(c[1:]))
        dropped: set[str] = set()
        for col in remaining:
            if col in dropped:
                continue
            selected.append(col)
            correlated = corr.index[(corr[col] >= V_CORR_THRESHOLD) & (corr.index != col)]
            dropped.update(correlated.tolist())
    return sorted(selected, key=lambda c: int(c[1:]))


class FeatureEngineer:
    def __init__(self) -> None:
        self.freq_maps_: dict[str, pd.Series] = {}
        self.user_agg_: pd.DataFrame | None = None
        self.user_agg_fallback_: dict[str, float] = {}
        self.selected_v_cols_: list[str] = []
        self.fitted_ = False

    def fit(self, train_df: pd.DataFrame) -> "FeatureEngineer":
        for col in FREQ_ENCODE_COLS:
            if col in train_df.columns:
                self.freq_maps_[col] = train_df[col].value_counts(normalize=True, dropna=False)

        user_key = self._user_key(train_df)
        tmp = train_df.assign(_user=user_key, _amt=train_df["TransactionAmt"])
        self.user_agg_ = tmp.groupby("_user")["_amt"].agg(
            user_txn_count="count", user_avg_amount="mean",
        )
        self.user_agg_fallback_ = {
            "user_txn_count": float(self.user_agg_["user_txn_count"].median()),
            "user_avg_amount": float(train_df["TransactionAmt"].mean()),
        }

        v_cols = [c for c in train_df.columns if c.startswith("V")]
        self.selected_v_cols_ = _select_v_columns(train_df, v_cols)

        self.fitted_ = True
        return self

    @staticmethod
    def _user_key(df: pd.DataFrame) -> pd.Series:
        return df["card1"].astype(str) + "_" + df["addr1"].fillna(-1).astype(str)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FeatureEngineer.fit() must be called before transform()")

        parts = [_time_features(df), _amount_features(df), _m_col_features(df)]

        freq_df = pd.DataFrame(index=df.index)
        for col, freq_map in self.freq_maps_.items():
            freq_df[f"{col}_freq"] = df[col].map(freq_map).fillna(0.0)
        parts.append(freq_df)

        user_key = self._user_key(df)
        user_df = user_key.to_frame("_user").join(self.user_agg_, on="_user")
        user_df["user_txn_count"] = user_df["user_txn_count"].fillna(
            self.user_agg_fallback_["user_txn_count"]
        )
        user_df["user_avg_amount"] = user_df["user_avg_amount"].fillna(
            self.user_agg_fallback_["user_avg_amount"]
        )
        parts.append(user_df[["user_txn_count", "user_avg_amount"]])

        parts.append(df[self.selected_v_cols_])
        parts.append(df[[c for c in BASE_NUMERIC_COLS if c in df.columns]])

        return pd.concat(parts, axis=1)

    def expected_raw_columns(self) -> list[str]:
        """Raw input columns this (fitted) instance needs from `transform`'s
        input DataFrame. Used by the API to reindex a single incoming
        transaction into the right shape before featurizing it."""
        if not self.fitted_:
            raise RuntimeError("FeatureEngineer.fit() must be called before expected_raw_columns()")
        cols = {"TransactionDT", *M_COLS, *self.freq_maps_.keys(), *BASE_NUMERIC_COLS, *self.selected_v_cols_}
        return sorted(cols)


if __name__ == "__main__":
    from src.data.load import load_labeled_data
    from src.data.split import time_based_split

    df = load_labeled_data()
    train, val, test = time_based_split(df)

    fe = FeatureEngineer().fit(train)
    X_train = fe.transform(train)

    print(f"V columns: {len([c for c in df.columns if c.startswith('V')])} -> {len(fe.selected_v_cols_)}")
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Any NaNs left: {X_train.isna().any().any()}")
