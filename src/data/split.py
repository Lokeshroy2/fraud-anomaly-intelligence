"""Chronological train/val/test split on TransactionDT.

A random split would leak future transaction patterns into training and
overstate performance, since fraud patterns drift over time. We sort by
TransactionDT and cut 70/15/15, mirroring how the model would actually be
deployed: trained on the past, evaluated on the future.
"""

import pandas as pd

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remainder (0.15) is test


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    assert train["TransactionDT"].max() <= val["TransactionDT"].min()
    assert val["TransactionDT"].max() <= test["TransactionDT"].min()
    assert len(set(train["TransactionID"]) & set(val["TransactionID"])) == 0
    assert len(set(val["TransactionID"]) & set(test["TransactionID"])) == 0

    return train, val, test


if __name__ == "__main__":
    from src.data.load import load_labeled_data

    df = load_labeled_data()
    train, val, test = time_based_split(df)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        days = (split["TransactionDT"].max() - split["TransactionDT"].min()) / 86400
        print(
            f"{name:5s}: {len(split):>7,} rows | "
            f"fraud rate {split['isFraud'].mean():.4%} | "
            f"span {days:.1f} days"
        )
