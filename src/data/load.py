"""Load and merge the IEEE-CIS fraud detection transaction + identity tables."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "ieee-fraud-detection"


def load_labeled_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the labeled training data (train_transaction + train_identity).

    This is the only portion of the Kaggle dataset with `isFraud` labels, so
    all of our train/val/test splits are carved out of it (see src/data/split.py).
    """
    transactions = pd.read_csv(data_dir / "train_transaction.csv")
    identity = pd.read_csv(data_dir / "train_identity.csv")
    return transactions.merge(identity, on="TransactionID", how="left")


def load_unlabeled_data(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load the unlabeled Kaggle test data (test_transaction + test_identity).

    Has no `isFraud` column. Chronologically later than the labeled data, so
    it's used as a real future-period sample for drift monitoring (src/drift/monitor.py)
    rather than for model evaluation.
    """
    transactions = pd.read_csv(data_dir / "test_transaction.csv")
    identity = pd.read_csv(data_dir / "test_identity.csv")
    identity = identity.rename(columns=lambda c: c.replace("-", "_"))
    return transactions.merge(identity, on="TransactionID", how="left")


if __name__ == "__main__":
    df = load_labeled_data()
    print(f"Labeled data: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Fraud rate: {df['isFraud'].mean():.4%}")
