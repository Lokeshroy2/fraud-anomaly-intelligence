"""Drift monitoring: compare train-period feature/score distributions
against later weeks using PSI and the Kolmogorov-Smirnov test.

Reference period = train split. Comparison periods = weekly buckets
spanning val + test + the unlabeled Kaggle test set (genuine future data,
~30 days after the labeled period ends - see src/data/load.py). No labels
are needed for drift detection, which is exactly why the unlabeled set is
useful here instead of being wasted.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data.load import load_labeled_data, load_unlabeled_data
from src.data.split import time_based_split

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
PSI_ALERT_THRESHOLD = 0.2
N_TOP_FEATURES = 8
WEEK_SECONDS = 7 * 86400


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. Bin edges are quantiles of `expected`.

    Rule of thumb: <0.1 no real shift, 0.1-0.2 moderate, >0.2 significant
    (PSI_ALERT_THRESHOLD below).
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    expected_pct = np.histogram(expected, bins=edges)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=edges)[0] / len(actual)

    eps = 1e-6
    expected_pct = np.clip(expected_pct, eps, None)
    actual_pct = np.clip(actual_pct, eps, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def top_features_by_importance(model, feature_names, n=N_TOP_FEATURES) -> list[str]:
    importances = pd.Series(model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False).head(n).index.tolist()


def weekly_drift_report(X_train, score_train, monitoring_df, fe, model, top_features) -> pd.DataFrame:
    weeks = ((monitoring_df["TransactionDT"] - monitoring_df["TransactionDT"].min()) // WEEK_SECONDS).astype(int)
    rows = []
    for week, idx in monitoring_df.groupby(weeks).groups.items():
        week_df = monitoring_df.loc[idx]
        if len(week_df) < 30:
            continue
        X_week = fe.transform(week_df)
        score_week = model.predict_proba(X_week)[:, 1]

        row = {
            "week": int(week),
            "n_transactions": len(week_df),
            "score_psi": psi(score_train, score_week),
            "score_ks_stat": ks_2samp(score_train, score_week).statistic,
        }
        for feature in top_features:
            row[f"{feature}_psi"] = psi(X_train[feature].values, X_week[feature].values)
        rows.append(row)

    report = pd.DataFrame(rows)
    feature_psi_cols = [c for c in report.columns if c.endswith("_psi") and c != "score_psi"]
    report["any_feature_alert"] = (report[feature_psi_cols] > PSI_ALERT_THRESHOLD).any(axis=1)
    report["score_alert"] = report["score_psi"] > PSI_ALERT_THRESHOLD
    return report


def render_html_report(report: pd.DataFrame, chart_path: Path, out_path: Path) -> None:
    n_alert_weeks = int(report["score_alert"].sum() + report["any_feature_alert"].sum())
    table_html = report.round(4).to_html(index=False, classes="drift-table")
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Fraud Model Drift Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Arial, sans-serif; margin: 2rem; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; }}
  .summary {{ background: #f5f5f5; padding: 1rem; border-radius: 6px; margin-bottom: 1.5rem; }}
  table.drift-table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
  table.drift-table th, table.drift-table td {{ border: 1px solid #ddd; padding: 4px 8px; text-align: right; }}
  table.drift-table th {{ background: #eee; }}
  img {{ max-width: 100%; }}
</style>
</head>
<body>
  <h1>Fraud Model Drift Report</h1>
  <div class="summary">
    Reference period: train split. Comparison: weekly buckets over val + test +
    the unlabeled future period. PSI alert threshold: {PSI_ALERT_THRESHOLD}.
    <br><b>{n_alert_weeks} alert condition(s) flagged</b> across {len(report)} weeks.
  </div>
  <img src="{chart_path.name}" alt="weekly score PSI">
  {table_html}
</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    import matplotlib.pyplot as plt

    fe = joblib.load(MODELS_DIR / "feature_engineer.joblib")
    model = joblib.load(MODELS_DIR / "lightgbm_best.joblib")

    df = load_labeled_data()
    train_df, val_df, test_df = time_based_split(df)
    unlabeled_df = load_unlabeled_data()

    X_train = fe.transform(train_df)
    score_train = model.predict_proba(X_train)[:, 1]

    monitoring_df = pd.concat([val_df, test_df, unlabeled_df], ignore_index=True)
    top_features = top_features_by_importance(model, X_train.columns)

    report = weekly_drift_report(X_train, score_train, monitoring_df, fe, model, top_features)

    print("=== Weekly drift report ===")
    print(report.round(4).to_string(index=False))
    print(f"\nTop features monitored: {top_features}")

    REPORTS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(report["week"], report["score_psi"], marker="o", color="tab:red")
    ax.axhline(PSI_ALERT_THRESHOLD, color="gray", linestyle="--", label=f"alert threshold ({PSI_ALERT_THRESHOLD})")
    ax.set_xlabel("Week (0 = start of val period)")
    ax.set_ylabel("Model score PSI vs train")
    ax.set_title("Weekly model score drift")
    ax.legend()
    plt.tight_layout()
    chart_path = REPORTS_DIR / "drift_weekly_psi.png"
    plt.savefig(chart_path, dpi=120)
    plt.close()

    render_html_report(report, chart_path, REPORTS_DIR / "drift_report.html")
    print(f"\nSaved {chart_path} and {REPORTS_DIR / 'drift_report.html'}")


if __name__ == "__main__":
    main()
