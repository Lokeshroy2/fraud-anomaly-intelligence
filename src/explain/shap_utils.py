"""SHAP-based explainability: global feature importance + per-transaction
'why flagged' reasons. `top_reasons` is also used by the API (src/api/main.py)
to explain a single scored transaction in the response.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"


def load_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def _binary_shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame):
    """Return a (n_samples, n_features) array of SHAP values for the
    positive (fraud) class, regardless of shap's output shape for this
    model/version (list-of-arrays vs single array vs 3D array)."""
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        return raw[1]
    if raw.ndim == 3:
        return raw[:, :, 1]
    return raw


def global_importance_plot(explainer: shap.TreeExplainer, X_sample: pd.DataFrame, save_path: Path) -> None:
    values = _binary_shap_values(explainer, X_sample)
    shap.summary_plot(values, X_sample, plot_type="bar", max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def top_reasons(explainer: shap.TreeExplainer, row: pd.DataFrame, n: int = 3) -> list[dict]:
    """Top-n features driving this single transaction's fraud score, by |SHAP value|."""
    values = _binary_shap_values(explainer, row)[0]
    contributions = pd.Series(values, index=row.columns).sort_values(key=abs, ascending=False)
    reasons = []
    for feature in contributions.index[:n]:
        raw_value = row.iloc[0][feature]
        reasons.append({
            "feature": feature,
            "value": None if pd.isna(raw_value) else float(raw_value),
            "shap_value": float(contributions[feature]),
            "direction": "increases" if contributions[feature] > 0 else "decreases",
        })
    return reasons


def main() -> None:
    from src.data.load import load_labeled_data
    from src.data.split import time_based_split

    fe = joblib.load(MODELS_DIR / "feature_engineer.joblib")
    model = joblib.load(MODELS_DIR / "lightgbm_best.joblib")

    df = load_labeled_data()
    _, _, test_df = time_based_split(df)
    X_test = fe.transform(test_df)

    explainer = load_explainer(model)

    sample = X_test.sample(n=min(3000, len(X_test)), random_state=42)
    REPORTS_DIR.mkdir(exist_ok=True)
    global_importance_plot(explainer, sample, REPORTS_DIR / "shap_global_importance.png")
    print(f"Saved global importance plot to {REPORTS_DIR / 'shap_global_importance.png'}")

    scores = model.predict_proba(X_test)[:, 1]
    top_flagged_idx = scores.argsort()[::-1][:3]

    print("\n=== 'Why flagged' for the 3 highest-scored test transactions ===")
    for i in top_flagged_idx:
        row = X_test.iloc[[i]]
        reasons = top_reasons(explainer, row, n=3)
        print(f"\nTransaction (score={scores[i]:.4f}):")
        for r in reasons:
            value_str = f"{r['value']:.3g}" if r["value"] is not None else "NaN"
            print(f"  - {r['feature']}={value_str} {r['direction']} fraud score "
                  f"(shap={r['shap_value']:+.4f})")


if __name__ == "__main__":
    main()
