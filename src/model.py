from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

try:
    from .preprocessing import load_feature_columns, prepare_features
except ImportError:  # Allows running this file directly during quick checks.
    from preprocessing import load_feature_columns, prepare_features


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "churn_prediction_model.pkl"
DEFAULT_FEATURE_COLUMNS_PATH = ROOT_DIR / "models" / "feature_columns.pkl"


def load_model(path: str | Path = DEFAULT_MODEL_PATH):
    """Load the trained churn prediction model."""
    return joblib.load(path)


def predict_batch(
    model,
    customers: pd.DataFrame,
    threshold: float = 0.5,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Predict churn probabilities for a batch of raw or encoded customers."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    columns = feature_columns or load_feature_columns(DEFAULT_FEATURE_COLUMNS_PATH)
    features = prepare_features(customers, columns)
    probabilities = model.predict_proba(features)[:, 1]

    predictions = customers.copy()
    predictions["churn_probability"] = probabilities.round(4)
    predictions["will_churn"] = probabilities >= threshold
    predictions["risk_level"] = pd.cut(
        probabilities,
        bins=[-0.01, 0.35, 0.65, 1.0],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return predictions


def predict_single(
    model,
    customer: dict,
    threshold: float = 0.5,
    feature_columns: list[str] | None = None,
) -> dict:
    """Predict churn for one customer dictionary."""
    result = predict_batch(
        model=model,
        customers=pd.DataFrame([customer]),
        threshold=threshold,
        feature_columns=feature_columns,
    ).iloc[0]
    return {
        "churn_probability": float(result["churn_probability"]),
        "will_churn": bool(result["will_churn"]),
        "risk_level": result["risk_level"],
    }
