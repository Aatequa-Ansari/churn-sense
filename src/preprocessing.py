from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]

DEFAULT_FEATURE_COLUMNS_PATH = ROOT_DIR / "models" / "feature_columns.pkl"
DEFAULT_SCALER_PATH = ROOT_DIR / "models" / "scaler.pkl"

BINARY_MAP = {"Yes": 1, "No": 0}
GENDER_MAP = {"Female": 0, "Male": 1}

MULTICLASS_FEATURES = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

BINARY_FEATURES = [
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
]


def load_feature_columns(path: str | Path = DEFAULT_FEATURE_COLUMNS_PATH) -> list[str]:
    """Load the trained model's expected feature order."""
    return list(joblib.load(path))


def load_scaler(path: str | Path = DEFAULT_SCALER_PATH):
    """Load the fitted StandardScaler used during model training."""
    return joblib.load(path)

def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric values and fill blanks with the median."""
    cleaned = df.copy()
    if "TotalCharges" not in cleaned.columns:
        raise ValueError("Input data must contain a TotalCharges column.")

    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    median_value = cleaned["TotalCharges"].median()
    if pd.isna(median_value):
        median_value = 0.0
    cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(median_value)
    return cleaned


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the feature engineering columns used during model training."""
    engineered = df.copy()
    required = {"TotalCharges", "tenure", "MonthlyCharges"}
    missing = required.difference(engineered.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {sorted(missing)}")

    engineered["AverageMonthlySpend"] = engineered["TotalCharges"] / (
        engineered["tenure"] + 1
    )
    engineered["HighValueCustomer"] = (
        engineered["MonthlyCharges"] > engineered["MonthlyCharges"].median()
    ).astype(int)
    engineered["LongTermCustomer"] = (engineered["tenure"] >= 24).astype(int)
    engineered["TenureGroup"] = pd.cut(
        engineered["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Growing", "Established", "Loyal"],
    )
    return engineered


def encode_raw_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook's binary and one-hot encoding logic to raw customer rows."""
    encoded = df.copy()
    encoded = encoded.drop(columns=["customerID"], errors="ignore")
    encoded = clean_total_charges(encoded)

    if "gender" in encoded.columns:
        encoded["gender"] = encoded["gender"].map(GENDER_MAP)

    for column in BINARY_FEATURES:
        if column in encoded.columns:
            encoded[column] = encoded[column].map(BINARY_MAP)

    if "Churn" in encoded.columns:
        encoded["Churn"] = encoded["Churn"].map(BINARY_MAP).fillna(encoded["Churn"])

    encoded = add_engineered_features(encoded)
    categorical_columns = [
        column for column in MULTICLASS_FEATURES if column in encoded.columns
    ]
    encoded = pd.get_dummies(encoded, columns=categorical_columns, drop_first=False, dtype=int)
    encoded = pd.get_dummies(encoded, columns=["TenureGroup"], drop_first=False, dtype=int)
    return encoded


def align_to_feature_columns(
    df: pd.DataFrame, feature_columns: Iterable[str] | None = None
) -> pd.DataFrame:
    """Return numeric model features in the exact order used for training."""
    columns = list(feature_columns) if feature_columns is not None else load_feature_columns()
    aligned = df.copy()
    for column in columns:
        if column not in aligned.columns:
            aligned[column] = 0
    aligned = aligned[columns]
    return aligned.astype(float)


def prepare_features(
    customers: pd.DataFrame, feature_columns: Iterable[str] | None = None
) -> pd.DataFrame:
    """Prepare raw or already-encoded customer data for model prediction."""
    columns = list(feature_columns) if feature_columns is not None else load_feature_columns()
    if set(columns).issubset(customers.columns):
        return align_to_feature_columns(customers, columns)
    encoded = encode_raw_customers(customers)
    aligned = align_to_feature_columns(encoded, columns)


    scaler = load_scaler()

    continuous_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AverageMonthlySpend",
]

    aligned[continuous_features] = scaler.transform(
        aligned[continuous_features]
)

    return aligned