from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"


def project_path(*parts: str) -> Path:
    """Build an absolute path inside the ChurnSense project."""
    return ROOT_DIR.joinpath(*parts)


def load_data(path: str | Path = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv") -> pd.DataFrame:
    """Load a CSV file as a pandas DataFrame."""
    return pd.read_csv(path)


def churn_rate(df: pd.DataFrame, target_column: str = "Churn") -> float:
    """Return churn rate as a percentage for Yes/No or 1/0 target columns."""
    if target_column not in df.columns:
        raise ValueError(f"{target_column} column was not found.")

    target = df[target_column]
    if target.dtype == object:
        return float((target == "Yes").mean() * 100)
    return float(target.mean() * 100)


def summarize_dataset(df: pd.DataFrame) -> dict[str, float | int]:
    """Return dashboard-ready summary statistics for the churn dataset."""
    return {
        "total_customers": int(len(df)),
        "churn_rate": round(churn_rate(df), 2),
        "avg_monthly_charges": round(float(df["MonthlyCharges"].mean()), 2),
        "avg_tenure": round(float(df["tenure"].mean()), 2),
    }
