from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model import load_model, predict_batch, predict_single
from src.preprocessing import load_feature_columns, prepare_features


SAMPLE_CUSTOMER = {
    "customerID": "TEST-001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 6,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.0,
    "TotalCharges": 570.0,
}


def test_model_loads():
    model = load_model()
    assert hasattr(model, "predict_proba")


def test_prepare_features_matches_training_columns():
    columns = load_feature_columns()
    features = prepare_features(pd.DataFrame([SAMPLE_CUSTOMER]), columns)
    assert list(features.columns) == columns
    assert features.shape == (1, len(columns))


def test_single_prediction_has_probability_and_risk_level():
    prediction = predict_single(load_model(), SAMPLE_CUSTOMER)
    assert 0.0 <= prediction["churn_probability"] <= 1.0
    assert prediction["risk_level"] in {"Low", "Medium", "High"}
    assert isinstance(prediction["will_churn"], bool)


def test_batch_prediction_returns_one_row_per_customer():
    customers = pd.DataFrame([SAMPLE_CUSTOMER, {**SAMPLE_CUSTOMER, "customerID": "TEST-002"}])
    predictions = predict_batch(load_model(), customers)
    assert len(predictions) == 2
    assert {"churn_probability", "will_churn", "risk_level"}.issubset(predictions.columns)


def test_different_customer_inputs_change_prediction_output():
    model = load_model()
    base_customer = dict(SAMPLE_CUSTOMER)
    changed_customer = {**base_customer, "Contract": "One year", "PaymentMethod": "Mailed check", "InternetService": "No"}

    base_features = prepare_features(pd.DataFrame([base_customer]), load_feature_columns())
    changed_features = prepare_features(pd.DataFrame([changed_customer]), load_feature_columns())

    assert base_features["Contract_One year"].iloc[0] != changed_features["Contract_One year"].iloc[0]
    assert base_features["PaymentMethod_Mailed check"].iloc[0] != changed_features["PaymentMethod_Mailed check"].iloc[0]

    base_prediction = predict_single(model, base_customer)
    changed_prediction = predict_single(model, changed_customer)
    assert base_prediction["churn_probability"] != changed_prediction["churn_probability"]
