from __future__ import annotations

import shap
import pandas as pd

from .preprocessing import prepare_features


def load_explainer(model):
    """
    Create a SHAP TreeExplainer for the trained XGBoost model.
    """
    return shap.TreeExplainer(model)


def explain_customer(
    explainer,
    customer: dict,
):
    """
    Generate SHAP values for a single customer.
    """
    features = prepare_features(
        pd.DataFrame([customer])
    )

    shap_values = explainer(features)

    return features, shap_values