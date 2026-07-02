from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import streamlit as st
import shap
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components import (render_sidebar,configure_page,footer,soft_card,)
from src.model import load_model
from src.preprocessing import prepare_features

configure_page("Explainability")
render_sidebar()

st.title("Model Explainability")
st.write(
    "Understand which features influence churn predictions across the entire model."
)

@st.cache_resource
def cached_model():
    return load_model()

model = cached_model()

data = pd.read_csv("data/final_telco_churn.csv")
X = prepare_features(data)


@st.cache_resource
def cached_explainer():
    return shap.TreeExplainer(
        model,
        feature_perturbation="tree_path_dependent"
    )
explainer = cached_explainer()
shap_values = explainer(X)
## st.write("SHAP Values Shape:", shap_values.values.shape)

st.markdown("## Global Feature Importance")

fig, ax = plt.subplots(figsize=(12, 8))

shap.summary_plot(
    shap_values,
    X,
    show=False
)

st.pyplot(fig)
plt.close(fig)


st.markdown("## Top 15 Most Important Features")

fig, ax = plt.subplots(figsize=(10, 7))

shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)

st.pyplot(fig)
plt.close(fig)




st.markdown("## Top Feature Insights")

soft_card(
    "Business Insights",

    """

    • MonthlyCharges has the highest impact on churn prediction.<br><br>
    • Customers with shorter tenure are more likely to churn.<br><br>
    • Fiber Optic internet users have a higher churn probability.<br><br>
    • Customers with Two-Year Contracts are much less likely to churn.<br><br>
    • Electronic Check payment is associated with higher churn.
    """,
    "SHAP"
)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance Score": np.mean(np.abs(np.asarray(shap_values.values)), axis=0)
})

importance = importance.sort_values(
    "Importance Score",
    ascending=False
)

st.dataframe(
    importance.head(15),
    use_container_width=True,
    hide_index=True
)