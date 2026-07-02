from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components import BRAND_BLUE, BRAND_RED, BRAND_TEAL, MODEL_RESULTS, apply_plot_theme, chart_card, configure_page, footer, metric_card, render_sidebar, soft_card


configure_page("Model")
render_sidebar()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Model card and production notes</div>
        <h1 style="margin:.35rem 0;">Model Information</h1>
        <p class="muted">A business and technical view of the selected churn prediction model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(4)
with cols[0]:
    metric_card("Architecture", "XGBoost", "Gradient boosted decision trees", "MODEL")
with cols[1]:
    metric_card("Target", "Churn", "Binary classification", "TARGET")
with cols[2]:
    metric_card("Test ROC-AUC", "0.821", "Final tuned model score", "AUC")
with cols[3]:
    metric_card("CV ROC-AUC", "0.917", "5-fold validation mean", "CV")

st.markdown("### Business Problem and Target")
business_col, data_col = st.columns(2)
with business_col:
    soft_card(
        "Business problem",
        "Predict churn risk early enough for retention teams to intervene with plan reviews, loyalty offers, service fixes, or proactive support.",
        "Problem",
    )
with data_col:
    soft_card(
        "Dataset and target",
        "The model is trained on the IBM Telco Customer Churn dataset. The target variable is Churn, encoded as retained or churned.",
        "Dataset",
    )

st.markdown("### Training Summary")
training = pd.DataFrame(
    [
        {"Step": "Data cleaning", "Summary": "Converted TotalCharges, handled blanks, removed customerID."},
        {"Step": "Feature engineering", "Summary": "Created spend, high-value, long-term, and tenure group features."},
        {"Step": "Encoding", "Summary": "Mapped binary fields and one-hot encoded multi-class categories."},
        {"Step": "Model training", "Summary": "Compared Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, and Tuned XGBoost."},
        {"Step": "Selection", "Summary": "Selected Tuned XGBoost for the best overall deployment profile and cross-validation score."},
    ]
)
st.dataframe(training, width="stretch", hide_index=True)

st.markdown("### Evaluation Metrics")
metrics = MODEL_RESULTS[MODEL_RESULTS["Model"] == "Tuned XGBoost"].melt(
    id_vars=["Model"], value_vars=["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "CV_ROC_AUC"], var_name="Metric", value_name="Score"
)
chart_card("Tuned XGBoost evaluation", px.bar(metrics, x="Metric", y="Score", text_auto=".3f", color_discrete_sequence=[BRAND_BLUE]))

st.markdown("### Confusion Matrix and ROC Curve")
cm_col, roc_col = st.columns(2)
with cm_col:
    cm = pd.DataFrame([[787, 248], [127, 247]], index=["Actual Retain", "Actual Churn"], columns=["Predicted Retain", "Predicted Churn"])
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Confusion Matrix - Tuned XGBoost")
    st.plotly_chart(apply_plot_theme(fig), width="stretch")
with roc_col:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 0.08, 0.18, 0.36, 1], y=[0, 0.42, 0.66, 0.82, 1], mode="lines", name="Tuned XGBoost", line=dict(color=BRAND_TEAL, width=4)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(color=BRAND_RED, dash="dash")))
    fig.update_layout(title="Illustrative ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(apply_plot_theme(fig), width="stretch")

model = joblib.load(ROOT_DIR / "models" / "churn_prediction_model.pkl")
feature_columns = joblib.load(ROOT_DIR / "models" / "feature_columns.pkl")
importance = pd.DataFrame({"Feature": feature_columns, "Importance": model.feature_importances_}).sort_values("Importance", ascending=False).head(15)
chart_card("Feature importance", px.bar(importance, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[BRAND_TEAL]))

st.markdown("### Strengths, Limitations, and Future Improvements")
s1, s2, s3 = st.columns(3)
with s1:
    soft_card(
        "Strengths",
        "Strong tabular performance, handles non-linear relationships, and produces feature importance for business explanation.",
        "Strengths",
    )
with s2:
    soft_card(
        "Limitations",
        "The model uses historical Telco data and does not include current support tickets, customer sentiment, or real-time usage behavior.",
        "Limitations",
    )
with s3:
    soft_card(
        "Future improvements",
        "Add CRM integration, automated retraining, SHAP explanations, cost-sensitive thresholds, and campaign outcome tracking.",
        "Roadmap",
    )

footer()
