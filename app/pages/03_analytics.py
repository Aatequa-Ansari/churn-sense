from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components import (
    BRAND_BLUE,
    BRAND_RED,
    BRAND_TEAL,
    MODEL_RESULTS,
    apply_plot_theme,
    chart_card,
    configure_page,
    footer,
    metric_card,
    model_comparison_chart,
    render_sidebar,
    soft_card,
)
from src.utils import load_data


configure_page("Analytics")
render_sidebar()


@st.cache_data
def cached_raw_data() -> pd.DataFrame:
    return load_data(ROOT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")


@st.cache_data
def cached_final_data() -> pd.DataFrame:
    return pd.read_csv(ROOT_DIR / "data" / "final_telco_churn.csv")


df = cached_raw_data()
final_df = cached_final_data()

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Exploratory and model analytics</div>
        <h1 style="margin:.35rem 0;">Analytics Command Center</h1>
        <p class="muted">Explore churn patterns, customer segments, model comparison, and feature importance in one place.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Distribution Overview")
overview_cols = st.columns(4)
with overview_cols[0]:
    metric_card("Gender Groups", f"{df['gender'].nunique()}", "Female and male customer segments", "SEG")
with overview_cols[1]:
    metric_card("Contract Types", f"{df['Contract'].nunique()}", "Month-to-month, one-year, two-year", "PLAN")
with overview_cols[2]:
    metric_card("Payment Methods", f"{df['PaymentMethod'].nunique()}", "Billing behavior segments", "PAY")
with overview_cols[3]:
    metric_card("Internet Services", f"{df['InternetService'].nunique()}", "DSL, fiber optic, or none", "WEB")

tab_dist, tab_churn, tab_model = st.tabs(["Customer Distributions", "Churn Drivers", "Model Comparison"])

with tab_dist:
    c1, c2 = st.columns(2)
    with c1:
        chart_card("Churn distribution", px.pie(df, names="Churn", hole=0.55, color="Churn", color_discrete_map={"Yes": BRAND_RED, "No": BRAND_TEAL}))
        chart_card("Gender distribution", px.histogram(df, x="gender", color="gender", color_discrete_sequence=[BRAND_BLUE, BRAND_TEAL]))
        chart_card("Internet service distribution", px.histogram(df, x="InternetService", color="InternetService"))
        chart_card("Monthly charges", px.histogram(df, x="MonthlyCharges", nbins=35, color_discrete_sequence=[BRAND_BLUE]))
    with c2:
        chart_card("Contract distribution", px.histogram(df, x="Contract", color="Contract"))
        fig = px.histogram(df,x="PaymentMethod",color="PaymentMethod")

        fig.update_layout(showlegend=False)

        chart_card(
            "Payment method distribution",
                fig
            )
        total = df.assign(TotalChargesNumeric=pd.to_numeric(df["TotalCharges"], errors="coerce"))
        chart_card("Total charges", px.histogram(total, x="TotalChargesNumeric", nbins=35, color_discrete_sequence=[BRAND_TEAL]))
        chart_card("Tenure distribution", px.histogram(df, x="tenure", nbins=30, color_discrete_sequence=[BRAND_BLUE]))

with tab_churn:
    def churn_rate_by(column: str) -> pd.DataFrame:
        result = (
            df.assign(ChurnFlag=(df["Churn"] == "Yes").astype(int))
            .groupby(column, as_index=False)["ChurnFlag"]
            .mean()
        )
        result["Churn Rate"] = result["ChurnFlag"] * 100
        return result

    c1, c2 = st.columns(2)
    with c1:
        chart_card("Churn by gender", px.bar(churn_rate_by("gender"), x="gender", y="Churn Rate", text_auto=".1f", color_discrete_sequence=[BRAND_BLUE]))
        chart_card("Churn by contract", px.bar(churn_rate_by("Contract"), x="Contract", y="Churn Rate", text_auto=".1f", color_discrete_sequence=[BRAND_RED]))
        chart_card("Churn by payment method", px.bar(churn_rate_by("PaymentMethod"), x="PaymentMethod", y="Churn Rate", text_auto=".1f"))
    with c2:
        chart_card("Churn by internet service", px.bar(churn_rate_by("InternetService"), x="InternetService", y="Churn Rate", text_auto=".1f"))
        chart_card("Churn by senior citizen", px.bar(churn_rate_by("SeniorCitizen"), x="SeniorCitizen", y="Churn Rate", text_auto=".1f", color_discrete_sequence=[BRAND_BLUE]))
        tenure_bins = df.assign(TenureGroup=pd.cut(df["tenure"], bins=[-1, 12, 24, 48, 72], labels=["New", "Growing", "Established", "Loyal"]))
        result = (
            tenure_bins.assign(ChurnFlag=(tenure_bins["Churn"] == "Yes").astype(int))
            .groupby("TenureGroup", observed=False, as_index=False)["ChurnFlag"]
            .mean()
        )
        result["Churn Rate"] = result["ChurnFlag"] * 100
        chart_card("Churn by tenure group", px.bar(result, x="TenureGroup", y="Churn Rate", text_auto=".1f", color_discrete_sequence=[BRAND_TEAL]))

    corr = final_df.select_dtypes(include="number").corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, title="Correlation Heatmap")
    st.plotly_chart(apply_plot_theme(fig), width="stretch")

with tab_model:
    st.markdown("### All Trained Models")
    styled_results = MODEL_RESULTS.copy()
    styled_results["Selected"] = styled_results["Model"].eq("Tuned XGBoost")
    st.dataframe(styled_results, width="stretch", hide_index=True)

    cols = st.columns(3)
    with cols[0]:
        metric_card("Selected Model", "Tuned XGBoost", "Best deployment candidate", "ML")
    with cols[1]:
        metric_card("Best CV ROC-AUC", "0.917", "Highest cross-validation score", "CV")
    with cols[2]:
        metric_card("Best Recall", "0.701", "Gradient Boosting caught most churners", "REC")

    soft_card(
        "Why XGBoost was selected",
        "The tuned XGBoost model delivered the strongest overall deployment profile: highest accuracy, strongest precision, strong recall balance, and the best cross-validation ROC-AUC among validated finalists.",
        "Model rationale",
    )

    for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "CV_ROC_AUC"]:
        chart_card(f"{metric} comparison", model_comparison_chart(metric))

    model = joblib.load(ROOT_DIR / "models" / "churn_prediction_model.pkl")
    feature_columns = joblib.load(ROOT_DIR / "models" / "feature_columns.pkl")
    importance = pd.DataFrame(
        {"Feature": feature_columns, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False).head(15)
    chart_card(
        "Top feature importance",
        px.bar(importance, x="Importance", y="Feature", orientation="h", color_discrete_sequence=[BRAND_TEAL]),
    )

footer()
