from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components import (
    BRAND_BLUE,
    BRAND_RED,
    BRAND_TEAL,
    configure_page,
    dataset_summary,
    footer,
    metric_card,
    render_sidebar,
    soft_card,
    apply_plot_theme,
)
from src.utils import churn_rate, load_data


configure_page("Dashboard")
render_sidebar()


@st.cache_data
def cached_data() -> pd.DataFrame:
    return load_data(ROOT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")


df = cached_data()
summary = dataset_summary(df)
churn_customers = int((df["Churn"] == "Yes").sum())
active_customers = int((df["Churn"] == "No").sum())
churn_pct = churn_rate(df)
retention_pct = 100 - churn_pct

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Production AI analytics dashboard</div>
        <h1 style="font-size:3rem;margin:.35rem 0 .6rem 0;">ChurnSense</h1>
        <p class="muted" style="font-size:1.08rem;max-width:780px;">
        A premium customer churn intelligence dashboard for telecom retention teams.
        It turns customer records into risk scores, business explanations, and
        retention actions using a trained XGBoost model.
        </p>
        <span class="badge">Tuned XGBoost</span>
        <span class="badge">ROC-AUC 0.82+</span>
        <span class="badge">7,043 customers</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Executive KPI Dashboard")
cols = st.columns(5)
with cols[0]:
    metric_card("Total Customers", f"{len(df):,}", "All records in the IBM Telco dataset", "DATA")
with cols[1]:
    metric_card("Active Customers", f"{active_customers:,}", "Customers retained in the dataset", "LIVE")
with cols[2]:
    metric_card("Churn Customers", f"{churn_customers:,}", "Customers marked as churned", "RISK")
with cols[3]:
    metric_card("Retention Rate", f"{retention_pct:.1f}%", "Share of customers retained", "OK")
with cols[4]:
    metric_card("Churn Rate", f"{churn_pct:.1f}%", "Business risk baseline", "ALERT")

cols = st.columns(5)
with cols[0]:
    metric_card("Avg Tenure", f"{df['tenure'].mean():.1f} mo", "Customer relationship length", "TIME")
with cols[1]:
    metric_card("Avg Monthly", f"${df['MonthlyCharges'].mean():.2f}", "Average recurring charge", "MRR")
with cols[2]:
    total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
    metric_card("Avg Total", f"${total_charges.mean():.0f}", "Average lifetime billing", "LTV")
with cols[3]:
    metric_card("Best Model", "XGBoost", "Selected deployment model", "ML")
with cols[4]:
    metric_card("ROC-AUC", "0.82+", "Final test discrimination score", "AUC")

st.markdown("### Business Context")
left, mid, right = st.columns(3)
with left:
    soft_card(
        "Business problem",
        "Telecom providers lose revenue when high-risk customers are discovered too late. ChurnSense helps teams identify risk before cancellation.",
        "Problem",
    )
with mid:
    soft_card(
        "AI solution",
        "A trained XGBoost classifier scores each customer and converts model output into risk levels and retention recommendations.",
        "Solution",
    )
with right:
    soft_card(
        "Dataset summary",
        f"{summary['rows']:,} rows, {summary['columns']} columns, {summary['missing_values']} missing values, and {summary['duplicate_rows']} duplicate rows.",
        "Data",
    )

st.markdown("### Project Workflow")
workflow = [
    "Dataset",
    "Preprocessing",
    "Feature Engineering",
    "Model Training",
    "Model Selection",
    "Saved Model",
    "Streamlit",
    "Prediction",
    "Business Decision",
]
st.markdown(
    "<div class='soft-card'><strong>"
    + " -> ".join(workflow)
    + "</strong><p class='muted'>A reproducible path from raw customer records to retention decisions.</p></div>",
    unsafe_allow_html=True,
)

st.markdown("### Quick Business Analytics")
chart_cols = st.columns(2)
with chart_cols[0]:
    churn_counts = df["Churn"].value_counts().rename_axis("Churn").reset_index(name="Customers")
    fig = px.pie(
        churn_counts,
        names="Churn",
        values="Customers",
        hole=0.55,
        color="Churn",
        color_discrete_map={"Yes": BRAND_RED, "No": BRAND_TEAL},
        title="Churn Distribution",
    )
    st.plotly_chart(apply_plot_theme(fig), width="stretch")
with chart_cols[1]:
    contract = (
        df.assign(ChurnFlag=(df["Churn"] == "Yes").astype(int))
        .groupby("Contract", as_index=False)["ChurnFlag"]
        .mean()
    )
    contract["Churn Rate"] = contract["ChurnFlag"] * 100
    fig = px.bar(
        contract,
        x="Contract",
        y="Churn Rate",
        text_auto=".1f",
        color_discrete_sequence=[BRAND_BLUE],
        title="Churn Rate by Contract",
    )
    st.plotly_chart(apply_plot_theme(fig), width="stretch")

st.markdown("### Quick Navigation")
nav_cols = st.columns(4)
with nav_cols[0]:
    soft_card("Predict", "Score a single customer and generate a retention strategy.", "Use now")
with nav_cols[1]:
    soft_card("Batch", "Upload a CSV and prioritize high-risk accounts at scale.", "Workflow")
with nav_cols[2]:
    soft_card("Analytics", "Explore churn patterns, model comparison, and feature signals.", "Explore")
with nav_cols[3]:
    soft_card("Model", "Review the model card, strengths, limitations, and future scope.", "Inspect")

footer()
