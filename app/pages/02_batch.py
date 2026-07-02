from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components import BRAND_BLUE, BRAND_RED, BRAND_TEAL, apply_plot_theme, configure_page, footer, metric_card, render_sidebar
from src.model import load_model, predict_batch


configure_page("Batch Prediction")
render_sidebar()


@st.cache_resource
def cached_model():
    return load_model()


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Portfolio scoring workflow</div>
        <h1 style="margin:.35rem 0;">Batch Churn Prediction</h1>
        <p class="muted">Upload customer records, score churn risk at scale, and export prioritized retention lists.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_col, control_col = st.columns([2, 1])
with upload_col:
    uploaded_file = st.file_uploader("Upload customer CSV", type=["csv"])
with control_col:
    threshold = st.slider("Churn threshold", 0.1, 0.9, 0.5, 0.05)

if uploaded_file:
    customers = pd.read_csv(uploaded_file)
else:
    sample_path = ROOT_DIR / "data" / "sample_new_customers.csv"
    customers = pd.read_csv(sample_path) if sample_path.exists() else pd.DataFrame()
    st.info("Previewing the bundled sample_new_customers.csv file. Upload your own CSV to replace it.")

if not customers.empty:
    predictions = predict_batch(cached_model(), customers, threshold=threshold)
    high_risk = int((predictions["risk_level"] == "High").sum())
    medium_risk = int((predictions["risk_level"] == "Medium").sum())
    avg_probability = float(predictions["churn_probability"].mean())
    expected_churners = int(predictions["will_churn"].sum())

    st.markdown("### Batch Summary")
    cols = st.columns(5)
    with cols[0]:
        metric_card("Rows Scored", f"{len(predictions):,}", "Customers processed", "CSV")
    with cols[1]:
        metric_card("Expected Churners", f"{expected_churners:,}", "Above selected threshold", "ALERT")
    with cols[2]:
        metric_card("High Risk", f"{high_risk:,}", "Probability above 65%", "HIGH")
    with cols[3]:
        metric_card("Medium Risk", f"{medium_risk:,}", "Probability from 35% to 65%", "MED")
    with cols[4]:
        metric_card("Avg Probability", f"{avg_probability:.1%}", "Portfolio risk average", "AVG")

    chart_col_1, chart_col_2 = st.columns(2)
    with chart_col_1:
        risk_counts = predictions["risk_level"].value_counts().rename_axis("Risk Level").reset_index(name="Customers")
        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Customers",
            color="Risk Level",
            color_discrete_map={"High": BRAND_RED, "Medium": BRAND_BLUE, "Low": BRAND_TEAL},
            title="Risk Distribution",
            text_auto=True,
        )
        st.plotly_chart(apply_plot_theme(fig), width="stretch")
    with chart_col_2:
        fig = px.histogram(
            predictions,
            x="churn_probability",
            nbins=20,
            color_discrete_sequence=[BRAND_BLUE],
            title="Churn Probability Histogram",
        )
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(apply_plot_theme(fig), width="stretch")

    st.markdown("### Interactive Prediction Results")
    st.dataframe(
        predictions.sort_values("churn_probability", ascending=False),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download Predictions CSV",
        predictions.to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv",
    )

footer()
