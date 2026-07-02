from __future__ import annotations

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import streamlit as st
import shap
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components import configure_page, footer, metric_card, render_sidebar, soft_card
from src.model import load_model, predict_single
from src.recommendations import build_recommendations, score_customer_health
from src.explainability import load_explainer, explain_customer

configure_page("Predict")
render_sidebar()


@st.cache_resource
def cached_model():
    return load_model()


@st.cache_resource
def cached_explainer():
    return load_explainer(cached_model())


st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Single customer intelligence</div>
        <h1 style="margin:.35rem 0;">Predict Customer Churn</h1>
        <p class="muted">Score a customer, understand the risk level, and convert the prediction into retention actions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("predict_form"):
    st.markdown("### Customer Profile")
    profile_1, profile_2, profile_3 = st.columns(3)

    with profile_1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with profile_2:
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    with profile_3:
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
        total = st.number_input("Total Charges", min_value=0.0, value=840.0, step=10.0)
        threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)

    st.markdown("### Product Add-ons")
    add_1, add_2, add_3 = st.columns(3)
    with add_1:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    with add_2:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    with add_3:
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    submitted = st.form_submit_button("Generate Churn Intelligence")

if submitted:
    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }

    prediction = predict_single(
    cached_model(),
    customer,
    threshold=threshold,
    )

    features, shap_values = explain_customer(
    cached_explainer(),
    customer,
    )

    probability = prediction["churn_probability"]
    scores = score_customer_health(probability)
    recommendations = build_recommendations(customer, probability)

    st.markdown("### Prediction Result")
    result_cols = st.columns(5)
    with result_cols[0]:
        metric_card("Prediction", "Will Churn" if prediction["will_churn"] else "Retain", "Decision based on selected threshold", "OUTCOME")
    with result_cols[1]:
        metric_card("Probability", f"{probability:.1%}", "Estimated churn likelihood", "PROB")
    with result_cols[2]:
        metric_card("Risk Level", prediction["risk_level"], "Low, medium, or high risk", "RISK")
    with result_cols[3]:
        metric_card("Health Score", f"{scores['health_score']}/100", "Higher is healthier", "HEALTH")
    with result_cols[4]:
        metric_card(
        "Retention Score",
        f"{scores['retention_score']}/100",
        "Estimated save opportunity",
        "SAVE",
    )

    st.markdown("### Churn Probability Gauge")

    gauge = go.Figure(

    go.Indicator(
        mode="gauge+number",    
        value=probability * 100,
        number={"suffix": "%"},
        title={"text": "Predicted Churn Probability"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#4f8dfd"},
            "steps": [
                {"range": [0, 30], "color": "#16351f"},
                {"range": [30, 70], "color": "#4d430d"},
                {"range": [70, 100], "color": "#4d1212"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 4},
                "value": threshold * 100,
            },
        },
    )
    )

    gauge.update_layout(
    height=330,
    margin=dict(l=20, r=20, t=50, b=10),
    )

    st.plotly_chart(
    gauge,
    use_container_width=True,
    config={"displayModeBar": False},
    )


    st.markdown("### Recommendation Engine")
    rec_cols = st.columns(3)
    with rec_cols[0]:
        soft_card("Suggested Strategy", recommendations["strategy"], "Retention playbook")
    with rec_cols[1]:
        soft_card("Confidence Score", f"{scores['confidence_score']}/100 confidence based on model probability distance from the decision boundary.", "Model signal")
    with rec_cols[2]:
        soft_card("Next Action", recommendations["actions"][0], "Priority")

    risk_col, positive_col = st.columns(2)
    with risk_col:
        st.markdown("#### Risk Factors")
        for item in recommendations["risk_factors"]:
            st.markdown(f"- {item}")
    with positive_col:
        st.markdown("#### Positive Factors")
        for item in recommendations["positive_factors"]:
            st.markdown(f"- {item}")

    st.markdown("#### Actionable Recommendations")
    for item in recommendations["actions"]:
        st.markdown(f"- {item}")



    tab1, tab2 = st.tabs(
    ["📊 Model Explanation", "📋 Customer Summary"]
    )   

    with tab1:
        st.markdown("### Why did the model make this prediction?")

        plt.figure(figsize=(10, 6))

        shap.plots.waterfall(
        shap_values[0],
        show=False
        )

        fig = plt.gcf()
        st.pyplot(fig)
        plt.close(fig)

    with tab2:
        st.markdown("### Customer Summary")
        st.json(customer)
footer()
