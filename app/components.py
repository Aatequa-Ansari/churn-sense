from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
BRAND_BLUE = "#3D8EFF"
BRAND_TEAL = "#00C9A7"
BRAND_RED = "#FF5A7A"
BRAND_AMBER = "#FFD166"
CARD_BG = "#111827"
MUTED = "#94A3B8"


MODEL_RESULTS = pd.DataFrame(
    [
        {"Model": "Logistic Regression", "Accuracy": 0.740241, "Precision": 0.508097, "Recall": 0.671123, "F1": 0.578341, "ROC_AUC": 0.823901, "CV_ROC_AUC": None},
        {"Model": "Decision Tree", "Accuracy": 0.717530, "Precision": 0.473799, "Recall": 0.580214, "F1": 0.521635, "ROC_AUC": 0.673479, "CV_ROC_AUC": None},
        {"Model": "Random Forest", "Accuracy": 0.762952, "Precision": 0.546083, "Recall": 0.633690, "F1": 0.586634, "ROC_AUC": 0.824825, "CV_ROC_AUC": None},
        {"Model": "Gradient Boosting", "Accuracy": 0.757984, "Precision": 0.533605, "Recall": 0.700535, "F1": 0.605780, "ROC_AUC": 0.830511, "CV_ROC_AUC": 0.898307},
        {"Model": "XGBoost", "Accuracy": 0.754436, "Precision": 0.529536, "Recall": 0.671123, "F1": 0.591981, "ROC_AUC": 0.824217, "CV_ROC_AUC": None},
        {"Model": "Tuned XGBoost", "Accuracy": 0.767211, "Precision": 0.551339, "Recall": 0.660428, "F1": 0.600973, "ROC_AUC": 0.820827, "CV_ROC_AUC": 0.917423},
    ]
)

FEATURE_DICTIONARY = [
    ("customerID", "object", "Unique customer identifier", "7590-VHVEG", "Removed before modeling to avoid identifier leakage."),
    ("gender", "object", "Customer gender", "Female", "Useful for segment-level churn monitoring."),
    ("SeniorCitizen", "int", "Whether customer is a senior citizen", "0", "Senior customers showed a different churn pattern."),
    ("Partner", "object", "Whether customer has a partner", "Yes", "Household context can influence retention."),
    ("Dependents", "object", "Whether customer has dependents", "No", "Family dependency may affect plan stability."),
    ("tenure", "int", "Months with the company", "12", "One of the strongest loyalty and churn indicators."),
    ("PhoneService", "object", "Whether phone service is active", "Yes", "Defines telecom product bundle."),
    ("MultipleLines", "object", "Multiple phone lines status", "No", "Captures product depth."),
    ("InternetService", "object", "Internet service type", "Fiber optic", "Fiber optic users showed elevated churn."),
    ("OnlineSecurity", "object", "Online security subscription", "No", "Add-on adoption can increase stickiness."),
    ("OnlineBackup", "object", "Online backup subscription", "Yes", "Add-on adoption can reduce churn risk."),
    ("DeviceProtection", "object", "Device protection subscription", "No", "Measures service bundle maturity."),
    ("TechSupport", "object", "Tech support subscription", "No", "Support availability affects customer confidence."),
    ("StreamingTV", "object", "Streaming TV subscription", "Yes", "Entertainment bundle depth."),
    ("StreamingMovies", "object", "Streaming movies subscription", "Yes", "Entertainment bundle depth."),
    ("Contract", "object", "Contract term", "Month-to-month", "Month-to-month contracts are high churn risk."),
    ("PaperlessBilling", "object", "Paperless billing status", "Yes", "Billing preference can reveal engagement behavior."),
    ("PaymentMethod", "object", "Customer payment method", "Electronic check", "Electronic check users showed high churn."),
    ("MonthlyCharges", "float", "Current monthly amount billed", "70.35", "Higher charges can increase price sensitivity."),
    ("TotalCharges", "float", "Lifetime charges", "1397.48", "Combines tenure and customer value."),
    ("Churn", "object", "Target variable", "Yes", "The label the model predicts."),
]


def configure_page(title: str) -> None:
    st.set_option("client.showSidebarNavigation", False)
    st.set_page_config(page_title=f"ChurnSense | {title}", page_icon="CS", layout="wide")
    inject_css()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #060B14;
            --card: #0F172A;
            --card-2: #101B31;
            --line: rgba(148, 163, 184, 0.24);
            --text: #F8FAFC;
            --muted: #94A3B8;
            --blue: #3D8EFF;
            --teal: #00C9A7;
            --red: #FF5A7A;
            --amber: #FFD166;
            --purple: #8B5CF6;
        }
        .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1320px; }
        h1, h2, h3, h4 { letter-spacing: -0.01em; color: var(--text); font-weight: 700; }
        h1 {
            background: linear-gradient(90deg, #7DD3FC 0%, #3D8EFF 45%, #A78BFA 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        h3 { color: #E2E8F0; margin: 1.35rem 0 0.5rem; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #07111F 0%, #0B1424 100%);
            border-right: 1px solid var(--line);
        }
        .sidebar-top-banner {
            position: sticky;
            top: 0;
            z-index: 1000;
            background: linear-gradient(180deg, rgba(7,17,31,0.98), rgba(11,20,36,0.98));
            padding: 0.55rem 0 0.8rem;
            margin: -0.2rem -0.4rem 0.7rem;
            border-bottom: 1px solid rgba(61,142,255,.24);
            box-shadow: 0 10px 24px rgba(0,0,0,.18);
        }
        .sidebar-heading {
            font-size: 0.95rem;
            font-weight: 800;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            color: #8DDCFD;
            white-space: nowrap;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.2rem 0.25rem;
        }
        .sidebar-heading-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 10px;
            background: linear-gradient(135deg, var(--blue), var(--purple));
            box-shadow: 0 6px 16px rgba(61,142,255,.28);
            font-size: 0.95rem;
        }
        .sidebar-divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(61,142,255,.9), rgba(0,201,167,.2));
            margin-top: 0.5rem;
            margin-bottom: 0.1rem;
        }
        .sidebar-section-spacer {
            height: 0.6rem;
        }
        [data-testid="stSidebarNav"] ul {
            padding-top: 0.2rem;
        }
        [data-testid="stSidebarNav"] li {
            margin: 0.2rem 0;
        }
        [data-testid="stSidebarNav"] a {
            border-radius: 12px;
            padding: 0.72rem 0.8rem;
            transition: all 180ms ease;
            color: #E2E8F0;
            font-weight: 600;
            text-decoration: none;
        }
        [data-testid="stSidebarNav"] a:hover {
            background: rgba(61,142,255,.16);
            color: #FFFFFF;
            transform: translateX(2px);
        }
        [data-testid="stSidebarNav"] a[aria-current="page"] {
            background: linear-gradient(135deg, rgba(61,142,255,.28), rgba(139,92,246,.24));
            color: #FFFFFF;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.14), 0 8px 18px rgba(61,142,255,.16);
            border: 1px solid rgba(125,211,252,.26);
        }
        .hero {
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 34px;
            background:
                radial-gradient(circle at top right, rgba(61,142,255,.24), transparent 34%),
                linear-gradient(135deg, #101B31 0%, #0B1220 55%, #07111F 100%);
            box-shadow: 0 18px 45px rgba(0,0,0,.28);
            margin-bottom: 1.25rem;
        }
        .eyebrow {
            color: var(--teal);
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .12em;
            font-size: .78rem;
        }
        .muted { color: var(--muted); }
        .metric-card, .soft-card {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px;
            background: linear-gradient(180deg, rgba(17,24,39,.98), rgba(15,23,42,.98));
            box-shadow: 0 12px 32px rgba(0,0,0,.22);
            min-height: 126px;
            margin-bottom: 0.85rem;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            width: 100%;
            min-width: 0;
            overflow: visible;
        }
        .metric-card--blue { background: linear-gradient(135deg, rgba(61,142,255,.22), rgba(15,23,42,.98)); }
        .metric-card--teal { background: linear-gradient(135deg, rgba(0,201,167,.22), rgba(15,23,42,.98)); }
        .metric-card--red { background: linear-gradient(135deg, rgba(255,90,122,.24), rgba(15,23,42,.98)); }
        .metric-card--amber { background: linear-gradient(135deg, rgba(255,209,102,.24), rgba(15,23,42,.98)); }
        .metric-card--purple { background: linear-gradient(135deg, rgba(139,92,246,.24), rgba(15,23,42,.98)); }
        .metric-top {
            display: grid;
            grid-template-columns: auto minmax(0, 1fr);
            align-items: center;
            gap: 0.7rem;
            margin-bottom: 10px;
            width: 100%;
        }
        .metric-icon {
            min-width: 44px;
            width: auto;
            height: 36px;
            padding: 0 12px;
            border-radius: 12px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            color: white;
            background: linear-gradient(135deg, var(--blue), var(--purple));
            box-shadow: inset 0 1px 0 rgba(255,255,255,.24);
            white-space: nowrap;
            flex-shrink: 0;
        }
        .metric-label {
            color: var(--muted);
            font-size: .84rem;
            font-weight: 600;
            line-height: 1.2;
            white-space: nowrap;
            overflow: visible;
        }
        .metric-value {
            color: var(--text);
            font-size: clamp(1.2rem, 1.6vw, 1.7rem);
            font-weight: 800;
            line-height: 1.12;
            margin-bottom: 6px;
            white-space: nowrap;
        }
        .metric-note {
            color: var(--muted);
            font-size: .78rem;
            line-height: 1.35;
            white-space: nowrap;
        }
        .soft-card-title {
            color: var(--text);
            font-size: 1.02rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }
        .badge {
            display: inline-block;
            border: 1px solid rgba(61,142,255,.35);
            border-radius: 999px;
            padding: 6px 12px;
            color: #DCEBFF;
            background: rgba(61,142,255,.12);
            font-weight: 700;
            font-size: .78rem;
            margin-bottom: 0.7rem;
            white-space: nowrap;
            width: fit-content;
            max-width: 100%;
        }
        .chart-shell {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 12px 12px 4px;
            background: linear-gradient(180deg, rgba(17,24,39,.96), rgba(15,23,42,.96));
            box-shadow: 0 10px 26px rgba(0,0,0,.18);
            margin-bottom: 0.6rem;
        }
        .chart-title {
            color: var(--text);
            font-weight: 700;
            margin-bottom: 0.3rem;
            padding-left: 2px;
        }
        .footer {
            border-top: 1px solid var(--line);
            margin-top: 34px;
            padding-top: 18px;
            color: var(--muted);
            font-size: .9rem;
        }
        .footer a {
            color: #8DDCFD;
            text-decoration: none;
        }
        .footer a:hover { color: #F8FAFC; }
        div[data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 16px;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 12px;
            border: 1px solid rgba(61,142,255,.45);
            background: linear-gradient(135deg, #2563EB, #00A7A0);
            color: white;
            font-weight: 700;
            box-shadow: 0 10px 24px rgba(37,99,235,.18);
        }
        .stTextInput > div > div > input, .stSelectbox > div > div > div, .stNumberInput > div > div > input, .stTextArea > div > div > textarea {
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.28);
            background: rgba(15,23,42,.88);
        }
        .stDataFrame, .stTable { overflow: auto; }
        .st-emotion-cache-1y4p8pa { padding-top: 0.4rem; }
        /* Hide Streamlit default multipage navigation */
            [data-testid="stSidebarNav"] {
            display: none;
            }

        /* Hide the Pages heading */
        [data-testid="stSidebarNavItems"] {
        display: none;
        }
        </style>
        """,
        
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    st.sidebar.markdown(
        "<div class='sidebar-top-banner'><div class='sidebar-heading'><span class='sidebar-heading-icon'>✨</span><span>Navigation Menu</span></div><div class='sidebar-divider'></div></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.page_link("streamlit_app.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/01_predict.py", label="Predict", icon="🔮")
    st.sidebar.page_link("pages/02_batch.py", label="Batch", icon="📦")
    st.sidebar.page_link("pages/03_analytics.py", label="Analytics", icon="📈")
    st.sidebar.page_link("pages/04_dataset_overview.py", label="Dataset Overview", icon="🗂️")
    st.sidebar.page_link("pages/05_model_information.py", label="Model Information", icon="🧠")
    st.sidebar.page_link("pages/06_Explainability.py",label="Explainability",icon="🔍")
    st.sidebar.markdown("<div class='sidebar-section-spacer'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("## ChurnSense")
    st.sidebar.caption("Production AI analytics dashboard")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Built by**  \nAatequa Ansari")
    st.sidebar.caption("AI / Machine Learning Engineer")


def metric_card(label: str, value: str, note: str, icon: str = "AI") -> None:
    palette = {
        "DATA": "metric-card--blue",
        "LIVE": "metric-card--teal",
        "RISK": "metric-card--red",
        "OK": "metric-card--teal",
        "ALERT": "metric-card--red",
        "TIME": "metric-card--purple",
        "MRR": "metric-card--blue",
        "LTV": "metric-card--teal",
        "ML": "metric-card--purple",
        "AUC": "metric-card--amber",
        "SEG": "metric-card--blue",
        "PLAN": "metric-card--teal",
        "PAY": "metric-card--purple",
        "WEB": "metric-card--amber",
        "MODEL": "metric-card--purple",
        "TARGET": "metric-card--blue",
        "CV": "metric-card--teal",
        "REC": "metric-card--amber",
        "ROWS": "metric-card--blue",
        "COLS": "metric-card--teal",
        "NUM": "metric-card--purple",
        "CAT": "metric-card--amber",
        "DUP": "metric-card--red",
        "CSV": "metric-card--blue",
        "HIGH": "metric-card--red",
        "MED": "metric-card--amber",
        "AVG": "metric-card--teal",
        "OUTCOME": "metric-card--red",
        "PROB": "metric-card--blue",
        "HEALTH": "metric-card--teal",
        "SAVE": "metric-card--purple",
    }
    card_class = palette.get(icon, "metric-card--blue")
    st.markdown(
        f"""
        <div class="metric-card {card_class}">
            <div class="metric-top">
                <div class="metric-icon">{icon}</div>
                <div class="metric-label">{label}</div>
            </div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def soft_card(title: str, body: str, badge: str | None = None) -> None:
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="soft-card">
            {badge_html}
            <div class="soft-card-title">{title}</div>
            <p class="muted">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown(
        """
        <div class="footer">
        Built with Python, Streamlit, Scikit-learn, XGBoost, Pandas, Plotly, NumPy, and Joblib.<br>
        Built by <strong>Aatequa Ansari</strong> | AI / Machine Learning Engineer |
        <a href="https://github.com/Aatequa-Ansari" target="_blank" rel="noopener noreferrer">GitHub</a> |
        <a href="https://www.linkedin.com/in/aatequa-ansari-873639326" target="_blank" rel="noopener noreferrer">LinkedIn</a> |
        <a href="https://share.streamlit.io/user/aatequa-ansari" target="_blank" rel="noopener noreferrer">Streamlit</a> |
        <a href="https://public.tableau.com/app/profile/aatequa.ansari/vizzes" target="_blank" rel="noopener noreferrer">Tableau Public</a>
        </div>
        """,

        unsafe_allow_html=True,
    )


def apply_plot_theme(fig: go.Figure, title: str | None = None) -> go.Figure:
    if title:
        fig.update_layout(
            title=dict(text=title, x=0.05, xanchor="left", font=dict(size=18, color="#F8FAFC")),
        )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB", family="Segoe UI, Inter, sans-serif"),
        title_font_color="#F8FAFC",
        margin=dict(l=20, r=20, t=60, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(size=12)),
        hoverlabel=dict(bgcolor="#0F172A", font_size=12, font_color="#F8FAFC"),
    )
    for trace in fig.data:
        if hasattr(trace, "marker") and getattr(trace.marker, "line", None) is not None:
            trace.marker.line.color = "rgba(255,255,255,0.14)"
            trace.marker.line.width = 1
    return fig


def chart_card(title: str, fig: go.Figure) -> None:
    st.markdown(f"<div class='chart-shell'><div class='chart-title'>{title}</div></div>", unsafe_allow_html=True)
    st.plotly_chart(apply_plot_theme(fig, title), width="stretch")


def dataset_summary(df: pd.DataFrame) -> dict[str, int | float]:
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numerical_features": int(len(df.select_dtypes(include="number").columns)),
        "categorical_features": int(len(df.select_dtypes(exclude="number").columns)),
    }


def model_comparison_chart(metric: str) -> go.Figure:
    color = [BRAND_TEAL if model == "Tuned XGBoost" else BRAND_BLUE for model in MODEL_RESULTS["Model"]]
    fig = px.bar(MODEL_RESULTS, x="Model", y=metric, text_auto=".3f", color_discrete_sequence=[BRAND_BLUE])
    fig.update_traces(marker_color=color)
    fig.update_yaxes(range=[0, 1])
    return fig
