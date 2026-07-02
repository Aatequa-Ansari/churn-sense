from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.components import FEATURE_DICTIONARY, configure_page, dataset_summary, footer, metric_card, render_sidebar, soft_card
from src.utils import load_data


configure_page("Dataset")
render_sidebar()


@st.cache_data
def cached_data() -> pd.DataFrame:
    return load_data(ROOT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")


df = cached_data()
summary = dataset_summary(df)

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Data foundation</div>
        <h1 style="margin:.35rem 0;">Dataset Overview</h1>
        <p class="muted">A clean business reference for the IBM Telco Customer Churn dataset used by ChurnSense.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(6)
with cols[0]:
    metric_card("Rows", f"{summary['rows']:,}", "Customer records", "ROWS")
with cols[1]:
    metric_card("Columns", f"{summary['columns']}", "Raw features plus target", "COLS")
with cols[2]:
    metric_card("Target", "Churn", "Yes or No", "TARGET")
with cols[3]:
    metric_card("Numerical", f"{summary['numerical_features']}", "Numeric raw fields", "NUM")
with cols[4]:
    metric_card("Categorical", f"{summary['categorical_features']}", "Category fields", "CAT")
with cols[5]:
    metric_card("Duplicates", f"{summary['duplicate_rows']}", "Duplicate rows found", "DUP")


cols2 = st.columns(6)

with cols2[0]:
    metric_card("Missing",f"{summary['missing_values']}","Missing values","MISS",)

with cols2[1]:
    metric_card("Classes","Yes / No","Binary target","CLS",)

with cols2[2]:
    churn_rate = (df["Churn"] == "Yes").mean() * 100
    metric_card("Churn Rate",f"{churn_rate:.1f}%","Positive class","RATE",)

with cols2[3]:
    metric_card("Avg Tenure",f"{df['tenure'].mean():.1f}","Months","TEN",)

with cols2[4]:
    metric_card("Avg Monthly",f"${df['MonthlyCharges'].mean():.1f}","Monthly charge","MON",)

with cols2[5]:
    total_avg = pd.to_numeric(df["TotalCharges"],errors="coerce"
).mean()

    metric_card(
        "Avg Total",
        f"${total_avg:.0f}",
        "Lifetime charges",
        "TOTAL",
    )


st.markdown("## Dataset Snapshot")

c1, c2, c3 = st.columns(3)

with c1:
    soft_card(
        "Dataset",
        "IBM Telco Customer Churn",
        "Dataset name",
    )

with c2:
    soft_card(
        "Industry",
        "Telecommunications",
        "Business domain",
    )

with c3:
    soft_card(
        "Prediction",
        "Binary Classification",
        "ML task",
    )

c4, c5, c6 = st.columns(3)

with c4:
    soft_card(
        "Target Column",
        "Churn",
        "Prediction target",
    )

with c5:
    soft_card(
        "File Format",
        "CSV",
        "Source file",
    )

with c6:
    soft_card(
        "Dataset Size",
        f"{summary['rows']:,} × {summary['columns']}",
        "Rows × Columns",
    )



st.markdown("### Dataset Composition")

left, right = st.columns(2)
with left:

    churn_fig = px.pie(
        df,
        names="Churn",
        hole=0.6,
        color="Churn",
    )

    st.plotly_chart(churn_fig, use_container_width=True)
    gender_fig = px.histogram(
        df,
        x="gender",
        color="gender"
    )

    st.plotly_chart(gender_fig, use_container_width=True)

with right:

    contract_fig = px.histogram(
        df,
        x="Contract",
        color="Contract"
    )

    st.plotly_chart(contract_fig, use_container_width=True)

    internet_fig = px.pie(
        df,
        names="InternetService",
        hole=0.6
    )

    st.plotly_chart(internet_fig, use_container_width=True)

st.markdown("### Data Quality")
quality_cols = st.columns(4)
with quality_cols[0]:
    soft_card(
        "Missing Values",
        f"{summary['missing_values']}",
        "Dataset is complete",
    )
with quality_cols[1]:
    soft_card(
        "Duplicate Rows",
        f"{summary['duplicate_rows']}",
        "No duplicate records",
    )
with quality_cols[2]:
    soft_card(
        "Target Variable",
        "Churn",
        "Binary Classification",
    )
with quality_cols[3]:
    soft_card(
        "Identifier",
        "customerID",
        "Removed before training",
    )

st.markdown("### Dataset Description")
left, right = st.columns(2)
with left:
    soft_card(
        "Source",
        "IBM Telco Customer Churn dataset. It contains customer profile, account, billing, service subscription, and churn label information.",
        "Dataset source",
    )
with right:
    soft_card(
        "Modeling purpose",
        "The dataset supports binary classification: estimate whether a telecom customer is likely to churn and convert that risk into retention action.",
        "Business use",
    )

st.markdown("### Feature Dictionary")

feature_table = pd.DataFrame(
    FEATURE_DICTIONARY,
    columns=[
        "Feature name",
        "Data type",
        "Business meaning",
        "Example value",
        "Why important",
    ],
)

search = st.text_input(
    "🔍 Search a feature",
    placeholder="Example: tenure, Contract, MonthlyCharges..."
)

if search:
    feature_table = feature_table[
        feature_table.apply(
            lambda row: row.astype(str)
            .str.lower()
            .str.contains(search.lower())
            .any(),
            axis=1,
        )
    ]

st.caption(f"Showing {len(feature_table)} feature(s)")

st.dataframe(
    feature_table,
    use_container_width=True,
    hide_index=True,
)



st.markdown("#### Dataset Explorer")

f1, f2, f3 = st.columns(3)

with f1:
    dataset_search = st.text_input(
        "Customer ID",
        placeholder="Search customer..."
    )

with f2:
    gender_filter = st.selectbox(
        "Gender",
        ["All"] + sorted(df["gender"].unique().tolist())
    )

with f3:
    churn_filter = st.selectbox(
        "Churn",
        ["All"] + sorted(df["Churn"].unique().tolist())
    )

f4, f5, f6 = st.columns(3)

with f4:
    contract_filter = st.selectbox(
        "Contract",
        ["All"] + sorted(df["Contract"].unique().tolist())
    )

with f5:
    internet_filter = st.selectbox(
        "Internet",
        ["All"] + sorted(df["InternetService"].unique().tolist())
    )

with f6:
    rows = st.selectbox(
        "Rows",
        [10,25,50,100],
        index=1,
    )

preview = df.copy()

if dataset_search:
    preview = preview[
        preview["customerID"]
        .astype(str)
        .str.contains(dataset_search, case=False)
    ]

if gender_filter != "All":
    preview = preview[
        preview["gender"] == gender_filter
    ]

if churn_filter != "All":
    preview = preview[
        preview["Churn"] == churn_filter
    ]

if contract_filter != "All":
    preview = preview[
        preview["Contract"] == contract_filter
    ]

if internet_filter != "All":
    preview = preview[
        preview["InternetService"] == internet_filter
    ]

st.caption(
    f"Showing {min(rows,len(preview))} of {len(preview)} matching customers"
)

st.download_button(
    "📥 Download Dataset",
    data=df.to_csv(index=False),
    file_name="telco_churn_dataset.csv",
    mime="text/csv",
)

st.dataframe(
    preview.head(rows),
    use_container_width=True,
    hide_index=True,
)

footer()
