"""
CineVision AI - Analytics Dashboard Page.
"""

import sys
from pathlib import Path

import joblib
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.charts import (
    create_category_pie_chart,
    create_comparison_chart,
    create_feature_importance_chart,
    create_gauge_chart,
    create_revenue_bar_chart,
    create_roi_indicator,
    create_radar_chart,
)
from utils.helpers import (
    METRICS_PATH,
    format_currency,
    format_percentage,
    get_session_prediction,
    load_custom_css,
    models_exist,
)
from utils.insights import get_industry_benchmarks
from utils.preprocessing import load_or_create_dataset

st.set_page_config(page_title="Dashboard | CineVision AI", page_icon="📊", layout="wide")
load_custom_css()

st.title("📊 Analytics Dashboard")

prediction = get_session_prediction()

if prediction is None:
    st.info(
        "👋 No prediction data yet. Head to the **Prediction** page, "
        "fill in movie details, and click **Predict** to populate this dashboard."
    )

    # Show dataset overview even without prediction
    st.markdown("### 📁 Dataset Overview")
    df = load_or_create_dataset()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Movies", f"{len(df):,}")
    c2.metric("Avg Budget", format_currency(df["budget"].median()))
    c3.metric("Avg Revenue", format_currency(df["revenue"].median()))
    c4.metric("Avg ROI", f"{df['roi'].median():.0f}%")
    st.stop()

# Extract prediction data
input_data = prediction.get("input_data", {})
pred = prediction.get("prediction", {})
category = pred.get("category", "N/A")
success_prob = pred.get("success_probability", 0)
revenue = pred.get("predicted_revenue", 0)
roi = pred.get("roi", 0)
budget = float(input_data.get("budget", 0))
marketing = float(input_data.get("marketing_budget", 0))
probs = pred.get("category_probabilities", {})

# Top metrics
st.markdown(f"### 🎬 Analysis for: **{input_data.get('title', 'Your Movie')}**")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Category", category)
m2.metric("Success Prob.", format_percentage(success_prob))
m3.metric("Revenue", format_currency(revenue))
m4.metric("ROI", f"{roi:.1f}%")
m5.metric("Confidence", format_percentage(pred.get("confidence_score", 0)))

st.markdown("---")

# Charts row 1
col_a, col_b = st.columns(2)
with col_a:
    st.plotly_chart(create_gauge_chart(success_prob), use_container_width=True)
with col_b:
    st.plotly_chart(
        create_revenue_bar_chart(revenue, budget, marketing),
        use_container_width=True,
    )

# Charts row 2
col_c, col_d = st.columns(2)
with col_c:
    st.plotly_chart(create_roi_indicator(roi), use_container_width=True)
with col_d:
    if probs:
        st.plotly_chart(create_category_pie_chart(probs), use_container_width=True)

# Charts row 3
col_e, col_f = st.columns(2)
with col_e:
    st.plotly_chart(create_radar_chart(input_data), use_container_width=True)
with col_f:
    if METRICS_PATH.exists() and models_exist():
        metrics = joblib.load(METRICS_PATH)
        fi = metrics.get("feature_importance", {})
        if fi:
            names = list(fi.keys())
            values = __import__("numpy").array(list(fi.values()))
            st.plotly_chart(
                create_feature_importance_chart(names, values),
                use_container_width=True,
            )

# Comparison chart
df = load_or_create_dataset()
benchmarks = get_industry_benchmarks(df)
movie_values = {
    "Budget (M)": budget / 1e6,
    "Marketing (M)": marketing / 1e6,
    "Lead Actor Pop.": input_data.get("lead_actor_popularity", 50),
    "Screens": input_data.get("num_screens", 2000),
    "Runtime": input_data.get("runtime", 120),
    "ROI %": roi,
}
st.plotly_chart(
    create_comparison_chart(movie_values, benchmarks),
    use_container_width=True,
)

# AI Insights on dashboard
insights = prediction.get("insights", [])
if insights:
    st.markdown("### 🧠 AI Insights")
    for insight in insights:
        st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
