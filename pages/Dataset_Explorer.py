"""
CineVision AI - Dataset Explorer Page.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.charts import (
    create_category_distribution,
    create_correlation_heatmap,
    create_distribution_chart,
)
from utils.helpers import DATASET_PATH, format_currency, load_custom_css
from utils.preprocessing import (
    CATEGORIES,
    CATEGORICAL_COLS,
    GENRES,
    LANGUAGES,
    NUMERIC_COLS,
    OUTCOME_COLS,
    load_or_create_dataset,
)

st.set_page_config(page_title="Dataset Explorer | CineVision AI", page_icon="🔍", layout="wide")
load_custom_css()

st.title("🔍 Dataset Explorer")
st.markdown("Explore, filter, and analyze the movie training dataset.")

# Load dataset
df = load_or_create_dataset()

# Summary metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Records", f"{len(df):,}")
c2.metric("Features", len(df.columns))
c3.metric("Missing Values", int(df.isnull().sum().sum()))
c4.metric("Duplicates Removed", "✓")
c5.metric("Avg Budget", format_currency(df["budget"].mean()))

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Preview", "🔎 Search & Filter", "📊 Statistics", "📈 Visualizations"])

with tab1:
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(100), use_container_width=True, height=400)

    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Full Dataset (CSV)",
        data=csv_data,
        file_name="movies.csv",
        mime="text/csv",
        use_container_width=True,
    )

with tab2:
    st.markdown("### Search & Filter")

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        search = st.text_input("🔍 Search by Title", "")
        genre_filter = st.multiselect("Genre", GENRES, default=[])
    with fcol2:
        category_filter = st.multiselect("Category", CATEGORIES, default=[])
        language_filter = st.multiselect("Language", LANGUAGES, default=[])
    with fcol3:
        budget_range = st.slider(
            "Budget Range ($M)",
            float(df["budget"].min() / 1e6),
            float(df["budget"].max() / 1e6),
            (float(df["budget"].min() / 1e6), float(df["budget"].max() / 1e6)),
        )

    filtered = df.copy()
    if search:
        filtered = filtered[filtered["title"].str.contains(search, case=False, na=False)]
    if genre_filter:
        filtered = filtered[filtered["genre"].isin(genre_filter)]
    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]
    if language_filter:
        filtered = filtered[filtered["language"].isin(language_filter)]
    filtered = filtered[
        (filtered["budget"] >= budget_range[0] * 1e6)
        & (filtered["budget"] <= budget_range[1] * 1e6)
    ]

    st.markdown(f"**Showing {len(filtered):,} of {len(df):,} records**")
    st.dataframe(filtered, use_container_width=True, height=400)

with tab3:
    st.markdown("### Descriptive Statistics")
    numeric_display = [c for c in NUMERIC_COLS + OUTCOME_COLS if c in df.columns]
    st.dataframe(df[numeric_display].describe().T, use_container_width=True)

    st.markdown("### Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        st.success("✅ No missing values in the dataset.")
    else:
        st.dataframe(missing.reset_index().rename(columns={"index": "Column", 0: "Missing Count"}))

    st.markdown("### Categorical Value Counts")
    cat_col = st.selectbox("Select categorical column", CATEGORICAL_COLS + ["category"])
    if cat_col in df.columns:
        st.dataframe(df[cat_col].value_counts().reset_index(), use_container_width=True)

with tab4:
    st.markdown("### Data Visualizations")

    vcol1, vcol2 = st.columns(2)
    with vcol1:
        st.plotly_chart(create_category_distribution(df), use_container_width=True)
        dist_col = st.selectbox("Distribution column", numeric_display, index=0)
        st.plotly_chart(create_distribution_chart(df, dist_col), use_container_width=True)
    with vcol2:
        corr_cols = st.multiselect(
            "Correlation columns",
            numeric_display,
            default=numeric_display[:8],
        )
        if len(corr_cols) >= 2:
            st.plotly_chart(create_correlation_heatmap(df, corr_cols), use_container_width=True)
