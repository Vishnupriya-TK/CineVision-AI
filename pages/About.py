"""
CineVision AI - About Page.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import load_custom_css

st.set_page_config(page_title="About | CineVision AI", page_icon="ℹ️", layout="wide")
load_custom_css()

st.title("ℹ️ About CineVision AI")

st.markdown(
    """
    ## 🎬 CineVision AI – Movie Success Predictor

    CineVision AI is a production-ready machine learning application that predicts the
    commercial success of movies based on production, marketing, and release features.
    It empowers filmmakers, producers, and analysts with data-driven insights before
    a film hits the box office.
    """
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 What It Predicts")
    st.markdown(
        """
        | Output | Description |
        |--------|-------------|
        | **Category** | Flop, Average, Hit, Super Hit, Blockbuster |
        | **Success Probability** | Model confidence in the predicted class |
        | **Revenue** | Estimated box office gross |
        | **ROI** | Return on investment percentage |
        | **Confidence Score** | Overall prediction certainty |
        """
    )

    st.markdown("### 🛠️ Tech Stack")
    st.markdown(
        """
        - **Python 3.10+** — Core language
        - **Streamlit** — Multi-page web application
        - **Scikit-learn** — Machine learning models
        - **XGBoost** — Gradient boosting (optional)
        - **Pandas & NumPy** — Data processing
        - **Plotly** — Interactive visualizations
        - **ReportLab** — PDF report generation
        - **Joblib** — Model serialization
        """
    )

with col2:
    st.markdown("### 🧠 ML Pipeline")
    st.markdown(
        """
        1. **Data Collection** — TMDB dataset or synthetic generation
        2. **Preprocessing** — Missing values, duplicates, encoding
        3. **Feature Engineering** — ROI, budget tiers, seasons, genre popularity
        4. **Model Training** — 5 algorithms compared via cross-validation
        5. **Selection** — Best model saved based on F1 score
        6. **Inference** — Real-time prediction with revenue estimation
        7. **Insights** — Rule-based AI recommendations
        """
    )

    st.markdown("### 📁 Project Structure")
    st.code(
        """
CineVision-AI/
├── app.py                  # Home page
├── train_model.py          # Training pipeline
├── predict.py              # Prediction engine
├── dataset/movies.csv      # Training data
├── model/                  # Saved artifacts
├── pages/                  # Streamlit pages
├── utils/                  # Utility modules
├── assets/                 # Logo & banner
└── reports/                # Generated PDFs
        """,
        language="text",
    )

st.markdown("---")

st.markdown("### 👥 Use Cases")
use_cases = [
    ("🎥 Pre-Production", "Evaluate greenlight decisions before committing budget"),
    ("📊 Marketing", "Optimize marketing spend based on predicted ROI"),
    ("📅 Release Planning", "Choose optimal release windows and screen counts"),
    ("💼 Investment", "Provide data-backed projections to investors"),
    ("🔬 Research", "Analyze trends in genre, season, and cast popularity"),
]
for title, desc in use_cases:
    st.markdown(f"**{title}** — {desc}")

st.markdown("---")
st.markdown(
    """
    ### 📜 License & Credits

    Built as an educational and demonstration project showcasing end-to-end ML application
    development with Python and Streamlit.

    **Dataset Sources:** TMDB 5000 Movies (GitHub mirror) with synthetic feature augmentation,
    or fully synthetic dataset for reproducibility.

    ---
    *CineVision AI © 2026*
    """
)
