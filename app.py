"""
CineVision AI - Main Streamlit Application (Home Page).
Run with: streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.helpers import (
    BANNER_PATH,
    LOGO_PATH,
    METRICS_PATH,
    ensure_directories,
    load_custom_css,
    models_exist,
)
import joblib

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CineVision AI – Movie Success Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

ensure_directories()
load_custom_css()


def _load_accuracy() -> float:
    """Load model accuracy from saved metrics."""
    if METRICS_PATH.exists():
        try:
            metrics = joblib.load(METRICS_PATH)
            return metrics.get("accuracy", 0.0)
        except Exception:
            pass
    return 0.0


def main():
    """Render the home page."""
    # Sidebar branding
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("## 🎬 CineVision AI")
        st.markdown("---")
        st.markdown("**Navigate**")
        st.page_link("app.py", label="🏠 Home", icon="🏠")
        st.page_link("pages/Dashboard.py", label="📊 Dashboard", icon="📊")
        st.page_link("pages/Prediction.py", label="🎯 Prediction", icon="🎯")
        st.page_link("pages/Dataset_Explorer.py", label="🔍 Dataset Explorer", icon="🔍")
        st.page_link("pages/Model_Performance.py", label="📈 Model Performance", icon="📈")
        st.page_link("pages/About.py", label="ℹ️ About", icon="ℹ️")

        st.markdown("---")
        if models_exist():
            st.success("✅ Model Ready")
        else:
            st.warning("⚠️ Run train_model.py")

    # Hero banner
    if BANNER_PATH.exists():
        st.image(str(BANNER_PATH), use_container_width=True)

    st.markdown('<p class="hero-title">CineVision AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="hero-subtitle">Predict movie success with machine learning — '
        "before the cameras roll.</p>",
        unsafe_allow_html=True,
    )

    accuracy = _load_accuracy()

    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.1%}" if accuracy else "N/A")
    with col2:
        st.metric("Categories", "5 Classes")
    with col3:
        st.metric("Algorithms", "5 Compared")
    with col4:
        st.metric("Features", "23+")

    st.markdown("---")

    # Description and features
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 🎬 About the Project")
        st.markdown(
            """
            **CineVision AI** is a production-ready machine learning application that forecasts
            the commercial success of movies before release. By analyzing budget, genre, cast
            popularity, marketing spend, release timing, and more, our ensemble model classifies
            films into five success tiers and estimates box office revenue and ROI.

            Built with **Python**, **Scikit-learn**, and **Streamlit**, CineVision delivers
            studio-grade analytics in a beautiful Netflix-inspired interface.
            """
        )

        st.markdown("### ✨ Key Features")
        features = [
            "🎯 **5-Class Prediction** — Flop, Average, Hit, Super Hit, Blockbuster",
            "💰 **Revenue & ROI Estimation** — Data-driven financial projections",
            "📊 **Interactive Analytics** — Gauges, radar charts, and comparison views",
            "🧠 **AI Insights** — Automated recommendations for marketing & release strategy",
            "📄 **PDF Reports** — Professional downloadable analysis documents",
            "🔍 **Dataset Explorer** — Full data exploration with filters and correlations",
        ]
        for feat in features:
            st.markdown(f'<div class="feature-item">{feat}</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown("### 🚀 Quick Start")
        st.code(
            "pip install -r requirements.txt\n"
            "python train_model.py\n"
            "streamlit run app.py",
            language="bash",
        )

        st.markdown("### 📋 Prediction Outputs")
        outputs = [
            ("Category", "Success tier classification"),
            ("Probability", "Confidence-weighted success %"),
            ("Revenue", "Predicted box office gross"),
            ("ROI", "Return on investment %"),
            ("Confidence", "Model certainty score"),
        ]
        for label, desc in outputs:
            st.markdown(
                f'<div class="glass-card"><b>{label}</b><br><small>{desc}</small></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center; color:rgba(255,255,255,0.4);">'
        "CineVision AI © 2026 | Built with ❤️ using Python & Streamlit</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
