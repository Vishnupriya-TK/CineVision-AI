"""
CineVision AI - Helper utilities for paths, styling, and common operations.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "dataset" / "movies.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "movie_model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METRICS_PATH = MODEL_DIR / "metrics.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.pkl"
ASSETS_DIR = PROJECT_ROOT / "assets"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGO_PATH = ASSETS_DIR / "logo.png"
BANNER_PATH = ASSETS_DIR / "banner.jpg"


def ensure_directories() -> None:
    """Create required project directories if they do not exist."""
    for directory in [MODEL_DIR, ASSETS_DIR, REPORTS_DIR, PROJECT_ROOT / "dataset"]:
        directory.mkdir(parents=True, exist_ok=True)


def models_exist() -> bool:
    """Return True if all trained model artifacts are present."""
    return all(
        path.exists()
        for path in [MODEL_PATH, ENCODER_PATH, SCALER_PATH, METRICS_PATH]
    )


def format_currency(value: float, currency: str = "USD") -> str:
    """Format a numeric value as currency string."""
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a float as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def category_color(category: str) -> str:
    """Return hex color for a movie success category."""
    colors = {
        "Flop": "#e50914",
        "Average": "#f5a623",
        "Hit": "#46d369",
        "Super Hit": "#00b4d8",
        "Blockbuster": "#ffd700",
    }
    return colors.get(category, "#ffffff")


def category_emoji(category: str) -> str:
    """Return emoji representing movie success category."""
    emojis = {
        "Flop": "📉",
        "Average": "📊",
        "Hit": "🎯",
        "Super Hit": "⭐",
        "Blockbuster": "🏆",
    }
    return emojis.get(category, "🎬")


def load_custom_css() -> None:
    """Inject Netflix-inspired dark theme with glassmorphism styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
            background-attachment: fixed;
        }

        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1200px;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d0d0d 0%, #1a1a2e 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        [data-testid="stSidebar"] .stMarkdown h1,
        [data-testid="stSidebar"] .stMarkdown h2,
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #e50914 !important;
        }

        /* Glass cards */
        .glass-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .glass-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(229, 9, 20, 0.15);
            border-color: rgba(229, 9, 20, 0.3);
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(229,9,20,0.15) 0%, rgba(255,255,255,0.05) 100%);
            border: 1px solid rgba(229, 9, 20, 0.25);
            border-radius: 16px;
            padding: 1.25rem;
            text-align: center;
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: scale(1.02);
            box-shadow: 0 8px 32px rgba(229, 9, 20, 0.2);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 800;
            color: #ffffff;
            margin: 0.5rem 0;
        }

        .metric-label {
            font-size: 0.85rem;
            color: rgba(255,255,255,0.6);
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(90deg, #e50914, #ff6b6b, #ffd700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }

        .hero-subtitle {
            font-size: 1.2rem;
            color: rgba(255,255,255,0.7);
            margin-bottom: 2rem;
        }

        .feature-item {
            background: rgba(255,255,255,0.03);
            border-left: 3px solid #e50914;
            padding: 0.75rem 1rem;
            margin: 0.5rem 0;
            border-radius: 0 8px 8px 0;
            color: rgba(255,255,255,0.85);
        }

        .insight-box {
            background: rgba(0, 180, 216, 0.1);
            border: 1px solid rgba(0, 180, 216, 0.3);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            color: rgba(255,255,255,0.9);
        }

        .stButton > button {
            background: linear-gradient(90deg, #e50914, #b20710);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.6rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
        }

        .stButton > button:hover {
            background: linear-gradient(90deg, #ff1a1a, #e50914);
            box-shadow: 0 6px 20px rgba(229, 9, 20, 0.4);
            transform: translateY(-2px);
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 1rem;
        }

        div[data-testid="stMetric"] label {
            color: rgba(255,255,255,0.6) !important;
        }

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #ffffff !important;
        }

        h1, h2, h3 {
            color: #ffffff !important;
        }

        p, li, span {
            color: rgba(255,255,255,0.85);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            color: rgba(255,255,255,0.7);
        }

        .stTabs [aria-selected="true"] {
            background: rgba(229, 9, 20, 0.3) !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, emoji: str = "") -> None:
    """Render a styled metric card in Streamlit."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{emoji} {label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_glass_card(content: str) -> None:
    """Render content inside a glassmorphism card."""
    st.markdown(
        f'<div class="glass-card">{content}</div>',
        unsafe_allow_html=True,
    )


def get_session_prediction() -> Optional[Dict[str, Any]]:
    """Retrieve the latest prediction from session state."""
    return st.session_state.get("last_prediction")


def set_session_prediction(prediction: Dict[str, Any]) -> None:
    """Store prediction result in session state."""
    st.session_state["last_prediction"] = prediction
