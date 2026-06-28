"""
CineVision AI - Movie Prediction Page.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from predict import get_predictor, predict_movie
from utils.helpers import (
    category_color,
    category_emoji,
    format_currency,
    format_percentage,
    load_custom_css,
    models_exist,
    render_metric_card,
    set_session_prediction,
)
from utils.insights import generate_insights
from utils.pdf_report import generate_pdf_report
from utils.preprocessing import (
    CERTIFICATIONS,
    COMPETITION_LEVELS,
    COUNTRIES,
    GENRES,
    LANGUAGES,
    load_or_create_dataset,
)

st.set_page_config(page_title="Prediction | CineVision AI", page_icon="🎯", layout="wide")
load_custom_css()

st.title("🎯 Movie Success Prediction")
st.markdown("Enter your movie details below to get AI-powered success forecasts.")

if not models_exist():
    st.error(
        "⚠️ **Model not trained yet.** Please run the following command in your terminal:\n\n"
        "```\npython train_model.py\n```"
    )
    st.stop()

# ---------------------------------------------------------------------------
# Prediction Form
# ---------------------------------------------------------------------------
with st.form("prediction_form", clear_on_submit=False):
    st.markdown("### 🎬 Movie Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        title = st.text_input("Movie Title", value="The Last Horizon", placeholder="Enter movie title")
        budget = st.number_input("Movie Budget ($)", min_value=100_000, max_value=500_000_000, value=25_000_000, step=500_000)
        genre = st.selectbox("Genre", GENRES)
        language = st.selectbox("Language", LANGUAGES)
        runtime = st.number_input("Runtime (minutes)", min_value=60, max_value=240, value=128)
        release_month = st.selectbox(
            "Release Month",
            list(range(1, 13)),
            format_func=lambda m: [
                "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
            ][m - 1],
            index=5,
        )

    with col2:
        director_experience = st.number_input("Director Experience (years)", min_value=0, max_value=40, value=12)
        lead_actor_popularity = st.slider("Lead Actor Popularity (0-100)", 0, 100, 72)
        supporting_cast_rating = st.slider("Supporting Cast Rating (0-10)", 0.0, 10.0, 7.5, 0.1)
        marketing_budget = st.number_input("Marketing Budget ($)", min_value=0, max_value=100_000_000, value=8_000_000, step=100_000)
        production_house_rating = st.slider("Production House Rating (0-10)", 0.0, 10.0, 8.0, 0.1)
        competition_level = st.selectbox("Competition Level", COMPETITION_LEVELS)

    with col3:
        num_screens = st.number_input("Number of Screens", min_value=100, max_value=10000, value=2800, step=100)
        franchise = st.selectbox("Franchise / Sequel", ["No", "Yes"])
        ott_release = st.selectbox("OTT Simultaneous Release", ["No", "Yes"])
        country = st.selectbox("Country", COUNTRIES)
        certification_rating = st.selectbox("Certification Rating", CERTIFICATIONS)

    submitted = st.form_submit_button("🚀 Predict Movie Success", use_container_width=True)

# ---------------------------------------------------------------------------
# Run Prediction
# ---------------------------------------------------------------------------
if submitted:
    genre_popularity_map = {
        "Action": 85, "Adventure": 78, "Animation": 72, "Comedy": 80,
        "Crime": 65, "Drama": 70, "Fantasy": 75, "Horror": 68,
        "Mystery": 62, "Romance": 74, "Sci-Fi": 77, "Thriller": 73,
    }
    input_data = {
        "title": title,
        "budget": budget,
        "genre": genre,
        "genre_popularity": genre_popularity_map.get(genre, 70),
        "language": language,
        "runtime": runtime,
        "release_month": release_month,
        "director_experience": director_experience,
        "lead_actor_popularity": lead_actor_popularity,
        "supporting_cast_rating": supporting_cast_rating,
        "marketing_budget": marketing_budget,
        "production_house_rating": production_house_rating,
        "competition_level": competition_level,
        "num_screens": num_screens,
        "franchise": franchise,
        "ott_release": ott_release,
        "country": country,
        "certification_rating": certification_rating,
    }

    try:
        with st.spinner("Analyzing movie features..."):
            prediction = predict_movie(input_data)

        df = load_or_create_dataset()
        insights = generate_insights(input_data, prediction, df)

        # Store in session for Dashboard page
        set_session_prediction({
            "input_data": input_data,
            "prediction": prediction,
            "insights": insights,
        })

        st.success("✅ Prediction complete!")
        st.markdown("---")
        st.markdown("### 📊 Prediction Results")

        cat = prediction["category"]
        color = category_color(cat)
        emoji = category_emoji(cat)

        # Premium result cards
        r1, r2, r3 = st.columns(3)
        with r1:
            render_metric_card("Movie Category", f"{emoji} {cat}")
        with r2:
            render_metric_card("Success Probability", format_percentage(prediction["success_probability"]))
        with r3:
            render_metric_card("Expected Revenue", format_currency(prediction["predicted_revenue"]))

        r4, r5, r6 = st.columns(3)
        with r4:
            render_metric_card("ROI", f"{prediction['roi']:.1f}%")
        with r5:
            render_metric_card("Confidence Score", format_percentage(prediction["confidence_score"]))
        with r6:
            render_metric_card("Prediction Time", f"{prediction['prediction_time_ms']:.1f} ms")

        # Category probability breakdown
        st.markdown("### 📈 Category Probabilities")
        probs = prediction.get("category_probabilities", {})
        prob_cols = st.columns(len(probs))
        for i, (label, prob) in enumerate(sorted(probs.items(), key=lambda x: -x[1])):
            with prob_cols[i % len(prob_cols)]:
                st.progress(prob, text=f"{label}: {prob:.1%}")

        # AI Insights
        st.markdown("### 🧠 AI Insights & Recommendations")
        for insight in insights:
            st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

        # PDF Download
        st.markdown("---")
        pdf_bytes = generate_pdf_report(input_data, prediction, insights)
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_bytes,
            file_name=f"cinevision_report_{title.replace(' ', '_')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
