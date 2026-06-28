"""
CineVision AI - AI-powered insight generation for movie predictions.
"""

from typing import Any, Dict, List

import pandas as pd

from utils.preprocessing import CATEGORIES


def _season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def generate_insights(
    input_data: Dict[str, Any],
    prediction: Dict[str, Any],
    dataset: pd.DataFrame,
) -> List[str]:
    """
    Generate intelligent, context-aware recommendations based on
    input features, prediction results, and historical dataset statistics.
    """
    insights: List[str] = []

    budget = float(input_data.get("budget", 0))
    marketing = float(input_data.get("marketing_budget", 0))
    runtime = int(input_data.get("runtime", 120))
    genre = input_data.get("genre", "Drama")
    season = _season_name(int(input_data.get("release_month", 6)))
    lead_pop = float(input_data.get("lead_actor_popularity", 50))
    screens = int(input_data.get("num_screens", 2000))
    franchise = input_data.get("franchise", "No")
    ott = input_data.get("ott_release", "No")
    competition = input_data.get("competition_level", "Medium")
    category = prediction.get("category", "Average")
    roi = prediction.get("roi", 0)
    confidence = prediction.get("confidence_score", 0)

    # Dataset benchmarks
    avg_marketing_ratio = (dataset["marketing_budget"] / dataset["budget"]).median()
    avg_screens = dataset["num_screens"].median()
    avg_lead_pop = dataset["lead_actor_popularity"].median()
    avg_runtime = dataset["runtime"].median()

    marketing_ratio = marketing / max(budget, 1)
    if marketing_ratio < avg_marketing_ratio * 0.8:
        insights.append(
            f"📢 Marketing budget is below average ({marketing_ratio:.1%} vs "
            f"industry median {avg_marketing_ratio:.1%}). Consider increasing spend by "
            f"{(avg_marketing_ratio - marketing_ratio) * budget / 1e6:.1f}M to improve visibility."
        )
    elif marketing_ratio > avg_marketing_ratio * 1.3:
        insights.append(
            "💰 Marketing spend is above average — budget allocation appears aggressive. "
            "Monitor ROI closely to ensure returns justify the investment."
        )
    else:
        insights.append("✅ Marketing budget allocation is within optimal range for this budget tier.")

    # Genre-season performance
    season_genre = dataset[
        (dataset["release_season"] == season) & (dataset["genre"] == genre)
    ]
    if len(season_genre) > 10:
        hit_rate = (season_genre["category"].isin(["Hit", "Super Hit", "Blockbuster"]).mean())
        if hit_rate > 0.4:
            insights.append(
                f"🎬 {genre} genre performs well in {season} season "
                f"({hit_rate:.0%} success rate historically). Favorable release timing."
            )
        else:
            insights.append(
                f"⚠️ {genre} has a lower success rate in {season} ({hit_rate:.0%}). "
                "Consider shifting release window or intensifying marketing."
            )

    # Screen count
    if screens < avg_screens * 0.7:
        insights.append(
            f"🎥 Screen count ({screens:,}) is below median ({avg_screens:,.0f}). "
            "Increasing distribution may significantly improve box office revenue."
        )
    elif screens > avg_screens * 1.3:
        insights.append(
            f"📺 Wide release strategy with {screens:,} screens — strong distribution "
            " footprint supports higher revenue potential."
        )

    # Lead actor
    if lead_pop > avg_lead_pop * 1.2:
        insights.append(
            f"⭐ Lead actor popularity ({lead_pop:.0f}/100) is above average — "
            "star power positively impacts prediction confidence."
        )
    elif lead_pop < avg_lead_pop * 0.8:
        insights.append(
            "👤 Lead actor popularity is below average. Pair with strong marketing "
            "or supporting cast to compensate."
        )

    # Runtime
    if 90 <= runtime <= 130:
        insights.append(
            f"⏱️ Runtime ({runtime} min) falls within the successful movie sweet spot "
            f"(industry median: {avg_runtime:.0f} min)."
        )
    elif runtime > 150:
        insights.append(
            f"⏱️ Long runtime ({runtime} min) may affect show frequency. "
            "Ensure content quality justifies extended duration."
        )

    # Franchise
    if franchise == "Yes":
        insights.append(
            "🏛️ Franchise IP provides built-in audience — historically associated "
            "with 30-40% higher revenue multiples."
        )

    # OTT
    if ott == "Yes":
        insights.append(
            "📱 Simultaneous OTT release detected — may reduce theatrical window "
            "but expand total reach. Hybrid strategy can work for mid-budget films."
        )

    # Competition
    if competition == "High":
        insights.append(
            "⚔️ High competition level at release — differentiate with unique marketing "
            "angles and pre-release buzz campaigns."
        )
    elif competition == "Low":
        insights.append(
            "🎯 Low competition window — excellent opportunity to maximize opening weekend."
        )

    # Category-specific
    if category in ("Super Hit", "Blockbuster"):
        insights.append(
            f"🏆 Model predicts '{category}' — all key indicators align favorably. "
            "Maintain current production and marketing strategy."
        )
    elif category == "Flop":
        insights.append(
            "📉 Predicted category suggests risk. Review budget, cast selection, "
            "and release timing for potential improvements before greenlight."
        )

    # ROI insight
    if roi > 200:
        insights.append(f"💵 Exceptional ROI projection ({roi:.0f}%) — strong investor appeal.")
    elif roi < 0:
        insights.append(
            f"📊 Negative ROI projected ({roi:.0f}%). Re-evaluate budget or "
            "revenue expectations before proceeding."
        )

    # Confidence
    if confidence >= 0.85:
        insights.append(
            f"🎯 High model confidence ({confidence:.0%}) — prediction is well-supported by training patterns."
        )
    elif confidence < 0.6:
        insights.append(
            f"🔍 Moderate confidence ({confidence:.0%}) — input combination is less common in training data. "
            "Treat prediction as directional guidance."
        )

    return insights[:10]  # Cap at 10 insights


def get_industry_benchmarks(dataset: pd.DataFrame) -> Dict[str, float]:
    """Compute industry average benchmarks for comparison charts."""
    return {
        "Budget (M)": dataset["budget"].median() / 1e6,
        "Marketing (M)": dataset["marketing_budget"].median() / 1e6,
        "Lead Actor Pop.": dataset["lead_actor_popularity"].median(),
        "Screens": dataset["num_screens"].median(),
        "Runtime": dataset["runtime"].median(),
        "ROI %": dataset["roi"].median(),
    }
