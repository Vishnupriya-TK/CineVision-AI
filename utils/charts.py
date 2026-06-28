"""
CineVision AI - Plotly and Matplotlib chart utilities.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Dark Netflix-inspired color palette
COLORS = {
    "primary": "#e50914",
    "secondary": "#b20710",
    "accent": "#ffd700",
    "success": "#46d369",
    "info": "#00b4d8",
    "warning": "#f5a623",
    "background": "#0a0a0a",
    "card": "rgba(255,255,255,0.05)",
}

CATEGORY_COLORS = {
    "Flop": "#e50914",
    "Average": "#f5a623",
    "Hit": "#46d369",
    "Super Hit": "#00b4d8",
    "Blockbuster": "#ffd700",
}

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.85)", family="Inter"),
        colorway=["#e50914", "#00b4d8", "#46d369", "#ffd700", "#f5a623"],
    )
)


def _apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent dark theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.85)"),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig


def create_gauge_chart(value: float, title: str = "Success Probability") -> go.Figure:
    """Create a gauge chart for success probability (0-100%)."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=value * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": title, "font": {"size": 18, "color": "white"}},
            number={"suffix": "%", "font": {"size": 36, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": COLORS["primary"]},
                "bgcolor": "rgba(255,255,255,0.1)",
                "bordercolor": "rgba(255,255,255,0.2)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(229,9,20,0.3)"},
                    {"range": [30, 60], "color": "rgba(245,166,35,0.3)"},
                    {"range": [60, 100], "color": "rgba(70,211,105,0.3)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": value * 100,
                },
            },
        )
    )
    return _apply_dark_theme(fig)


def create_revenue_bar_chart(
    predicted_revenue: float,
    budget: float,
    marketing_budget: float,
) -> go.Figure:
    """Bar chart comparing budget, marketing, and predicted revenue."""
    labels = ["Budget", "Marketing", "Predicted Revenue"]
    values = [budget, marketing_budget, predicted_revenue]
    colors = [COLORS["warning"], COLORS["info"], COLORS["success"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors,
                text=[f"${v/1e6:.1f}M" if v >= 1e6 else f"${v/1e3:.0f}K" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(title="Revenue Analysis", yaxis_title="Amount (USD)")
    return _apply_dark_theme(fig)


def create_roi_indicator(roi: float) -> go.Figure:
    """Create ROI bullet chart."""
    color = COLORS["success"] if roi > 0 else COLORS["primary"]
    fig = go.Figure(
        go.Indicator(
            mode="number+delta",
            value=roi,
            title={"text": "Return on Investment (%)", "font": {"color": "white"}},
            number={"suffix": "%", "font": {"size": 40, "color": color}},
            delta={"reference": 100, "relative": False, "font": {"color": "white"}},
        )
    )
    return _apply_dark_theme(fig)


def create_feature_importance_chart(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 12,
) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    indices = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker=dict(
                color=values,
                colorscale=[[0, COLORS["primary"]], [1, COLORS["accent"]]],
            ),
        )
    )
    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Importance Score",
        yaxis=dict(autorange="reversed"),
    )
    return _apply_dark_theme(fig)


def create_radar_chart(input_data: Dict[str, Any]) -> go.Figure:
    """Radar chart comparing input feature values against normalized benchmarks."""
    categories = [
        "Lead Actor Pop.",
        "Cast Rating",
        "Director Exp.",
        "Prod. House",
        "Marketing Ratio",
        "Screens",
        "Genre Pop.",
    ]
    marketing_ratio = input_data.get("marketing_budget", 0) / max(input_data.get("budget", 1), 1)
    values = [
        input_data.get("lead_actor_popularity", 50) / 100,
        input_data.get("supporting_cast_rating", 5) / 10,
        input_data.get("director_experience", 10) / 30,
        input_data.get("production_house_rating", 5) / 10,
        min(marketing_ratio / 0.5, 1.0),
        input_data.get("num_screens", 2000) / 5000,
        input_data.get("genre_popularity", 70) / 100,
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(229, 9, 20, 0.3)",
            line=dict(color=COLORS["primary"], width=2),
            name="Your Movie",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[0.6] * (len(categories) + 1),
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor="rgba(255,255,255,0.05)",
            line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
            name="Industry Avg.",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        title="Feature Profile Radar",
        showlegend=True,
    )
    return _apply_dark_theme(fig)


def create_category_pie_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Pie chart of class probabilities."""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors = [CATEGORY_COLORS.get(l, "#888") for l in labels]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                marker=dict(colors=colors),
                hole=0.4,
                textinfo="label+percent",
                textfont=dict(color="white"),
            )
        ]
    )
    fig.update_layout(title="Category Probability Distribution")
    return _apply_dark_theme(fig)


def create_comparison_chart(
    movie_values: Dict[str, float],
    industry_avg: Dict[str, float],
) -> go.Figure:
    """Grouped bar chart comparing movie vs industry averages."""
    metrics = list(movie_values.keys())
    fig = go.Figure(
        data=[
            go.Bar(name="Your Movie", x=metrics, y=list(movie_values.values()), marker_color=COLORS["primary"]),
            go.Bar(name="Industry Avg.", x=metrics, y=list(industry_avg.values()), marker_color=COLORS["info"]),
        ]
    )
    fig.update_layout(title="Movie vs Industry Comparison", barmode="group")
    return _apply_dark_theme(fig)


def create_confusion_matrix_heatmap(cm: np.ndarray, labels: List[str]) -> go.Figure:
    """Plotly heatmap for confusion matrix."""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        color_continuous_scale=[[0, "rgba(0,0,0,0.5)"], [1, COLORS["primary"]]],
        text_auto=True,
    )
    fig.update_layout(title="Confusion Matrix")
    return _apply_dark_theme(fig)


def create_roc_curves(
    fpr_dict: Dict[str, np.ndarray],
    tpr_dict: Dict[str, np.ndarray],
    auc_dict: Dict[str, float],
) -> go.Figure:
    """Multi-class ROC curve plot."""
    fig = go.Figure()
    for label in fpr_dict:
        fig.add_trace(
            go.Scatter(
                x=fpr_dict[label],
                y=tpr_dict[label],
                mode="lines",
                name=f"{label} (AUC={auc_dict[label]:.3f})",
            )
        )
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="gray"), showlegend=False)
    )
    fig.update_layout(
        title="ROC Curves (One-vs-Rest)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return _apply_dark_theme(fig)


def create_distribution_chart(df: pd.DataFrame, column: str) -> go.Figure:
    """Histogram with KDE-style distribution for a numeric column."""
    fig = px.histogram(
        df,
        x=column,
        nbins=40,
        color_discrete_sequence=[COLORS["primary"]],
        opacity=0.85,
    )
    fig.update_layout(title=f"Distribution: {column.replace('_', ' ').title()}")
    return _apply_dark_theme(fig)


def create_correlation_heatmap(df: pd.DataFrame, numeric_cols: List[str]) -> go.Figure:
    """Correlation matrix heatmap."""
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
    )
    fig.update_layout(title="Feature Correlation Matrix")
    return _apply_dark_theme(fig)


def create_category_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of category counts in dataset."""
    counts = df["category"].value_counts().reset_index()
    counts.columns = ["category", "count"]
    counts["color"] = counts["category"].map(CATEGORY_COLORS)

    fig = go.Figure(
        go.Bar(
            x=counts["category"],
            y=counts["count"],
            marker_color=counts["color"],
        )
    )
    fig.update_layout(title="Movie Category Distribution", xaxis_title="Category", yaxis_title="Count")
    return _apply_dark_theme(fig)
