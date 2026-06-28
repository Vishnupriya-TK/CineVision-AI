"""
CineVision AI - Professional PDF report generation using ReportLab.
"""

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Brand colors
BRAND_RED = colors.HexColor("#e50914")
BRAND_DARK = colors.HexColor("#1a1a2e")
BRAND_GOLD = colors.HexColor("#ffd700")


def _build_styles():
    """Create custom paragraph styles for the PDF."""
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontSize=24,
            textColor=BRAND_RED,
            spaceAfter=12,
            alignment=TA_CENTER,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionHeader",
            parent=styles["Heading2"],
            fontSize=14,
            textColor=BRAND_DARK,
            spaceBefore=16,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTextCustom",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="InsightText",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            leftIndent=12,
            spaceAfter=4,
            textColor=colors.HexColor("#333333"),
        )
    )
    return styles


def generate_pdf_report(
    movie_details: Dict[str, Any],
    prediction: Dict[str, Any],
    insights: List[str],
) -> bytes:
    """
    Generate a professional PDF report with movie details, prediction,
    metrics, and AI insights. Returns PDF as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
    )
    styles = _build_styles()
    story = []

    # Title
    story.append(Paragraph("🎬 CineVision AI", styles["ReportTitle"]))
    story.append(Paragraph("Movie Success Prediction Report", styles["Heading3"]))
    story.append(
        Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}",
            styles["BodyTextCustom"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=2, color=BRAND_RED))
    story.append(Spacer(1, 0.2 * inch))

    # Movie Details
    story.append(Paragraph("Movie Details", styles["SectionHeader"]))
    detail_rows = [
        ["Field", "Value"],
        ["Title", movie_details.get("title", "Untitled Project")],
        ["Genre", movie_details.get("genre", "N/A")],
        ["Language", movie_details.get("language", "N/A")],
        ["Country", movie_details.get("country", "N/A")],
        ["Budget", f"${movie_details.get('budget', 0):,.0f}"],
        ["Marketing Budget", f"${movie_details.get('marketing_budget', 0):,.0f}"],
        ["Runtime", f"{movie_details.get('runtime', 0)} minutes"],
        ["Release Month", str(movie_details.get("release_month", "N/A"))],
        ["Certification", movie_details.get("certification_rating", "N/A")],
        ["Franchise", movie_details.get("franchise", "No")],
        ["OTT Release", movie_details.get("ott_release", "No")],
    ]
    detail_table = Table(detail_rows, colWidths=[2.5 * inch, 3.5 * inch])
    detail_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(detail_table)
    story.append(Spacer(1, 0.3 * inch))

    # Prediction Results
    story.append(Paragraph("Prediction Results", styles["SectionHeader"]))
    pred_rows = [
        ["Metric", "Value"],
        ["Predicted Category", prediction.get("category", "N/A")],
        ["Success Probability", f"{prediction.get('success_probability', 0) * 100:.1f}%"],
        ["Expected Revenue", f"${prediction.get('predicted_revenue', 0):,.0f}"],
        ["ROI", f"{prediction.get('roi', 0):.1f}%"],
        ["Confidence Score", f"{prediction.get('confidence_score', 0) * 100:.1f}%"],
        ["Prediction Time", f"{prediction.get('prediction_time_ms', 0):.1f} ms"],
    ]
    pred_table = Table(pred_rows, colWidths=[2.5 * inch, 3.5 * inch])
    pred_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), BRAND_RED),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff5f5")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(pred_table)
    story.append(Spacer(1, 0.3 * inch))

    # Category Probabilities
    probs = prediction.get("category_probabilities", {})
    if probs:
        story.append(Paragraph("Category Probabilities", styles["SectionHeader"]))
        prob_rows = [["Category", "Probability"]] + [
            [cat, f"{prob * 100:.1f}%"] for cat, prob in sorted(probs.items(), key=lambda x: -x[1])
        ]
        prob_table = Table(prob_rows, colWidths=[2.5 * inch, 3.5 * inch])
        prob_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), BRAND_DARK),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ])
        )
        story.append(prob_table)
        story.append(Spacer(1, 0.3 * inch))

    # AI Insights
    story.append(Paragraph("AI Insights & Recommendations", styles["SectionHeader"]))
    for insight in insights:
        # Strip emoji for PDF compatibility
        clean = insight.encode("ascii", "ignore").decode("ascii").strip()
        if clean:
            story.append(Paragraph(f"• {clean}", styles["InsightText"]))
    story.append(Spacer(1, 0.3 * inch))

    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(
        Paragraph(
            "CineVision AI — Movie Success Predictor | Confidential Analysis Report",
            ParagraphStyle(name="Footer", fontSize=8, textColor=colors.grey, alignment=TA_CENTER),
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
