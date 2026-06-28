"""
CineVision AI - Model Performance Page.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.charts import (
    create_confusion_matrix_heatmap,
    create_feature_importance_chart,
    create_roc_curves,
)
from utils.helpers import METRICS_PATH, load_custom_css, models_exist

st.set_page_config(page_title="Model Performance | CineVision AI", page_icon="📈", layout="wide")
load_custom_css()

st.title("📈 Model Performance")
st.markdown("Detailed evaluation metrics and model comparison results.")

if not models_exist() or not METRICS_PATH.exists():
    st.error(
        "⚠️ **No trained model found.** Run `python train_model.py` to train and evaluate models."
    )
    st.stop()

metrics = joblib.load(METRICS_PATH)

# Header info
st.markdown(f"### 🏆 Best Model: **{metrics.get('best_model_name', 'N/A')}**")

# Core metrics
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
m2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
m3.metric("Recall", f"{metrics.get('recall', 0):.4f}")
m4.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
m5.metric("CV F1 (mean)", f"{metrics.get('cv_mean', 0):.4f}")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Classification Report",
    "🔲 Confusion Matrix",
    "📉 ROC Curves",
    "🌳 Feature Importance",
    "⚖️ Model Comparison",
])

with tab1:
    st.markdown("### Classification Report")
    report = metrics.get("classification_report", {})
    if report:
        rows = []
        for label, values in report.items():
            if isinstance(values, dict):
                rows.append({
                    "Class": label,
                    "Precision": f"{values.get('precision', 0):.4f}",
                    "Recall": f"{values.get('recall', 0):.4f}",
                    "F1-Score": f"{values.get('f1-score', 0):.4f}",
                    "Support": values.get("support", 0),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

with tab2:
    st.markdown("### Confusion Matrix")
    cm = np.array(metrics.get("confusion_matrix", []))
    labels = metrics.get("class_labels", [])
    if cm.size > 0 and labels:
        st.plotly_chart(
            create_confusion_matrix_heatmap(cm, labels),
            use_container_width=True,
        )

with tab3:
    st.markdown("### ROC Curves (One-vs-Rest)")
    fpr_dict = metrics.get("roc_fpr", {})
    tpr_dict = metrics.get("roc_tpr", {})
    auc_dict = metrics.get("roc_auc", {})
    if fpr_dict and tpr_dict:
        import numpy as np
        fpr_np = {k: np.array(v) for k, v in fpr_dict.items()}
        tpr_np = {k: np.array(v) for k, v in tpr_dict.items()}
        st.plotly_chart(create_roc_curves(fpr_np, tpr_np, auc_dict), use_container_width=True)
    else:
        st.info("ROC data not available. Re-run train_model.py to generate.")

with tab4:
    st.markdown("### Feature Importance")
    fi = metrics.get("feature_importance", {})
    if fi:
        names = list(fi.keys())
        values = np.array(list(fi.values()))
        st.plotly_chart(
            create_feature_importance_chart(names, values, top_n=15),
            use_container_width=True,
        )

        st.markdown("#### Full Feature Ranking")
        fi_df = pd.DataFrame(
            sorted(fi.items(), key=lambda x: -x[1]),
            columns=["Feature", "Importance"],
        )
        fi_df["Importance"] = fi_df["Importance"].round(4)
        st.dataframe(fi_df, use_container_width=True, hide_index=True)

with tab5:
    st.markdown("### Algorithm Comparison")
    all_scores = metrics.get("all_model_scores", {})
    if all_scores:
        comparison_rows = []
        for name, scores in all_scores.items():
            comparison_rows.append({
                "Algorithm": name,
                "Accuracy": f"{scores.get('accuracy', 0):.4f}",
                "Precision": f"{scores.get('precision', 0):.4f}",
                "Recall": f"{scores.get('recall', 0):.4f}",
                "F1 Score": f"{scores.get('f1', 0):.4f}",
            })
        comp_df = pd.DataFrame(comparison_rows)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

        # Highlight best
        best = metrics.get("best_model_name", "")
        st.success(f"✅ **{best}** was automatically selected as the best performing model.")

    st.markdown("### Training Details")
    d1, d2, d3 = st.columns(3)
    d1.metric("Training Samples", metrics.get("train_size", "N/A"))
    d2.metric("Test Samples", metrics.get("test_size", "N/A"))
    d3.metric("Dataset Size", metrics.get("dataset_size", "N/A"))

    if metrics.get("has_xgboost"):
        st.info("XGBoost was included in model comparison.")
    else:
        st.info("XGBoost was not available; 4 algorithms were compared.")
